#!Reference

using NNlib: softmax
# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES AND TYPE DEFINITIONS
# ==============================================================================

# Define simple structs for geometric primitives
struct Point2D
    x::Float64
    y::Float64
end

struct Point4D
    x::Float64
    y::Float64
    z::Float64
    t::Float64
end

struct Triangle
    v1::Int
    v2::Int
    v3::Int
end


abstract type Component end
abstract type ComponentModel <: Component end
abstract type ComponentOperator <: Component end

struct NoneComponent <: ComponentModel end

struct IID <: ComponentModel; sigma::UnivariateDistribution; end
struct ICAR <: ComponentModel; sigma::UnivariateDistribution; end
struct Besag <: ComponentModel; sigma::UnivariateDistribution; end

"""
    BYM2 <: ComponentModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and well-identified
parameterization for spatial effects by separating them into a structured (ICAR)
and an unstructured (IID) component.

# Fields
- `rho`: The prior for the mixing parameter, controlling the proportion of
  variance attributed to the structured spatial effect.
- `sigma`: The prior for the overall marginal standard deviation of the
  total spatial effect.
"""
struct BYM2 <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end

struct Leroux <: ComponentModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end
struct SAR <: ComponentModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end
struct DAG <: ComponentModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end



# --- Continuous, Spectral, and Advanced Components ---
struct GP <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; kernel::String; end
struct RFF <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_features::Int; kernel::String; end
struct FITC <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct SVGP <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Nystrom <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct SPDE <: ComponentModel; sigma::UnivariateDistribution; kappa::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; end
struct FFT <: ComponentModel; sigma::UnivariateDistribution; nbins::Int; kernel::String; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; end
struct Warp <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_features::Int; kernel::String; end
struct ExponentialDecay <: ComponentModel; sigma::UnivariateDistribution; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; end
struct Kriging <: ComponentModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; kernel::String; end

# --- Specialized & Basis Manifolds ---
struct Wavelet <: ComponentModel; family::Symbol; nbins::Int; sigma::UnivariateDistribution; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; end
struct Hyperbolic <: ComponentModel; curvature::Real; sigma::UnivariateDistribution; end

struct BCGN <: ComponentModel; sigma::UnivariateDistribution; bipartite_adj::AbstractMatrix; end
struct NetworkFlow <: ComponentModel; sigma::UnivariateDistribution; adjacency_matrix::AbstractMatrix; flow_direction::Symbol; end

"""
    LocalAdaptive <: ComponentModel

A component for localized spatial effects. It combines a Leroux-style precision matrix
with cluster-specific means, allowing the spatial field to have different average
levels in different regions of the domain.
"""
struct LocalAdaptive <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end


 
struct Eigen <: ComponentModel
    n_vars::Int
    n_factors::Int
    pca_sd::UnivariateDistribution
    pdef_sd::UnivariateDistribution
    ltri_indices::Vector{Int}
end
struct Moran <: ComponentModel; sigma::UnivariateDistribution; end
struct Spherical <: ComponentModel; sigma::UnivariateDistribution; range::UnivariateDistribution; end
struct Barycentric <: ComponentModel; sigma::UnivariateDistribution; end
struct Mosaic <: ComponentModel; sigma::UnivariateDistribution; n_regions::Int; end
struct TensorProductSmooth <: ComponentModel; sigma::UnivariateDistribution; Q_template::AbstractMatrix; end

struct TPS <: ComponentModel; nbins::Int; sigma::UnivariateDistribution; end
struct BSpline <: ComponentModel; nbins::Int; degree::Int; sigma::UnivariateDistribution; end

struct PSpline <: ComponentModel
    nbins::Int
    degree::Int
    diff_order::Int
    sigma::UnivariateDistribution
end

struct AR1 <: ComponentModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end

struct AR2 <: ComponentModel
    rho1::UnivariateDistribution
    rho2::UnivariateDistribution
    sigma::UnivariateDistribution
end

struct RW1 <: ComponentModel; sigma::UnivariateDistribution; end
struct RW2 <: ComponentModel; sigma::UnivariateDistribution; end

"""
    Harmonic <: ComponentModel

A component for modeling periodic effects as a sum of sinusoids.

# Fields
- `nharmonics`: The number of harmonic pairs (sine and cosine) to include.
- `amplitude`: Prior for the amplitudes of the waves.
- `phase`: Prior for the phase shifts of the waves.
- `period`: The period of each wave. Can be a single `Real` or `UnivariateDistribution`
  (if all harmonics share the same period), or a `Vector{<:UnivariateDistribution}`
  to specify an independent prior for each harmonic's period.
"""
struct Harmonic <: ComponentModel
    nharmonics::Int
    amplitude::UnivariateDistribution
    phase::UnivariateDistribution
    period::Union{Real, UnivariateDistribution, Vector{<:UnivariateDistribution}}
end


struct Cyclic <: ComponentModel; period::Int; sigma::UnivariateDistribution; end

struct ST_I <: ComponentModel; sigma::UnivariateDistribution; end
struct ST_II <: ComponentModel; sigma::UnivariateDistribution; end
struct ST_III <: ComponentModel; sigma::UnivariateDistribution; end
struct ST_IV <: ComponentModel; sigma::UnivariateDistribution; end

struct CustomComponent <: ComponentModel
    code_fragment::String
    params::Dict{Symbol, Any}
end

struct DynamicsComponent <: ComponentModel; model::String; params::Dict{Symbol, Any}; end


struct ComposedComponent <: ComponentOperator; components::Vector{Component}; operator::Symbol; end
struct SVCComponent <: ComponentOperator
    covariate::Symbol
    model::ComponentModel
end

"""
    MixedComponent <: ComponentOperator

Represents a random effect (intercept or slope) for a specified grouping variable.
The `lhs` field is now a `Vector{String}` to correctly handle multiple correlated
effects (e.g., `(1 + cov1 | group)`), fixing a bug where it was treated as a single string.
"""
struct MixedComponent <: ComponentOperator
    group_var::Symbol
    lhs::Vector{String}
    model::ComponentModel
end

abstract type ComponentSupervisor <: Component end
struct NestedComponent <: ComponentSupervisor
    var::Symbol
    formula::String
    data_source::Symbol
end


# SVAR allows the temporal correlation to vary across space
struct SVAR <: ComponentModel
    rho_spatial::ComponentModel # The model for the spatial distribution of rho
    sigma::UnivariateDistribution
end

"""
    TVCComponent <: ComponentOperator

Represents a Temporally Varying Coefficient model, created by the syntax:
`covariate |> random(time, model=...)`. It links a covariate to a temporal model.
"""
struct TVCComponent <: ComponentOperator
    covariate::Symbol
    model::ComponentModel
end

  
# AdaptiveSmooth learns a coordinate transformation via a simple MLP before smoothing
struct AdaptiveSmooth <: ComponentModel
    hidden_dim::Int
    nbins::Int
    sigma::UnivariateDistribution
end


# Threshold Autoregressive (TAR): Implements regime-switching temporal dynamics where model parameters depend on a covariate threshold logic.
struct TAR <: ComponentModel
    threshold_var::Symbol
    rho_regimes::Vector{UnivariateDistribution}
    sigma_regimes::Vector{UnivariateDistribution}
end

# Define LGCP Struct
# Rationale: LGCP models point patterns by assuming the intensity function lambda(s) 
# is a realization of a Log-Gaussian process: log(lambda(s)) = Z(s).
struct LGCP <: ComponentModel
    model::ComponentModel
    sigma::UnivariateDistribution
    inner_model_node::NamedTuple 
end



"""
    LogGammaCoxProcess <: ComponentModel

A component for Log-Gamma Cox Processes. It models point patterns by assuming the
intensity function Λ(s) is a realization of a Gamma-distributed random field.

# Fields
- `model::ComponentModel`: The inner spatial model for the latent field (e.g., ICAR).
- `shape::UnivariateDistribution`: The prior for the shape parameter of the Gamma distribution.
"""
struct LogGammaCoxProcess <: ComponentModel
    model::ComponentModel
    shape::UnivariateDistribution
end


"""
    ShotNoiseCoxProcess <: ComponentModel

A component for Shot-Noise Cox Processes. It models point patterns by assuming the
intensity function is a sum of kernel functions centered at random "parent" points.

# Fields
- `n_parents`: The number of parent points, which can be a fixed integer or a distribution.
- `kernel`: The name of the kernel function for the shots (e.g., "se").
- `lengthscale`: The prior for the kernel's lengthscale (bandwidth).
- `amplitude`: The prior for the amplitude of each shot.
"""
struct ShotNoiseCoxProcess <: ComponentModel
    n_parents::Union{Int, UnivariateDistribution}
    kernel::String
    lengthscale::UnivariateDistribution
    amplitude::UnivariateDistribution
end


struct Kriging <: ComponentModel
    lengthscale::UnivariateDistribution
    sigma::UnivariateDistribution
    kernel::String
end


"""
    NonStationaryVariance <: ComponentOperator

A dedicated component representing a non-stationary variance model, created by
the composition operator (`∘`), e.g., `random(structure=:spatial) ∘ random(structure=:smooth)`.

This component models a process where the variance of a `base_model` (typically spatial)
is modulated by a `modifier_model` (typically a smoother).

# Fields
- `base_model::ComponentModel`: The underlying component whose variance is being modulated (e.g., `ICAR`).
- `modifier_model::ComponentModel`: The component that defines the non-stationary (log) variance field (e.g., `PSpline`).
"""
struct NonStationaryVariance <: ComponentOperator
    base_model::ComponentModel
    modifier_model::ComponentModel
end


 
abstract type AbstractModelArchitecture end
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end


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
struct DirichletMultinomialFamily <: AbstractBSTM_Family end
struct OrdinalFamily <: AbstractBSTM_Family end

abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end


abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end
struct LeftCensored <: AbstractCensoringState end
struct RightCensored <: AbstractCensoringState end
struct IntervalCensored <: AbstractCensoringState end


struct bstm_Likelihood{F, Z, C, W, P, R, S, T, TR, TL, TU, HT, EX} <: ContinuousMultivariateDistribution
    family::F
    y_obs::TR
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    phi_hurdle::P
    r_nb::R
    sigma_y::S
    trial::T
    censor_lower::TL
    censor_upper::TU
    hurdle::HT
    extra_params::EX
end


# ==============================================================================
# SECTION 2: CONSTANTS, REGISTRIES, AND OPERATOR OVERLOADS
# ==============================================================================

const BSTM_MODULE_KEYWORDS = Set([ 
    :intercept, :fixed, :mixed, :random, :nested, :eigen, :dynamics, :pointprocess, :custom,
    :zscore, :log, :center, :scale
]);
 


const TRANSFORMATION_FUNCTIONS = Set([:zscore, :log, :center, :scale])


"""
    apply_transformation(func_name::Symbol, data_vector::AbstractVector)

Applies a specified data transformation to a vector.

# Arguments
- `func_name::Symbol`: The name of the transformation (e.g., `:zscore`).
- `data_vector::AbstractVector`: The input data column.

# Returns
- A new vector with the transformation applied.
"""
function apply_transformation(func_name::Symbol, data_vector::AbstractVector)
    if func_name == :zscore
        return standardize(ZScoreTransform, data_vector)
    elseif func_name == :log
        # Add offset to handle non-positive values before log transform.
        offset = 1.0 - minimum(data_vector)
        return log.(data_vector .+ offset)
    elseif func_name == :center
        return data_vector .- mean(data_vector)
    elseif func_name == :scale
        # Scales data to the [0, 1] range.
        return standardize(UnitRangeTransform, data_vector)
    else
        @warn "Unknown transformation function ':$func_name'. Returning original data."
        return data_vector
    end
end


"""
    _rewrite_transformations!(nodes::Vector, data::DataFrame)

Recursively traverses the formula's AST, applies data transformations, and rewrites
the AST nodes to use the newly created data columns.
"""
function _rewrite_transformations!(nodes::Vector, data::DataFrame)
    new_nodes = []
    for node in nodes
        if hasproperty(node, :type) && node.type == :operator && node.op == :pipe && length(node.children) == 2
            lhs, rhs = node.children
            if hasproperty(lhs, :module_type) && lhs.module_type in TRANSFORMATION_FUNCTIONS
                # This is a transformation pipe, e.g., `zscore(temp) |> fixed()`
                transform_func_name = lhs.module_type
                vars_to_transform = get(lhs.args, :positional_args, [])
                if isempty(vars_to_transform)
                    @warn "Transformation module '$(transform_func_name)' called without a variable. Skipping."
                    push!(new_nodes, rhs) # Keep the RHS component
                    continue
                end
                var_sym = vars_to_transform[1]

                if !hasproperty(data, var_sym)
                    error("Variable ':$var_sym' for transformation not found in data.")
                end
                
                # Apply transformation and add new column to the DataFrame
                original_data = data[!, var_sym]
                transformed_data = apply_transformation(transform_func_name, original_data)
                
                new_col_name = Symbol("$(var_sym)_$(transform_func_name)")
                counter = 1
                while hasproperty(data, new_col_name)
                    counter += 1
                    new_col_name = Symbol("$(var_sym)_$(transform_func_name)_$(counter)")
                end
                data[!, new_col_name] = transformed_data

                # Rewrite the RHS node to use the new variable
                new_rhs_args = deepcopy(rhs.args)
                new_rhs_args[:positional_args] = [new_col_name]
                new_rhs = (module_type=rhs.module_type, args=new_rhs_args)
                
                # Recursively process the rewritten node
                rewritten_children = _rewrite_transformations!([new_rhs], data)
                append!(new_nodes, rewritten_children)
            else
                # Not a transformation pipe, so process children recursively
                rewritten_children = _rewrite_transformations!(node.children, data)
                push!(new_nodes, (type=:operator, op=:pipe, children=rewritten_children))
            end
        elseif hasproperty(node, :type) && node.type == :operator
            # Handle other operators like ⊗ and ∘
            rewritten_children = _rewrite_transformations!(node.children, data)
            push!(new_nodes, (type=node.type, op=node.op, children=rewritten_children))
        else
            # It's a terminal node (a standard component)
            push!(new_nodes, node)
        end
    end
    return new_nodes
end



"""
    MODEL_TO_STRUCTURE_MAP

The updated mapping from unambiguous model names to their corresponding structure type.
Models like `kriging`, `pspline`, `tps`, `wavelet`, `spherical`, and `barycentric` have
been removed as their structure is context-dependent and better handled by the inference engine.
"""
const MODEL_TO_STRUCTURE_MAP = Dict{Symbol, Symbol}(
    # Spatial Models (primarily for discrete, graph-based structures)
    :icar => :spatial,
    :proper_car => :spatial,
    :besag => :spatial,
    :bym2 => :spatial,
    :leroux => :spatial,
    :sar => :spatial,
    :spde => :spatial,
    :dag => :spatial,
    :moran => :spatial,
    :bcgn => :spatial,
    :networkflow => :spatial,
    :localadaptive => :spatial,
    :mosaic => :spatial,
    :lgcp => :spatial,

    # Temporal Models (for time-series structures)
    :rw1 => :temporal,
    :rw2 => :temporal,
    :ar1 => :temporal,
    :ar2 => :temporal,
    :cyclic => :temporal,
    :harmonic => :temporal,
    :tar => :temporal,

    # Smooth Models (unambiguous cases)
    :tensorproductsmooth => :smooth,

    # Other specific structures
    :svar => :svar
);

"""
    AMBIGUOUS_MODELS

The updated set of model names whose structure is ambiguous and must be inferred from
the input variables. This set has been expanded to include versatile smoothers and
continuous-space models that were previously in the static map.
"""
const AMBIGUOUS_MODELS = Set([
    # General purpose
    :iid, 
    :gp, 
    
    # Spectral & Sparse GP Approximations
    :rff, 
    :fft,
    :fitc, 
    :svgp, 
    :nystrom, 

    # Continuous-space models
    :kriging,
    :spherical,
    :barycentric,
    :exponentialdecay,
    :hyperbolic,
    :warp,

    # Spline and Wavelet smoothers
    :pspline, 
    :bspline, 
    :tps,
    :wavelet
]);


const PC_PRIORS = Dict(
    "sigma" => Exponential(1.0),
    "rho" => Beta(1, 1),
    "rho1" => Normal(0, 0.5),
    "rho2" => Normal(0, 0.5),
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
    "rho1" => Normal(0, 1.0),
    "rho2" => Normal(0, 1.0),
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
    "rho" => Uniform(-1, 1),
    "rho1" => Normal(0, 10),
    "rho2" => Normal(0, 10),
    "lengthscale" => InverseGamma(0.01, 0.01),
    "kappa" => Exponential(10.0),
    "amplitude" => Normal(0, 100),
    "phase" => Uniform(0, 1),
    "pca_sd" => Normal(0, 1e6), 
    "pdef_sd" => Normal(0, 1e6),
    "range" => InverseGamma(0.01, 0.01)
)


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
    "inverse_wishart" => InverseWishartFamily(),
    "dirichlet_multinomial" => DirichletMultinomialFamily(),
    "ordinal" => OrdinalFamily()
)

const STATSMODELS_CONTRASTS = Dict(
    :dummy => StatsModels.DummyCoding(),
    :effects => StatsModels.EffectsCoding(),
    :helmert => StatsModels.HelmertCoding(),
    :treatment => StatsModels.DummyCoding()
)

const COMPONENT_CONSTRUCTORS = Dict{Symbol, Function}(
    :none => (p, params) -> NoneComponent(),
    :iid => (p, params) -> IID(p.sigma),
    :icar => (p, params) -> ICAR(p.sigma),
    :besag => (p, params) -> Besag(p.sigma),
    :bym2 => (p, params) -> BYM2(p.rho, p.sigma),
    :leroux => (p, params) -> Leroux(p.rho, p.sigma),
    :proper_car => (p, params) -> SAR(p.rho, p.sigma), # Alias for SAR
    :sar => (p, params) -> SAR(p.rho, p.sigma),
    :dag => (p, params) -> DAG(p.rho, p.sigma),
    :ar1 => (p, params) -> AR1(p.rho, p.sigma),
    :ar2 => (p, params) -> AR2(p.rho1, p.rho2, p.sigma),  
    :rw1 => (p, params) -> RW1(p.sigma),
    :rw2 => (p, params) -> RW2(p.sigma),
    :fitc => (p, params) -> FITC(p.lengthscale, p.sigma, get(params, :n_inducing, 20), string(get(params, :kernel, "se"))),
    :svgp => (p, params) -> SVGP(p.lengthscale, p.sigma, get(params, :n_inducing, 20), string(get(params, :kernel, "se"))),
    :nystrom => (p, params) -> Nystrom(p.lengthscale, p.sigma, get(params, :n_inducing, 20), string(get(params, :kernel, "se"))),
    :warp => (p, params) -> Warp(p.lengthscale, p.sigma, get(params, :n_features, 20), string(get(params, :kernel, "se"))),
    :hyperbolic => (p, params) -> Hyperbolic(get(params, :curvature, -1.0), p.sigma),
    :decay => (p, params) -> ExponentialDecay(p.sigma, p.lengthscale),
    :gp => (p, params) -> GP(p.lengthscale, p.sigma, string(get(params, :kernel, "se"))),
    :rff => (p, params) -> RFF(p.lengthscale, p.sigma, get(params, :n_features, 20), string(get(params, :kernel, "se"))),
    :fft => (p, params) -> FFT(p.sigma, get(params, :nbins, 20), string(get(params, :kernel, "se")), p.lengthscale),
    :spde => (p, params) -> SPDE(p.sigma, p.kappa),
    :cyclic => (p, params) -> Cyclic(get(params, :period, 12), p.sigma),
    :harmonic => (p, params) -> begin
        nharmonics = get(params, :nharmonics, 1)
        period_param = get(params, :period, 12.0)
        
        if nharmonics > 1
            if period_param isa Real
                error("For `nharmonics > 1`, `period` must be a `Distribution` or a `Vector{<:Distribution}` to be estimated, not a fixed Real value.")
            end
            if period_param isa UnivariateDistribution
                period_param = [period_param for _ in 1:nharmonics]
            end
            if period_param isa Vector && length(period_param) != nharmonics
                error("Length of `period` vector ($(length(period_param))) must match `nharmonics` ($(nharmonics)).")
            end
        else # nharmonics == 1
            if period_param isa Vector
                error("For `nharmonics = 1`, `period` must be a Real or a single UnivariateDistribution, not a Vector.")
            end
        end
        
        Harmonic(nharmonics, p.amplitude, p.phase, period_param)
    end,
    :pspline => (p, params) -> PSpline(get(params, :nbins, 20), get(params, :degree, 3), get(params, :diff_order, 2), p.sigma),
    :bspline => (p, params) -> BSpline(get(params, :nbins, 10), get(params, :degree, 3), p.sigma),
    :tps => (p, params) -> TPS(get(params, :nbins, 20), p.sigma),
    :wavelet => (p, params) -> Wavelet(get(params, :family, :db4), get(params, :nbins, 32), p.sigma, p.lengthscale),
    :eigen => (p, params) -> Eigen(get(params, :n_vars, 0), get(params, :n_factors, 1), p.pca_sd, p.pdef_sd, get(params, :ltri_indices, Int[])),
    :moran => (p, params) -> Moran(p.sigma),
    :spherical => (p, params) -> Spherical(p.sigma, p.range),
    :barycentric => (p, params) -> Barycentric(p.sigma),
    :bcgn => (p, params) -> BCGN(p.sigma, get(params, :bipartite_adj, sparse(zeros(1,1)))),
    :networkflow => (p, params) -> NetworkFlow(p.sigma, get(params, :adjacency_matrix, sparse(zeros(1,1))), get(params, :flow_direction, :bidirectional)),
    :svar => (p, params) -> SVAR(get(params, :rho_spatial_obj, ICAR(p.sigma)), p.sigma),
    :kriging => (p, params) -> Kriging(p.lengthscale, p.sigma, string(get(params, :kernel, "se"))),
    :localadaptive => (p, params) -> LocalAdaptive(p.rho, p.sigma),
    :mosaic => (p, params) -> Mosaic(p.sigma, get(params, :n_regions, 4)),
    :tar => (p, params) -> TAR(
        get(params, :threshold_var, error("TAR model requires a `threshold_var` parameter.")),
        get(params, :rho_regimes, [Beta(1,1), Beta(1,1)]),
        get(params, :sigma_regimes, [Exponential(1.0), Exponential(1.0)])
    ),
    :tensorproductsmooth => (p, params) -> TensorProductSmooth(p.sigma, get(params, :Q_template, sparse(zeros(1,1)))),
    :dynamics => (p, params) -> DynamicsComponent(string(get(params, :model, "none")), params),
    :custom => (p, params) -> CustomComponent(get(params, :code_fragment, ""), get(params, :params, Dict{Symbol, Any}()))
    # Note: LGCP and LogGammaCoxProcess are handled specially in `resolve_technical_primitive`
    # and do not need a direct constructor here.

)

function Base.:|>(m1::Component, m2::Component)
    return ComposedComponent([m1, m2], :pipe)
end

composition(m1::Component, m2::Component) = ComposedComponent([m1, m2], :composition)
∘(m1::Component, m2::Component) = ComposedComponent([m1, m2], :composition)

otimes(m1::Component, m2::Component) = ComposedComponent([m1, m2], :kronecker_product)
⊗(m1::Component, m2::Component) = ComposedComponent([m1, m2], :kronecker_product)

# ==============================================================================
# SECTION 3: FORMULA PARSING ENGINE
# ==============================================================================

"""
    split_terms_at_depth(input::AbstractString, sep::AbstractString)

Splits a string by a separator, but only when the separator is not inside parentheses or brackets.

# Rationale for Update
The original implementation was not robust to multi-byte characters (like `∘` or `⊗`),
as it mixed character-based iteration with byte-based substring indexing, leading to
`StringIndexError`.

This corrected version uses a more robust iteration pattern:
1.  It iterates through the string using byte indices (`ncodeunits`) and advances to the
    next character's start byte using `nextind`.
2.  It uses `startswith` to check for the separator, which is safe for multi-byte strings.
3.  It uses an `IOBuffer` to efficiently build the string for each term.
4.  It correctly handles leading, trailing, and consecutive separators by filtering out
    empty strings from the final result.

This ensures correct parsing of formulas containing special operators.
"""
function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    terms = String[]
    current_term = IOBuffer()
    depth = 0
    
    i = 1
    while i <= ncodeunits(input)
        # Check for separator at current position, but only if not inside parentheses.
        if depth == 0 && startswith(SubString(input, i), sep)
            # Finalize the current term and add it to the list.
            push!(terms, strip(String(take!(current_term))))
            
            # Advance index past the separator.
            # `length(sep)` is character count, which is correct for `nextind`.
            i = nextind(input, i, length(sep))
            continue
        end
        
        # Append character to current term and update depth.
        char = input[i]
        if char == '(' || char == '['
            depth += 1
        elseif char == ')' || char == ']'
            depth -= 1
        end
        
        write(current_term, char)
        
        # Move to the next character's starting byte index.
        i = nextind(input, i)
    end
    
    # Add the final term after the last separator.
    push!(terms, strip(String(take!(current_term))))
    
    # Filter out any empty strings that might result from leading/trailing separators.
    return filter!(!isempty, terms)
end


"""
    _infer_structure_from_args(args::Dict)

Infers the `structure` of a `random()` module call based on its arguments. This function
implements the core logic for consolidating multiple modules into `random()`.

# Inference Logic
1.  If `structure` or the legacy `domain` keyword is explicitly provided, it is used.
2.  If the `model` is a tuple (e.g., `model=(icar, ar1)`), it is inferred as a `:spacetime` interaction.
3.  The `model` is checked against a map of unambiguous models (e.g., `:bym2` is always `:spatial`).
4.  For ambiguous models (e.g., `:gp`, `:iid`), the number and names of the input variables are inspected to infer the structure. For example, two variables are assumed to be spatial coordinates, while a single variable named 'year' implies a temporal structure. The presence of a `W` matrix with an `:iid` model also implies a spatial structure.
5.  If inference fails, an error is thrown, prompting the user to specify the `structure` explicitly.
"""
function _infer_structure_from_args(args::Dict)
    # Purpose: Infers the `structure` of a `random()` module call based on its arguments.
    # Rationale: This version is updated to prioritize the model type over the variable name
    #            when inferring the structure. If a model is explicitly a smoother (e.g., :pspline, :gp),
    #            the structure is correctly set to `:smooth`, regardless of the variable name (e.g., 'year').
    #            This resolves a key ambiguity in the parser.
    # v1.0.1 (2026-07-30)
    # Inputs:
    #   - args: A dictionary of parsed arguments from the module call.
    # Outputs: The inferred structure as a Symbol (e.g., :spatial, :temporal, :smooth).

    # Priority 1: Explicit user-provided structure.
    if haskey(args, :structure)
        return args[:structure]
    end
    if haskey(args, :domain) # Support legacy 'domain' keyword
        return args[:domain]
    end

    model = get(args, :model, nothing)
    vars = get(args, :vars, [])

    # Priority 2: Spacetime interaction syntax `random(s, t, model=(m1, m2))`.
    if model isa Tuple && length(vars) >= 2
        return :spacetime
    end

    # --- FIX: Explicitly handle aliases and missing models ---
    if model == :dag
        return :spatial
    end
    if model == :proper_car # Alias for SAR
        return :spatial
    end
    if model == :decay # Alias for ExponentialDecay
        model = :exponentialdecay # Remap to the canonical name
    end

    # Priority 3: Unambiguous model-to-structure mapping.
    if haskey(MODEL_TO_STRUCTURE_MAP, model)
        return MODEL_TO_STRUCTURE_MAP[model]
    end

    # Priority 4: Ambiguous models requiring variable inspection.
    if model in AMBIGUOUS_MODELS
        # --- NEW: Check for explicit smoother models first ---
        smoother_models = Set([:pspline, :bspline, :tps, :gp, :rff, :fitc, :svgp, :nystrom, :warp, :kriging, :wavelet, :fft, :spherical, :barycentric, :exponentialdecay])
        if model in smoother_models
            return :smooth
        end
        # --- End new check ---

        num_vars = length(vars)
        if num_vars >= 2
            if num_vars == 2
                var1_name = string(vars[1])
                var2_name = string(vars[2])
                
                is_sp_1 = occursin(r"spatial|space|s_idx|region|area|location"i, var1_name)
                is_t_1 = occursin(r"year|time|t_idx|month|day"i, var1_name)
                
                is_sp_2 = occursin(r"spatial|space|s_idx|region|area|location"i, var2_name)
                is_t_2 = occursin(r"year|time|t_idx|month|day"i, var2_name)

                if (is_sp_1 && is_t_2) || (is_t_1 && is_sp_2)
                    return :spacetime
                end
            end
            return :smooth
        elseif num_vars == 1
            var_name = string(first(vars))
            if haskey(args, :W) && model == :iid
                return :spatial
            end
            if occursin(r"year|time|t_idx|month|day"i, var_name)
                return :temporal
            elseif occursin(r"spatial|space|s_idx|region|area|location"i, var_name)
                return :spatial
            else
                return :smooth
            end
        else 
            error("Ambiguous model '$model' used with no variables. Please specify a `structure` (e.g., structure=:spatial).")
        end
    end
    
    # Priority 5: Special keyword arguments like `point_process`.
    if haskey(args, :point_process) && args[:point_process] == :lgcp
        return :spatial
    end

    # Fallback: Inference failed.
    error("Could not infer structure for model '$model'. Please specify `structure` explicitly (e.g., structure=:spatial).")
end



"""
    _parse_arguments_from_expr(args::Vector{Any})

Parses the arguments from a Julia expression (specifically, the `args` field of a `:call` expression)
into a dictionary of keyword arguments and a list of positional arguments.

# Rationale
This function is a core part of the formula parsing engine that works directly with
Julia's Abstract Syntax Tree (AST). It correctly distinguishes between positional
arguments (like variable names) and keyword arguments (like `model=:bym2`). It preserves
expressions (e.g., `prior=Normal(0,1)`) for later evaluation, which is crucial for
allowing complex objects as parameters.

# Arguments
- `args`: A vector of arguments from an `Expr` object.

# Returns
- A `Dict{Symbol, Any}` where keyword arguments are stored by their key, and positional
  arguments are stored under the `:vars` key.
"""
function _parse_arguments_from_expr(args::Vector{Any})
    parsed_args = Dict{Symbol, Any}()
    positional_args = []

    for arg in args
        if arg isa Expr && arg.head == :kw
            # This is a keyword argument, e.g., `model=:bym2`.
            # arg.args is the key (e.g., :model)
            # arg.args[1] is the key (e.g., :model)
            # arg.args[2] is the value (e.g., :bym2 as a QuoteNode or `Normal(0,1)` as an Expr)
            key = arg.args[1]
            value = arg.args[2]
            
            # If the value is a QuoteNode, extract the inner symbol.
            # Otherwise, keep it as is (it could be a literal or another expression).
            if value isa QuoteNode
                parsed_args[key] = value.value # Extract the Symbol from QuoteNode
            else
                parsed_args[key] = value
            end
        else
            # This is a positional argument, e.g., `s_idx`.
            push!(positional_args, arg)
        end
    end

    # Store positional arguments under the conventional `:vars` key.
    if !isempty(positional_args)
        parsed_args[:vars] = positional_args
    end

    return parsed_args
end





"""
    _parse_value(val_str::String)

Parses a string value from a formula argument into an appropriate Julia type.

# Rationale for Update
This function has been updated to be more robust in parsing symbol literals. The
original implementation could, in some contexts, misinterpret a symbol like `:foo`
as the string `":foo"`. The new logic explicitly checks if the string starts with
a colon (`:`) and, if so, correctly converts the rest of the string to a `Symbol`.
This ensures that `model=:bym2` is always parsed as the symbol `:bym2`, resolving
the `KeyError`. It also correctly handles bare words (e.g., `log_offsets=my_offsets_col`)
and other value types.
"""
function _parse_value(val_str::String)
    val_str = strip(val_str)

    # New logic: explicitly handle symbol literals first.
    if startswith(val_str, ":")
        # This handles `:foo` -> Symbol("foo") -> :foo
        return Symbol(val_str[2:end])
    elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str, "\"") && endswith(val_str, "\""))
        # It's a string literal like "foo"
        return String(val_str[2:end-1])
    elseif val_str == "true"
        return true
    elseif val_str == "false"
        return false
    elseif occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", val_str)
        # It's a bare word, treat as a symbol (variable name)
        return Symbol(val_str)
    else
        # It could be a number, or a complex expression like Normal(0,1)
        try
            return Meta.parse(val_str)
        catch
            # If all else fails, it's just a string that wasn't quoted
            return String(val_str)
        end
    end
end


_parse_value(val_str::SubString{String}) = _parse_value(String(val_str))


function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::String)
    # Purpose: A helper to add a parsed argument to either the keyword or positional argument list.
    # Rationale: Centralizes the logic for distinguishing between `key=value` and positional arguments.
    # Assumptions: arg_val is a single argument string.
    # Inputs:
    #   - args_dict: Dictionary for keyword arguments.
    #   - positional_args: Vector for positional arguments.
    #   - arg_val: The argument string to parse.
    # Outputs: None (mutates input collections).
    if contains(arg_val, "=")
        key_val = Base.split(arg_val, "=", limit=2)
        key = Symbol(Base.strip(key_val[1]))
        val_str = String(Base.strip(key_val[2]))
        args_dict[key] = _parse_value(val_str)
    else
        push!(positional_args, _parse_value(arg_val))
    end
end

function _parse_arguments_string(args_str::String)
    # Purpose: Parses the inner content of a module call, e.g., "s_idx, model=bym2".
    # Rationale: Handles a mix of positional and keyword arguments, respecting nested structures.
    # Assumptions: Input is the string between the parentheses of a module call.
    # Inputs:
    #   - args_str: The argument string.
    # Outputs: A dictionary containing parsed keyword arguments and a `:positional_args` key for positional ones.
    args_dict = Dict{Symbol, Any}()
    positional_args = []
    current_arg = ""
    depth = 0

    for char in args_str
        if char == ',' && depth == 0
            arg_val = String(Base.strip(current_arg))
            if !isempty(arg_val)
                _add_parsed_arg!(args_dict, positional_args, arg_val)
            end
            current_arg = ""
        else
            current_arg *= char
            if char == '(' || char == '['; depth += 1; elseif char == ')' || char == ']'; depth -= 1; end
        end
    end

    arg_val = String(Base.strip(current_arg))
    if !isempty(arg_val)
        _add_parsed_arg!(args_dict, positional_args, arg_val)
    end

    if !isempty(positional_args)
        args_dict[:positional_args] = positional_args
    end

    return args_dict
end

function _sanitize_variablename(name::String)
    # Purpose: Sanitizes a string to be a valid Julia variable name.
    # Rationale: The formula parser can generate module keys with characters (e.g., "|", " ")
    #            that are invalid in variable names. This function cleans those keys before
    #            they are used by the code generator to create variable names for latent effects.
    # Inputs:
    #   - name: The raw string from the formula module key.
    # Outputs: A sanitized version of the name suitable for use as a variable.
    s = name
    # Replace pipe, parentheses, spaces, and other invalid characters with an underscore
    s = replace(s, r"[\s\|()\+\*&:]" => "_")
    # Replace kronecker product symbol
    s = replace(s, "⊗" => "_kron_")
    # Replace composition symbol
    s = replace(s, "∘" => "_comp_")
    # Remove any leading or trailing underscores that may result
    s = Base.strip(s, '_')
    # Consolidate sequences of multiple underscores into a single one
    return replace(s, r"__+" => "_")
end

function _parse_single_component_term(term_str::AbstractString)
    # Purpose: Parses a single module call string like "spatial(s_idx, model=bym2)".
    # Rationale: Extracts the module name and its arguments.
    # Assumptions: The input is a single, well-formed module call.
    # Inputs:
    #   - term_str: The module call string.
    # Outputs: A NamedTuple `(module_type, args)`.
    term_str = Base.strip(term_str)
    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", term_str)
    
    if m === nothing
        # This error is thrown when a term without parentheses is incorrectly passed to this function.
        error("Internal Parser Error: _parse_single_component_term was called with a non-module term '$term_str'. This indicates a logic error in the parent parser.")
    end

    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)

    return (module_type = module_name, args = args_dict)
end


"""
    resolve_hyperpriors(...)

# Rationale for Update
The `possible_priors` list has been updated to include `:rho1` and `:rho2`. This
ensures that the function will correctly search for and resolve priors for the
two coefficients of an `AR2` model, following the standard 3-level precedence
(local, global, scheme).
"""
function resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict, scheme::Symbol, calling_mod::Module)
    prior_defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end
    is_anisotropic = get(local_params, :anisotropic, false)
    in_dims = get(local_params, :in_dims, 0)

    possible_priors = [:sigma, :rho, :rho1, :rho2, :lengthscale, :kappa, :amplitude, :phase, :pca_sd, :pdef_sd, :range]

    resolved = Dict{Symbol, Any}()

    for p_sym in possible_priors
        p_base_name = p_sym

        is_ard_param = p_sym in [:lengthscale, :kappa]

        if is_ard_param && is_anisotropic
            if in_dims == 0; error("Cannot resolve anisotropic prior for '$p_sym' because input dimensionality is unknown."); end
            
            prior_val = get(local_params, p_sym, nothing)
            
            if prior_val isa Expr && prior_val.head == :vect
                # Case: lengthscale=[Normal(0,1), Normal(0,1)]
                resolved_priors = [Core.eval(calling_mod, p) for p in prior_val.args]
                if length(resolved_priors) != in_dims; error("Anisotropic prior vector for '$p_sym' has length $(length(resolved_priors)), but expected $in_dims."); end
                resolved[p_sym] = resolved_priors
            else
                # Case: lengthscale=Normal(0,1) or default
                single_prior = if !isnothing(prior_val)
                    prior_val isa Expr ? Core.eval(calling_mod, prior_val) : prior_val
                elseif haskey(global_priors, Symbol(model_name, "_", p_sym)); global_priors[Symbol(model_name, "_", p_sym)]
                elseif haskey(global_priors, p_sym); global_priors[p_sym]
                else; prior_defaults[string(p_sym)]; end
                
                if !(single_prior isa Distribution); error("Resolved prior for '$p_sym' is not a Distribution."); end
                resolved[p_sym] = [single_prior for _ in 1:in_dims]
            end
            continue
        end

        if haskey(local_params, p_sym)
            prior_val = local_params[p_sym]
            if prior_val isa Tuple
                resolved[p_sym] = create_pc_prior(p_base_name, prior_val)
            elseif prior_val isa Expr
                try
                    resolved[p_sym] = Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for component '$model_name'. Error: $e")
                end
            else
                resolved[p_sym] = prior_val
            end
            continue
        end

        global_key_model = Symbol(model_name, "_", p_base_name)
        global_key_param = p_base_name
        
        if haskey(global_priors, global_key_model)
            resolved[p_sym] = global_priors[global_key_model]
            continue
        elseif haskey(global_priors, global_key_param)
            resolved[p_sym] = global_priors[global_key_param]
            continue
        elseif haskey(prior_defaults, string(p_base_name))
            resolved[p_sym] = prior_defaults[string(p_base_name)]
        end
    end

    return NamedTuple(resolved)
end


function _is_outermost_grouping_parentheses(s::AbstractString)
    if !startswith(s, "(") || !endswith(s, ")")
        return false
    end
    depth = 0
    # Iterate from the second character to the second-to-last to check the balance
    # of parentheses within the outer pair.
    for i in 2:ncodeunits(s)-1
        char = s[i]
        if char == '('
            depth += 1
        elseif char == ')'
            depth -= 1
        end
        # If depth becomes negative, it means a closing parenthesis appeared before its opening one.
        if depth < 0
            return false
        end
        # If depth returns to 0 before the end of the string, it means the outer
        # parentheses are not grouping the entire expression.
        if depth == 0
            return false
        end
    end
    # The depth should be exactly 0 after checking all internal characters.
    return depth == 0
end



function _parse_rhs_expression(term_str::AbstractString)
    term_str_stripped = Base.strip(term_str)

    # If the expression is wrapped in balanced parentheses, parse the inner content recursively.
    if startswith(term_str_stripped, "(") && endswith(term_str_stripped, ")")
        # Safely get the inner content, respecting multi-byte characters
        inner_content = SubString(term_str_stripped, nextind(term_str_stripped, 1), prevind(term_str_stripped, lastindex(term_str_stripped)))
        
        # Check if the parentheses around the inner content are balanced.
        # This is a simple check to avoid parsing malformed strings.
        depth = 0
        is_balanced = true
        for char in inner_content
            if char == '('; depth += 1; elseif char == ')'; depth -= 1; end
            if depth < 0; is_balanced = false; break; end
        end
        if depth != 0; is_balanced = false; end

        # If the parentheses are balanced, it's a grouped expression.
        # We recurse on the inner content to handle nested parentheses and expressions.
        if is_balanced
            return _parse_rhs_expression(inner_content)
        end
    end

    # Proceed with parsing based on operator precedence.
    parts = split_terms_at_depth(term_str_stripped, " |> ")
    if length(parts) > 1
        return (type=:operator, op=:pipe, children=[_parse_rhs_expression(parts[1]), _parse_rhs_expression(join(parts[2:end], " |> "))])
    end
    parts = split_terms_at_depth(term_str_stripped, " ⊗ ")
    if length(parts) > 1
        return (type=:operator, op=:kronecker_product, children=[_parse_rhs_expression(p) for p in parts])
    end
    parts = split_terms_at_depth(term_str_stripped, " ∘ ")
    if length(parts) > 1
        return (type=:operator, op=:composition, children=[_parse_rhs_expression(p) for p in parts])
    end

    # If no operators are found, parse as a single module or a fixed effect.
    if occursin(r"\(.*\)", term_str_stripped)
        return _parse_single_component_term(term_str_stripped)
    else
        return (module_type = :fixed, args = Dict(:positional_args => [term_str_stripped]))
    end
end



# This new constant maps legacy module names to their corresponding structure type.
const LEGACY_MODULES = Dict(
    :spatial => :spatial,
    :temporal => :temporal,
    :smooth => :smooth,
    :seasonal => :seasonal,
    :spacetime => :spacetime
);

 

function _categorize_rhs_nodes!(nodes, modules, fixed_effects)
    for node in nodes
        is_pp_composition = false
        if hasproperty(node, :type) && node.type == :operator && node.op == :composition && length(node.children) == 2
            outer_node, inner_node = node.children[1], node.children[2]
            if outer_node.module_type == :pointprocess && inner_node.module_type == :random
                is_pp_composition = true
                
                pp_module_type = get(outer_node.args, :model, :lgcp)
                final_params = copy(inner_node.args)
                for (k, v) in outer_node.args
                    if k != :model; final_params[k] = v; end
                end
                
                vars = get(inner_node.args, :positional_args, [])
                new_module_data = (module_type = pp_module_type, args = final_params)
                
                key_parts = [string(pp_module_type), join(string.(vars), "_")]
                raw_key = join(filter(!isempty, key_parts), "_")
                sanitized_base_key = _sanitize_variablename(raw_key)
                module_key = sanitized_base_key
                counter = 1
                while haskey(modules, module_key)
                    counter += 1
                    module_key = "$(sanitized_base_key)_$(counter)"
                end
                modules[module_key] = new_module_data
            end
        end

        if is_pp_composition
            continue
        end

        if hasproperty(node, :type) && node.type == :operator
            # Simplified key generation for composed nodes.
            function _get_simplified_composed_node_key(n)
                if hasproperty(n, :type) && n.type == :operator
                    op_str = string(n.op)
                    child_keys = [_get_simplified_composed_node_key(child) for child in n.children]
                    return join(child_keys, "_$(op_str)_")
                elseif hasproperty(n, :module_type)
                    pos_args = get(n.args, :positional_args, [])
                    return isempty(pos_args) ? string(n.module_type) : join(string.(pos_args), "_")
                else
                    return "unknown"
                end
            end
            raw_key = _get_simplified_composed_node_key(node)
            sanitized_base_key = _sanitize_variablename(raw_key)
            module_key = isempty(sanitized_base_key) ? "composed_$(length(modules)+1)" : sanitized_base_key
            
            final_key = module_key
            counter = 1
            while haskey(modules, final_key)
                counter += 1
                final_key = "$(module_key)_$(counter)"
            end
            modules[final_key] = (module_type = :interact, args = Dict(:operator => node.op, :components => node.children))

        elseif hasproperty(node, :module_type)
            m_type = node.module_type
            if m_type in BSTM_MODULE_KEYWORDS
                local raw_key
                pos_args = get(node.args, :positional_args, [])
                
                if m_type == :mixed
                    # For `mixed(effect | group)`, the key is based on the group variable.
                    if !isempty(pos_args) && pos_args[1] isa Expr && pos_args[1].head == :call && pos_args[1].args[1] == :|
                        group_var_str = string(pos_args[1].args[3])
                        raw_key = group_var_str
                    else
                        raw_key = isempty(pos_args) ? "mixed_unknown" : string(pos_args[1])
                    end
                elseif !isempty(pos_args)
                    # For most modules, the key is based on the variables it acts on.
                    raw_key = join([string(a) for a in pos_args], "_")
                else
                    # Fallback for modules with no positional args (e.g., intercept).
                    raw_key = string(m_type)
                end

                sanitized_base_key = _sanitize_variablename(raw_key)
                module_key = sanitized_base_key
                counter = 1
                while haskey(modules, module_key)
                    counter += 1
                    module_key = "$(sanitized_base_key)_$(counter)"
                end
                modules[module_key] = node
            else
                push!(fixed_effects, string(m_type))
            end
        end
    end
end





function _extract_symbols_from_expr!(sym_set::Set{Symbol}, ex::Expr)
    # Purpose: Recursively extracts all symbols from a Julia expression.
    # Rationale: Used to find all variable names mentioned in the formula to check for their presence in the data.
    # Assumptions: Input is a valid Julia expression.
    # Inputs:
    #   - sym_set: A set to store the found symbols.
    #   - ex: The expression to traverse.
    # Outputs: None (mutates `sym_set`).
    for arg in ex.args
        if arg isa Symbol
            push!(sym_set, arg)
        elseif arg isa Expr
            _extract_symbols_from_expr!(sym_set, arg)
        end
    end
end

function _parse_lhs_term(term::String)
    # Purpose: Parses a single term from the left-hand side of the formula.
    # Rationale: Handles both `likelihood()` blocks and bare outcome variables,
    #            including multiple outcomes specified with `+`.
    # Inputs:
    #   - term: A string representing one part of the LHS.
    # Outputs: A vector of outcome specification dictionaries.
    term = Base.strip(term)
    m = match(r"likelihood\((.*)\)", term)
    specs = Dict{Symbol, Any}[]
    if !isnothing(m)
        inner_content = m.captures[1]
        args = split_terms_at_depth(inner_content, ",")
        if isempty(args); return specs; end
        outcome_var_str = Base.strip(args[1])
        params_str = join(args[2:end], ",")
        params = _parse_arguments_string(params_str)
        for ov in [Base.strip(s) for s in Base.split(outcome_var_str, '+')]; push!(specs, Dict(:var => ov, :params => params)); end
    else
        for ov in [Base.strip(s) for s in Base.split(term, '+')]; push!(specs, Dict(:var => ov, :params => Dict())); end
    end
    return specs
end


"""
    decompose_bstm_formula(formula_str::String, data::DataFrame)

This updated version now accepts the `data` DataFrame to enable in-place modification
during the new AST rewriting step for data transformations.
"""
function decompose_bstm_formula(formula_str::String, data::DataFrame)
    parts = Base.split(formula_str, "~")
    lhs_str = Base.strip(parts[1])
    rhs_str = Base.strip(parts[2])
    outcome_specs = vcat([_parse_lhs_term(term) for term in split_terms_at_depth(lhs_str, "+")]...)

    rhs_normalized = replace(rhs_str, r"\s*-\s*" => " + -")
    rhs_terms = split_terms_at_depth(rhs_normalized, " + ")
    
    has_intercept = !("0" in rhs_terms || "-1" in rhs_terms)
    intercept_prior = nothing
    module_terms = String[]
    intercept_module_found = false

    for term in rhs_terms
        term_stripped = Base.strip(term)
        if term_stripped == "0" || term_stripped == "-1" || term_stripped == "1"
            continue
        elseif startswith(term_stripped, "intercept(")
            if !intercept_module_found
                intercept_module_found = true
                intercept_data = _parse_single_component_term(term_stripped)
                has_intercept = get(intercept_data.args, :positional_args, [true])[1] != false
                if haskey(intercept_data.args, :prior); intercept_prior = intercept_data.args[:prior]; end
            end
        else
            push!(module_terms, term_stripped)
        end
    end

    top_level_nodes = [_parse_rhs_expression(term) for term in module_terms]
    
    # New Step: Rewrite the AST to handle transformations, modifying `data` in place.
    rewritten_nodes = _rewrite_transformations!(top_level_nodes, data)

    modules = Dict{String, Any}()
    fixed_effects = String[]
    _categorize_rhs_nodes!(rewritten_nodes, modules, fixed_effects)
    
    return (outcomes=outcome_specs, modules=modules, fixed_effects=unique(fixed_effects), has_intercept=has_intercept, intercept_prior=intercept_prior)
end


# ==============================================================================
# SECTION 4: MODEL CONFIGURATION ENGINE
# ==============================================================================


function _precompute_likelihood_params!(M::Dict)
    # Purpose: Ensures all observation-level likelihood parameters are consistently formatted as matrices.
    # Rationale: This function standardizes all likelihood parameters (per-observation and per-outcome)
    #            into matrices of size `(N, K)` (observations x outcomes). This simplifies the downstream
    #            code generator, which can then access these parameters with simple indexing `[i, k]`
    #            inside the model's observation loop, avoiding inefficient runtime conditional checks.
    # v1.0.0 (2026-07-18) - Aligned with new parser logic for scalar-per-outcome parameters.
    # Assumptions: M[:y_N] (number of observations) and M[:outcomes_N] (number of outcomes) are set.
    # Inputs:
    #   - M: The model configuration dictionary, which is mutated by this function.
    # Outputs: None.

    N = M[:y_N]
    K = M[:outcomes_N]

    param_defaults = [
        (key=:censor_lower, default=-Inf, is_scalar_per_outcome=true),
        (key=:censor_upper, default=Inf, is_scalar_per_outcome=true),
        (key=:hurdle, default=-Inf, is_scalar_per_outcome=true),
        (key=:trials, default=1, is_scalar_per_outcome=false),
        (key=:weights, default=1.0, is_scalar_per_outcome=false),
        (key=:log_offsets, default=0.0, is_scalar_per_outcome=false)
    ]

    for spec in param_defaults
        key = spec.key
        default_val = spec.default
        is_scalar_per_outcome = spec.is_scalar_per_outcome

        final_matrix = Matrix{typeof(default_val)}(undef, N, K)

        if !haskey(M, key)
            fill!(final_matrix, default_val)
        else
            val = M[key]
            if is_scalar_per_outcome
                if val isa Real
                    fill!(final_matrix, val)
                elseif val isa AbstractVector && length(val) == K # Multivariate case
                    final_matrix = repeat(val', N, 1)
                else
                    @warn "Scalar-per-outcome parameter `:$key` has unexpected type or dimensions. Using default."
                    fill!(final_matrix, default_val)
                end
            else # Per-observation parameter
                if val isa Real
                    fill!(final_matrix, val)
                elseif val isa AbstractVector && length(val) == N
                    final_matrix = repeat(val, 1, K)
                elseif val isa AbstractMatrix && size(val) == (N, K)
                    final_matrix = val
                else
                    @warn "Per-observation parameter `:$key` has incorrect dimensions. Expected scalar, vector of length $N, or matrix ($N, $K). Using default."
                    fill!(final_matrix, default_val)
                end
            end
        end
        M[key] = final_matrix
    end
end


function bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)
    # Purpose: Constructs the complete model configuration from a formula and data.
    # Rationale: This version corrects a FieldError by changing `M.likelihood_specs` to
    #            `M[:likelihood_specs]`, as `M` is a Dictionary at this stage of execution.
    # v1.0.5 (2026-07-31)

    df_processed = deepcopy(data)
    decomposed_formula = decompose_bstm_formula(formula, df_processed)

    all_vars = Set{Symbol}()
    for out_spec in decomposed_formula.outcomes
        push!(all_vars, Symbol(out_spec[:var]))
        for key in [:log_offsets, :offsets, :weights, :trials, :censor_lower, :censor_upper, :threshold_var]
            if haskey(out_spec[:params], key)
                val = out_spec[:params][key]
                if val isa Symbol; push!(all_vars, val); end
            end
        end
    end
    for fe in decomposed_formula.fixed_effects; push!(all_vars, Symbol(fe)); end
    for (_, mod_data) in pairs(decomposed_formula.modules)
        if haskey(mod_data.args, :positional_args)
            for arg in mod_data.args[:positional_args]
                if arg isa Symbol; push!(all_vars, arg);
                elseif arg isa Expr; _extract_symbols_from_expr!(all_vars, arg); end
            end
        end
        if haskey(mod_data.args, :threshold_var)
            val = mod_data.args[:threshold_var]
            if val isa Symbol; push!(all_vars, val); end
        end
    end

    df_processed = deepcopy(data)
    vars_to_categorize = Set{Symbol}()
    for (_, mod_data_nt) in decomposed_formula.modules
        if mod_data_nt.module_type == :fixed
            if haskey(mod_data_nt.args, :positional_args)
                for var_sym in mod_data_nt.args[:positional_args]
                    if var_sym isa Symbol
                        if haskey(mod_data_nt.args, :contrast) || get(mod_data_nt.args, :model, nothing) == :categorical
                            push!(vars_to_categorize, var_sym)
                        end
                    end
                end
            end
        end
    end
    for var_sym in vars_to_categorize
        if hasproperty(df_processed, var_sym) && !(eltype(df_processed[!, var_sym]) <: CategoricalValue)
            df_processed[!, var_sym] = categorical(df_processed[!, var_sym])
        end
    end

    valid_vars_in_data = filter(v -> hasproperty(df_processed, v), all_vars)
    if !isempty(valid_vars_in_data)
        filtered_data = DataFrame(df_processed[completecases(df_processed, collect(valid_vars_in_data)), :])
        if nrow(filtered_data) < nrow(data)
            @warn "Missing values detected. $(nrow(data) - nrow(filtered_data)) rows were removed."
        end
    else
        filtered_data = df_processed
        if !isempty(all_vars)
            @warn "None of the variables specified in the formula were found in the data: $(all_vars). Proceeding without removing rows for missing data."
        end
    end

    M = _initialize_config(filtered_data, merge(Dict(kwargs), Dict(:calling_module => calling_module)))
    M[:formula] = formula
    
    _process_lhs!(M, decomposed_formula.outcomes)
    
    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"
    if is_multivariate
        for (key, mod_data_nt) in decomposed_formula.modules
            model_name = get(mod_data_nt.args, :model, :none)
            if mod_data_nt.module_type == :dynamics && model_name in [:leslie_matrix, :delay_difference, :generalized_lotka_volterra]
                M[:is_multivariate_dynamics] = true
                M[:multivariate_dynamics_key] = key
                break
            end
        end
    end

    non_prop_effects = Symbol[]
    # FIX: Changed M.likelihood_specs to M[:likelihood_specs]
    is_ordinal = any(spec -> string(get(spec, :family, "")) == "ordinal", M[:likelihood_specs])
    if is_ordinal
        for (_, mod_data_nt) in decomposed_formula.modules
            if mod_data_nt.module_type == :fixed && get(mod_data_nt.args, :non_proportional_effects, false) == true
                vars = get(mod_data_nt.args, :positional_args, [])
                for v in vars
                    if v isa Symbol
                        push!(non_prop_effects, v)
                    end
                end
            end
        end
    end
    M[:non_proportional_effects] = unique(non_prop_effects)

    _precompute_likelihood_params!(M)

    M[:N_cov] = 0
    M[:add_intercept] = decomposed_formula.has_intercept
    if !isnothing(decomposed_formula.intercept_prior)
        prior_val = decomposed_formula.intercept_prior
        if prior_val isa Expr
            try; M[:intercept_prior] = Core.eval(calling_module, prior_val);
            catch e; error("Could not evaluate `prior` argument `$(prior_val)` in intercept() module. Error: $e"); end
        else; M[:intercept_prior] = prior_val; end
    end

    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        mod_data_dict = Dict(:type => mod_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)
        
        effective_processor_key = mod_type
        
        if mod_type == :random
            params = mod_data_dict[:params]
            if !haskey(params, :structure)
                args_for_inference = copy(params)
                args_for_inference[:vars] = get(mod_data_dict, :variables, [])
                params[:structure] = _infer_structure_from_args(args_for_inference)
            end
            mod_data_dict[:type] = params[:structure]
        end

        processor! = get(MODULE_PROCESSORS, effective_processor_key, nothing)
        
        create_component_for_module = true
        if !isnothing(processor!)
            create_component_for_module = processor!(M, mod_data_dict, M, M[:hyperpriors])
        end
        
        if mod_type == :fixed || !create_component_for_module
            continue
        end
        
        component_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
        component_spec_built = build_model(component_obj, M, mod_data_dict)
        
        spec = (
            key=Symbol(key), 
            structure=mod_data_dict[:type], 
            var=join(string.(mod_data_dict[:variables]), "_"), 
            component_obj=component_obj, 
            params=mod_data_dict[:params], 
            Q_template=component_spec_built.Q_template, 
            scaling_factor=component_spec_built.scaling_factor,
            hyper=component_spec_built.hyper
        )
        push!(M[:components], spec)
    end

    _process_fixed_effects!(M, unique(decomposed_formula.fixed_effects))
    _process_fixed_effects_priors!(M)
    _precompute_static_components!(M)
    _finalize_config!(M)
    return NamedTuple(M)
end






function generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates a standardized set of variable names for a given component.
    # Rationale: This version is updated to include the full set of biological parameters
    #            used by the `dynamics()` module (e.g., `r`, `K`, `q`, `alpha`). This resolves
    #            a `FieldError` that occurred because the code generator for dynamics models
    #            was attempting to access variable names that were not being created.
    # v1.0.1 (2026-07-31)
    # Inputs: Standard code generation arguments.
    # Outputs: A NamedTuple containing all possible variable names for a component.

    base_key = string(spec.key)
    full_key = isempty(prefix) ? base_key : "$(prefix)_$(base_key)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    hyperparam_suffix = (is_multivariate && !is_shared) ? "_$(outcome_idx)" : ""
    latent_field_suffix = is_multivariate ? "_$(outcome_idx)" : ""

    names = Dict{Symbol, Symbol}()
    
    # --- Standard Hyperparameters ---
    names[:sigma] = Symbol("sigma_$(full_key)$(hyperparam_suffix)")
    names[:rho] = Symbol("rho_$(full_key)$(hyperparam_suffix)")
    names[:rho1] = Symbol("rho1_$(full_key)$(hyperparam_suffix)")
    names[:rho2] = Symbol("rho2_$(full_key)$(hyperparam_suffix)")
    names[:kappa] = Symbol("kappa_$(full_key)$(hyperparam_suffix)")
    names[:ls] = Symbol("ls_$(full_key)$(hyperparam_suffix)")
    names[:range] = Symbol("range_$(full_key)$(hyperparam_suffix)")
    names[:period] = Symbol("period_$(full_key)$(hyperparam_suffix)")
    names[:amplitude] = Symbol("amplitude_$(full_key)$(hyperparam_suffix)")
    names[:phase] = Symbol("phase_$(full_key)$(hyperparam_suffix)")
    names[:velocity] = Symbol("velocity_$(full_key)$(hyperparam_suffix)")
    names[:diffusion] = Symbol("diffusion_$(full_key)$(hyperparam_suffix)")
    names[:pca_sd] = Symbol("pca_sd_$(full_key)$(hyperparam_suffix)")
    names[:pdef_sd] = Symbol("pdef_sd_$(full_key)$(hyperparam_suffix)")
    names[:L_corr] = Symbol("L_corr_$(full_key)$(hyperparam_suffix)")
    names[:sigma_effects] = Symbol("sigma_effects_$(full_key)$(hyperparam_suffix)")

    # --- Biological Model Hyperparameters ---
    names[:r] = Symbol("r_$(full_key)$(hyperparam_suffix)")
    names[:K] = Symbol("K_$(full_key)$(hyperparam_suffix)")
    names[:q] = Symbol("q_$(full_key)$(hyperparam_suffix)")
    names[:M_nat] = Symbol("M_nat_$(full_key)$(hyperparam_suffix)")
    names[:alpha] = Symbol("alpha_$(full_key)$(hyperparam_suffix)")
    names[:beta] = Symbol("beta_$(full_key)$(hyperparam_suffix)")
    names[:gamma] = Symbol("gamma_$(full_key)$(hyperparam_suffix)")
    names[:delta] = Symbol("delta_$(full_key)$(hyperparam_suffix)")

    # --- Latent Fields and Innovations ---
    names[:raw] = Symbol("$(full_key)_raw$(latent_field_suffix)")
    names[:innov] = Symbol("$(full_key)_innov$(latent_field_suffix)")
    names[:latent] = Symbol("$(full_key)_latent$(latent_field_suffix)")
    names[:rho_field] = Symbol("$(full_key)_rho_field$(latent_field_suffix)")
    names[:struct] = Symbol("$(full_key)_struct$(latent_field_suffix)")
    names[:iid] = Symbol("$(full_key)_iid$(latent_field_suffix)")
    names[:beta_cos] = Symbol("$(full_key)_beta_cos$(latent_field_suffix)")
    names[:beta_sin] = Symbol("$(full_key)_beta_sin$(latent_field_suffix)")

    return NamedTuple(names)
end







"""
    _generate_component_code_fragments(m::MixedComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for mixed effects models, handling both simple (uncorrelated)
and complex (correlated) random effects.

# Rationale for Update
This function has been updated with more detailed comments to clarify the mathematical
logic behind the reconstruction of correlated random effects. The implementation uses
a non-centered parameterization where the final effects matrix `β` is constructed from
standard normal innovations `Z` and the Cholesky factors of the group-level (`L_g`) and
effect-level (`L_e`) covariance matrices.

The core transformation is `vec(β) = (L_e ⊗ L_g) * vec(Z)`. For computational efficiency,
this is implemented with matrix operations as `β = (L_g')⁻¹ * Z * L_e'`. This approach
correctly induces the desired `Σ_g ⊗ Σ_e` covariance structure while being efficient
within the MCMC sampler. The logic for pre-computing the Cholesky factor for static
group structures (e.g., ICAR) is also correctly handled to improve performance.
"""
function _generate_component_code_fragments(m::MixedComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing code fragments for mixed effects models, handling both simple (uncorrelated)
    #          and complex (correlated) random effects.
    # Rationale: This function has been updated to robustly handle cases where the formula parser
    #            incorrectly combines multiple effects (e.g., "1 + cov") into a single string.
    #            It also corrects the logic for generating the inverse Cholesky factor for the group-level
    #            effect, prioritizing the efficient identity matrix for IID models to resolve a CanonicalIndexError.
    # v1.0.2 (2026-07-30)
    # Inputs: Standard code generation arguments.
    # Outputs: A NamedTuple containing `priors` and `update` code strings.
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    inner_model = m.model
    group_var = m.group_var 
    lhs_effects_raw = m.lhs
    
    # Handle cases where multiple terms are parsed as a single string
    lhs_effects = vcat([Base.split(s, r"\s*\+\s*") for s in lhs_effects_raw]...)
    
    k_effects = length(lhs_effects)
    n_groups = get(spec.params, :n_cat, 0) 

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "mixed_idx_$(group_var)"

    if k_effects == 1
        # Logic for a single, uncorrelated random effect (e.g., random intercept only).
        inner_frags = _generate_component_code_fragments(inner_model, spec, arch, outcome_idx, prefix=prefix)
        update_inner_cleaned = replace(inner_frags.update, Regex("\\s*$(eta_target)\\s*\\.\\+=\\s*.*") => "")
        lhs_str = lhs_effects[1]
        is_intercept = (lhs_str == "1" || lhs_str == "intercept()" || lhs_str == "(Intercept)")

        application_code = if is_intercept
            "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        else
            "$(eta_target) .+= M.data[!, :$(Symbol(lhs_str))] .* view($(v.latent), M.$(index_var))"
        end

        update_str = """
        begin
            # Mixed Effect Logic (Single): $(lhs_str) | $(group_var)
            $(update_inner_cleaned)
            $(application_code)
        end
        """
        return (priors=inner_frags.priors, update=update_str)
    else
        # Logic for multiple, correlated random effects (e.g., random intercept and slope).
        priors_acc = String[]
        push!(priors_acc, "$(v.L_corr) ~ NamedDist(LKJCholesky($(k_effects), 1.0), :$(v.L_corr))")
        push!(priors_acc, "$(v.sigma_effects) ~ NamedDist(filldist(Exponential(1.0), $(k_effects)), :$(v.sigma_effects))")
        push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_groups * k_effects)), I), :$(v.raw))")

        # --- FIX: Prioritize IID model check to avoid CanonicalIndexError ---
        # The IID model is static, but its Cholesky factor (from a sparse identity matrix)
        # does not support the `F.U \ I` operation. By checking for IID first, we
        # ensure the correct and efficient identity matrix is used for L_groups_inv_t,
        # resolving the error.
        group_chol_logic = if inner_model isa IID || inner_model isa NoneComponent
            "local L_groups_inv_t_$(spec.key) = sparse(I, $(n_groups), $(n_groups))"
        elseif get(spec, :is_static, false)
            "local F_groups_$(spec.key) = spec_registry[\"$(spec.key)\"].cholesky_factor\nlocal L_groups_inv_t_$(spec.key) = F_groups_$(spec.key).U \\ I"
        else
            "local Q_groups_$(spec.key) = spec_registry[\"$(spec.key)\"].Q_template\nlocal F_groups_$(spec.key) = cholesky(Symmetric(Q_groups_$(spec.key) + noise * I))\nlocal L_groups_inv_t_$(spec.key) = F_groups_$(spec.key).U \\ I"
        end

        application_loop_acc = String[]
        for i in 1:k_effects
            term = lhs_effects[i]
            is_int = (term == "1" || term == "intercept()" || term == "(Intercept)")

            # Generate the correct application logic for each term (intercept or slope).
            term_application = if is_int
                "$(eta_target) .+= view(effects_matrix_$(spec.key), M.$(index_var), $(i))"
            else
                "$(eta_target) .+= M.data[!, :$(Symbol(term))] .* view(effects_matrix_$(spec.key), M.$(index_var), $(i))"
            end
            push!(application_loop_acc, "# Effect Component: $(term)")
            push!(application_loop_acc, term_application)
        end

        update_str = """
        begin
            # Correlated Mixed Effects Construction for $(group_var)
            local L_effects_t_$(spec.key) = ($(v.L_corr).L' * Diagonal($(v.sigma_effects)))
            $(group_chol_logic)
            local innovations_matrix_$(spec.key) = reshape($(v.raw), $(n_groups), $(k_effects))
            local effects_matrix_$(spec.key) = L_groups_inv_t_$(spec.key) * innovations_matrix_$(spec.key) * L_effects_t_$(spec.key)
            $(join(application_loop_acc, "\n        "))
        end
        """
        return (priors=join(priors_acc, "\n    "), update=update_str)
    end
end



function _generate_component_code_fragments(m::SVCComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing code for Spatially Varying Coefficients (SVC).
    # Rationale: Ensures covariates are correctly indexed from the DataFrame while 
    #            preventing invalid indexing if an intercept term is passed as a covariate.
    # v1.0.0 (2026-07-20) - Corrected intercept handling.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    cov_var = m.covariate
    
    # Generate base spatial field logic
    inner_frags = _generate_component_code_fragments(inner_model, spec, arch, outcome_idx, prefix=prefix)
    priors_str = inner_frags.priors
    update_inner = inner_frags.update

    # Remove the standard effect application from the inner model
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_app_regex = Regex("\\s*$(eta_target)\\s*\\.\\+=\\s*.*")
    update_inner_cleaned = replace(update_inner, effect_app_regex => "")
    
    # Application Logic: Check if covariate is an intercept indicator
    is_intercept = (string(cov_var) == "1" || string(cov_var) == "intercept()")
    
    application_code = if is_intercept
        "$(eta_target) .+= view($(v.latent), M.s_idx)"
    else
        "$(eta_target) .+= M.data[!, :$(cov_var)] .* view($(v.latent), M.s_idx)"
    end
    update_str = """
    begin
        # SVC Logic for variable: $(cov_var)
        $(update_inner_cleaned)
        $(application_code)
    end
    """

    return (priors=priors_str, update=update_str)
end



function _generate_component_code_fragments(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="", generate_eta_update::Bool=true)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma)
            push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        end
        if hasproperty(m, :rho)
            push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
        end
    end

    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    local index_var
    if spec.structure == :spatial
        index_var = "s_idx"
    elseif spec.structure == :temporal
        index_var = (typeof(m) <: Union{Cyclic, Harmonic}) ? "u_idx" : "t_idx"
    elseif spec.structure == :mixed
        index_var = "mixed_idx_$(spec.var)"
    else
        index_var = string(spec.structure) * "_idx"
    end

    local eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    local effect_app_str = if generate_eta_update
        if spec.structure == :smooth
            "$(eta_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
        else
            "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        end
    else
        ""
    end

    local update_str
    if get(spec, :is_static, false)
        update_str = """
        begin
            # Static Component Solve: $(spec.key)
            # FIX: Access pre-computed Cholesky factor from the spec_registry.
            F_$(spec.key) = spec_registry["$(spec.key)"].cholesky_factor
            $(v.latent) = $(v.sigma) .* (F_$(spec.key).U \\ $(v.raw))
            $(effect_app_str)
        end
        """
    else
        local flow_direction_kwarg = (m isa NetworkFlow) ? ", flow_direction=:$(m.flow_direction)" : ""

        update_str = """
        begin
            # Dynamic Component Solve: $(spec.key)
            local Q_temp_$(spec.key) = spec_registry["$(spec.key)"].Q_template
            local m_type_$(spec.key) = spec_registry["$(spec.key)"].component_obj |> typeof |> Symbol
            local rho_val_$(spec.key) = $(hasproperty(m, :rho) ? v.rho : "nothing")

            local Q_final_$(spec.key) = recompose_precision(m_type_$(spec.key), Q_temp_$(spec.key), 1.0; extra_param=rho_val_$(spec.key)$(flow_direction_kwarg))
            local F_$(spec.key) = cholesky(Symmetric(Q_final_$(spec.key) + noise * I))

            $(v.latent) = $(v.sigma) .* (F_$(spec.key).U \\ $(v.raw))
            $(effect_app_str)
        end
        """
    end

    return (priors=priors_str, update=update_str)
end



"""
    _generate_component_code_fragments(m::DynamicsComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for process-based `dynamics` models.

# Rationale for Update
This function implements a state-space model where the latent field at time `t` is a
function of the field at `t-1`. This updated version includes more detailed comments
clarifying the use of an implicit Euler scheme for numerical stability.

The model supports different evolution operators:
- **`advection`**: Uses a first-order directed operator (`A_template`) and requires LU decomposition for the non-symmetric propagator.
- **`diffusion`**: Uses the symmetric graph Laplacian (`L_template`) and can use an efficient Cholesky decomposition.
- **`advection_diffusion`**: Combines both operators, resulting in a non-symmetric system solved with LU decomposition.

The state-space equation is of the form `u_t = P⁻¹ * u_{t-1} + innov_t`, where `P` is the
propagator matrix derived from the operators. This formulation is robust and correctly
models the temporal evolution of the spatial field.
"""
function _generate_component_code_fragments(m::DynamicsComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing code fragments for process-based `dynamics` models.
    # Rationale: This version corrects a `MethodError` caused by a local variable `params`
    #            shadowing the `Distributions.params` function. The call is now explicitly
    #            qualified as `Distributions.params` to correctly extract parameters from
    #            prior distributions.
    # v1.1.3 (2026-07-31)

    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = m.params
    model_type = m.model
    is_multivariate = arch == "multivariate"
    spatially_varying_K = get(params, :spatially_varying_K, false)
    spatially_varying_r = get(params, :spatially_varying_r, false)

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Define priors for the dynamics parameters (biological and spatial transport)
    has_propagator = model_type in ["advection", "diffusion", "advection_diffusion"]
    if has_propagator
        if model_type in ["advection", "advection_diffusion"]
            vel = get(params, :velocity, Normal(0, 0.5))
            push!(priors_acc, "$(v.velocity) ~ NamedDist($(_distribution_to_string(vel)), :$(v.velocity))")
        end
        if model_type in ["diffusion", "advection_diffusion"]
            diff = get(params, :diffusion, LogNormal(-1, 1))
            push!(priors_acc, "$(v.diffusion) ~ NamedDist($(_distribution_to_string(diff)), :$(v.diffusion))")
        end
    end
    
    sigma = get(params, :sigma, Exponential(1.0))
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(sigma)), :$(v.sigma))")

    # Priors for biological model parameters
    if model_type in ["logistic_basic", "logistic_exploitation", "delay_difference", "leslie_logistic"]
        if model_type != "leslie_logistic"
            if spatially_varying_r
                log_r_mean_prior = get(params, :log_r_mean, haskey(params, :r) && params[:r] isa LogNormal ? Normal(Distributions.params(params[:r])...) : Normal(0.0, 0.5))
                sigma_r_prior = get(params, :sigma_r, Exponential(1.0))
                push!(priors_acc, "sigma_r_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_r_prior)), :sigma_r_$(prefixed_key))")
                push!(priors_acc, "log_r_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_r_mean_prior)), :log_r_mean_$(prefixed_key))")
                push!(priors_acc, "r_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :r_raw_$(prefixed_key))")
            else
                r = get(params, :r, LogNormal(0, 1))
                push!(priors_acc, "$(v.r) ~ NamedDist($(_distribution_to_string(r)), :$(v.r))")
            end
        end

        if spatially_varying_K
            log_K_mean_prior = get(params, :log_K_mean, haskey(params, :K) && params[:K] isa LogNormal ? Normal(Distributions.params(params[:K])...) : Normal(log(100.0), 0.5))
            sigma_K_prior = get(params, :sigma_K, Exponential(1.0))
            push!(priors_acc, "sigma_K_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_K_prior)), :sigma_K_$(prefixed_key))")
            push!(priors_acc, "log_K_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_K_mean_prior)), :log_K_mean_$(prefixed_key))")
            push!(priors_acc, "K_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :K_raw_$(prefixed_key))")
        else
            K = get(params, :K, LogNormal(log(100.0), 1))
            push!(priors_acc, "$(v.K) ~ NamedDist($(_distribution_to_string(K)), :$(v.K))")
        end
    end
    if model_type == "logistic_exploitation"; q = get(params, :q, LogNormal(-2, 1)); push!(priors_acc, "$(v.q) ~ NamedDist($(_distribution_to_string(q)), :$(v.q))"); end
    if model_type == "delay_difference"; M_nat = get(params, :M_nat, LogNormal(-1, 0.5)); push!(priors_acc, "$(v.M_nat) ~ NamedDist($(_distribution_to_string(M_nat)), :$(v.M_nat))"); end
    if model_type == "lotka_volterra"; alpha = get(params, :alpha, LogNormal(0, 1)); beta = get(params, :beta, LogNormal(-1, 1)); gamma = get(params, :gamma, LogNormal(-1, 1)); delta = get(params, :delta, LogNormal(0, 1)); push!(priors_acc, "$(v.alpha) ~ NamedDist($(_distribution_to_string(alpha)), :$(v.alpha))"); push!(priors_acc, "$(v.beta) ~ NamedDist($(_distribution_to_string(beta)), :$(v.beta))"); push!(priors_acc, "$(v.gamma) ~ NamedDist($(_distribution_to_string(gamma)), :$(v.gamma))"); push!(priors_acc, "$(v.delta) ~ NamedDist($(_distribution_to_string(delta)), :$(v.delta))"); push!(priors_acc, "$(v.innov)_predator ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(Symbol(string(v.innov, "_predator"))))"); end
    if model_type == "leslie_logistic"; n_classes = get(params, :n_age_classes, error("`leslie_logistic` model requires `n_age_classes` parameter.")); push!(priors_acc, "survival_rates ~ NamedDist(filldist(Beta(9, 1), $(n_classes-1)), :survival_rates)"); push!(priors_acc, "fecundity_rates ~ NamedDist(filldist(LogNormal(0, 1), $(n_classes)), :fecundity_rates)"); end

    innov_name = v.innov
    push!(priors_acc, "$(innov_name) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(innov_name))")
    priors_str = join(priors_acc, "\n    ")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    L_op = "spec_registry[\"$(key_str)\"].hyper.L_template"
    A_op = "spec_registry[\"$(key_str)\"].hyper.A_template"
    grid_areas_access = "spec_registry[\"$(key_str)\"].hyper.areas"

    local evolution_loop_body, propagator_setup, field_setup, K_setup_block, r_setup_block
    
    propagator_setup = ""
    if has_propagator
        if model_type == "advection"; propagator_setup = "local propagator = lu(I(M.s_N) - $(v.velocity) * $(A_op) + noise * I)";
        elseif model_type == "diffusion"; propagator_setup = "local propagator = cholesky(Symmetric(I(M.s_N) - $(v.diffusion) * $(L_op) + noise * I))";
        elseif model_type == "advection_diffusion"; propagator_setup = "local propagator = lu(I(M.s_N) - $(v.velocity) * $(A_op) - $(v.diffusion) * $(L_op) + noise * I)"; end
    end

    field_setup = "dyn_field = zeros(T, M.s_N, M.t_N)\n    innov_matrix = reshape($(innov_name), M.s_N, M.t_N)\n    dyn_field[:, 1] = innov_matrix[:, 1]"
    
    K_setup_block = ""; K_variable_name = spatially_varying_K ? "K_spatial" : string(v.K)
    if spatially_varying_K; K_setup_block = "local Q_K = spec_registry[\"$(key_str)\"].hyper.L_template\nlocal F_K = cholesky(Symmetric(Q_K + noise * I))\nlocal K_field_raw = F_K.U \\ K_raw_$(prefixed_key)\nTuring.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(K_field_raw))\nlocal K_spatial = exp.(log_K_mean_$(prefixed_key) .+ K_field_raw .* sigma_K_$(prefixed_key))"; end

    r_setup_block = ""; r_variable_name = spatially_varying_r ? "r_spatial" : string(v.r)
    if spatially_varying_r; r_setup_block = "local Q_r = spec_registry[\"$(key_str)\"].hyper.L_template\nlocal F_r = cholesky(Symmetric(Q_r + noise * I))\nlocal r_field_raw = F_r.U \\ r_raw_$(prefixed_key)\nTuring.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(r_field_raw))\nlocal r_spatial = exp.(log_r_mean_$(prefixed_key) .+ r_field_raw .* sigma_r_$(prefixed_key))"; end

    propagator_logic = has_propagator ? "dyn_field[:, t] = (propagator \\ N_intermediate) .+ innov_matrix[:, t]" : "dyn_field[:, t] = N_intermediate .+ innov_matrix[:, t]"

    if model_type == "logistic_basic"; evolution_loop_body = "local areas = $(grid_areas_access)\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]\n    local D_prev = N_prev ./ areas\n    local K_density = $(K_variable_name) ./ areas\n    local growth = $(r_variable_name) .* D_prev .* (1.0 .- D_prev ./ K_density)\n    local N_intermediate = N_prev .+ (growth .* areas)\n    $(propagator_logic)\n    dyn_field[:, t] = max.(0.0, dyn_field[:, t])\nend"
    elseif model_type == "logistic_exploitation"; evolution_loop_body = "local areas = $(grid_areas_access)\nlocal effort_matrix = spec_registry[\"$(key_str)\"].hyper.processed_params[:effort]\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]\n    local D_prev = N_prev ./ areas\n    local K_density = $(K_variable_name) ./ areas\n    local growth = $(r_variable_name) .* D_prev .* (1.0 .- D_prev ./ K_density)\n    local exploitation = $(v.q) .* effort_matrix[:, t] .* N_prev\n    local N_intermediate = N_prev .+ (growth .* areas) .- exploitation\n    $(propagator_logic)\n    dyn_field[:, t] = max.(0.0, dyn_field[:, t])\nend"
    elseif model_type == "delay_difference"; catch_data_col_sym = spec.hyper.catch_data_col_sym; evolution_loop_body = "local areas = $(grid_areas_access)\nlocal catch_data_matrix = spec_registry[\"$(key_str)\"].hyper.processed_params[:$(catch_data_col_sym)]\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]\n    local D_prev = N_prev ./ areas\n    local K_density = $(K_variable_name) ./ areas\n    local growth = $(r_variable_name) .* D_prev .* (1.0 .- D_prev ./ K_density)\n    local C_prev = catch_data_matrix[:, t-1]\n    local N_intermediate = (N_prev .+ (growth .* areas) .- C_prev) .* exp.(-$(v.M_nat))\n    $(propagator_logic)\n    dyn_field[:, t] = max.(0.0, dyn_field[:, t])\nend"
    elseif model_type == "lotka_volterra"; output_species = get(params, :output_species, :prey); interaction_cov_sym = get(params, :interaction_covariate, nothing); field_setup = "dyn_field_prey = zeros(T, M.s_N, M.t_N)\ndyn_field_predator = zeros(T, M.s_N, M.t_N)\ninnov_matrix_prey = reshape($(v.innov), M.s_N, M.t_N)\ninnov_matrix_predator = reshape($(v.innov)_predator, M.s_N, M.t_N)\ndyn_field_prey[:, 1] = innov_matrix_prey[:, 1]\ndyn_field_predator[:, 1] = innov_matrix_predator[:, 1]"; evolution_loop_body = "local predator_pop_matrix = if !isnothing(Symbol(\"$(interaction_cov_sym)\"))\n    spec_registry[\"$(key_str)\"].hyper.processed_params[:$(interaction_cov_sym)]\nelse\n    nothing\nend\nfor t in 2:M.t_N\n    local N_prey_prev = dyn_field_prey[:, t-1]\n    local N_pred_prev = isnothing(predator_pop_matrix) ? dyn_field_predator[:, t-1] : predator_pop_matrix[:, t-1]\n    local d_prey = ($(v.alpha) .* N_prey_prev) .- ($(v.beta) .* N_prey_prev .* N_pred_prev)\n    local d_pred = ($(v.gamma) .* N_prey_prev .* N_pred_prev) .- ($(v.delta) .* N_pred_prev)\n    dyn_field_prey[:, t] = max.(0.0, N_prey_prev .+ d_prey .+ innov_matrix_prey[:, t])\n    dyn_field_predator[:, t] = max.(0.0, N_pred_prev .+ d_pred .+ innov_matrix_predator[:, t])\nend\nlocal dyn_field = $(output_species == :prey ? "dyn_field_prey" : "dyn_field_predator")"
    elseif model_type == "leslie_logistic"; n_classes = get(params, :n_age_classes, 1); evolution_loop_body = "local L = zeros(T, $(n_classes), $(n_classes))\nfor i in 1:($(n_classes)-1); L[i+1, i] = survival_rates[i]; end\nL[1, :] = fecundity_rates\nlocal r_leslie = log(maximum(abs.(eigen(L).values)))\nlocal areas = $(grid_areas_access)\nfor t in 2:M.t_N\n    local N_prev = dyn_field[:, t-1]\n    local D_prev = N_prev ./ areas\n    local K_density = $(K_variable_name) ./ areas\n    local growth = r_leslie .* D_prev .* (1.0 .- D_prev ./ K_density)\n    local N_intermediate = N_prev .+ (growth .* areas)\n    $(propagator_logic)\n    dyn_field[:, t] = max.(0.0, dyn_field[:, t])\nend"
    else; evolution_loop_body = "for t in 2:M.t_N\n    dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]\nend"; end

    update_str = "begin\n# Dynamics model: $(model_type) for $(key_str)\n$(K_setup_block)\n$(r_setup_block)\n$(propagator_setup)\n$(field_setup)\n$(evolution_loop_body)\ndyn_field .*= $(v.sigma)\nfor i in 1:N\n    $(eta_update_target)[i] += dyn_field[M.s_idx[i], M.t_idx[i]]\nend\nend"
    
    return (priors=priors_str, update=update_str)
end






"""
    _generate_component_code_fragments(m::RFF, ...)

Updated code generator for the `RFF` component to support ARD and fix `spec` access.

# Rationale for Update
This version corrects the `UndefVarError: spec not defined` by accessing component
properties via `spec_registry  key_str instead of directly using `spec`.
It also ensures the `lengthscale` is no longer used to scale the projection matrix
after sampling, relying on `build_model` to generate `W_fixed` correctly.
"""
function _generate_component_code_fragments(m::RFF, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_features = m.n_features
    
    in_dims = size(spec.hyper.coords, 2)
    
    W_name = Symbol("$(string(spec.key))_W")
    b_name = Symbol("$(string(spec.key))_b")
    beta_name = v.innov

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(W_name) ~ NamedDist(MvNormal(vec(spec_registry[\"$(key_str)\"].hyper.W_fixed), 0.1), :$(W_name))")
    push!(priors_acc, "$(b_name) ~ NamedDist(MvNormal(spec_registry[\"$(key_str)\"].hyper.b_fixed, 0.1), :$(b_name))")
    
    # --- FIX ---
    # The `n_features` variable is now correctly interpolated into the string using `$(n_features)`.
    # This ensures the generated code uses the actual number of basis functions for the prior.
    push!(priors_acc, "$(beta_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(beta_name))")
    
    priors_str = join(priors_acc, "\n")
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update = """
    begin
        # RFF GP model for $(key_str)
        local X_coords = spec_registry["$(key_str)"].hyper.coords
        local W_matrix = reshape($(W_name), $(in_dims), $(n_features))
        local Phi = sqrt(2.0 / $(n_features)) .* cos.((X_coords * W_matrix) .+ $(b_name)')
        local scaled_beta = $(beta_name) .* $(v.sigma)
        local rff_effect = Phi * scaled_beta
        $(eta_target) .+= rff_effect
    end
    """
    return (priors=priors_str, update=update)
end


 
function _generate_component_code_fragments(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing code fragments for the `Eigen` (Bayesian PCA) component.
    # Rationale: This version is updated to use the `_distribution_to_string` helper function
    #            when generating code for the priors on `pca_sd` and `pdef_sd`. This resolves
    #            a `MethodError` that occurred with versions of `Distributions.jl` that do not
    #            support keyword arguments (e.g., `θ=...`) for the `Exponential` constructor.
    # v1.0.1 (2026-07-31)
    # Inputs: Standard code generation arguments.
    # Outputs: A NamedTuple containing `priors` and `update` code strings.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_vars = m.n_vars
    n_factors = m.n_factors
    n_obs = size(spec.hyper.eigen_data, 1)

    # Convert prior distributions to their correct string representations.
    pca_sd_prior_str = _distribution_to_string(m.pca_sd)
    pdef_sd_prior_str = _distribution_to_string(m.pdef_sd)

    # Priors for the factor model parameters
    priors_str = """
    # Priors for eigen component: $(key_str)
    v_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, $(length(m.ltri_indices))), 1.0), :v_raw_$(prefixed_key))
    
    # Priors for factor standard deviations (related to eigenvalues)
    pca_sds_$(prefixed_key) ~ NamedDist(filldist($(pca_sd_prior_str), $(n_factors)), :pca_sds_$(prefixed_key))
    
    # Priors for uniquenesses (residual standard deviations)
    pdef_sds_$(prefixed_key) ~ NamedDist(filldist($(pdef_sd_prior_str), $(n_vars)), :pdef_sds_$(prefixed_key))
    
    # Latent factors (scores) are sampled from a standard normal
    factors_flat_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, $(n_obs * n_factors)), I), :factors_flat_$(prefixed_key))
    """

    # Main logic for the factor model and its contribution to eta
    update_str = """
    begin
        # --- Factor Model for Eigen Component: $(key_str) ---
        
        # 1. Construct orthonormal loadings matrix U from Householder reflectors
        local v_mat_$(prefixed_key) = zeros(T, $(n_vars), $(n_factors))
        v_mat_$(prefixed_key)[$(m.ltri_indices)] .= v_raw_$(prefixed_key)
        local U_$(prefixed_key) = householder_to_eigenvector(v_mat_$(prefixed_key), $(n_vars), $(n_factors))
        
        # 2. Construct the full loadings matrix L = U * diag(pca_sds)
        local L_$(prefixed_key) = U_$(prefixed_key) * Diagonal(pca_sds_$(prefixed_key))
        
        # 3. Reshape latent factors F into a matrix of scores
        local F_$(prefixed_key) = reshape(factors_flat_$(prefixed_key), $(n_obs), $(n_factors))
        
        # 4. Calculate the reconstructed mean Y_hat = F * L'
        local Y_hat_$(prefixed_key) = F_$(prefixed_key) * L_$(prefixed_key)'
        
        # 5. Define the diagonal matrix of uniquenesses (residual covariance)
        local Psi_$(prefixed_key) = Diagonal(pdef_sds_$(prefixed_key).^2 .+ noise)
        
        # 6. Add the factor model likelihood to the total log probability.
        #    This models the data passed to the eigen() module.
        local Y_eigen_data = spec_registry["$(key_str)"].hyper.eigen_data
        for i in 1:$(n_obs)
            Turing.@addlogprob! logpdf(MvNormal(Y_hat_$(prefixed_key)[i, :], Psi_$(prefixed_key)), Y_eigen_data[i, :])
        end

        # 7. Add the sum of all latent factors to the main model's linear predictor.
        #    This uses the combined shared signal from all factors as a predictor.
        eta .+= sum(F_$(prefixed_key), dims=2)
    end
    """
    return (priors=priors_str, update=update_str)
end



function _generate_component_code_fragments(m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # v1.0.0 (2026-07-20) - Added state-space implementation for RW1.
    # Specialized implementation for RW1 processes using a state-space formulation.
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_latent = size(spec.Q_template, 1)

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # RW1 state-space evolution for $(key_str)
        innovations = $(v.innov)
        latent_field_raw = cumsum(innovations)
        
        # Apply sum-to-zero constraint for identifiability
        # Apply soft sum-to-zero constraint for identifiability
        if $(n_latent) > 0
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(latent_field_raw))
            $(v.latent) = latent_field_raw .* $(v.sigma)
            $(eta_update_target) .+= view($(v.latent), M.$(index_var))
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end


function _generate_component_code_fragments(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # v1.0.0 (2026-07-20) - Added state-space implementation for RW2.
    # Specialized implementation for RW2 processes using a state-space formulation.
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_latent = size(spec.Q_template, 1)

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # RW2 state-space evolution for $(key_str)
        innovations = $(v.innov)
        latent_field_raw = Vector{T}(undef, $(n_latent))
        
        if $(n_latent) > 0; latent_field_raw[1] = innovations[1]; end
        if $(n_latent) > 1; latent_field_raw[2] = 2*latent_field_raw[1] + innovations[2]; end

        for t in 3:$(n_latent)
            latent_field_raw[t] = 2*latent_field_raw[t-1] - latent_field_raw[t-2] + innovations[t]
        end
        
        # Apply sum-to-zero constraint for identifiability
        # Apply soft sum-to-zero constraint for identifiability
        if $(n_latent) > 0
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(latent_field_raw))
            $(v.latent) = latent_field_raw .* $(v.sigma)
            $(eta_update_target) .+= view($(v.latent), M.$(index_var))
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end



function _generate_component_code_fragments(m::Union{Besag, ICAR, BCGN, Cyclic}, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="", generate_eta_update::Bool=true)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma)
            push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        end
    end

    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    index_var = if spec.structure == :spatial
        "s_idx"
    elseif spec.structure == :temporal
        (typeof(m) <: Cyclic) ? "u_idx" : "t_idx"
    else
        string(spec.structure) * "_idx"
    end
    
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"

    effect_application_str = if generate_eta_update
        if spec.structure == :smooth
            "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
        else
            "$(eta_update_target) .+= view($(v.latent), M.$(index_var))"
        end
    else
        ""
    end

    update_str = """
    begin
        # --- Intrinsic Component Solve: $(key_str) ---
        local Q_template_$(key_str) = spec_registry["$(key_str)"].Q_template
        local F_$(key_str) = cholesky(Symmetric(Q_template_$(key_str) + noise * I))
        
        local latent_field_raw_$(key_str) = F_$(key_str).U \\ $(v.raw)

        Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(latent_field_raw_$(key_str)))
        
        $(v.latent) = latent_field_raw_$(key_str) .* $(v.sigma)
        $(effect_application_str)
    end
    """

    return (priors=priors_str, update=update_str)
end



function _generate_component_code_fragments(m::Wavelet, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_basis = m.nbins
    
    # --- FIX: Use `spec.hyper` instead of `spec_registry` ---
    # The `spec` object is available during code generation.
    nbins_per_dim_str = if haskey(spec.hyper, :nbins_per_dim)
        string(spec.hyper.nbins_per_dim)
    elseif haskey(spec.hyper, :coords)
        string([round(Int, n_basis^(1/size(spec.hyper.coords, 2)))])
    else
        string([n_basis])
    end

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_basis)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update = """
    begin
        # --- Wavelet Component: $(key_str) ---
        local coords = spec_registry["$(key_str)"].hyper.coords
        local nbins_per_dim = $(nbins_per_dim_str)
        
        local B_wavelet = bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, Symbol("$(m.family)"), $(v.ls))
        local Q_penalty = build_structure_template(:rw2, $(n_basis)).matrix
        local F_penalty = cholesky(Symmetric(Q_penalty + noise * I))
        
        local wavelet_coeffs = F_penalty.U \\ $(v.raw)
        local scaled_coeffs = wavelet_coeffs .* $(v.sigma)
        local wavelet_effect = B_wavelet * scaled_coeffs
        
        $(eta_target) .+= wavelet_effect
    end
    """
    return (priors=priors, update=update)
end 


"""
    _generate_component_code_fragments(m::FFT, ...)

Updated code generator for the `FFT` component to support ARD and fix `spec` access.

# Rationale for Update
This version corrects the `UndefVarError: spec not defined` by accessing component
properties via `spec_registry key_str` instead of directly using `spec`.
"""
function _generate_component_code_fragments(m::FFT, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_basis = m.nbins
    
    coords = spec.hyper.coords
    n_dims = size(coords, 2)
    n_obs = size(coords, 1)

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    # --- FIX ---
    # The `n_basis` variable is now correctly interpolated into the string using `$(n_basis)`.
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_basis)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    local basis_gen_code
    if n_dims == 1
        basis_gen_code = """
        local vals = spec_registry["$(key_str)"].hyper.coords[:, 1]
        local ls_val = $(v.ls) isa Real ? $(v.ls) : $(v.ls)[1]
        local t_coords = vals ./ ls_val
        local B_fft = zeros(T, $(n_obs), $(n_basis))
        for m_fft in 1:div($(n_basis), 2)
            if (2*m_fft) <= $(n_basis)
                local arg = (2.0 * pi * m_fft) .* t_coords
                B_fft[:, 2*m_fft-1] = sin.(arg)
                B_fft[:, 2*m_fft]   = cos.(arg)
            end
        end
        """
    elseif n_dims >= 2
        nbins_per_dim = get(spec.hyper, :nbins_per_dim, [round(Int, n_basis^(1/n_dims)) for _ in 1:n_dims])
        n_marginal_x = nbins_per_dim[1]
        n_marginal_y = nbins_per_dim[2]
        basis_gen_code = """
        local coords = spec_registry["$(key_str)"].hyper.coords
        local ls_vec = $(v.ls) isa Real ? fill($(v.ls), $n_dims) : $(v.ls)
        local nx = coords[:, 1] ./ ls_vec[1]
        local ny = coords[:, 2] ./ ls_vec[2]
        local B_fft = zeros(T, $(n_obs), $(n_basis))
        local idx = 1
        for my in 1:$(n_marginal_y), mx in 1:$(n_marginal_x)
            if idx + 1 <= $(n_basis)
                local arg = mx .* nx .+ my .* ny
                B_fft[:, idx]   = sin.(2.0 * pi * arg)
                B_fft[:, idx+1] = cos.(2.0 * pi * arg)
                idx += 2
            end
        end
        """
    end

    update = """
    begin
        $(basis_gen_code)
        local Q_penalty = build_structure_template(:rw2, $(n_basis)).matrix
        local F_penalty = cholesky(Symmetric(Q_penalty + noise * I))
        local fft_coeffs = F_penalty.U \\ $(v.raw)
        local scaled_coeffs = fft_coeffs .* $(v.sigma)
        local fft_effect = B_fft * scaled_coeffs
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= fft_effect
    end
    """
    return (priors=priors, update=update)
end


function _generate_component_code_fragments(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # This is a specialized implementation for AR(1) processes.
    # It uses a state-space formulation for numerical stability and efficiency,
    # avoiding the construction and Cholesky decomposition of a dense precision matrix.
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)

    key_str = string(spec.key)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        # The prior for rho is truncated to ensure stationarity and numerical stability.
        push!(priors_acc, "$(v.rho) ~ NamedDist(truncated($(_distribution_to_string(m.rho)), -0.9999, 0.9999), :$(v.rho))")
    end

    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx" # AR1 is always temporal

    update_str = """
    begin
        $(v.latent) = ar1_statespace($(v.rho), $(v.sigma), $(v.innov), T, $(n_latent), noise)
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_component_code_fragments(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `Moran` spatial component.

# Rationale for Update
This is a new function that implements the spectral decomposition for the Moran component.
Instead of using a precision matrix (GMRF), it models the latent spatial effect as a
linear combination of the pre-computed Moran eigenvectors. The coefficients of this
combination are sampled from a Normal distribution, with their scale controlled by the
component's `sigma` hyperparameter. This correctly implements the `Moran's I Basis Component`
as a spectral model.
"""
function _generate_component_code_fragments(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_latent = size(spec.hyper.moran_eigenvectors, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    local sigma_name_str, coeffs_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        coeffs_name_str = "$(prefixed_key)_coeffs_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        if is_multivariate
            coeffs_name_str = "$(prefixed_key)_coeffs_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            coeffs_name_str = "$(prefixed_key)_coeffs"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
    end
    
    # Priors for the coefficients of the Moran eigenvectors
    push!(priors_acc, "$(Symbol(coeffs_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(coeffs_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    update_str = """
    begin
        # Moran eigenvector spectral model for $(key_str)
        local moran_eigenvectors = spec_registry["$(key_str)"].hyper.moran_eigenvectors
        
        # The latent effect is a linear combination of the eigenvectors,
        # with coefficients scaled by sigma.
        $(v.latent) = moran_eigenvectors * ($(coeffs_name) .* $(v.sigma))
        
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_component_code_fragments(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `Spherical` Gaussian Process model.

# Rationale for Update
This new function implements the code generation for a full GP with a spherical
covariance function. It computes the pairwise distance matrix, evaluates the
spherical kernel based on the sampled `range` and `sigma` parameters, and then
samples the latent field from the resulting `MvNormal` distribution. This correctly
implements the `Spherical` component as a continuous GP model.
"""
function _generate_component_code_fragments(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.range) ~ NamedDist($(_distribution_to_string(m.range)), :$(v.range))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Spherical GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # Compute pairwise Euclidean distances
        dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute spherical kernel matrix
        h = dist_matrix ./ $(v.range)
        K = zeros(T, size(h))
        mask = h .< 1.0
        K[mask] = ($(v.sigma)^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
        K += (noise * I)
        
        F = cholesky(Symmetric(K))
        $(v.latent) = F.L * $(v.raw)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_component_code_fragments(m::LocalAdaptive, ...)

A specialized code generator for the `LocalAdaptive` component.

# Rationale
This function generates the Turing model code required to implement the `LocalAdaptive`
model correctly. It defines priors for the cluster-specific means and ensures they are
incorporated into the sampling of the latent spatial field.

Key features of the generated code include:
1.  **Priors for Cluster Means**: Samples a vector of raw cluster means from a standard
    normal distribution.
2.  **Sum-to-Zero Constraint**: Centers the raw cluster means to ensure the model is
    identifiable with respect to the global intercept.
3.  **Non-Zero Mean GMRF**: Constructs the full mean vector for the GMRF by mapping the
    centered cluster means to their corresponding spatial units.
4.  **Non-Centered Parameterization**: Samples the latent field using a non-centered
    parameterization for a non-zero mean GMRF (`x = μ + L'⁻¹z`), which improves
    sampler efficiency.
"""
function _generate_component_code_fragments(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    # For this model, 'innov' will store the raw cluster means.
    mu_clusters_raw_name = v.innov

    priors_acc = String[]

    # Generate priors only once for shared parameters in multivariate models.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
        
        # Prior for the raw cluster means. They will be centered later for identifiability.
        n_clusters = spec.hyper.n_clusters
        push!(priors_acc, "$(mu_clusters_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_clusters)), I), :$(mu_clusters_raw_name))")
    end

    # Prior for the main latent field (non-centered innovations)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx" # LocalAdaptive is always spatial

    update_str = """
    begin
        # LocalAdaptive model for $(key_str)
        
        # 1. Apply soft sum-to-zero constraint on cluster means for identifiability.
        local n_clusters = spec_registry["$(key_str)"].hyper.n_clusters
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_clusters), sum($(mu_clusters_raw_name)))
        
        # 2. Construct the mean vector for the GMRF by mapping cluster means to spatial units.
        local mean_vector = $(mu_clusters_raw_name)[M.cluster_assignments]

        # 3. Recompose the Leroux precision matrix.
        local Q_template = spec_registry["$(key_str)"].Q_template
        local m_type = spec_registry["$(key_str)"].component_obj |> typeof |> Symbol
        local rho_val = $(v.rho)
        local Q_final = recompose_precision(m_type, Q_template, 1.0; extra_param=rho_val)
        
        # 4. Sample from the non-zero mean GMRF using a non-centered parameterization.
        #    For x ~ N(μ, Q⁻¹), we can sample z ~ N(0,I) and compute x = μ + L'⁻¹z,
        #    where Q = LL'. Here, Q = F.U' * F.U, so L = F.U'. Then L'⁻¹ = (F.U)⁻¹ = F.U \\ I.
        #    This gives x = μ + (F.U \\ z).
        local F = cholesky(Symmetric(Q_final + noise * I))
        local latent_field_centered_part = F.U \\ $(v.raw)
        $(v.latent) = mean_vector .+ latent_field_centered_part

        # 5. Scale by sigma and apply to eta.
        $(v.latent) .*= $(v.sigma)
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end




"""
    _generate_component_code_fragments(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `BYM2` component.

# Rationale for Correction
This version corrects the implementation to strictly follow the Riebler et al. (2016)
parameterization. It removes a redundant and potentially unstable scaling step where
the structured component was scaled by its sample standard deviation. The precision
matrix `Q_template` is already scaled in `build_structure_template`, ensuring the
`struct_latent` component has approximately unit variance. This change makes the
implementation more robust and theoretically sound.
"""
function _generate_component_code_fragments(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    params = spec.params
    n_latent = isnothing(spec.Q_template) ? 0 : size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    # Generate priors only once for shared parameters in multivariate models.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
    end
    
    # Priors for the raw innovations for the structured and unstructured components.
    push!(priors_acc, "$(v.struct) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.struct))")
    push!(priors_acc, "$(v.iid) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.iid))")
    priors_str = join(priors_acc, "\n")
    
    index_var = spec.structure == :spatial ? "s_idx" : string(spec.structure) * "_idx"
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    # Update block to combine the components
    update_str = """
    begin
        # BYM2 model for $(key_str)
        # 1. Reconstruct the structured (ICAR) component from its raw innovations.
        local Q_template = spec_registry["$(spec.key)"].Q_template
        if !isnothing(Q_template) && $(n_latent) > 0
            # Use sparse matrix for Cholesky factorization for efficiency.
            local F = cholesky(Symmetric(sparse(Q_template) + noise * I))
            local struct_latent = F.U \\ $(v.struct)
            
            # 2. Apply soft sum-to-zero constraint for identifiability.
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(struct_latent))
            
            # 3. Combine structured and unstructured components using Riebler parameterization.
            # The `struct_latent` field already has approximately unit variance due to the scaling of Q.
            local bym2_effect = $(v.sigma) .* (sqrt($(v.rho)) .* struct_latent .+ sqrt(1.0 - $(v.rho)) .* $(v.iid))
            
            # 4. Add the final effect to the linear predictor.
            $(eta_update_target) .+= view(bym2_effect, M.$(index_var))
        end
    end
    """    
    
    return (priors=priors_str, update=update_str)
end



function _generate_component_code_fragments(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Warped Gaussian Process model.
    # This model first applies a non-linear warping function (approximated by RFFs) to the
    # input coordinates, and then applies a standard GP (also approximated by RFFs) to the
    # warped coordinates. This allows for modeling non-stationary spatial/temporal effects.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # This model has many specific parameters not in the standard naming convention.
    # Manual naming is clearer here.
    beta_main_name = Symbol("$(prefixed_key)_beta_main_$(outcome_idx)")
    W_main_name = Symbol("$(prefixed_key)_W_main_$(outcome_idx)")
    b_main_name = Symbol("$(prefixed_key)_b_main_$(outcome_idx)")
    beta_warp_name = Symbol("$(prefixed_key)_beta_warp_$(outcome_idx)")
    W_warp_name = Symbol("$(prefixed_key)_W_warp_$(outcome_idx)")
    b_warp_name = Symbol("$(prefixed_key)_b_warp_$(outcome_idx)")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end

    n_features = m.n_features
    in_dims = size(spec.hyper.coords, 2)

    # Priors for the warping function's RFF parameters
    push!(priors_acc, "$(W_warp_name) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(W_warp_name))")
    push!(priors_acc, "$(b_warp_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(b_warp_name))")
    push!(priors_acc, "$(beta_warp_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(beta_warp_name))")

    # Priors for the main GP's RFF parameters
    push!(priors_acc, "$(W_main_name) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(W_main_name))")
    push!(priors_acc, "$(b_main_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(b_main_name))")
    push!(priors_acc, "$(beta_main_name) ~ NamedDist(MvNormal(zeros(T, $(n_features)), $(v.sigma)^2 * I), :$(beta_main_name))")
    
    priors_str = join(priors_acc, "\n")
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # 1. Construct and apply the warping function
        W_warp_matrix = reshape($(W_warp_name), $(in_dims), $(n_features))
        Phi_warp = sqrt(2.0 / $(n_features)) .* cos.((coords * W_warp_matrix) .+ $(b_warp_name)')
        warping_effect = Phi_warp * $(beta_warp_name)
        coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(v.ls)
        Phi_main = sqrt(2.0 / $(n_features)) .* cos.((coords_warped * W_main_matrix) .+ $(b_main_name)')
        main_effect = Phi_main * $(beta_main_name)

        $(eta_update_target) .+= main_effect
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 

"""
    _generate_component_code_fragments(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

A new, specialized code generator for the `SVGP` (Sparse Variational Gaussian Process) component.

# Rationale for New Implementation
This function is introduced to resolve a `DimensionMismatch` error that occurred when the `svgp`
model was incorrectly handled by a generic GMRF code generator. This implementation provides
the correct logic for a sparse GP, similar to the `FITC` model. It ensures that the model
correctly computes the kernel matrices based on observation and inducing point coordinates,
constructs the conditional mean and variance, and samples the final latent field using a
non-centered parameterization. This prevents the fallback to the incorrect GMRF logic and
resolves the error.
"""
function _generate_component_code_fragments(m::SVGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the SVGP (Sparse Variational Gaussian Process) model.
    # This implementation is functionally similar to FITC for the purpose of MCMC sampling,
    # providing a non-centered parameterization for a sparse GP.
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    u_raw_name = v.raw
    f_raw_name = v.innov

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.lengthscale isa Vector
            ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
            push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
        else
            push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
        end
    end

    n_inducing = m.n_inducing
    n_latent = size(spec.Q_template, 1)
    
    push!(priors_acc, "$(u_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(u_raw_name))")
    push!(priors_acc, "$(f_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(f_raw_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # SVGP sparse GP model for $(key_str)
        X_coords = spec_registry["$(key_str)"].Q_template
        Z_coords = spec_registry["$(key_str)"].hyper.Z_inducing
        
        K_UU = evaluate_kernel_matrix(Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"), noise)
        K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"))
        
        L_UU = cholesky(Symmetric(K_UU)).L
        u_latent = L_UU * $(u_raw_name)
        
        K_UU_inv_u = K_UU \\ u_latent
        mean_f = K_XU * K_UU_inv_u
        
        diag_K_XX = fill($(v.sigma)^2, $(n_latent))
        tmp = (L_UU' \\ K_XU')'
        diag_Q_ff = sum(tmp.^2, dims=2)
        lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        $(v.latent) = mean_f + sqrt.(max.(lambda_diag, 0.0) .+ noise) .* $(f_raw_name)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end


 
function _generate_component_code_fragments(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Nystrom sparse Gaussian Process model.
    # This method uses a low-rank approximation based on inducing points.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    # For Nystrom, 'raw' holds the innovations for the inducing points.
    v_latent_name = v.raw

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.lengthscale isa Vector; ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", "); push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))");
        else; push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))"); end
    end

    n_inducing = m.n_inducing
    push!(priors_acc, "$(v_latent_name) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(v_latent_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Nystrom sparse GP model for $(key_str)
        X_coords = spec_registry["$(key_str)"].hyper.coords
        Z_coords = spec_registry["$(key_str)"].hyper.Z_inducing
        
        K_UU = evaluate_kernel_matrix(Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"), noise)
        K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"))
        
        L_UU = cholesky(Symmetric(K_UU)).L
        
        # Project standard normal noise through the Nystrom approximation
        # f(X) ≈ K_XU * inv(K_UU) * u, where u ~ N(0, K_UU)
        # Using non-centered parameterization: u = L_UU * v, where v ~ N(0, I)
        # f(X) ≈ K_XU * inv(K_UU) * L_UU * v = K_XU * inv(L_UU' * L_UU) * L_UU * v = K_XU * (L_UU' \\ v)
        $(v.latent) = K_XU * (L_UU' \\ $(v_latent_name))
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_component_code_fragments(m::FITC, ...)

A specialized code generator for the `FITC` sparse Gaussian Process model.

# Rationale
This function implements the FITC approximation, which is crucial for scaling GPs
to large datasets. The key steps are:
1.  **Priors**: Defines priors for the kernel hyperparameters (`sigma`, `lengthscale`) and
    for the raw innovations for the latent values at the inducing points (`u_raw`) and
    for the final latent field (`f_raw`).
2.  **Kernel Matrices**: Computes the required kernel matrices: `K_UU` (covariance between
    inducing points) and `K_XU` (cross-covariance between data and inducing points).
3.  **Inducing Point Sampling**: Samples the latent values at the inducing points (`u_latent`)
    using a non-centered parameterization for improved MCMC efficiency.
4.  **Conditional Distribution**: Calculates the conditional mean and the diagonal of the
    conditional covariance of the GP at the observation points, given the values at the
    inducing points. This is the core of the FITC approximation.
5.  **Final Latent Field**: Samples the final latent field `f` from this conditional
    distribution, again using a non-centered parameterization.
"""
function _generate_component_code_fragments(m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    # For FITC, 'raw' holds innovations for inducing points, 'innov' for the final field.
    u_raw_name = v.raw
    f_raw_name = v.innov

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.lengthscale isa Vector
            ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
            push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
        else
            push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
        end
    end

    n_inducing = m.n_inducing
    n_latent = size(spec.Q_template, 1) # Number of data points
    
    # Priors for the latent values at inducing points and the final field innovations
    push!(priors_acc, "$(u_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(u_raw_name))")
    push!(priors_acc, "$(f_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(f_raw_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # FITC sparse GP model for $(key_str)
        X_coords = spec_registry["$(key_str)"].Q_template
        Z_coords = spec_registry["$(key_str)"].hyper.Z_inducing
        
        # 1. Compute kernel matrices
        K_UU = evaluate_kernel_matrix(Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"), noise)
        K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(v.sigma), $(v.ls), Symbol("$(m.kernel)"))
        
        # 2. Sample latent values at inducing points (non-centered)
        L_UU = cholesky(Symmetric(K_UU)).L
        u_latent = L_UU * $(u_raw_name)
        
        # 3. Compute conditional mean and variance for FITC
        #    μ_f = K_XU * inv(K_UU) * u_latent
        #    diag_cov_f = diag(K_XX - K_XU * inv(K_UU) * K_XU')
        
        K_UU_inv_u = K_UU \\ u_latent
        mean_f = K_XU * K_UU_inv_u
        
        # Compute diagonal of K_XX - Q_ff efficiently
        # diag(K_XX) is sigma^2 for stationary kernels.
        diag_K_XX = fill($(v.sigma)^2, $(n_latent))
        
        # diag(K_XU * inv(K_UU) * K_XU') = sum((K_XU / L_UU.U).^2, dims=2)
        tmp = (L_UU' \\ K_XU')'
        diag_Q_ff = sum(tmp.^2, dims=2)
        
        lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        # 4. Sample final latent field (non-centered)
        $(v.latent) = mean_f + sqrt.(max.(lambda_diag, 0.0) .+ noise) .* $(f_raw_name)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end

 
function _generate_component_code_fragments(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Hyperbolic Gaussian Process model.
    # This model computes distances in a hyperbolic space (Poincaré disk) before applying a kernel.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        # Curvature is fixed for now, but could be given a prior.
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Hyperbolic GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        curvature = $(m.curvature) # Fixed curvature
        
        K = evaluate_hyperbolic_kernel_matrix(coords, $(v.sigma), curvature, noise)
        F = cholesky(Symmetric(K))
        $(v.latent) = F.L * $(v.raw)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 

function _generate_component_code_fragments(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Exponential Decay GP model.
    # This model uses an exponential kernel based on Euclidean distances between coordinates.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Exponential Decay GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # Compute pairwise Euclidean distances
        dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute exponential decay kernel matrix
        K = ($(v.sigma)^2) .* exp.(-dist_matrix ./ $(v.ls)) .+ (noise * I)
        
        F = cholesky(Symmetric(K))
        $(v.latent) = F.L * $(v.raw)
        
        $(eta_update_target) .+= $(v.latent)
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 

function _generate_component_code_fragments(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        if m.kappa isa Vector
            kappa_priors_str = join([_distribution_to_string(p) for p in m.kappa], ", ")
            push!(priors_acc, "$(v.kappa) ~ NamedDist(Product([$(kappa_priors_str)]), :$(v.kappa))")
        else
            push!(priors_acc, "$(v.kappa) ~ NamedDist($(_distribution_to_string(m.kappa)), :$(v.kappa))")
        end
    end

    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")
    
    index_var = spec.structure == :spatial ? "s_idx" : string(spec.structure) * "_idx"
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_application_str = spec.structure == :smooth ? "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)" : "$(eta_update_target) .+= view($(v.latent), M.$(index_var))"

    update = """
    begin
        local Q_template = spec_registry["$(key_str)"].Q_template
        local m_type = spec_registry["$(key_str)"].component_obj |> typeof |> Symbol
        local kappa_val = $(v.kappa)
        
        local Q_final = recompose_precision(m_type, Q_template, 1.0; extra_param=kappa_val)
        local F = cholesky(Symmetric(Q_final + noise * I))
        $(v.latent) = $(v.sigma) .* (F.U \\ $(v.raw))
        $(effect_application_str)
    end
    """
    
    return (priors=priors, update=update)
end


"""
    _generate_component_code_fragments(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for a Directed Acyclic Graph (DAG) spatial component.

# Rationale for Update
This version corrects a `MethodError` that occurred during the forward substitution
step. The original implementation used `view(vector, index)` to access parent node
values, which returns a 0-dimensional array. This has been changed to standard scalar
indexing (`vector[index]`), which returns a `Float64`, ensuring that the accumulation
of `parent_effect` is mathematically correct and resolves the error.
"""
function _generate_component_code_fragments(m::DAG, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n    ")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx"

    update_str = """
    begin
        # DAG model for $(key_str) using forward substitution.
        # NOTE: This is a simplified DAG model that assumes IID innovations. A full
        # DAGAR (Directed Acyclic Graph Auto-Regressive) model would adjust the
        # innovation variance at each node to ensure constant marginal variance.
        local W_dag = spec_registry["$(key_str)"].Q_template
        local $(v.latent) = zeros(T, $(n_latent))
        local innovations = $(v.innov)
        local rho_val = $(v.rho)
        local sigma_val = $(v.sigma)

        # Assumes W_dag is lower triangular, representing a valid DAG ordering.
        for i in 1:$(n_latent)
            local parent_effect = 0.0
            # Efficiently iterate over non-zero elements in the row of the sparse matrix
            for j_ptr in nzrange(W_dag, i)
                parent_idx = W_dag.rowval[j_ptr]
                parent_effect += W_dag.nzval[j_ptr] * $(v.latent)[parent_idx]
            end
            $(v.latent)[i] = rho_val * parent_effect + innovations[i]
        end
        $(v.latent) .*= sigma_val
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end



"""
    _generate_component_code_fragments(m::ComposedComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for composed components, such as state-space models (`|>`)
and non-stationary variance models (`∘`).

# Rationale for Update
This function has been updated to correctly handle the `:composition` operator. It now
retrieves the custom `priors` and `update` code strings that were injected into the
module's parameters by `process_interact_module!`. This completes the implementation
for non-stationary variance models where a smoother modulates a spatial field. The
existing logic for the `:pipe` operator (spatially-varying curves) is preserved.
"""
function _generate_component_code_fragments(m::ComposedComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    op = m.operator 
    key_str = string(spec.key)

    if op == :pipe
        state_component = m.components[1]
        dynamic_component = get(spec.params, :dynamic_component_obj, nothing)
        structure_str = string(spec.structure)
        is_multivariate = arch == "multivariate"
        is_first_outcome = outcome_idx == 1
        is_shared = get(spec.params, :shared, false)
        eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
        
        v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
        coeffs_raw_name = v.raw

        state_spec = spec.hyper.state_spec
        n_spatial = size(state_spec.Q_template, 1)
        n_basis = dynamic_component.nbins
        basis_key = get(spec.params, :dynamic_basis_key, nothing)
        if isnothing(basis_key); error("Could not find basis matrix key for piped component $(key_str)."); end
        
        priors_acc = String[]
        if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
            if hasproperty(state_component, :sigma); push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(state_component.sigma)), :$(v.sigma))"); end
            if hasproperty(state_component, :rho); push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(state_component.rho)), :$(v.rho))"); end
        end
        push!(priors_acc, "$(coeffs_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_spatial * n_basis)), I), :$(coeffs_raw_name))")
        priors_str = join(priors_acc, "\n        ")
        
        is_state_static = get(state_spec, :is_static, false)
        local cholesky_block
        if is_state_static
            cholesky_block = "F_spatial = spec_registry[\"$(key_str)\"].hyper.state_spec.cholesky_factor"
        else
            cholesky_block = "Q_spatial_template = spec_registry[\"$(key_str)\"].hyper.state_spec.Q_template\n state_m_type = spec_registry[\"$(key_str)\"].hyper.state_spec.model_type\n rho_val = $(hasproperty(state_component, :rho) ? v.rho : "nothing")\n Q_spatial = recompose_precision(state_m_type, Q_spatial_template, 1.0; extra_param=rho_val)\n F_spatial = cholesky(Symmetric(Q_spatial + noise * I))"
        end
        
        update_str = "begin\n $(cholesky_block)\n coeffs_raw_matrix = reshape($(coeffs_raw_name), $(n_spatial), $(n_basis))\n spatial_coeffs = $(v.sigma) .* (F_spatial.U \\ coeffs_raw_matrix)\n B_smooth = M.basis_matrices[:$(basis_key)]\n $(eta_update_target) .+= sum(B_smooth .* spatial_coeffs[M.s_idx, :], dims=2)\nend"
        return (priors=priors_str, update=update_str)
    end
    
    # The `:composition` operator is handled by dispatching to the specific component
    # type's code generator (e.g., `NonStationaryVariance`). This branch is no longer needed.
    @warn "Code generation for ComposedComponent with operator ':$op' is not explicitly handled here. Relying on downstream dispatch."
    return (priors="", update="")
end






function _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String})
    # Purpose: Creates the design matrix for all fixed effects, excluding the intercept.
    # Rationale: Consolidates fixed effects from bare terms and `fixed()` modules into a single design matrix.
    #            The intercept is handled separately by the `intercept()` module and its assembler block.
    # v1.0.0 (2026-07-16)
    # Assumptions: `fixed_effects_vars` contains the names of fixed effect variable names.
    # Inputs:
    #   - M: The model configuration dictionary.
    #   - fixed_effects_vars: A vector of fixed effect variable names.
    # Outputs: None (mutates `M`).
    
    if isempty(fixed_effects_vars)
        M[:Xfixed] = zeros(M[:y_N], 0)
        M[:Xfixed_N] = 0
        M[:Xfixed_names] = Symbol[]
        M[:Xfixed_applied_formula] = nothing
        return
    end

    rhs_vars = join(fixed_effects_vars, " + ")
    # We explicitly add "0" to the formula to tell StatsModels.jl NOT to create an intercept column.
    # The intercept is handled separately by the `intercept()` module logic in the assembler.
    rhs = "0 + " * rhs_vars
    
    Xfixed_named, applied_formula = create_fixed_design(rhs, M[:data]; contrasts=get(M, :contrasts, Dict()))
    if size(Xfixed_named, 1) != M[:y_N]
        @warn "Dimension mismatch in fixed effects design matrix: Expected $(M[:y_N]) rows, but got $(size(Xfixed_named, 1)). Attempting to reconcile."
        if size(Xfixed_named, 1) < M[:y_N]
            padded_Xfixed = zeros(M[:y_N], size(Xfixed_named, 2))
            padded_Xfixed[1:size(Xfixed_named, 1), :] = Matrix(Xfixed_named)
            M[:Xfixed] = padded_Xfixed
        else
            M[:Xfixed] = Matrix(Xfixed_named[1:M[:y_N], :])
        end
    else
        M[:Xfixed] = Matrix(Xfixed_named)
    end
    M[:Xfixed_N] = size(M[:Xfixed], 2)
    M[:Xfixed_names] = size(Xfixed_named, 2) > 0 ? names(Xfixed_named, 2) : Symbol[]
    M[:Xfixed_applied_formula] = applied_formula
end

function _canonical_term_string(term::StatsModels.AbstractTerm)
    if term isa StatsModels.InteractionTerm
        # Sort term names for canonical representation, e.g., "a&b" is the same as "b&a"
        term_names = sort([string(t.sym) for t in term.terms])
        return join(term_names, "&")
    elseif term isa StatsModels.Term
        return string(term.sym)
    elseif term isa StatsModels.ConstantTerm
        return "(Intercept)"
    else
        # Fallback for other term types like FunctionTerm, etc.
        # This might not be perfectly canonical but is a reasonable default.
        return string(term)
    end
end




function _precompute_static_components!(M::Dict)
    # Purpose: Pre-computes the Cholesky factorization for static components.
    # Rationale: Moves constant computations out of the MCMC loop to improve sampling speed.
    #            A component is "static" if its precision matrix structure does not depend on a
    #            hyperparameter that is sampled within the model (e.g., a `rho` parameter).
    # v1.0.1 (2026-07-27) - Corrected access to nested component objects.
    # Inputs:
    #   - M: The model configuration dictionary, which is mutated.
    # Outputs: None.
    noise = M[:noise]
    new_components = []
    # Define component types that do not have dynamic structure parameters like `rho`.
    static_component_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Spherical, Barycentric, TensorProductSmooth]

    for spec_in in M[:components]
        current_spec = spec_in
        m_obj = current_spec.component_obj

        # --- v1.0.0 (2026-07-20) ---
        # Rationale: This block is added to correctly handle wrapper components like
        #            MixedComponent and SVCComponent. It checks if their *inner* model
        #            is static. If so, it pre-computes the Cholesky factor and attaches
        #            it to the wrapper's spec. This resolves a FieldError where the
        #            code generator for the inner model would look for a `cholesky_factor`
        #            on the wrapper's spec, which didn't exist.
        if m_obj isa MixedComponent || m_obj isa SVCComponent
            inner_model = m_obj.model
            is_inner_static = any(T -> inner_model isa T, static_component_types)
            if is_inner_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
                try
                    Q_concrete = sparse(current_spec.Q_template)
                    F = cholesky(Symmetric(Q_concrete + noise * I))
                    final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                    push!(new_components, final_spec)
                    continue # Proceed to the next component in the loop
                catch e
                    @warn "Cholesky factorization failed for static inner model in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                end
            end
        end

        if m_obj isa ComposedComponent && m_obj.operator == :pipe
            state_spec = get(current_spec.hyper, :state_spec, nothing)
            if !isnothing(state_spec)
                # The state component is the second element in a `dynamic |> state` pipe.
                state_m_obj = m_obj.components[2]
                is_state_static = any(T -> state_m_obj isa T, static_component_types)

                if is_state_static && !isnothing(state_spec.Q_template) && size(state_spec.Q_template, 1) > 0
                    try
                        Q_concrete = sparse(state_spec.Q_template)
                        F = cholesky(Symmetric(Q_concrete + noise * I))
                        new_state_spec = merge(state_spec, (is_static=true, cholesky_factor=F))
                        new_hyper = merge(current_spec.hyper, (state_spec=new_state_spec,))
                        current_spec = merge(current_spec, (hyper=new_hyper,))
                    catch e
                        @warn "Cholesky factorization failed for static state component in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                    end
                end
            end
        end

        # Now check the main component object of the (potentially updated) current_spec
        is_main_static = !(current_spec.component_obj isa ComposedComponent) && any(T -> current_spec.component_obj isa T, static_component_types)

        if is_main_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
            try
                # Ensure Q_template is a concrete sparse matrix for Cholesky
                Q_concrete = sparse(current_spec.Q_template)
                F = cholesky(Symmetric(Q_concrete + noise * I))
                final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                push!(new_components, final_spec)
            catch e
                @warn "Cholesky factorization failed for static component $(current_spec.key). Reverting to dynamic computation. Error: $e"
                final_spec = merge(current_spec, (is_static=false,))
                push!(new_components, final_spec)
            end
        else
            # For composed components or dynamic components, just add them.
            # The is_static flag for the composed component itself remains false.
            final_spec = merge(current_spec, (is_static=get(current_spec, :is_static, false),))
            push!(new_components, final_spec)
        end
    end
    M[:components] = new_components
end


function _generate_component_code_fragments(m::ComposedComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    op = m.operator 
    key_str = string(spec.key)

    if op == :pipe
        dynamic_component = get(spec.hyper, :dynamic_component_obj, nothing)
        if isnothing(dynamic_component)
            # This check prevents the FieldError if build_model failed to populate the hyper registry.
            error("Internal Error: dynamic_component_obj not found for piped component '$(key_str)'. Check build_model logic for ComposedComponent.")
        end
        
        state_component = m.components[2] # The state component is the second one in a `dynamic |> state` pipe
        structure_str = string(spec.structure)
        is_multivariate = arch == "multivariate"
        is_first_outcome = outcome_idx == 1
        is_shared = get(spec.params, :shared, false)
        eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
        
        v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
        coeffs_raw_name = v.raw

        state_spec = spec.hyper.state_spec
        n_spatial = size(state_spec.Q_template, 1)
        n_basis = dynamic_component.nbins
        basis_key = get(spec.hyper, :dynamic_basis_key, nothing)
        if isnothing(basis_key); error("Could not find basis matrix key for piped component $(key_str)."); end
        
        priors_acc = String[]
        if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
            if hasproperty(state_component, :sigma); push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(state_component.sigma)), :$(v.sigma))"); end
            if hasproperty(state_component, :rho); push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(state_component.rho)), :$(v.rho))"); end
        end
        push!(priors_acc, "$(coeffs_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_spatial * n_basis)), I), :$(coeffs_raw_name))")
        priors_str = join(priors_acc, "\n        ")
        
        is_state_static = get(state_spec, :is_static, false)
        local cholesky_block
        if is_state_static
            cholesky_block = "F_spatial = spec_registry[\"$(key_str)\"].hyper.state_spec.cholesky_factor"
        else
            cholesky_block = "Q_spatial_template = spec_registry[\"$(key_str)\"].hyper.state_spec.Q_template\n state_m_type = spec_registry[\"$(key_str)\"].hyper.state_spec.model_type\n rho_val = $(hasproperty(state_component, :rho) ? v.rho : "nothing")\n Q_spatial = recompose_precision(state_m_type, Q_spatial_template, 1.0; extra_param=rho_val)\n F_spatial = cholesky(Symmetric(Q_spatial + noise * I))"
        end
        
        update_str = "begin\n $(cholesky_block)\n coeffs_raw_matrix = reshape($(coeffs_raw_name), $(n_spatial), $(n_basis))\n spatial_coeffs = $(v.sigma) .* (F_spatial.U \\ coeffs_raw_matrix)\n B_smooth = M.basis_matrices[:$(basis_key)]\n $(eta_update_target) .+= sum(B_smooth .* spatial_coeffs[M.s_idx, :], dims=2)\nend"
        return (priors=priors_str, update=update_str)
    end
    
    # The `:composition` operator is handled by dispatching to the specific component
    # type's code generator (e.g., `NonStationaryVariance`). This branch is no longer needed.
    @warn "Code generation for ComposedComponent with operator ':$op' is not explicitly handled here. Relying on downstream dispatch."
    return (priors="", update="")
end
 


function _initialize_config(data::DataFrame, kwargs)
    # Purpose: Creates the initial model configuration dictionary.
    # Rationale: Centralizes the creation of the `M` object.
    # Assumptions: `data` is a DataFrame.
    # Inputs:
    #   - data: The input DataFrame.
    #   - kwargs: Keyword arguments passed from the main `bstm` call.
    # Outputs: A dictionary `M`.
    M = Dict{Symbol, Any}()
    M[:data] = data
    M[:y_N] = nrow(data)
    
    # Set defaults that can be overridden by user kwargs
    M[:noise] = 1e-6
    M[:hyperpriors] = Dict{String, Any}()
    M[:prior_scheme] = :pcpriors
    M[:fixed_effects_priors] = Dict{Symbol, Any}()
    M[:spectral_orientation] = true

    for (k, v) in kwargs; M[k] = v; end
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:components] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end



function _process_lhs!(M::Dict, outcome_specs::Vector{Dict{Symbol, Any}})
    # Purpose: Processes the LHS of the formula, setting up outcomes and likelihoods.
    # Rationale: This version adds special handling for the `dirichlet_multinomial` family,
    #            which treats multiple variables specified with `+` as categories of a single
    #            response, rather than as separate outcome variables. This resolves an error
    #            where the parser would incorrectly try to validate each category as a
    #            standalone outcome.
    # v1.0.1 (2026-07-31)
    # Inputs:
    #   - M: The model configuration dictionary.
    #   - outcome_specs: A vector of parsed outcome specifications from the formula.
    # Outputs: None (mutates `M`).

    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    
    # Check the family from the first spec, assuming it's consistent for a `+` separated group.
    family_type = string(get(likelihood_specs[1], :family, "gaussian"))

    if family_type == "dirichlet_multinomial"
        # --- Special Handling for Dirichlet-Multinomial ---
        # In this case, `y1 + y2 + y3` represents categories of one response.
        
        if length(outcomes) <= 1
            error("The `dirichlet_multinomial` family requires multiple outcome variables specified with `+`, e.g., `likelihood(y1 + y2, ...)`.")
        end
        
        for out_sym in outcomes
            if !hasproperty(M[:data], out_sym)
                error("Outcome category variable ':$out_sym' for dirichlet_multinomial not found in the data frame.")
            end
        end

        M[:outcomes] = outcomes
        M[:outcomes_N] = length(outcomes)
        M[:model_arch] = "multivariate"
        M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        
        merged_params = Dict{Symbol, Any}()
        for spec in reverse(likelihood_specs); merge!(merged_params, spec); end
        M[:likelihood_specs] = [merged_params]

    else
        # --- Standard Handling for Other Families ---
        M[:outcomes] = outcomes
        M[:outcomes_N] = length(outcomes)
        M[:likelihood_specs] = likelihood_specs

        for (i, spec) in enumerate(M[:likelihood_specs])
            if !haskey(spec, :family)
                spec[:family] = "gaussian"
                @warn "Likelihood `family` not specified. Defaulting to `family=gaussian`."
            end
            
            if string(get(spec, :family, "")) == "ordinal"
                if M[:outcomes_N] > 1; error("The `ordinal` family is currently only supported for univariate models."); end
                outcome_var = M[:outcomes][i]
                if !hasproperty(M[:data], outcome_var); error("Ordinal outcome variable ':$outcome_var' not found in data."); end
                
                outcome_data = M[:data][!, outcome_var]
                if !(eltype(outcome_data) <: Integer)
                    @warn "Ordinal outcome variable ':$outcome_var' is not of integer type. Attempting to convert."
                    try; outcome_data = round.(Int, outcome_data); catch; error("Could not convert ordinal outcome variable ':$outcome_var' to integers."); end
                end
                
                unique_levels = sort(unique(outcome_data))
                K = length(unique_levels)
                if K < 2; error("Ordinal outcome variable ':$outcome_var' must have at least 2 unique levels."); end
                
                spec[:latent_dist] = get(spec, :latent_dist, :logistic)
                spec[:K] = K
                
                level_map = Dict(level => i for (i, level) in enumerate(unique_levels))
                M[:data][!, outcome_var] = [level_map[val] for val in outcome_data]
                @info "Ordinal outcome '$outcome_var' recoded to integers 1:$K."
            end
        end

        for out_sym in M[:outcomes]
            if !hasproperty(M[:data], out_sym)
                error("Outcome variable ':$out_sym' specified in the formula was not found as a column in the provided data frame. Please check for typos or ensure the column exists.")
            end
        end

        if M[:outcomes_N] > 1
            M[:model_arch] = "multivariate"
            M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        else
            M[:model_arch] = get(M, :model_arch, "univariate")
            M[:y_obs] = M[:data][!, M[:outcomes][1]]
        end
    end

    # --- Resolve Observation-Level Parameters ---
    calling_mod = get(M, :calling_module, Main)
    merged_params = Dict{Symbol, Any}()
    for spec_params in M[:likelihood_specs]; merge!(merged_params, spec_params); end

    _resolve_obs_param!(M, merged_params, M[:data], [:log_offsets, :offsets], :log_offsets)
    _resolve_obs_param!(M, merged_params, M[:data], [:weights], :weights)
    _resolve_obs_param!(M, merged_params, M[:data], [:trials], :trials)

    scalar_param_keys = [:censor_lower, :censor_upper, :hurdle]
    for key in scalar_param_keys
        values_per_outcome = []
        any_provided = false
        for spec_params in M[:likelihood_specs]
            val = _resolve_outcome_scalar_param!(spec_params, key, calling_mod)
            if !isnothing(val); any_provided = true; end
            push!(values_per_outcome, val)
        end

        if any_provided
            default_val = if key == :censor_lower || key == :hurdle; -Inf; else Inf; end
            final_values = [isnothing(v) ? default_val : v for v in values_per_outcome]
            M[key] = M[:outcomes_N] == 1 ? final_values[1] : final_values
            M[Symbol("user_provided_", key)] = true
        end
    end

    _resolve_boolean_obs_param!(M, merged_params, :zero_inflated, :use_zi)
    if get(M, :user_provided_hurdle, false) && get(M, :use_zi, false)
        @warn "Both `hurdle` and `zero_inflated` were specified. The hurdle model will be used and zero-inflation will be ignored."
        M[:use_zi] = false
    end
    _resolve_boolean_obs_param!(M, merged_params, :volatility, :volatility)
end





function _resolve_outcome_scalar_param!(params::Dict, key::Symbol, calling_mod::Module)
    # Purpose: Resolves a likelihood parameter that must be a scalar value for a given outcome.
    # Rationale: This enforces that parameters like `censor_lower` cannot be specified as
    #            per-observation vectors from the data, only as single scalar values.
    # v1.0.0 (2026-07-18)
    # Inputs:
    #   - params: The likelihood parameters from the formula.
    #   - key: The symbol for the parameter (e.g., `:censor_lower`).
    #   - calling_mod: The module context for evaluating symbols.
    # Outputs: The resolved scalar value, or `nothing`.
    if !haskey(params, key); return nothing; end

    val = params[key]
    if val isa Number
        return val
    elseif val isa Symbol || val isa Expr
        try
            evaluated_val = Core.eval(calling_mod, val)
            if evaluated_val isa Number
                return evaluated_val
            else
                @warn "Parameter '$val' for '$key' must be a scalar number, but evaluated to type '$(typeof(evaluated_val))'. Ignoring."
                return nothing
            end
        catch
            @warn "Parameter '$val' for '$key' could not be evaluated as a scalar variable in the calling module. Ignoring."
            return nothing
        end
    else
        @warn "Parameter for '$key' has an unsupported type '$(typeof(val))'. It must be a scalar number or a variable that evaluates to one. Ignoring."
        return nothing
    end
end


function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    # Purpose: Finds an observation-level parameter (like offsets or weights) in the data and adds it to the config.
    # Rationale: This version is updated to correctly handle cases where the provided argument is a variable
    #            in the calling scope that itself contains a Symbol or String referring to a data column.
    #            This resolves a common failure point when using the macro non-interactively.
    # v1.0.1 (2026-07-27)
    # Assumptions: `data` is a DataFrame.
    # Inputs:
    #   - opt_dict: The configuration dictionary to update.
    #   - params: The likelihood parameters from the formula.
    #   - data: The input DataFrame.
    #   - param_keys: A list of possible keys for the parameter (e.g., [:log_offsets, :offsets]).
    #   - target_key: The key to use in `opt_dict`.
    # Outputs: None (mutates `opt_dict`).
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol
                if hasproperty(data, val)
                    opt_dict[target_key] = data[!, val]
                    opt_dict[Symbol("user_provided_", target_key)] = true
                else
                    # The symbol might refer to a variable in the calling scope.
                    calling_mod = get(opt_dict, :calling_module, Main)
                    try
                        evaluated_val = Core.eval(calling_mod, val)
                        if evaluated_val isa Symbol && hasproperty(data, evaluated_val)
                            # Case: log_offsets = :my_col
                            opt_dict[target_key] = data[!, evaluated_val]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa String && hasproperty(data, Symbol(evaluated_val))
                            # Case: log_offsets = "my_col"
                            opt_dict[target_key] = data[!, Symbol(evaluated_val)]
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        elseif evaluated_val isa Number || evaluated_val isa AbstractVector
                            # Case: log_offsets = my_vector
                            opt_dict[target_key] = evaluated_val
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        else
                            @warn "Parameter '$val' for '$target_key' evaluated to an unsupported type '$(typeof(evaluated_val))'. Ignoring."
                        end
                    catch
                        @warn "Parameter '$val' for '$target_key' is not a valid column name and could not be evaluated as a variable in the calling module '$(calling_mod)'. Ignoring."
                    end
                end
            elseif val isa Number || val isa AbstractVector
                opt_dict[target_key] = val # The value was parsed directly as a number/vector.
                opt_dict[Symbol("user_provided_", target_key)] = true
            else
                @warn "Observation parameter '$val' for '$target_key' is not a valid column name, vector, or scalar. Ignoring."
            end
            return
        end
    end
end

 
function _resolve_boolean_obs_param!(opt_dict, params, param_key, target_key) 
    # Purpose: Resolves a boolean flag from the likelihood parameters.
    # Rationale: Handles boolean flags like `zero_inflated=true`. This version is more robust,
    #            handling symbols that evaluate to booleans and issuing warnings for invalid types.
    # v1.0.0 (2026-07-19)
    # Inputs:
    #   - opt_dict: The configuration dictionary to update.
    #   - params: The likelihood parameters from the formula.
    #   - param_key: The key for the boolean flag.
    #   - target_key: The key to set in `opt_dict`.
    # Outputs: None (mutates `opt_dict`).
    if haskey(params, param_key)
        val = params[param_key]
        if val isa Bool
            opt_dict[target_key] = val
        elseif val isa Symbol || val isa Expr
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                evaluated_val = Core.eval(calling_mod, val)
                if evaluated_val isa Bool
                    opt_dict[target_key] = evaluated_val
                else
                    @warn "Parameter '$val' for '$param_key' evaluated to a non-boolean type '$(typeof(evaluated_val))'. Ignoring."
                end
            catch
                @warn "Parameter '$val' for '$param_key' could not be evaluated as a boolean variable in the calling module. Ignoring."
            end
        else
            @warn "Parameter for '$param_key' has an unsupported type '$(typeof(val))'. It must be a boolean or a variable that evaluates to one. Ignoring."
        end
    end
end



function _process_fixed_effects_priors!(M::Dict) 
    # Purpose: Resolves and stores the prior distributions for each fixed effect coefficient.
    # Rationale: This version is updated to correctly evaluate `Expr` objects for priors,
    #            allowing users to specify distributions directly in the formula string.
    # v1.0.0 (2026-07-17)
    # Assumptions: `M[:Xfixed_applied_formula]` is populated by `_process_fixed_effects!`.
    # Inputs:
    #   - M: The model configuration dictionary.
    # Outputs: None (mutates `M`).
    n_fixed = get(M, :Xfixed_N, 0) 
    if n_fixed == 0
        M[:Xfixed_priors_vec] = UnivariateDistribution[]
        return
    end

    calling_mod = get(M, :calling_module, Main)
    custom_priors = M[:fixed_effects_priors]
    intercept_prior_val = get(M, :intercept_prior, nothing)
    default_prior = Normal(0, 5)
    priors_vec = Vector{Union{UnivariateDistribution, Nothing}}(undef, n_fixed)
    fill!(priors_vec, nothing)

    normalized_priors = Dict{String, Any}()
    for (key, prior) in custom_priors
        norm_key = replace(string(key), r"\s*[\*:]\s*" => "&")
        norm_key = replace(norm_key, r"\s*&\s*" => "&")
        
        if occursin("&", norm_key)
            parts = sort(Base.split(norm_key, '&'))
            norm_key = join(parts, '&')
        end
        normalized_priors[norm_key] = prior
    end

    applied_formula = get(M, :Xfixed_applied_formula, nothing)
    if isnothing(applied_formula)
        @warn "Could not find the applied formula for fixed effects. Prior assignment may be incomplete. This is an internal issue."
        M[:Xfixed_priors_vec] = fill(default_prior, n_fixed)
        return
    end

    all_coef_names = string.(coefnames(applied_formula.rhs))
    coef_name_to_idx = Dict(name => i for (i, name) in enumerate(all_coef_names))
    processed_indices = Set{Int}()

    for term in applied_formula.rhs.terms
        canonical_name = _canonical_term_string(term)
        
        if haskey(normalized_priors, canonical_name)
            prior_val = normalized_priors[canonical_name]
            local prior_obj
            if prior_val isa Expr
                try; prior_obj = Core.eval(calling_mod, prior_val);
                catch e; error("Could not evaluate `prior` argument `$(prior_val)` for fixed effect '$canonical_name'. Error: $e"); end
            else
                prior_obj = prior_val
            end

            term_coef_names = coefnames(term)
            
            term_coef_names_vec = term_coef_names isa AbstractString ? [term_coef_names] : term_coef_names

            for coef_name in term_coef_names_vec
                if haskey(coef_name_to_idx, coef_name)
                    idx = coef_name_to_idx[coef_name]
                    priors_vec[idx] = prior_obj isa Tuple ? create_pc_prior(Symbol(canonical_name), prior_obj) : prior_obj
                    push!(processed_indices, idx)
                else
                    @warn "Coefficient name '$coef_name' for term '$canonical_name' not found in the full coefficient list. Prior may not be applied."
                end
            end
        end
    end

    for i in 1:n_fixed
        if !(i in processed_indices)
            coef_name_str = all_coef_names[i]
            if coef_name_str == "(Intercept)"
                prior = intercept_prior_val
                priors_vec[i] = isnothing(prior) ? default_prior : (prior isa Tuple ? create_pc_prior(:intercept, prior) : prior)
            else
                priors_vec[i] = default_prior
            end
        end
    end

    M[:Xfixed_priors_vec] = convert(Vector{UnivariateDistribution}, priors_vec) 
end 


function _finalize_config!(M::Dict)
    # Purpose: Ensures the configuration dictionary has all necessary keys with default values.
    # Rationale: Prevents `KeyError` exceptions in the model assembler and execution.
    # Assumptions: `M` is a valid configuration dictionary.
    # Inputs:
    #   - M: The model configuration dictionary.
    # Outputs: None (mutates `M`).
    defaults = Dict(
        :s_N => 0, :t_N => 0, :u_N => 0,
        :s_idx => ones(Int, M[:y_N]),
        :t_idx => ones(Int, M[:y_N]),
        :u_idx => ones(Int, M[:y_N]),
        :log_offset => zeros(M[:y_N]),
        :weights => ones(M[:y_N]),
        :trials => ones(Int, M[:y_N]),
        :hyperpriors => Dict(),
        :prior_scheme => :pcpriors,
        :intercept_prior => Normal(0, 5)
    )
    for (key, val) in defaults
        if !haskey(M, key)
            M[key] = val
        end
    end
end


# ==============================================================================
# SECTION 5: MODULE-SPECIFIC PROCESSORS
# ==============================================================================

 

function process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `spatial()` module call.
    # Rationale: Handles the setup of the adjacency matrix `W` and spatial indices `s_idx`.
    # v1.0.0 (2026-07-16)
    #            If `W` is not provided, it attempts to infer it from coordinates.
    # Assumptions: `data` is present in `opt_dict`.
    # Inputs:
    #   - opt_dict: The main configuration dictionary.
    #   - mod_data: The parsed data for this specific module.
    #   - registries, hyperpriors: Not used here, but part of the standard processor signature.
    # Outputs: None (mutates `opt_dict`).
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]
    
    # # Check if this is a specialized point process like LGCP
    point_process = get(params, :point_process, nothing)
    
    if point_process == :lgcp
        # # Re-tag module type for correct builder dispatch
        mod_data[:type] = :lgcp
        # # Ensure the underlying spatial model is specified
        if !haskey(params, :model)
            params[:model] = :icar
            @warn "LGCP point process requested without a model. Defaulting to :icar."
        end
    end

    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                opt_dict[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` in spatial module. Error: $e")
            end
        else
            opt_dict[:W] = w_val
        end
    end
    
    if !haskey(opt_dict, :W)
        @warn "Adjacency matrix 'W' not provided for spatial module. Attempting to infer from coordinates."
        if hasproperty(data, :s_x) && hasproperty(data, :s_y)
            au = assign_spatial_units(Matrix(data[!, [:s_x, :s_y]]); target_units=get(params, :target_units, 50))
            opt_dict[:W] = au.W
            opt_dict[:s_idx] = au.s_idx
            opt_dict[:s_N] = size(au.W, 1)
            opt_dict[:centroids] = au.centroids
        else
            error("Cannot infer spatial structure without 'W' or coordinate columns 's_x', 's_y'.")
        end
    else
        opt_dict[:s_N] = size(opt_dict[:W], 1)
        if !isempty(variables)
            s_var_sym = Symbol(variables[1])
            if hasproperty(data, s_var_sym)
                opt_dict[:s_idx] = data[!, s_var_sym]
            else
                @warn "Spatial index variable ':$s_var_sym' not found. Ensure data is aligned with W."
            end
        end
    end
    
    # # Grid Area Resolution for LGCP
    if point_process == :lgcp
        if haskey(params, :grid_areas)
            ga_val = params[:grid_areas]
            if ga_val isa Symbol && hasproperty(data, ga_val)
                opt_dict[:grid_areas] = data[!, ga_val]
            elseif ga_val isa AbstractVector
                opt_dict[:grid_areas] = ga_val
            else
                # # Fallback to evaluating symbol in calling module
                calling_mod = get(opt_dict, :calling_module, Main)
                try
                    opt_dict[:grid_areas] = Core.eval(calling_mod, ga_val)
                catch
                    @warn "Could not resolve grid_areas. Defaulting to unit areas."
                    opt_dict[:grid_areas] = ones(opt_dict[:s_N])
                end
            end
        else
            opt_dict[:grid_areas] = ones(opt_dict[:s_N])
        end
    end

    return true
end
 
"""
    process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, components_registry)

Processes the `smooth()` module call, with updated logic for dynamic basis models.

# Rationale for Update
This version introduces a distinction between static basis models (like `pspline`) and
dynamic basis models (`wavelet`, `fft`). For dynamic models, where the basis functions
depend on a hyperparameter like `lengthscale`, this function no longer pre-computes the
basis matrix. Instead, it stores the coordinate data in the module's parameters. This
allows a specialized code generator to construct the basis matrix dynamically inside the
Turing model using the sampled `lengthscale` value, thus making `lengthscale` an
estimated parameter. It also correctly computes and stores the number of bins per
dimension (`nbins_per_dim`) for use by the code generators.
"""
function process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, components_registry)
    local registry_to_use
    if basis_matrices_registry isa Dict && haskey(basis_matrices_registry, :basis_matrices)
        registry_to_use = basis_matrices_registry[:basis_matrices]
    else
        registry_to_use = basis_matrices_registry
    end
    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")
    original_nbins_param = get(params, :nbins, 20)
    variables = mod_data[:variables]
    n_vars = length(variables)
    
    local total_bins_for_component_obj
    local nbins_per_dim_vec
    if n_vars > 0
        if original_nbins_param isa Int
            nbins_per_dim_vec = fill(original_nbins_param, n_vars)
            total_bins_for_component_obj = original_nbins_param^n_vars
        elseif original_nbins_param isa Vector{Int}
            if length(original_nbins_param) != n_vars
                error("`nbins` vector length must match number of variables for smooth. Got $(length(original_nbins_param)) for $n_vars variables.")
            end
            nbins_per_dim_vec = original_nbins_param
            total_bins_for_component_obj = prod(original_nbins_param)
        else
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                evaluated_nbins = Core.eval(calling_mod, original_nbins_param)
                temp_params = copy(params)
                temp_params[:nbins] = evaluated_nbins
                temp_mod_data = Dict(:type => mod_data[:type], :params => temp_params, :variables => variables)
                return process_smooth_module!(opt_dict, temp_mod_data, basis_matrices_registry, components_registry)
            catch e
                error("Could not evaluate `nbins` parameter `$(original_nbins_param)`. Error: $e")
            end
        end
        mod_data[:params][:nbins] = total_bins_for_component_obj
        if @isdefined nbins_per_dim_vec; mod_data[:params][:nbins_per_dim] = nbins_per_dim_vec; end
    end
    basis_models = ["pspline", "bspline", "tps", "moran", "spherical", "barycentric", "decay", "linear", "invdist", "kriging"]
    dynamic_basis_models = ["wavelet", "fft"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay", "rff", "kriging"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]
    model_str = string(model_param)
    if model_str in dynamic_basis_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
        else
            error("Coordinate variables for smooth model not found in data: $(mod_data[:variables])")
        end
        return true
    end
    if model_str in basis_models
        if !isempty(mod_data[:variables]) 
            reg_key = Symbol(join(mod_data[:variables], "_"))
            if all(hasproperty(data, Symbol(v)) for v in mod_data[:variables])
                n_vars = length(mod_data[:variables])
                
                local_kwargs = Dict(params)
                delete!(local_kwargs, :nbins)
                local B_smooth_matrix
                local actual_nbins_for_component = total_bins_for_component_obj # Default, will be updated for 1D
                if n_vars == 1
                    v_vec = data[!, Symbol(mod_data[:variables][1])]
                    B_smooth_matrix, actual_nbins_for_component = bstm_smooth_basis_1D(model_str, v_vec, nbins_per_dim_vec[1], get(params, :degree, 3); local_kwargs...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
                    if n_vars == 2; B_smooth_matrix = bstm_smooth_basis_2D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...);
                    elseif n_vars == 3; B_smooth_matrix = bstm_smooth_basis_3D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...);
                    elseif n_vars == 4; B_smooth_matrix = bstm_smooth_basis_4D(model_str, c_mat, nbins_per_dim_vec; local_kwargs...);
                    end
                    actual_nbins_for_component = size(B_smooth_matrix, 2) # For multi-D, actual is size of matrix
                end
                registry_to_use[reg_key] = B_smooth_matrix
                mod_data[:params][:nbins] = actual_nbins_for_component # Update nbins with the actual count
            end
        end
    elseif model_str in continuous_kernel_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            coords = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            mod_data[:params][:coords] = coords
            if model_str in ["fitc", "svgp", "nystrom", "gp"]
                n_inducing_default = min(100, size(coords, 1))
                n_inducing = get(mod_data[:params], :n_inducing, n_inducing_default)
                Z_inducing = generate_inducing_points(coords, n_inducing; seed=42, method="kmeans")
                mod_data[:params][:Z_inducing] = Z_inducing
            end
        else
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Component may be misspecified."
        end
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) != 1; @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."; return true; end
        
        index_key = Symbol("mixed_idx_$(string(vars[1]))")
        var_sym = Symbol(vars[1])
        nbins = get(mod_data[:params], :nbins, 20)
        _, indices = apply_discretization_logic(data[!, var_sym], nbins)
        opt_dict[index_key] = indices
        mod_data[:params][:indices] = indices
        mod_data[:params][:n_cat] = length(unique(indices))
        mod_data[:type] = :mixed
    end    
    mod_data[:params][:model] = model_param
    return true
end


 

function process_lgcp_module!(opt_dict, mod_data, registries, hyperpriors)
    # The LGCP module itself doesn't directly set spatial indices, its inner model does.
    # We need to ensure the spatial context for the inner model is established in opt_dict.

    # Create a temporary mod_data_dict for the inner spatial component.
    # This will be passed to process_spatial_module! to set opt_dict[:s_N], opt_dict[:s_idx], opt_dict[:W].
    inner_spatial_mod_data = Dict(
        :type => :spatial, # Explicitly process as a spatial module
        :params => mod_data[:params], # Pass all parameters from the LGCP module, including W, s_idx, etc.
        :variables => get(mod_data[:params], :positional_args, []) # Positional args for the inner random()
    )
    
    # Call the spatial processor. This will set opt_dict[:s_N], opt_dict[:s_idx], opt_dict[:W]
    # based on the parameters provided in the LGCP module (which came from the inner random()).
    process_spatial_module!(opt_dict, inner_spatial_mod_data, registries, hyperpriors)

    # Now handle LGCP-specific parameters like grid_areas, which depend on s_N.
    if haskey(mod_data[:params], :grid_areas)
        ga_val = mod_data[:params][:grid_areas]
        if ga_val isa Symbol && hasproperty(opt_dict[:data], ga_val)
            opt_dict[:grid_areas] = opt_dict[:data][!, ga_val]
        elseif ga_val isa AbstractVector
            opt_dict[:grid_areas] = ga_val
        else
            calling_mod = get(opt_dict, :calling_module, Main)
            try
                opt_dict[:grid_areas] = Core.eval(calling_mod, ga_val)
            catch
                @warn "Could not resolve grid_areas. Defaulting to unit areas."
                opt_dict[:grid_areas] = ones(opt_dict[:s_N])
            end
        end
    else
        # If grid_areas is not specified, default to ones(s_N).
        # This requires s_N to be set by process_spatial_module! already.
        opt_dict[:grid_areas] = ones(opt_dict[:s_N])
    end

    return true # Continue to create the LGCP component object via resolve_technical_primitive
end



function process_dynamics_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    # Purpose: Processes the `dynamics()` module, ensuring spatial and temporal contexts are established.
    # Rationale: This version is updated to correctly identify and store `grid_areas` in the model
    #            configuration. These areas are essential for biological models that operate on
    #            population densities, allowing conversion between total population and density.
    # v1.0.1 (2026-07-31)

    # 1. Initialize Spatial Context
    process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)

    params = mod_data[:params]
    data = opt_dict[:data]
    
    # 2. Model Type Verification
    model_type = string(get(params, :model, "none"))
    if model_type == "none"
        error("Dynamics module requires a 'model' parameter (e.g., model='advection').")
    end

    # 3. Covariate and Parameter Validation (existing logic)
    if model_type in ["advection", "advection_diffusion"]
        if !haskey(params, :velocity_prior) && !haskey(opt_dict[:hyperpriors], "velocity")
            @warn "Advection model specified without explicit velocity priors. Using system defaults."
        end
    end

    if model_type in ["diffusion", "advection_diffusion"]
        if !haskey(params, :diffusion_prior) && !haskey(opt_dict[:hyperpriors], "diffusion")
            @warn "Diffusion model specified without explicit diffusion priors. Using system defaults."
        end
    end

    # 4. Spatiotemporal Indexing (existing logic)
    if !haskey(opt_dict, :t_idx)
        @warn "Dynamics module detected but no temporal indices found. Attempting default temporal resolution."
        if hasproperty(data, :year)
            opt_dict[:t_idx] = data[!, :year] .- minimum(data[!, :year]) .+ 1
            opt_dict[:t_N] = length(unique(opt_dict[:t_idx]))
        else
            error("Dynamics models require temporal indices. Provide a time variable via temporal() or ensure 'year' is in data.")
        end
    end

    # 5. Ensure `grid_areas` are available for density-based models
    if !haskey(opt_dict, :grid_areas)
        if haskey(params, :grid_areas)
            ga_val = params[:grid_areas]
            if ga_val isa Symbol && hasproperty(data, ga_val)
                opt_dict[:grid_areas] = data[!, ga_val]
            elseif ga_val isa AbstractVector
                opt_dict[:grid_areas] = ga_val
            else
                calling_mod = get(opt_dict, :calling_module, Main)
                try
                    opt_dict[:grid_areas] = Core.eval(calling_mod, ga_val)
                catch
                    @warn "Could not resolve grid_areas. Defaulting to unit areas."
                    opt_dict[:grid_areas] = ones(opt_dict[:s_N])
                end
            end
        else
            # Default to unit areas if not specified
            opt_dict[:grid_areas] = ones(opt_dict[:s_N])
        end
    end

    # 6. Mapping Spatiotemporal State (existing logic)
    s_idx = opt_dict[:s_idx]
    t_idx = opt_dict[:t_idx]
    s_N = opt_dict[:s_N]
    
    opt_dict[:st_idx] = [(t - 1) * s_N + s for (s, t) in zip(s_idx, t_idx)]

    return true
end




"""
    process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `eigen()` module call.

# Rationale for Update
The original implementation did not extract the data for the variables specified in the
`eigen()` call, making it impossible to perform PCA. This updated version correctly
extracts the relevant columns from the data frame, centers them, and stores the
resulting matrix in the module's parameters for use by the model builder and code
generator. It also validates the number of factors against the number of variables.
"""
function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    params = mod_data[:params]
    vars_str = mod_data[:variables]
    vars_sym = Symbol.(vars_str)
    
    if isempty(vars_sym)
        error("The `eigen()` module was called without any variables specified.")
    end

    data = opt_dict[:data]
    if !all(hasproperty(data, v) for v in vars_sym)
        missing_vars = filter(v -> !hasproperty(data, v), vars_sym)
        error("Eigen module variables not found in data: $(missing_vars)")
    end
    
    # Extract the data and center it (a standard assumption for PCA).
    eigen_data_matrix = Matrix(data[!, vars_sym])
    eigen_data_matrix .-= mean(eigen_data_matrix, dims=1)
    
    # Store the data matrix in the module's parameters for the builder to access.
    mod_data[:params][:eigen_data] = eigen_data_matrix
    
    n_vars = length(vars_sym)
    n_factors = get(params, :n_factors, 1)
    if n_factors >= n_vars
        @warn "Number of factors ($n_factors) for eigen() module should be less than the number of variables ($n_vars). Setting to $(n_vars - 1)."
        n_factors = n_vars - 1
    end
    
    # Pre-calculate indices for the lower-triangular part of the Householder matrix.
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))
    
    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors
    mod_data[:params][:n_vars] = n_vars
    
    return true # Proceed with component creation.
end


"""
    _replace_bstm_modules_in_expr(ex)

Recursively traverses a Julia expression and replaces `bstm`-specific modules
with their `StatsModels.jl` equivalents for parsing within other modules like `mixed()`.
- `intercept()` becomes `1`.
- `fixed(x)` becomes `x`.

# Arguments
- `ex`: A Julia expression, symbol, or literal.

# Returns
- The modified expression.
"""
function _replace_bstm_modules_in_expr(ex)
    if ex isa Expr && ex.head == :call
        if ex.args[1] == :intercept
            return 1
        elseif ex.args[1] == :fixed && length(ex.args) > 1
            return ex.args[2] # Return the variable inside fixed()
        end
        return Expr(ex.head, _replace_bstm_modules_in_expr.(ex.args)...)
    elseif ex isa Expr
        return Expr(ex.head, _replace_bstm_modules_in_expr.(ex.args)...)
    else
        return ex
    end
end


"""
    process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)

Processes the `mixed()` module for random effects. This version uses `StatsModels.jl`
to correctly parse the effects part of the mixed model formula (e.g., `intercept() + cov1`).
It handles `intercept()` by converting it to `1` and correctly identifies all terms, which
are then passed to the code generator as a vector of strings. This fixes a bug where
`intercept()` was treated as a string literal, causing a `ParseError`.

# Arguments
- `opt_dict`: The main model configuration dictionary.
- `mod_data`: The parsed data for the `mixed()` module.
- `registries`, `hyperpriors`: Additional configuration dictionaries.

# Returns
- `true` to indicate that a `MixedComponent` object should be created.
"""
function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    response_var = Symbol(opt_dict[:outcomes][1])

    local effect_expr, group_var_str
    if !isempty(vars) && vars[1] isa Expr && vars[1].head == :call && vars[1].args[1] == :|
        effect_expr = vars[1].args[2]
        group_expr = vars[1].args[3]
        group_var_str = string(group_expr)
    elseif length(vars) >= 2
        effect_expr = vars[1]
        group_var_str = string(vars[2])
    else
        @warn "The mixed() module requires syntax `mixed(effect | group)` or `mixed(effect, group)`. Skipping."
        return false
    end

    effect_expr_mod = _replace_bstm_modules_in_expr(effect_expr)
    schema = StatsModels.schema(data)
    local terms

    if effect_expr_mod isa Number
        terms = StatsModels.term(effect_expr_mod)
    else
        calling_mod = get(opt_dict, :calling_module, Main)
        form = Core.eval(calling_mod, :(@formula($response_var ~ $(effect_expr_mod))))
        applied_form = StatsModels.apply_schema(form, schema)
        terms = applied_form.rhs
    end

    # Correctly handle Tuple and TupleTerm from StatsModels to decompose multi-term effects.
    term_vec = if terms isa StatsModels.TupleTerm
        terms.terms
    elseif terms isa StatsModels.AbstractTerm
        (terms,) # Wrap single AbstractTerm in a tuple for consistent iteration
    elseif terms isa Tuple # Fallback for raw tuples, though less common from StatsModels.jl
        collect(terms)
    else
        [terms]
    end
    
    effect_names = String[]
    for term in term_vec
        if term isa StatsModels.InterceptTerm{true}
            push!(effect_names, "1")
        elseif term isa StatsModels.InterceptTerm{false}
            continue
        else
            push!(effect_names, _canonical_term_string(term))
        end
    end

    group_var_sym = Symbol(group_var_str)
    if !hasproperty(data, group_var_sym)
        error("Grouping variable ':$group_var_sym' for mixed() module not found in dataset.")
    end
    
    group_data = data[!, group_var_sym]
    unique_levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(unique_levels))
    indices = [group_map[v] for v in group_data]

    index_key = Symbol("mixed_idx_$(group_var_str)")
    opt_dict[index_key] = indices
    
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = length(unique_levels)
    mod_data[:params][:lhs] = effect_names
    mod_data[:variables] = [group_var_str]
    
    return true
end



"""
    _generate_component_code_fragments(m::CustomComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for a user-defined `CustomComponent`.

# Rationale for New Implementation
The previous implementation incorrectly dispatched `CustomComponent` to a generic GMRF
generator, ignoring the user-provided code. This new, specialized function correctly
implements the intended behavior by directly injecting the user's code into the model.

The `code_fragment` provided by the user in the `custom()` module is expected to be a
complete and valid block of Turing model code. This block is inserted directly into the
model's main assembly block. The user is responsible for defining any necessary priors
and update logic within this fragment. The function returns an empty `priors` string
and places the entire user code into the `update` string, as Turing does not
distinguish between these contexts within the `@model` macro.
"""
function _generate_component_code_fragments(m::CustomComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # The user's code fragment is expected to be a self-contained block
    # that includes both prior definitions and update logic for the linear predictor.
    
    user_code = m.code_fragment
    
    if isempty(strip(user_code))
        @warn "Custom component '$(spec.key)' was specified but the `code_fragment` is empty. This component will have no effect."
        return (priors="", update="")
    end

    # The entire user code is treated as an update block.
    # Turing doesn't distinguish between prior and update sections inside the @model macro,
    # so this is a valid approach. The user must ensure their code is correct and
    # that any new parameter names are unique to avoid collisions.
    update_str = """
    begin
        # --- Custom Code Block for $(spec.key) ---
        $(user_code)
    end
    """
    
    # Return an empty priors string as all logic is contained in the update block.
    return (priors="", update=update_str)
end




function _generate_component_code_fragments(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    is_spatiotemporal = hasproperty(spec.hyper, :temporal_spec)
    n_latent_dims = is_spatiotemporal ? "M.s_N * M.t_N" : "M.s_N"

    priors = """
    # LGCP Intensity Field Priors
    $(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))
    $(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_dims)), I), :$(v.raw))
    """

    update = """
    begin
        # LGCP Model: $(key_str)
        local latent_field_st = zeros(T, M.s_N, M.t_N)
        
        # 1. Reconstruct the latent spatiotemporal field Z(s,t)
        if $(is_spatiotemporal)
            # Spatiotemporal case with Kronecker solver
            local s_spec = spec_registry["$(key_str)"].hyper.inner_spec
            local t_spec = spec_registry["$(key_str)"].hyper.temporal_spec
            
            local C_s = cholesky(Symmetric(s_spec.Q_template + noise * I))
            local C_t = cholesky(Symmetric(t_spec.Q_template + noise * I))
            
            local Z_matrix = reshape($(v.raw), M.s_N, M.t_N)
            
            # Solve X = L_s^{-T} * Z * L_t^{-1} via backslash with upper Cholesky factors
            local tmp_spatial = C_s.U \\ Z_matrix
            latent_field_st = (transpose(C_t.U \\ transpose(tmp_spatial))) .* $(v.sigma)
        else
            # Purely spatial case
            local Q_lgcp = spec_registry["$(key_str)"].hyper.inner_spec.Q_template
            local F_lgcp = cholesky(Symmetric(Q_lgcp + noise * I))
            local spatial_component = $(v.sigma) .* (F_lgcp.U \\ $(v.raw))
            # Broadcast spatial component across time
            latent_field_st = repeat(spatial_component, 1, M.t_N)
        end

        # 2. Assemble the full log-intensity surface.
        # This combines the main linear predictor `eta` (containing intercept, fixed effects, and SVCs)
        # with the latent spatiotemporal field `latent_field_st`.
        local log_intensity_surface = zeros(T, M.s_N, M.t_N)
        for t in 1:M.t_N, s in 1:M.s_N
            # Find all observation indices that fall into this space-time cell.
            obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:N)
            
            # Average the linear predictor `eta` for all observations in the cell.
            # This correctly incorporates observation-level covariate effects (including SVCs).
            base_contribution = isempty(obs_indices) ? 0.0 : mean(view(eta, obs_indices))
            
            log_intensity_surface[s, t] = base_contribution + latent_field_st[s, t]
        end

        # 4. Point Process Likelihood Evaluation
        # Integral of intensity over the domain is approximated by cell-wise summation
        # The data `M.y_obs` is assumed to be a matrix of counts of size (s_N, t_N).
        local grid_areas = spec_registry["$(key_str)"].hyper.areas
        for t in 1:M.t_N, s in 1:M.s_N
            local y_st = M.y_obs[s, t]
            local A_s = grid_areas[s]
            local Z_st = log_intensity_surface[s, t]
            
            Turing.@addlogprob! (y_st * (Z_st + log(A_s + noise)) - A_s * exp(Z_st))
        end

        # The flag M[:likelihood_handled] = true has been removed.
    end
    """

    return (priors=priors, update=update)
end


 

function process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)
    models = get(mod_data[:params], :model, (:iid, :iid))
    
    local spatial_model, temporal_model
    if models isa Expr && models.head == :tuple && length(models.args) == 2
        spatial_model = string(models.args[1])
        temporal_model = string(models.args[2])
    elseif models isa Tuple && length(models) == 2
        spatial_model = string(models[1])
        temporal_model = string(models[2])
    else
        error("The `model` for a spacetime interaction must be a tuple of two models, e.g., `model=(icar, ar1)`.")
    end

    has_structured_space = spatial_model != "iid"
    has_structured_time = temporal_model != "iid"

    if has_structured_space && has_structured_time
        opt_dict[:model_st] = "IV"
    elseif !has_structured_space && has_structured_time
        opt_dict[:model_st] = "II"
    elseif has_structured_space && !has_structured_time
        opt_dict[:model_st] = "III"
    else # !has_structured_space && !has_structured_time
        opt_dict[:model_st] = "I"
    end

    if haskey(mod_data[:params], :sigma)
        prior_val = mod_data[:params][:sigma]
        calling_mod = get(opt_dict, :calling_module, Main)
        if prior_val isa Tuple
            opt_dict[:st_interaction_sigma_prior] = create_pc_prior(:sigma, prior_val)
        elseif prior_val isa Expr
            try
                opt_dict[:st_interaction_sigma_prior] = Core.eval(calling_mod, prior_val)
            catch e
                error("Could not evaluate `prior` argument `$(prior_val)` for spacetime interaction. Error: $e")
            end
        else
            opt_dict[:st_interaction_sigma_prior] = prior_val
        end
    end
    
    # This module only sets flags; it does not create a component itself.
    return false
end
 



function process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `fixed()` module.
    # Rationale: Gathers information about fixed effects, including custom contrasts and priors.
    # v1.0.0 (2026-07-16)
    # Assumptions: `mod_data` contains variables and optional parameters.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `opt_dict`).
    # Returns: A boolean indicating whether a standard component object should be created for this module.
    if !haskey(opt_dict, :fixed_effects); opt_dict[:fixed_effects] = String[]; end
    if !haskey(opt_dict, :contrasts); opt_dict[:contrasts] = Dict{Symbol, Any}(); end
    if !haskey(opt_dict, :fixed_effects_priors); opt_dict[:fixed_effects_priors] = Dict{Symbol, Any}(); end
    if !haskey(opt_dict, :vars_to_categorize); opt_dict[:vars_to_categorize] = Set{Symbol}(); end
    params = mod_data[:params]
    vars = mod_data[:variables]
    # The fixed effects list is now populated in `_categorize_rhs_nodes!`.
    # This processor only handles side-effects like contrasts and priors.
    if haskey(params, :contrast)
        if !isempty(vars)
            contrast_sym = params[:contrast]
            if haskey(STATSMODELS_CONTRASTS, contrast_sym)
                opt_dict[:contrasts][Symbol(vars[1])] = STATSMODELS_CONTRASTS[contrast_sym]
            else
                @warn "Unknown contrast coding ':$contrast_sym'. Using default (DummyCoding)."
                opt_dict[:contrasts][Symbol(vars[1])] = StatsModels.DummyCoding()
            end
        else
            @warn "A 'contrast' was specified in a fixed() module with no variable. Ignoring."
        end
    end
    if haskey(params, :prior)
        for var in vars
            opt_dict[:fixed_effects_priors][Symbol(var)] = params[:prior]
        end
    end
    if get(params, :model, nothing) == :categorical || haskey(params, :contrast)
        for var in vars
            push!(opt_dict[:vars_to_categorize], Symbol(var))
        end
    end
    return false # fixed() modules do not create components in the main loop.
end

function process_custom_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `custom()` module.
    # Rationale: Placeholder for user-defined custom model components.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None.
    # Returns: A boolean indicating whether a standard component object should be created for this module.
    return true
end





function adjacency_to_bipartite(W::AbstractMatrix; force_bipartite::Bool=true)
    # # Process: Translates a unipartite graph into a bipartite representation.
    # # Rationale: Required for models like BCGN that operate on inter-set connectivity.
    
    rows, cols = size(W)
    if rows != cols
        error("Input matrix must be square to represent a unipartite adjacency structure.")
    end
    
    n = rows
    g = SimpleGraph(W)
    
    # # Coloring Algorithm: Attempt to find a natural 2-coloring (bipartition)
    # # nodes are assigned to set 0 or set 1
    colors = fill(-1, n)
    is_bipartite = true
    
    for start_node in 1:n
        if colors[start_node] != -1
            continue
        end
        
        colors[start_node] = 0
        queue = [start_node]
        
        while !isempty(queue)
            u = popfirst!(queue)
            for v in Neighbors(g, u)
                if colors[v] == -1
                    colors[v] = 1 - colors[u]
                    push!(queue, v)
                elseif colors[v] == colors[u]
                    is_bipartite = false
                    if !force_bipartite
                        error("Graph is not bipartite and force_bipartite is false.")
                    end
                end
            end
        end
    end
    
    # # Fallback: If not bipartite, use a greedy degree-based partition to maximize cut
    if !is_bipartite
        @warn "Graph is not naturally bipartite. Applying greedy partitioning to maximize inter-set edges."
        colors = fill(0, n)
        node_degrees = degree(g)
        sorted_nodes = sortperm(node_degrees, rev=true)
        
        for u in sorted_nodes
            # # Count neighbors already in set 0 and set 1
            n0 = 0
            n1 = 0
            for v in Neighbors(g, u)
                if colors[v] == 0
                    n0 += 1
                else
                    n1 += 1
                end
            end
            # # Assign to the set that maximizes connections to the other set
            colors[u] = n0 >= n1 ? 1 : 0
        end
    end
    
    # # Extraction: Construct the bipartite matrix B
    set1_indices = findall(==(0), colors)
    set2_indices = findall(==(1), colors)
    
    n1 = length(set1_indices)
    n2 = length(set2_indices)
    
    if n1 == 0 || n2 == 0
        error("Partitioning failed to create two non-empty sets. Check graph connectivity.")
    end
    
    # # B is n1 x n2 matrix representing connections from Set 1 to Set 2
    B = spzeros(Float64, n1, n2)
    
    for (i, u) in enumerate(set1_indices)
        for (j, v) in enumerate(set2_indices)
            if W[u, v] > 0
                B[i, j] = Float64(W[u, v])
            end
        end
    end
    
    return (
        bipartite_adj = B,
        set1 = set1_indices,
        set2 = set2_indices,
        is_natural = is_bipartite
    )
end


function process_bcgn_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `bcgn()` module for bipartite graphs.
    # Rationale: Validates the provided bipartite adjacency matrix.
    # v1.0.0 (2026-07-16)
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Boolean indicating if a component should be created.
    params = mod_data[:params]
    if !haskey(params, :W) || isempty(params[:W]) || all(iszero, params[:W])
        error("The `bcgn()` module requires a non-empty `:W` sparse matrix parameter.")
    end

    res = adjacency_to_bipartite(params[:W])
    params[:bipartite_adj] = res.bipartite_adj

    # Further validation could be added here (e.g., check if it's actually bipartite)
    return true # Proceed with component creation
end

function process_networkflow_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `networkflow()` module.
    # Rationale: Validates the provided adjacency matrix for the network flow model.
    # v1.0.0 (2026-07-16)
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Boolean indicating if a component should be created.
    params = mod_data[:params]
    if !haskey(params, :adjacency_matrix) || isempty(params[:adjacency_matrix]) || all(iszero, params[:adjacency_matrix])
        error("The `networkflow()` module requires a non-empty `:adjacency_matrix` sparse matrix parameter.")
    end
    return true # Proceed with component creation
end


function process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `localadaptive` module, ensuring centroids are computed for clustering.
    # Rationale: This version is updated to resolve an `UndefVarError` by explicitly qualifying
    #            the `combine` function call with its parent module, `DataFrames`. This is necessary
    #            to disambiguate it from other functions of the same name that may be exported by
    #            other loaded packages.
    # v1.0.1 (2026-07-31)
    # Inputs/Outputs: Standard module processor arguments.

    # First, run the standard spatial processor to set up W, s_idx, s_N, etc.
    process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)

    # The localadaptive model requires centroids for clustering. If they weren't
    # computed by process_spatial_module! (e.g., because W was provided directly),
    # we must compute them now from coordinate data.
    if !haskey(opt_dict, :centroids)
        data = opt_dict[:data]
        if hasproperty(data, :s_x) && hasproperty(data, :s_y)
            @info "Centroids not found, computing from s_x and s_y for localadaptive model."
            
            s_idx_col = get(opt_dict, :s_idx, nothing)
            if isnothing(s_idx_col)
                error("Cannot compute centroids for localadaptive model without a spatial index column (`s_idx`).")
            end

            # Create a temporary DataFrame for aggregation
            temp_df = DataFrame(s_idx = s_idx_col, s_x = data.s_x, s_y = data.s_y)
            
            # Group by s_idx and calculate the mean of s_x and s_y
            gdf = groupby(temp_df, :s_idx)
            
            # --- FIX: Explicitly qualify `combine` with `DataFrames.` ---
            # This resolves the UndefVarError caused by function name ambiguity.
            centroids_df = DataFrames.combine(gdf, :s_x => mean => :s_x, :s_y => mean => :s_y)
            
            # Sort by s_idx to ensure order matches the areal unit indices
            sort!(centroids_df, :s_idx)
            
            # Convert to the expected Vector{Point2D} format
            opt_dict[:centroids] = [Point2D(row.s_x, row.s_y) for row in eachrow(centroids_df)]

            # Ensure the number of centroids matches s_N
            if length(opt_dict[:centroids]) != opt_dict[:s_N]
                @warn "Number of computed centroids ($(length(opt_dict[:centroids]))) does not match number of spatial units s_N ($(opt_dict[:s_N])). This may indicate inconsistent spatial indexing."
            end
        else
            error("The `localadaptive()` model requires centroids for clustering, but they were not found. Ensure spatial coordinates (s_x, s_y) are provided in the data frame.")
        end
    end
    
    centroids = opt_dict[:centroids]
    params = mod_data[:params]
    
    n_clusters = get(params, :n_clusters, 5)
    
    if length(centroids) < n_clusters
        @warn "Number of spatial units ($(length(centroids))) is less than the requested number of clusters ($n_clusters). Adjusting n_clusters to $(length(centroids))."
        n_clusters = length(centroids)
    end
    
    # Clustering.jl expects a [dims x n_points] matrix
    centroids_matrix = hcat([c.x for c in centroids], [c.y for c in centroids])'
    
    # Perform k-means clustering on the centroids
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    # The assignments map each of the s_N centroids to a cluster
    opt_dict[:cluster_assignments] = assignments(kmeans_result)
    opt_dict[:n_clusters] = nclusters(kmeans_result)
    
    return true
end



function process_mosaic_module!(opt_dict, mod_data, registries, hyperpriors)
    data = opt_dict[:data]
    params = mod_data[:params]
    n_regions = get(params, :n_regions, 4)

    if !hasproperty(data, :s_x) || !hasproperty(data, :s_y)
        error("The `mosaic` model requires continuous spatial coordinates `s_x` and `s_y` in the data, but they were not found.")
    end
    
    coords = hcat(data.s_x, data.s_y)'
    
    if size(coords, 2) < n_regions
        @warn "Number of observations ($(size(coords, 2))) is less than the requested number of regions ($n_regions). Adjusting n_regions to $(size(coords, 2))."
        n_regions = size(coords, 2)
    end
    
    kmeans_result = kmeans(coords, n_regions; maxiter=200, display=:none)
    
    mod_data[:params][:mosaic_centers] = kmeans_result.centers
    mod_data[:params][:n_regions] = nclusters(kmeans_result)
    
    return true
end

function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    if !haskey(opt_dict, :nested_components); opt_dict[:nested_components] = Dict{Symbol, Any}(); end
    
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)

    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return false
    end

    sub_data = opt_dict[data_source_sym]
    
    sub_config_kwargs = Dict(pairs(NamedTuple(opt_dict)))
    delete!(sub_config_kwargs, :data)
    
    sub_config = bstm_config(sub_formula, sub_data; sub_config_kwargs...)
    
    opt_dict[:nested_components][var] = sub_config
    
    return false
end



 

# ==============================================================================
# SECTION 6: MODEL BUILDING AND ASSEMBLY
# ==============================================================================

function _bstm_error_handler(e, model)
    println("\nERROR during prior predictive check (rand(m)):")
    showerror(stdout, e, stacktrace(catch_backtrace()))
    println("\n\n--- bstm Diagnosis ---")

    if e isa DimensionMismatch
        println("A `DimensionMismatch` error occurred. This often points to an issue in the model's structure.")
        println("Potential Causes:")
        println("  1. Latent Field vs. Index Mismatch: The number of latent variables for a spatial, temporal, or mixed effect does not match the number of unique levels in the corresponding index variable.")
        println("     - Check `s_N`, `t_N`, or the number of levels in your `mixed()` effect's grouping variable.")
        println("     - Ensure the data passed to `bstm()` is consistent with the dimensions inferred from the formula.")
        println("  2. Matrix Multiplication: An operation like `X * beta` has incompatible dimensions.")
        println("     - Verify the number of columns in your fixed effects design matrix `X` matches the length of the `beta` vector.")
    elseif e isa BoundsError
        println("A `BoundsError` occurred. This means an index is out of range for an array.")
        println("Potential Causes:")
        println("  - This is very common with `mixed()` or `spatial()` effects. The latent field vector is smaller than the maximum index in the corresponding index vector (e.g., `M.s_idx` or `M.mixed_idx_...`).")
        println("  - Review the configuration of the component that caused the error. The number of latent units (e.g., `n_cat` for a mixed effect, or `s_N` for a spatial effect) might be miscalculated.")
        println("  - Check for off-by-one errors in manual indexing if using a `custom()` component.")
    elseif e isa PosDefException
        println("A `PosDefException` occurred. This means a matrix that needs to be positive definite (e.g., for a Cholesky decomposition) is not.")
        println("Potential Causes:")
        println("  1. GMRF Precision Matrix: The precision matrix `Q` for a spatial or temporal model might be numerically unstable or not positive definite.")
        println("     - For `icar` or `besag` models, ensure your adjacency matrix `W` corresponds to a single connected graph. Disconnected spatial 'islands' will cause this error.")
        println("     - A small amount of diagonal jitter is added (`noise` parameter), but it might be insufficient. Try increasing the `noise` keyword argument in the `@bstm` call (e.g., `noise=1e-5`).")
        println("  2. GP Covariance Matrix: A kernel matrix `K` in a Gaussian Process model is not positive definite.")
        println("     - This can happen with very close data points. A small amount of 'nugget' or jitter is usually added to the diagonal. Check the `noise` parameter.")
    elseif e isa KeyError
        println("A `KeyError` occurred. The model tried to access a parameter in the configuration that does not exist.")
        println("Potential Causes:")
        println("  - A typo in a variable name within the formula string.")
        println("  - A required parameter (e.g., `W` for a spatial model, or a custom index like `s_idx`) was not passed as a keyword argument to the `@bstm` call.")
        println("  - An internal error where a parameter was not correctly propagated during model configuration. Check the generated model code for missing `M.` accesses.")
    else
        println("An unexpected error occurred. Here are some general debugging tips:")
        println("  - Carefully review the generated model code printed above the error for any obvious issues.")
        println("  - Use `show_model(m)` to inspect the full model configuration and ensure all parameters seem correct.")
        println("  - Simplify your model formula by removing components one by one to isolate the source of the error.")
    end
    println("------------------------")

    # Suggest simplified formulas to help the user debug.
    println("\n--- Suggested Debugging Steps ---")
    try
        formula_str = model.args.M.formula
        lhs, rhs_raw = split(formula_str, '~')
        lhs = Base.strip(lhs)

        # Use the same logic as the parser to split terms
        rhs_normalized = replace(Base.strip(rhs_raw), r"\s*-\s*" => " + -")
        all_terms = split_terms_at_depth(rhs_normalized, " + ")

        # Determine if the original model had an intercept
        has_intercept = !any(in.(Base.strip.(all_terms), (["0", "-1"],))) && !any(startswith.(Base.strip.(all_terms), "intercept(false"))

        # Suggest a base model
        base_rhs = has_intercept ? "1" : "0"
        println("1. Start with the simplest possible model to isolate the issue.")
        println("   This helps determine if the error is in your `likelihood()` definition or in the model components.")
        println("\n   Suggested base model:")
        println("   @bstm(\n       $lhs ~ $base_rhs,\n       data, ...\n   )")

        # Filter out intercept-related terms for the incremental build-up
        structural_terms = filter(t -> !in(Base.strip(t), ["1", "0", "-1"]) && !startswith(Base.strip(t), "intercept("), all_terms)

        if !isempty(structural_terms)
            println("\n2. If the base model works, add components back one by one to find the problematic term.")
            println("   For example, try the following formulas in order:")
            
            current_formula_rhs = base_rhs
            for (i, term) in enumerate(structural_terms)
                current_formula_rhs *= " + " * term
                println("\n   Step $i: Add '$term'")
                println("   @bstm(\n       $lhs ~ $current_formula_rhs,\n       data, ...\n   )")
            end
        end
    catch e_sugg
        println("\nCould not automatically generate debugging suggestions. Error: $e_sugg")
    end
    println("---------------------------------\n")
end

"""
    @bstm(exprs...)

The main user-facing macro for defining a `bstm` model. It supports two syntaxes:
1.  `m = @bstm(formula, data, ...)`: Returns the model object.
2.  `@bstm m = formula, data, ...`: Assigns the model to `m`.

# Rationale for Correction
The previous implementation used a simplistic method to parse arguments, which failed
when keyword arguments were passed out of order or with a semicolon, leading to a `MethodError`
or causing keyword arguments to be ignored. This updated version implements a more robust
parser that correctly distinguishes between positional arguments (like `formula` and `data`)
and keyword arguments (like `W=...`), regardless of their order. This resolves the `MethodError`
and the issue of `W` not being found, making the macro's behavior more predictable and
aligned with standard Julia syntax.
"""
macro bstm(exprs...)
    local var_name = nothing
    local formula_expr = nothing
    local data_expr = nothing
    local collected_kwargs = []
    local current_positional_args = []

    # Handle assignment syntax: `@bstm m = formula, data, ...`
    local expressions_to_parse = exprs
    if !isempty(exprs) && exprs[1] isa Expr && exprs[1].head == :(=)
        var_name = exprs[1].args[1]
        expressions_to_parse = (exprs[1].args[2], exprs[2:end]...)
    end

    # Iterate through all expressions to separate positional and keyword arguments
    for ex in expressions_to_parse
        if ex isa Expr && ex.head == :parameters # This is the block after a semicolon
            append!(collected_kwargs, ex.args)
        elseif ex isa Expr && ex.head == :(=) # This is a keyword argument without a semicolon
            # Convert Expr(:(=), key, value) to Expr(:kw, key, value) for consistency
            push!(collected_kwargs, Expr(:kw, ex.args[1], ex.args[2]))
        elseif ex isa Expr && ex.head == :kw # This is a keyword argument within a :parameters block
            push!(collected_kwargs, ex)
        else # Positional argument
            push!(current_positional_args, ex)
        end
    end

    # Extract formula and data from positional arguments
    if length(current_positional_args) < 2
        error("The @bstm macro requires at least a formula and a data frame, e.g., `@bstm(y ~ 1, my_data)`.")
    end
    formula_expr = current_positional_args[1]
    data_expr = current_positional_args[2]

    # Warn if there are unexpected extra positional arguments
    if length(current_positional_args) > 2
        @warn "Ignoring extra positional arguments: $(current_positional_args[3:end])"
    end

    # Convert the formula expression to a string
    formula_str = string(formula_expr)

    # Escape all expressions for interpolation
    data_esc = esc(data_expr)
    kwargs_esc = [esc(kw) for kw in collected_kwargs]

    # Construct the core function call
    core_logic = :(bstm($formula_str, $data_esc, $(__module__); $(kwargs_esc...)))

    # Return the appropriate expression
    if !isnothing(var_name)
        return :($(esc(var_name)) = $core_logic)
    else
        return core_logic
    end
end



function _print_param(name, value, status; indent=4)
    # Purpose: Helper function to print a single parameter with its value and status.
    # Rationale: Ensures consistent formatting and handles truncation of long values safely.
    # v1.0.1 (2026-07-31) - Fixed StringIndexError by using `first(str, N)` for safe truncation.
    indent_str = " " ^ indent
    status_str = status == :user ? "(User-provided)" : "(Default)"
    
    value_str = string(value)
    # Safely truncate long values for cleaner display, handling multi-byte characters.
    if length(value_str) > 70
        value_str = first(value_str, 67) * "..."
    end
    
    println("$indent_str- $(rpad(name, 20)): $(value_str)  $status_str")
end
 

"""
    COMPONENT_CONFIG_ARGS

A registry holding the default values for non-prior configuration arguments
for various `random()` model types. This allows the parameter summary to show
all applicable settings, even those not explicitly set by the user.
"""
const COMPONENT_CONFIG_ARGS = Dict(
    # Spline models
    :pspline => Dict(:nbins => 20, :degree => 3, :diff_order => 2, :knot_method => :quantile),
    :bspline => Dict(:nbins => 10, :degree => 3, :knot_method => :quantile),
    :tps => Dict(:nbins => 20, :knot_method => :quantile),
    
    # Continuous kernel models
    :gp => Dict(:kernel => "se", :anisotropic => false),
    :kriging => Dict(:kernel => "se", :anisotropic => false),
    :rff => Dict(:n_features => 20, :kernel => "se", :anisotropic => false),
    :fitc => Dict(:n_inducing => 20, :kernel => "se", :anisotropic => false),
    :svgp => Dict(:n_inducing => 20, :kernel => "se", :anisotropic => false),
    :nystrom => Dict(:n_inducing => 20, :kernel => "se", :anisotropic => false),
    :warp => Dict(:n_features => 20, :kernel => "se", :anisotropic => false),
    :spde => Dict(:anisotropic => false),

    # Temporal models
    :harmonic => Dict(:nharmonics => 1, :period => 12.0),
    :cyclic => Dict(:period => 12),
    
    # Other models
    :dynamics => Dict(:model => "none"),
    :svar => Dict(),
    :tar => Dict(),
    :lgcp => Dict(:model => :icar, :grid_areas => "unit"),
    :sncp => Dict(:n_parents => 50, :kernel => "se"),
    :mosaic => Dict(:n_regions => 4),
    :localadaptive => Dict(:n_clusters => 5),
    :eigen => Dict(:n_factors => 1)
)


# --- Updated function to print finalized parameters ---
"""
    _print_finalized_parameters(config::NamedTuple)

Prints a comprehensive and well-formatted summary of all finalized parameters for each
module used in the `bstm` model. This function is called when `verbose=true`.

It details the configuration for the likelihood, intercept, fixed effects, and all
random/smooth components. For each parameter, it shows the final value that will be
used in the model and indicates whether this value was explicitly provided by the user
or if it was assigned a system default. This provides clarity on the model's exact
specification before sampling begins.
"""
function _print_finalized_parameters(config::NamedTuple)
    println("\n--- Finalized Model Configuration ---")

    # 1. Likelihood Configuration
    println("\n[ Likelihood ]")
    lik_param_defs = [
        (:family, "gaussian"), (:log_offsets, "0.0"), (:weights, "1.0"),
        (:trials, "1"), (:zero_inflated, false), (:volatility, false),
        (:censor_lower, -Inf), (:censor_upper, Inf), (:hurdle, -Inf)
    ]
    
    # Add latent_dist for ordinal family
    push!(lik_param_defs, (:latent_dist, :logistic))

    for (i, spec) in enumerate(config.likelihood_specs)
        outcome = config.outcomes[i]
        println("  Outcome: $outcome")
        user_params = spec # `spec` itself is already the params dictionary
        
        for (p_name, p_default) in lik_param_defs
            final_val = get(user_params, p_name, p_default)
            status = haskey(user_params, p_name) ? :user : :default
            _print_param(p_name, final_val, status)
        end
    end

    # 2. Intercept Configuration
    println("\n[ Intercept ]")
    if config.add_intercept
        is_user_provided = haskey(config, :intercept_prior) && config.intercept_prior != Normal(0, 5)
        _print_param(:prior, config.intercept_prior, is_user_provided ? :user : :default, indent=2)
    else
        println("  - Intercept removed from model.")
    end

    # 3. Fixed Effects Configuration
    if get(config, :Xfixed_N, 0) > 0
        println("\n[ Fixed Effects ]")
        println("  Formula: ~ $(config.Xfixed_applied_formula.rhs)")
        
        if haskey(config, :contrasts) && !isempty(config.contrasts)
            println("  Contrasts:")
            for (var, cont) in config.contrasts
                println("    - $var: $(typeof(cont))")
            end
        end

        println("  Priors per Coefficient:")
        for (i, name) in enumerate(config.Xfixed_names)
            prior_obj = config.Xfixed_priors_vec[i]
            is_default = prior_obj == Normal(0, 5)
            _print_param(name, prior_obj, is_default ? :default : :user, indent=4)
        end
    end

    # 4. Model Components (random, smooth, etc.)
    if !isempty(config.components)
        println("\n[ Model Components ]")
        for spec in config.components
            println("  --- Component: $(spec.key) ---")
            component_obj = spec.component_obj
            model_type_sym = Symbol(lowercase(string(typeof(component_obj))))
            println("    - Type: $(typeof(component_obj))")
            
            latent_dim_val = 0
            if !isnothing(spec.Q_template) && spec.Q_template isa AbstractMatrix; latent_dim_val = size(spec.Q_template, 1);
            elseif hasproperty(component_obj, :nbins); latent_dim_val = component_obj.nbins;
            elseif hasproperty(component_obj, :n_features); latent_dim_val = component_obj.n_features;
            elseif hasproperty(component_obj, :n_inducing); latent_dim_val = component_obj.n_inducing;
            end
            if latent_dim_val > 0; println("    - Latent Field Dimension: $(latent_dim_val)"); end

            println("    - Parameters:")
            user_provided_params_raw = spec.params 
            
            # Combine struct fields (priors) and config args
            all_param_names = Set(fieldnames(typeof(component_obj)))
            if haskey(COMPONENT_CONFIG_ARGS, model_type_sym)
                union!(all_param_names, keys(COMPONENT_CONFIG_ARGS[model_type_sym]))
            end

            for param_name in sort(collect(all_param_names))
                local final_val, status
                if param_name in fieldnames(typeof(component_obj))
                    # It's a hyperparameter (prior)
                    final_val = getfield(component_obj, param_name)
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                else
                    # It's a configuration argument
                    final_val = get(user_provided_params_raw, param_name, COMPONENT_CONFIG_ARGS[model_type_sym][param_name])
                    status = haskey(user_provided_params_raw, param_name) ? :user : :default
                end
                
                if final_val isa Vector{<:UnivariateDistribution}
                    println("      - $(rpad(param_name, 20)): [")
                    for (idx, p_dist) in enumerate(final_val)
                        println("        $idx: $p_dist")
                    end
                    println("      ] $(status == :user ? "(User-provided)" : "(Default)")")
                else
                    _print_param(param_name, final_val, status, indent=6)
                end
            end
        end
    end
    println("\n-------------------------------------\n")
end


# --- The bstm function that calls the print function ---
# This function needs to be updated in the original file to call the new print function.
"""
    bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)

The main entry point for the `@bstm` macro. This function orchestrates the configuration,
code generation, and instantiation of a Turing model. This version is updated to call
the detailed parameter summary function `_print_finalized_parameters` when `verbose=true`.
"""
function bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # Generate model configuration dictionary based on formula syntax and data schema
    options = bstm_config(formula, data; calling_module = calling_module, kwargs...)

    # Invoke the codegen engine to produce the model source string and expression
    model_func_name, expr, new_config, registry = bstm_codegen(options)

    if get(new_config, :verbose, true)
        println("\n--- Dynamically Generated Model Code ---")
        println(new_config.generated_model_code)
        println("----------------------------------------\n")

        # Call the new detailed parameter printing function
        _print_finalized_parameters(new_config)

        println("\n--- Running prior predictive check ---")
    end

    # Evaluate the generated @model macro expression in the target module scope
    calling_module.eval(expr)

    # Access the function binding from the module's global scope
    model_func = getfield(calling_module, model_func_name)

    # Instantiation of the Turing Model Object
    model_instance = Base.invokelatest(model_func, new_config, registry)

    prior_sample = nothing
    try
        # Prior Predictive Validation
        redirect_stderr(devnull) do
            prior_sample = Base.invokelatest(rand, model_instance)
        end
    
        if get(new_config, :verbose, true) && !isnothing(prior_sample)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
    catch e 
        # Error handling for structural or parameterization issues
        _bstm_error_handler(e, model_instance)
    end

    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end

    # Return the fully configured and validated model object
    return model_instance
end

"""
    bstm(formula::String, data::DataFrame; kwargs...)

A convenience overload for the `bstm` function that defaults to the `Main` module
as the calling context.

# Rationale for Update
This function has been corrected to fix a `MethodError` that would occur if it were
called directly. The call to the three-argument `bstm` method was missing a semicolon
(`;`) before the keyword arguments. The corrected call `bstm(formula, data, Main; kwargs...)`
ensures that keyword arguments are passed correctly.
"""
function bstm(formula::String, data::DataFrame; kwargs...)
    # Convenience overload defaulting to the Main execution scope
    # The semicolon below is the critical fix.
    return bstm(formula, data, Main; kwargs...)
end


function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # Purpose: Maps a parsed formula module to a concrete Component object.
    # Rationale: This version is updated to evaluate symbol-based parameters (like `n_age_classes=n_age_classes_ll`)
    #            within the `dynamics()` module. This ensures that the integer value is passed to the
    #            component constructor, resolving a `KeyError` or `MethodError` in the code generator.
    # v1.0.5 (2026-07-31)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]

    if m_type == :lgcp # Log-Gaussian Cox Process
        inner_model_name = get(m_params, :model, :icar)
        
        inner_mod_data = Dict(
            :type => :spatial,
            :params => merge(m_params, Dict(:model => inner_model_name)),
            :variables => get(module_metadata, :variables, [])
        )
        
        inner_component_obj = resolve_technical_primitive(inner_mod_data, M, priors_dict, scheme)
        
        if !(inner_component_obj isa ComponentModel)
            error("LGCP's inner spatial model is invalid. Expected a ComponentModel, but received $(typeof(inner_component_obj)).")
        end

        resolved_priors = resolve_hyperpriors("lgcp", priors_dict, m_params, scheme, M[:calling_module])
        
        dummy_inner_node = (module_type=:random, args=m_params, positional_args=get(m_params, :positional_args, []))
        return LGCP(inner_component_obj, resolved_priors.sigma, dummy_inner_node)

    elseif m_type == :lgammap
        inner_model_node = get(m_params, :inner_model_node, error("LogGammaCoxProcess is missing inner model node."))
        inner_mod_data = Dict(:type => inner_model_node.module_type, :params => inner_model_node.args, :variables => get(inner_model_node.args, :positional_args, []))
        inner_component_obj = resolve_technical_primitive(inner_mod_data, M, priors_dict, scheme)
        shape_prior = get(m_params, :shape, Gamma(2, 2))
        if shape_prior isa Expr; try; shape_prior = Core.eval(M[:calling_module], shape_prior); catch; end; end
        return LogGammaCoxProcess(inner_component_obj, shape_prior)

    elseif m_type == :sncp
        n_parents = get(m_params, :n_parents, 50)
        kernel = get(m_params, :kernel, "se")
        resolved_priors = resolve_hyperpriors("sncp", priors_dict, m_params, scheme, M[:calling_module])
        return ShotNoiseCoxProcess(n_parents, kernel, resolved_priors.lengthscale, resolved_priors.amplitude)

    elseif m_type == :tar
        model_name = "tar"
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        
        threshold_var = get(m_params, :threshold_var, nothing)
        if isnothing(threshold_var); error("TAR model requires a `threshold_var` parameter."); end
        if threshold_var isa Expr; threshold_var = Core.eval(M[:calling_module], threshold_var); end

        rho_regimes = get(m_params, :rho_regimes, [Beta(1,1), Beta(1,1)])
        if rho_regimes isa Expr; rho_regimes = Core.eval(M[:calling_module], rho_regimes); end
        if rho_regimes isa Tuple; rho_regimes = [create_pc_prior(:rho, r) for r in rho_regimes]; end

        sigma_regimes = get(m_params, :sigma_regimes, [Exponential(1.0), Exponential(1.0)])
        if sigma_regimes isa Expr; sigma_regimes = Core.eval(M[:calling_module], sigma_regimes); end
        if sigma_regimes isa Tuple; sigma_regimes = [create_pc_prior(:sigma, s) for s in sigma_regimes]; end

        return TAR(threshold_var, rho_regimes, sigma_regimes)
  
    elseif m_type == :svar
        inner_model_name = get(m_params, :model, nothing)
        if isnothing(inner_model_name)
            error("SVAR model is missing its inner spatial model specification. Use `model=...` to specify it (e.g., model=icar).")
        end
        
        inner_mod_data = Dict(
            :type => :spatial, 
            :params => merge(m_params, Dict(:model => inner_model_name)),
            :variables => get(module_metadata, :variables, []) 
        )
        
        inner_component_obj = resolve_technical_primitive(inner_mod_data, M, priors_dict, scheme)
        
        if !(inner_component_obj isa ComponentModel)
            error("SVAR model's inner spatial model specification is invalid. Expected a ComponentModel, but received $(typeof(inner_component_obj)).")
        end

        resolved_priors = resolve_hyperpriors("svar", priors_dict, m_params, scheme, M[:calling_module])
        return SVAR(inner_component_obj, resolved_priors.sigma)

    elseif m_type == :svc
        covariate_sym = Symbol(get(m_params, :covariate, :unknown))
        spatial_model_spec_node = get(m_params, :spatial_model_spec, nothing)
        spatial_mod_data = Dict(:type => spatial_model_spec_node.module_type, :params => spatial_model_spec_node.args, :variables => get(spatial_model_spec_node.args, :positional_args, []))
        inner_component_obj = resolve_technical_primitive(spatial_mod_data, M, priors_dict, scheme)
        return SVCComponent(covariate_sym, inner_component_obj)
  
    elseif m_type == :tvc
        covariate_sym = Symbol(get(m_params, :covariate, :unknown))
        temporal_model_spec_node = get(m_params, :temporal_model_spec, nothing)
        temporal_mod_data = Dict(:type => temporal_model_spec_node.module_type, :params => temporal_model_spec_node.args, :variables => get(temporal_model_spec_node.args, :positional_args, []))
        inner_component_obj = resolve_technical_primitive(temporal_mod_data, M, priors_dict, scheme)
        return TVCComponent(covariate_sym, inner_component_obj)
   
    elseif m_type == :mixed
        group_var_sym = Symbol(module_metadata[:variables][1])
        lhs_str = module_metadata[:params][:lhs]
        model_name = string(get(m_params, :model, "iid"))
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        model_key = Symbol(model_name)
        if !haskey(COMPONENT_CONSTRUCTORS, model_key); error("Component model ':$model_key' for mixed effect is not a recognized model type."); end
        constructor_func = COMPONENT_CONSTRUCTORS[model_key]
        return MixedComponent(group_var_sym, lhs_str, constructor_func(resolved_priors, m_params))

    elseif m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(:type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, [])), components_data)
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
        return ComposedComponent(resolved_components, op)

    elseif m_type == :spacetime 
        return NoneComponent()

    elseif m_type == :nonstationary_variance
        base_node = m_params[:base_node]
        modifier_node = m_params[:modifier_node]
        base_mod_data = Dict(:type => base_node.module_type, :params => base_node.args, :variables => get(base_node.args, :positional_args, []))
        base_component_obj = resolve_technical_primitive(base_mod_data, M, priors_dict, scheme)
        modifier_mod_data = Dict(:type => modifier_node.module_type, :params => modifier_node.args, :variables => get(modifier_node.args, :positional_args, []))
        modifier_component_obj = resolve_technical_primitive(modifier_mod_data, M, priors_dict, scheme)
        return NonStationaryVariance(base_component_obj, modifier_component_obj)

    elseif m_type == :adaptivesmooth
        resolved_priors = resolve_hyperpriors("adaptivesmooth", priors_dict, m_params, scheme, M[:calling_module])
        h_dim = get(m_params, :hidden_dim, 10)
        n_bins = get(m_params, :nbins, 20)
        return AdaptiveSmooth(h_dim, n_bins, resolved_priors.sigma)
    
    elseif m_type == :dynamics
        model_name_raw = get(m_params, :model, "none")
        model_name = String(lstrip(string(model_name_raw), ':'))
        
        calling_mod = get(M, :calling_module, Main)
        evaluated_params = Dict{Symbol, Any}()
        for (k, v) in m_params
            # Exclude symbols that are part of the DSL and not variables
            if k in [:model, :catch_data_col, :interaction_covariate, :output_species]
                evaluated_params[k] = v
                continue
            end

            # Try to evaluate expressions and symbols that are defined in the calling module
            if v isa Expr || (v isa Symbol && isdefined(calling_mod, v))
                try
                    evaluated_params[k] = Core.eval(calling_mod, v)
                catch e
                    # If evaluation fails, it might be a string-like symbol, so keep it as is.
                    if v isa Symbol
                        evaluated_params[k] = v
                    else
                        error("Could not evaluate parameter `$(v)` for dynamics model '$model_name'. Error: $e")
                    end
                end
            else
                evaluated_params[k] = v
            end
        end
        
        return DynamicsComponent(model_name, evaluated_params)

    elseif m_type == :eigen
        model_name = "eigen"
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        
        n_vars = get(m_params, :n_vars, 0)
        n_factors = get(m_params, :n_factors, 0)
        ltri_indices = get(m_params, :ltri_indices, Int[])

        return Eigen(n_vars, n_factors, resolved_priors.pca_sd, resolved_priors.pdef_sd, ltri_indices)

    elseif m_type == :custom
        code_fragment_val = get(m_params, :code_fragment, "")
        
        local final_code_fragment::String
        if code_fragment_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                evaluated_val = Core.eval(calling_mod, code_fragment_val)
                if evaluated_val isa String
                    final_code_fragment = evaluated_val
                else
                    error("The `code_fragment` argument for the custom() module must be a variable containing a String, but '$(code_fragment_val)' evaluated to a '$(typeof(evaluated_val))'.")
                end
            catch e
                error("Could not evaluate the variable `$(code_fragment_val)` for the `code_fragment` argument. Ensure it is defined in the calling scope. Error: $e")
            end
        elseif code_fragment_val isa String
            final_code_fragment = code_fragment_val
        elseif code_fragment_val isa Expr
             calling_mod = get(M, :calling_module, Main)
             try
                evaluated_val = Core.eval(calling_mod, code_fragment_val)
                if evaluated_val isa String
                    final_code_fragment = evaluated_val
                else
                    error("The `code_fragment` argument for the custom() module must be a variable containing a String, but '$(code_fragment_val)' evaluated to a '$(typeof(evaluated_val))'.")
                end
            catch e
                error("Could not evaluate the expression `$(code_fragment_val)` for the `code_fragment` argument. Error: $e")
            end
        else
            error("The `code_fragment` argument must be a String or a variable/expression that evaluates to a String. Got type: $(typeof(code_fragment_val))")
        end

        return CustomComponent(final_code_fragment, get(m_params, :params, Dict{Symbol, Any}()))

    else
        default_model = if m_type == :spatial; haskey(M, :W) ? "bym2" : "iid"; elseif m_type == :temporal; "rw2"; else "none"; end
        model_name_raw = get(m_params, :model, default_model)
        model_name = String(lstrip(string(model_name_raw), ':'))
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        model_key = Symbol(model_name)
        if !haskey(COMPONENT_CONSTRUCTORS, model_key); error("Component model ':$model_key' is not a recognized model type. Check for typos in the `model=` parameter."); end
        constructor_func = COMPONENT_CONSTRUCTORS[model_key]
        return constructor_func(resolved_priors, m_params)
    end
end



# This ensures that the BYM2 model correctly uses the graph Laplacian for its structured component.
function build_structure_template(model_type::Symbol, n::Int; W::Union{AbstractMatrix, Nothing}=nothing)
    Q_template = spzeros(Float64, n, n)
    scaling_factor = 1.0
    rank_deficiency = 0

    if n == 0
        return (matrix=Q_template, scaling_factor=scaling_factor)
    end

    if model_type == :icar || model_type == :besag || model_type == :bym2
        if isnothing(W)
            error("Spatial model '$model_type' requires an adjacency matrix `W`.")
        end
        if size(W, 1) != n || size(W, 2) != n
            error("Adjacency matrix `W` dimensions ($(size(W))) do not match `n` ($n).")
        end

        W_sym = sparse((W + W') .> 0)
        D = spdiagm(0 => vec(sum(W_sym, dims=2)))
        Q_template = D - W_sym
        rank_deficiency = 1

    elseif model_type == :rw1
        Q_template = spzeros(Float64, n, n)
        if n > 1
            Q_template[1, 1] = 1.0
            for i in 2:n
                Q_template[i, i] = 2.0
                Q_template[i, i-1] = -1.0
                Q_template[i-1, i] = -1.0
            end
            Q_template[n, n] = 1.0
        elseif n == 1
            Q_template[1,1] = 1.0
        end
        rank_deficiency = 1

    elseif model_type == :rw2
        Q_template = spzeros(Float64, n, n)
        if n > 1
            Q_template[1, 1] = 1.0; Q_template[2, 2] = 5.0
            Q_template[1, 2] = -2.0; Q_template[2, 1] = -2.0
            for i in 3:n
                Q_template[i,i] = 6.0
                Q_template[i,i-1] = -4.0
                Q_template[i-1,i] = -4.0
                Q_template[i,i-2] = 1.0
                Q_template[i-2,i] = 1.0
            end
            Q_template[n-1, n-1] = 5.0; Q_template[n, n] = 1.0
            Q_template[n-1, n] = -2.0; Q_template[n, n-1] = -2.0
        elseif n == 1
            Q_template[1,1] = 1.0
        end
        rank_deficiency = 2

    elseif model_type == :cyclic
        Q_template = spzeros(Float64, n, n)
        if n > 0
            for i in 1:n
                Q_template[i, i] = 2.0
                Q_template[i, mod1(i + 1, n)] = -1.0
                Q_template[i, mod1(i - 1, n)] = -1.0
            end
        end
        rank_deficiency = 1

    elseif model_type == :ar1
        # For AR1, the template is just the adjacency structure (tridiagonal).
        # The actual values depend on rho and are set in `recompose_precision`.
        Q_template = spzeros(Float64, n, n)
        if n > 1
            for i in 1:(n-1)
                Q_template[i, i+1] = -1.0
                Q_template[i+1, i] = -1.0
            end
        end
        rank_deficiency = 0

    elseif model_type == :iid
        Q_template = sparse(I, n, n)
        rank_deficiency = 0

    else
        @warn "Unknown model type '$model_type'. Returning identity matrix as template."
        Q_template = sparse(I, n, n)
        rank_deficiency = 0
    end

    if rank_deficiency > 0
        evals = eigen(Symmetric(Matrix(Q_template))).values
        scaling_factor = _compute_scaling_factor(evals, rank_deficiency)
        Q_template = Q_template ./ scaling_factor
    end

    return (matrix=Q_template, scaling_factor=scaling_factor)
end

 


function build_model(m::TVCComponent, data_inputs::Dict, module_metadata::Dict)
    temporal_model_spec_node = get(module_metadata[:params], :temporal_model_spec, nothing)
    if isnothing(temporal_model_spec_node); error("TVC builder is missing the inner temporal model specification."); end
    
    temporal_mod_data = Dict(:type => :temporal, :params => temporal_model_spec_node.args, :variables => get(temporal_model_spec_node.args, :positional_args, []))
    
    # Call the builder for the inner temporal model (e.g., RW2)
    inner_spec = build_model(m.model, data_inputs, temporal_mod_data)
    
    hyper_dict = Dict{Symbol, Any}(:inner_hyper => inner_spec.hyper)
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:tvc, hyper=NamedTuple(hyper_dict))
end


function _generate_component_code_fragments(m::TVCComponent, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    cov_var = m.covariate
    
    # Generate base temporal field logic
    inner_frags = _generate_component_code_fragments(inner_model, spec, arch, outcome_idx, prefix=prefix)
    
    # Remove the standard effect application from the inner model's code
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_app_regex = Regex("\\s*$(eta_target)\\s*\\.\\+=\\s*.*")
    update_inner_cleaned = replace(inner_frags.update, effect_app_regex => "")
    
    # Application Logic: Multiply the temporal field by the covariate
    application_code = "$(eta_target) .+= M.data[!, :$(cov_var)] .* view($(v.latent), M.t_idx)"
    
    update_str = """
    begin
        # TVC Logic for variable: $(cov_var)
        $(update_inner_cleaned)
        $(application_code)
    end
    """

    return (priors=inner_frags.priors, update=update_str)
end

function build_model(m::LogGammaCoxProcess, data_inputs::Dict, module_metadata::Dict)
    params = module_metadata[:params]
    
    inner_model_node = get(params, :inner_model_node, nothing)
    if isnothing(inner_model_node)
        error("LogGammaCoxProcess builder is missing the inner model specification node.")
    end

    inner_params = merge(params, inner_model_node.args)

    inner_mod_data = Dict(
        :type => get(inner_model_node.args, :structure, :spatial), 
        :params => inner_params, 
        :variables => get(inner_model_node.args, :positional_args, [])
    )
    inner_spec = build_model(m.model, data_inputs, inner_mod_data)
    
    temporal_spec_idx = findfirst(s -> s.structure == :temporal, data_inputs[:components])
    areas = get(data_inputs, :grid_areas, ones(data_inputs[:s_N]))

    hyper_dict = Dict(
        :inner_spec => inner_spec, 
        :areas => Float64.(areas), 
        :s_N => data_inputs[:s_N], 
        :t_N => get(data_inputs, :t_N, 1)
    )
    if !isnothing(temporal_spec_idx)
        hyper_dict[:temporal_spec] = data_inputs[:components][temporal_spec_idx]
    end

    return (Q_template=inner_spec.Q_template, scaling_factor=1.0, model_type=:lgammap, hyper=NamedTuple(hyper_dict))
end



"""
    _generate_component_code_fragments(m::LogGammaCoxProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `LogGammaCoxProcess` component.

# Rationale for Update
This version corrects a logical error in the likelihood implementation. The previous
version incorrectly used a Poisson log-likelihood after sampling from a Gamma
distribution. The correct implementation for a Poisson-Gamma mixture is a Negative
Binomial likelihood. This function now correctly constructs the parameters for a
Negative Binomial distribution using the `(r, p)` parameterization, where `r` is the
shape and `p` is derived from the mean and shape. This ensures statistically correct
inference for the Log-Gamma Cox Process.
"""
function _generate_component_code_fragments(m::LogGammaCoxProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    is_spatiotemporal = hasproperty(spec.hyper, :temporal_spec)
    n_latent_dims = is_spatiotemporal ? "M.s_N * M.t_N" : "M.s_N"

    # Priors for the Gamma shape and the raw innovations for the latent field
    priors = """
    # Log-Gamma Cox Process Priors
    $(v.innov) ~ NamedDist($(_distribution_to_string(m.shape)), :$(v.innov)) # Using 'innov' for the shape parameter
    $(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_dims)), I), :$(v.raw))
    """

    update = """
    begin
        # Log-Gamma Cox Process Model: $(key_str)
        local latent_field_st = zeros(T, M.s_N, M.t_N)
        
        # 1. Reconstruct the latent spatiotemporal field Z(s,t)
        if $(is_spatiotemporal)
            local s_spec = spec_registry["$(key_str)"].hyper.inner_spec
            local t_spec = spec_registry["$(key_str)"].hyper.temporal_spec
            local C_s = cholesky(Symmetric(s_spec.Q_template + noise * I))
            local C_t = cholesky(Symmetric(t_spec.Q_template + noise * I))
            local Z_matrix = reshape($(v.raw), M.s_N, M.t_N)
            local tmp_spatial = C_s.U \\ Z_matrix
            latent_field_st = exp.(transpose(C_t.U \\ transpose(tmp_spatial))) # Exponentiate to ensure positivity
        else
            local Q_inner = spec_registry["$(key_str)"].hyper.inner_spec.Q_template
            local F_inner = cholesky(Symmetric(Q_inner + noise * I))
            local spatial_component = exp.(F_inner.U \\ $(v.raw)) # Exponentiate
            latent_field_st = repeat(spatial_component, 1, M.t_N)
        end

        # 2. Assemble the full mean intensity surface.
        local mean_intensity_surface = zeros(T, M.s_N, M.t_N)
        local gamma_shape = $(v.innov) # The learned shape parameter

        for t in 1:M.t_N, s in 1:M.s_N
            obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:N)
            base_contribution = isempty(obs_indices) ? 0.0 : mean(view(eta, obs_indices))
            mean_intensity_surface[s, t] = exp(base_contribution) * latent_field_st[s, t]
        end

        # 3. Point Process Likelihood Evaluation using Negative Binomial
        # A Poisson-Gamma mixture results in a Negative Binomial distribution.
        # Mean (μ) = shape * scale = exp(eta) * latent_field
        # Variance = μ + μ^2/shape
        local grid_areas = spec_registry["$(key_str)"].hyper.areas
        for t in 1:M.t_N, s in 1:M.s_N
            local y_st = M.y_obs[s, t]
            local A_s = grid_areas[s]
            local mu = mean_intensity_surface[s, t] * A_s
            
            # Parameterize Negative Binomial with successes (r=shape) and probability (p)
            # The p parameter is r / (r + μ)
            local r_nb = gamma_shape
            local p_nb = r_nb / (r_nb + mu)
            local nb_dist = NegativeBinomial(r_nb, p_nb)
            Turing.@addlogprob! logpdf(nb_dist, y_st)
        end

        M[:likelihood_handled] = true
    end
    """

    return (priors=priors, update=update)
end



"""
    build_model(m::NonStationaryVariance, data_inputs::Dict, module_metadata::Dict)

A specialized model builder for the `NonStationaryVariance` component.

# Rationale
This builder constructs the technical specifications for both the `base_model` (spatial)
and `modifier_model` (smoother) by recursively calling `build_model` on them. The resulting
specifications, which include their respective precision matrix templates, are stored in
the `hyper` registry of the main component. This makes all necessary structural information
available to the code generator.
"""
function build_model(m::NonStationaryVariance, data_inputs::Dict, module_metadata::Dict)
    base_node = module_metadata[:params][:base_node]
    modifier_node = module_metadata[:params][:modifier_node]

    # Build the spec for the base model (e.g., ICAR).
    base_mod_data = Dict(:type => base_node.module_type, :params => base_node.args, :variables => get(base_node.args, :positional_args, []))
    base_spec = build_model(m.base_model, data_inputs, base_mod_data)

    # Build the spec for the modifier model (e.g., PSpline).
    modifier_mod_data = Dict(:type => modifier_node.module_type, :params => modifier_node.args, :variables => get(modifier_node.args, :positional_args, []))
    modifier_spec = build_model(m.modifier_model, data_inputs, modifier_mod_data)

    # Store these specs and the basis key in the hyper registry for the code generator.
    hyper_dict = Dict(
        :base_spec => base_spec,
        :modifier_spec => modifier_spec,
        :modifier_basis_key => module_metadata[:params][:modifier_basis_key]
    )

    # The NonStationaryVariance component itself does not have a Q_template.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:nonstationary_variance, hyper=NamedTuple(hyper_dict))
end



"""
    _generate_component_code_fragments(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

A specialized code generator for the `NonStationaryVariance` component.

# Rationale
This function generates the Turing model code for the non-stationary variance model.
It replaces the old code-injection method with a clean, structured implementation that:
1.  Recursively calls the code generator for the `modifier_model` (smoother) to get its priors and update logic.
2.  Defines a prior for the `base_model`'s raw innovations.
3.  In the update block, it first runs the modifier's logic to compute its latent field.
4.  It then uses this latent field to construct the spatially varying log-sigma field.
5.  Finally, it reconstructs the base spatial field and scales it by the exponentiated, spatially varying sigma, adding the result to the linear predictor.
"""
function _generate_component_code_fragments(m::NonStationaryVariance, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    is_multivariate = arch == "multivariate"
    
    # Retrieve specs for the child components from the hyper registry.
    base_spec = spec.hyper.base_spec
    modifier_spec = spec.hyper.modifier_spec
    basis_key = spec.hyper.modifier_basis_key

    # Generate prefixed variable names for the sub-components to avoid collisions.
    v_base = generate_full_variable_names(base_spec, arch, outcome_idx, prefix=key_str)
    v_modifier = generate_full_variable_names(modifier_spec, arch, outcome_idx, prefix=key_str)

    # --- Priors ---
    # Get the priors for the modifier (smoother) component by calling its code generator.
    modifier_frags = _generate_component_code_fragments(m.modifier_model, modifier_spec, arch, outcome_idx, prefix=key_str)
    
    # Define the prior for the base (spatial) component's raw innovations.
    # Its sigma is determined by the modifier, so we only need the raw innovations here.
    n_latent_base = size(base_spec.Q_template, 1)
    base_priors = "$(v_base.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_base)), I), :$(v_base.raw))"
    
    priors_str = """
    # Priors for NonStationaryVariance component: $(key_str)
    $(modifier_frags.priors)
    $(base_priors)
    """

    # --- Update Logic ---
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    # The modifier's update logic will define its latent field, `v_modifier.latent`.
    # We remove its final `eta .+= ...` line, as we will use the latent field differently.
    modifier_update_cleaned = replace(modifier_frags.update, Regex("\\s*$(eta_target)\\s*\\.\\+=\\s*.*") => "")

    update_str = """
    begin
        # --- Non-Stationary Variance Logic for $(key_str) ---
        
        # 1. Compute the latent field for the modifier (smoother).
        # This block will define the variable `$(v_modifier.latent)`.
        $(modifier_update_cleaned)
        
        # 2. The modifier's latent field, projected by its basis, becomes the log of the spatially varying sigma.
        local log_sigma_field = M.basis_matrices[:$(basis_key)] * $(v_modifier.latent)
        local spatially_varying_sigma = exp.(log_sigma_field)
        
        # 3. Reconstruct the base spatial field from its raw innovations.
        local Q_base_template = spec_registry["$(spec.key)"].hyper.base_spec.Q_template
        local F_base = cholesky(Symmetric(sparse(Q_base_template) + noise * I))
        local base_latent_raw = F_base.U \\ $(v_base.raw)
        
        # 4. Apply sum-to-zero constraint if the base model is intrinsic (e.g., ICAR).
        if $(m.base_model isa Union{ICAR, Besag})
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent_base)), sum(base_latent_raw))
        end
        
        # 5. Combine the base field and the spatially varying sigma.
        # The base model's own sigma is not used; it's replaced by the modifier.
        local final_effect = view(base_latent_raw, M.s_idx) .* spatially_varying_sigma
        
        # 6. Add the final effect to the linear predictor.
        $(eta_target) .+= final_effect
    end
    """
    
    return (priors=priors_str, update=update_str)
end


function _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
    # Purpose: Computes a robust scaling factor for a precision matrix from its eigenvalues.
    # Rationale: The scaling factor is the geometric mean of the non-zero eigenvalues.
    #            This method avoids using a fixed tolerance to identify zero eigenvalues,
    #            which can be sensitive to floating-point noise. Instead, it uses the known
    #            rank deficiency of the GMRF model to correctly identify the structural zero
    #            eigenvalues.
    # v1.0.0 (2026-07-21)
    # Inputs:
    #   - evals: A vector of eigenvalues.
    #   - rank_deficiency: The known rank deficiency of the precision matrix (e.g., 1 for ICAR, 2 for RW2).
    # Outputs: The scaling factor.
    
    # Sort eigenvalues in ascending order to easily discard the smallest ones.
    sorted_evals = sort(evals)
    
    n = length(sorted_evals)
    if n <= rank_deficiency
        return 1.0
    end
    
    # Select the eigenvalues that are not part of the null space.
    positive_evals = sorted_evals[(rank_deficiency + 1):end]
    
    if isempty(positive_evals)
        return 1.0
    end
    
    # The scaling factor is the geometric mean of the positive eigenvalues.
    return exp(mean(log.(positive_evals)))
end


function process_random_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    # Purpose: The main processor for the `random()` module, which dispatches to structure-specific handlers.
    # Rationale: This version is updated to correctly dispatch to the `process_localadaptive_module!`
    #            when `model=:localadaptive` is specified. The previous logic incorrectly defaulted
    #            to the generic `:spatial` handler, causing a `KeyError` because the necessary
    #            clustering information was never computed. This fix prioritizes the check for
    #            special model types before falling back to the general structure-based dispatch.
    # v1.0.1 (2026-07-31)
    # Inputs/Outputs: Standard module processor arguments.

    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]
    
    is_anisotropic = get(params, :anisotropic, false)
    ls_prior = get(params, :lengthscale, get(params, :kappa, nothing))
    
    if ls_prior isa Expr && ls_prior.head == :vect && length(ls_prior.args) > 1
        is_anisotropic = true
    end
    params[:anisotropic] = is_anisotropic

    if !isempty(variables); params[:in_dims] = length(variables); end

    if !haskey(params, :structure)
        args_for_inference = copy(params)
        args_for_inference[:vars] = get(mod_data, :variables, [])
        params[:structure] = _infer_structure_from_args(args_for_inference)
    end
    structure = params[:structure]
    
    model_name = get(params, :model, :iid)

    # --- FIX: Special handling for models that have their own processor logic ---
    if model_name == :localadaptive
        # This model has its own processor to handle clustering.
        process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)
        # We still return true so that the component object is created.
        return true
    end
    # --- END FIX ---

    point_process_type = get(params, :point_process, nothing)
    if point_process_type == :lgcp || point_process_type == :sncp
        if haskey(params, :grid_areas)
            ga_val = params[:grid_areas]
            if ga_val isa Symbol && hasproperty(data, ga_val)
                opt_dict[:grid_areas] = data[!, ga_val]
            elseif ga_val isa AbstractVector
                opt_dict[:grid_areas] = ga_val
            else
                try; opt_dict[:grid_areas] = Core.eval(get(opt_dict, :calling_module, Main), ga_val);
                catch; @warn "Could not resolve grid_areas for point process. Defaulting to unit areas."; opt_dict[:grid_areas] = ones(get(opt_dict, :s_N, 1)); end
            end
        else
            @warn "$(uppercase(string(point_process_type))) model specified, but `grid_areas` parameter is missing. Defaulting to unit areas."
            opt_dict[:grid_areas] = ones(get(opt_dict, :s_N, 1))
        end
    end

    if structure == :spatial
        if haskey(params, :W)
            w_val = params[:W]
            if w_val isa Expr || w_val isa Symbol
                calling_mod = get(opt_dict, :calling_module, Main)
                try; opt_dict[:W] = Core.eval(calling_mod, w_val); catch e; error("Could not evaluate `W` argument `$(w_val)`. Error: $e"); end
            else
                opt_dict[:W] = w_val
            end
        end
        if !haskey(opt_dict, :W); error("Spatial model requires an adjacency matrix `W`."); end
        
        opt_dict[:s_N] = size(opt_dict[:W], 1)
        if !isempty(variables)
            s_var_sym = Symbol(variables[1])
            if hasproperty(data, s_var_sym); opt_dict[:s_idx] = data[!, s_var_sym]; else; @warn "Spatial index ':$s_var_sym' not found."; end
        end

    elseif structure == :temporal
        if model_name == :tar
            mod_data[:type] = :tar
            if !haskey(params, :threshold_var)
                error("The `tar` model requires a `threshold_var` parameter, e.g., `random(time, model=tar, threshold_var=my_covariate)`.")
            end
        end
        continuous_kernel_models = [:gp, :rff, :fitc, :svgp, :nystrom, :warp, :spde, :kriging]
        is_continuous_kernel_on_time = model_name in continuous_kernel_models
        is_discrete_seasonal = model_name == :cyclic
        is_continuous_periodic = model_name == :harmonic || haskey(params, :period)

        if !isempty(variables)
            t_var_sym = Symbol(variables[1])
            if hasproperty(data, t_var_sym)
                if is_continuous_kernel_on_time
                    coords = Matrix{Float64}(data[!, [t_var_sym]])
                    params[:coords] = coords
                    mod_data[:type] = :smooth 
                    if model_name in [:fitc, :svgp, :nystrom]
                        n_inducing = get(params, :n_inducing, min(100, size(coords, 1)))
                        params[:n_inducing] = n_inducing
                        params[:Z_inducing] = generate_inducing_points(coords, n_inducing)
                    end
                elseif is_discrete_seasonal || is_continuous_periodic
                    opt_dict[:u_idx] = data[!, t_var_sym]
                    opt_dict[:u_N] = length(unique(opt_dict[:u_idx]))
                    opt_dict[:u_idx_var] = t_var_sym
                else
                    time_opts = Dict(:time_method => get(params, :time_method, "regular"))
                    tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
                    opt_dict[:t_idx] = tu_meta.idx
                    opt_dict[:t_N] = tu_meta.N_cat
                    opt_dict[:t_idx_var] = t_var_sym
                end
            else
                @warn "Temporal index ':$t_var_sym' not found."
            end
        end

    elseif structure == :svar
        # An SVAR model needs both spatial and temporal context.
        # 1. Process the spatial part to resolve W, s_idx, and s_N.
        process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)
        
        # 2. Ensure temporal context exists, as it's an autoregressive model.
        if !haskey(opt_dict, :t_idx)
            if hasproperty(data, :year)
                time_opts = Dict(:time_method => get(params, :time_method, "regular"))
                tu_meta = assign_time_units(data[!, :year]; time_opts...)
                opt_dict[:t_idx] = tu_meta.idx
                opt_dict[:t_N] = tu_meta.N_cat
                opt_dict[:t_idx_var] = :year
            else
                error("SVAR model requires a temporal index, but none was found (e.g., a 'year' column or a `random(..., structure=:temporal)` component).")
            end
        end
        
        # The SVAR model itself doesn't have a Q_template, but its inner spatial model does.
        # The main build process will handle this.
        return true # Proceed to create the SVAR component.

    elseif structure == :smooth
        continuous_kernel_models = [:gp, :rff, :fitc, :svgp, :nystrom, :warp, :spde, :kriging]
        if model_name in continuous_kernel_models
            if all(v -> hasproperty(data, Symbol(v)), variables)
                coords = Matrix{Float64}(data[!, Symbol.(variables)])
                params[:coords] = coords
                if model_name in [:fitc, :svgp, :nystrom]
                    n_inducing = get(params, :n_inducing, min(100, size(coords, 1)))
                    params[:n_inducing] = n_inducing
                    params[:Z_inducing] = generate_inducing_points(coords, n_inducing)
                end
            else
                error("Coordinate variables for smooth model not found in data: $(variables)")
            end
        else
            process_smooth_module!(opt_dict, mod_data, opt_dict[:basis_matrices], opt_dict[:components])
        end
    
    elseif string(structure) == "spacetime"
        process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)
        return false # Signal to bstm_config to not create a component
        
    else
        @warn "Processing for structure ':$structure' is not fully implemented in `process_random_module!`. A default component will be created."
    end

    return true
end
 



function process_interact_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes interaction modules like `|>` (pipe) and `⊗` (Kronecker product).
    # Rationale: This version is updated to correctly handle the `random |> random` syntax
    #            for spatially varying curves. It now detects this pattern, infers the
    #            structures of the child components, and explicitly calls the necessary
    #            processors (`process_smooth_module!` and `process_spatial_module!`) to
    #            ensure that basis matrices and spatial configurations are correctly set up
    #            before the model building phase. This resolves the `KeyError` for `:year`.
    # v1.0.1 (2026-07-30)
    # Inputs/Outputs: Standard module processor arguments.

    op = get(mod_data, :operator, get(mod_data[:params], :operator, nothing))
    components = get(mod_data, :components, get(mod_data[:params], :components, []))
    if isnothing(op) || isempty(components); return false; end
    
    if op == :composition && length(components) == 2
        outer_node, inner_node = components[1], components[2]
        
        if outer_node.module_type == :pointprocess
            return true
        end

        is_nonstationary_variance = outer_node.module_type == :random && get(outer_node.args, :structure, :none) == :spatial && inner_node.module_type == :random && get(inner_node.args, :structure, :none) == :smooth
        if is_nonstationary_variance
            modifier_vars = get(inner_node.args, :positional_args, [])
            if isempty(modifier_vars); @warn "The modifier component (smooth) of a composition operator is missing variables. Skipping."; return false; end
            
            smooth_mod_data = Dict(:type => :smooth, :variables => modifier_vars, :params => inner_node.args)
            process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:components])
            
            mod_data[:type] = :nonstationary_variance
            mod_data[:params][:base_node] = outer_node
            mod_data[:params][:modifier_node] = inner_node
            mod_data[:params][:modifier_basis_key] = Symbol(join(modifier_vars, "_"))
            
            return true 
        end
    end

    if op == :pipe && length(components) == 2
        node1, node2 = components[1], components[2]
        
        # Infer structures if not explicitly provided.
        if node1.module_type == :random && !haskey(node1.args, :structure)
            args1 = copy(node1.args); args1[:vars] = get(node1.args, :positional_args, [])
            node1.args[:structure] = _infer_structure_from_args(args1)
        end
        if node2.module_type == :random && !haskey(node2.args, :structure)
            args2 = copy(node2.args); args2[:vars] = get(node2.args, :positional_args, [])
            node2.args[:structure] = _infer_structure_from_args(args2)
        end

        # --- FIX: Added logic for spatially varying curves ---
        is_spatially_varying_curve = node1.module_type == :random && get(node1.args, :structure, :none) == :smooth &&
                                     node2.module_type == :random && get(node2.args, :structure, :none) == :spatial

        is_svc = node1.module_type == :fixed && node2.module_type == :random && get(node2.args, :structure, :none) == :spatial
        is_tvc = node1.module_type == :fixed && node2.module_type == :random && get(node2.args, :structure, :none) == :temporal
        is_svar = node1.module_type == :random && get(node1.args, :structure, :none) == :temporal && node2.module_type == :random && get(node2.args, :structure, :none) == :spatial

        if is_spatially_varying_curve
            # Process the dynamic (smooth) part to create the basis matrix.
            dynamic_node = node1
            dynamic_vars = get(dynamic_node.args, :positional_args, [])
            if isempty(dynamic_vars); error("The dynamic part of a pipe operator (e.g., a smoother) must have a variable."); end
            
            smooth_mod_data = Dict(:type => :smooth, :variables => dynamic_vars, :params => dynamic_node.args)
            process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:components])
            
            # Process the state (spatial) part to set up W, s_idx, etc.
            state_node = node2
            spatial_mod_data = Dict(:type => :spatial, :variables => get(state_node.args, :positional_args, []), :params => state_node.args)
            process_spatial_module!(opt_dict, spatial_mod_data, registries, hyperpriors)
            
            # Set up parameters for the ComposedComponent builder.
            mod_data[:params][:dynamic_component_node] = dynamic_node
            mod_data[:params][:state_component_node] = state_node
            
            return true # Proceed to build the ComposedComponent.

        elseif is_svc
            covariate_node = node1
            spatial_node = node2
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args); @warn "SVC model is missing a covariate. Skipping."; return false; end
            covariate_name = cov_args[1]
            
            mod_data[:type] = :svc
            mod_data[:variables] = [covariate_name, get(spatial_node.args, :positional_args, [])...]
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:spatial_model_spec] = spatial_node
            return true

        elseif is_tvc
            covariate_node = node1
            temporal_node = node2
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args); @warn "TVC model is missing a covariate. Skipping."; return false; end
            covariate_name = cov_args[1]

            mod_data[:type] = :tvc
            mod_data[:variables] = [covariate_name, get(temporal_node.args, :positional_args, [])...]
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:temporal_model_spec] = temporal_node
            return true
            
        elseif is_svar
            temporal_node = node1
            spatial_node = node2
            
            process_random_module!(opt_dict, Dict(:type => :temporal, :params => temporal_node.args, :variables => get(temporal_node.args, :positional_args, [])), registries, hyperpriors)
            process_random_module!(opt_dict, Dict(:type => :spatial, :params => spatial_node.args, :variables => get(spatial_node.args, :positional_args, [])), registries, hyperpriors)

            mod_data[:type] = :svar
            mod_data[:params][:rho_spatial_node] = spatial_node
            mod_data[:params][:base_temporal_node] = temporal_node
            return true
        end
    end

    if op == :kronecker_product
        if haskey(mod_data[:params], :sigma)
            prior_val = mod_data[:params][:sigma]
            calling_mod = get(opt_dict, :calling_module, Main)
            if prior_val isa Tuple
                opt_dict[:st_interaction_sigma_prior] = create_pc_prior(:sigma, prior_val)
            elseif prior_val isa Expr
                opt_dict[:st_interaction_sigma_prior] = Core.eval(calling_mod, prior_val)
            else
                opt_dict[:st_interaction_sigma_prior] = prior_val
            end
        end
        
        if length(components) == 2
            s_idx = get(opt_dict, :s_idx, nothing); t_idx = get(opt_dict, :t_idx, nothing); s_N = get(opt_dict, :s_N, nothing)
            if !isnothing(s_idx) && !isnothing(t_idx) && !isnothing(s_N); mod_data[:params][:indices] = [(t - 1) * s_N + s for (s, t) in zip(s_idx, t_idx)];
            else @warn "Could not compute Kronecker product indices for '$(mod_data[:variables])'. Ensure spatial and temporal components are defined."; end
            
            c1_type = get(components[1], :module_type, :unknown); c2_type = get(components[2], :module_type, :unknown)
            
            spatial_node = c1_type == :random && get(components[1].args, :structure, :none) == :spatial ? components[1] : (c2_type == :random && get(components[2].args, :structure, :none) == :spatial ? components[2] : nothing)
            temporal_node = c1_type == :random && get(components[1].args, :structure, :none) == :temporal ? components[1] : (c2_type == :random && get(components[2].args, :structure, :none) == :temporal ? components[2] : nothing)
            
            if !isnothing(spatial_node) && !isnothing(temporal_node)
                spatial_model_str = string(get(spatial_node.args, :model, :iid)); temporal_model_str = string(get(temporal_node.args, :model, :iid))
                has_structured_space = spatial_model_str != "iid"; has_structured_time = temporal_model_str != "iid"
                if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
                elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
                elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
                else opt_dict[:model_st] = "I"; end
            end
            return false
        else
            @warn "Kronecker product with more than 2 components is not yet supported in process_interact_module!."
        end
    end
    
    return true
end



# Generic builder for standard Component types
function build_model(m::Component, data_inputs::Dict, module_metadata::Dict)
    structure = get(module_metadata, :type, :spatial)
    return _build_from_template(m, data_inputs, structure, module_metadata)
end

function build_model(m::CustomComponent, data_inputs::Dict, module_metadata::Dict)
"""
    build_model(m::CustomComponent, data_inputs::Dict, module_metadata::Dict)

A model builder for the `CustomComponent`.

# Rationale
This is a new function that ensures `CustomComponent` is handled correctly by the
configuration engine. Since a custom component is defined entirely by user-provided
code, it does not have a predefined structure matrix (`Q_template`). This builder
uses the `_build_pass_through_model` helper to signal that no precision matrix
template is needed, preventing the framework from incorrectly trying to build one.
"""
    # This component is defined entirely by user code, so it doesn't have a Q_template.
    # We use the pass-through builder to indicate this.
    return _build_pass_through_model(m, data_inputs, module_metadata)
end


# Specialized builder for IID to ensure structure-specific template resolution
function build_model(m::IID, data_inputs::Dict, module_metadata::Dict)
    structure = get(module_metadata, :type, :spatial)
    return _build_from_template(m, data_inputs, structure, module_metadata)
end

# Builder for temporal Gaussian Markov Random Fields
function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict, module_metadata::Dict)
    # v1.0.0 (2026-07-20) - Context-aware structure resolution.
    # If used in a `smooth()` call, the structure is `:mixed` (on bins), not `:temporal`.
    structure = get(module_metadata, :type, :temporal)
    return _build_from_template(m, data_inputs, structure, module_metadata)
end



"""
    build_model(m::Union{Wavelet, FFT}, data_inputs::Dict, module_metadata::Dict)

A new model builder for `Wavelet` and `FFT` components.

# Rationale
This builder ensures that the coordinate data and the per-dimension bin counts,
which are resolved in `process_smooth_module!`, are correctly stored in the
component's `hyper` registry. This information is essential for the new dynamic
code generators for these models.
"""
function build_model(m::Union{Wavelet, FFT}, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("$(typeof(m)) component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:coords] = coords
    
    if haskey(module_metadata[:params], :nbins_per_dim)
        hyper_dict[:nbins_per_dim] = module_metadata[:params][:nbins_per_dim]
    end

    return (Q_template=nothing, scaling_factor=1.0, model_type=Symbol(lowercase(string(typeof(m)))), hyper=NamedTuple(hyper_dict))
end


function build_model(m::ShotNoiseCoxProcess, data_inputs::Dict, module_metadata::Dict)
    # Get spatial domain bounds from the data coordinates.
    # Assumes 's_x' and 's_y' are present from a `random(s_x, s_y, ...)` call.
    if !hasproperty(data_inputs[:data], :s_x) || !hasproperty(data_inputs[:data], :s_y)
        error("SNCP requires continuous spatial coordinates `s_x` and `s_y` to define the domain.")
    end
    
    coords = data_inputs[:data]
    x_min, x_max = extrema(coords.s_x)
    y_min, y_max = extrema(coords.s_y)
    
    areas = get(data_inputs, :grid_areas, ones(get(data_inputs, :s_N, 1)))

    hyper_dict = Dict(
        :domain_bounds => (x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        :areas => Float64.(areas),
        :s_N => get(data_inputs, :s_N, 1)
    )

    # SNCP does not use a GMRF precision matrix template.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:sncp, hyper=NamedTuple(hyper_dict))
end


# --- 5. New `_generate_component_code_fragments` method for `ShotNoiseCoxProcess` ---
# This function generates the core Turing code for the SNCP model.

function _generate_component_code_fragments(m::ShotNoiseCoxProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    # Determine if the number of parents is fixed or a random variable
    n_parents_str = if m.n_parents isa Int
        string(m.n_parents)
    else
        # Use 'innov' to sample the number of parents if it's a distribution
        "$(v.innov)_n_parents"
    end

    priors_list = String[]
    if m.n_parents isa UnivariateDistribution
        push!(priors_list, "$(n_parents_str) ~ NamedDist($(_distribution_to_string(m.n_parents)), :$(Symbol(n_parents_str)))")
    end
    
    bounds = spec.hyper.domain_bounds
    # Priors for parent locations (2D for spatial)
    push!(priors_list, "$(v.raw)_parent_locs_x ~ NamedDist(filldist(Uniform($(bounds.x_min), $(bounds.x_max)), $(n_parents_str)), :$(Symbol(string(v.raw, "_parent_locs_x"))))")
    push!(priors_list, "$(v.raw)_parent_locs_y ~ NamedDist(filldist(Uniform($(bounds.y_min), $(bounds.y_max)), $(n_parents_str)), :$(Symbol(string(v.raw, "_parent_locs_y"))))")
    
    # Priors for kernel parameters
    push!(priors_list, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    push!(priors_list, "$(v.amplitude) ~ NamedDist(filldist($(_distribution_to_string(m.amplitude)), $(n_parents_str)), :$(Symbol(string(v.amplitude))))")

    priors = join(priors_list, "\n    ")

    update = """
    begin
        # Shot-Noise Cox Process Model: $(key_str)
        
        # 1. Get observation locations (centroids of areal units)
        local obs_locs = M.centroids 
        
        # 2. Assemble parent locations
        local parent_locs = hcat($(v.raw)_parent_locs_x, $(v.raw)_parent_locs_y)
        
        # 3. Compute intensity at each observation location centroid
        local intensity_at_obs = zeros(T, M.s_N)
        for i in 1:M.s_N
            local intensity_i = 0.0
            for j in 1:$(n_parents_str)
                # Calculate squared Euclidean distance
                local dist_sq = (obs_locs[i].x - parent_locs[j, 1])^2 + (obs_locs[i].y - parent_locs[j, 2])^2
                
                # Evaluate the kernel (e.g., Squared Exponential)
                local kernel_val = exp(-0.5 * dist_sq / ($(v.ls)^2))
                
                # Add the "shot" to the intensity
                intensity_i += $(v.amplitude)[j] * kernel_val
            end
            intensity_at_obs[i] = intensity_i
        end

        # 4. Point Process Likelihood Evaluation for areal data
        local grid_areas = spec_registry["$(key_str)"].hyper.areas
        for s in 1:M.s_N
            # y_obs is assumed to be a vector of counts for each spatial unit
            local y_s = M.y_obs[s] 
            local A_s = grid_areas[s]
            
            # Approximate integrated intensity λ_s = ∫_{A_s} Λ(u)du ≈ Λ(centroid_s) * A_s
            local lambda_s = intensity_at_obs[s] * A_s
            
            Turing.@addlogprob! (y_s * log(lambda_s + noise) - lambda_s)
        end

        # Signal that the likelihood has been handled
        M[:likelihood_handled] = true
    end
    """
    
    return (priors=priors, update=update)
end

function build_model(m::MixedComponent, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `MixedComponent`.
    # Rationale: This function correctly constructs the technical specification for a mixed effect model.
    #            It recursively calls `build_model` on the inner component (e.g., IID, RW2)
    #            to obtain its precision matrix template (`Q_template`), which defines the
    #            correlation structure of the random effects.
    # v1.0.0 (2026-07-20)
    
    # The inner model determines the structure of the random effects.
    inner_mod_data = Dict(
        :type => :mixed,
        :params => module_metadata[:params]
    )
    # Purpose: A specialized model builder for the `MixedComponent`.
    # Rationale: This function correctly constructs the technical specification for a mixed effect model.
    #            It recursively calls `build_model` on the inner component (e.g., IID, RW2)
    #            to obtain its precision matrix template (`Q_template`), which defines the
    #            correlation structure of the random effects.
    # v1.0.0 (2026-07-20)
    
    # The inner model determines the structure of the random effects.
    inner_mod_data = Dict(
        :type => :mixed,
        :params => module_metadata[:params]
    )
    
    inner_spec = build_model(m.model, data_inputs, inner_mod_data)
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:inner_hyper] = inner_spec.hyper
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:mixed, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::Union{ICAR, Besag, BYM2, Leroux, SAR}, ...)

A model builder for standard spatial GMRF models.

# Rationale for Update
This version has been updated to remove `LocalAdaptive` from the type union.
`LocalAdaptive` now has its own specialized builder to handle the inclusion
of clustering information.
"""
function build_model(m::Union{ICAR, Besag, BYM2, Leroux, SAR}, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata) 
end

"""
    build_model(m::SPDE, data_inputs::Dict, module_metadata::Dict)

A model builder for the `SPDE` component.
"""
function build_model(m::SPDE, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata)
end


"""
    build_model(m::LocalAdaptive, data_inputs::Dict, module_metadata::Dict)

A specialized model builder for the `LocalAdaptive` component.

# Rationale
This function ensures that the `n_clusters` and `cluster_assignments`, which are
pre-computed by `process_localadaptive_module!`, are correctly stored in the
component's `hyper` registry. This information is essential for the specialized
code generator to construct the model with cluster-specific means.
"""
function build_model(m::LocalAdaptive, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `LocalAdaptive` component.
    # Rationale: This function ensures that the `n_clusters` and `cluster_assignments`, which are
    #            pre-computed by `process_localadaptive_module!`, are correctly stored in the
    #            component's `hyper` registry. This information is essential for the specialized
    #            code generator to construct the model with cluster-specific means.
    # v1.0.1 (2026-07-31) - Corrected dictionary type to handle mixed value types.
    # Inputs:
    #   - m: The LocalAdaptive component object.
    #   - data_inputs: The main model configuration dictionary.
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple containing the component's technical specification.

    # First, call the generic template builder to get the Q_template.
    base_spec = _build_from_template(m, data_inputs, :spatial, module_metadata)
    
    # Now, augment the hyper parameters with the clustering info.
    # FIX: Initialize as Dict{Symbol, Any} to allow for mixed types (Distributions and Ints).
    hyper_dict = Dict{Symbol, Any}(pairs(base_spec.hyper))
    
    if !haskey(data_inputs, :n_clusters) || !haskey(data_inputs, :cluster_assignments)
        error("LocalAdaptive model requires `n_clusters` and `cluster_assignments` to be pre-computed, but they were not found in the model configuration. This indicates an issue with `process_localadaptive_module!`.")
    end
    
    hyper_dict[:n_clusters] = data_inputs[:n_clusters]
    
    # The cluster_assignments are large and only needed by the code generator,
    # not the reconstruction engine, so they are attached directly to the main
    # model configuration `M` rather than the spec's hyper registry.
    # The code generator will access it via `M.cluster_assignments`.
    
    return merge(base_spec, (hyper=NamedTuple(hyper_dict),))
end




function build_model(m::SVCComponent, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `SVCComponent`.
    # Rationale: This function correctly constructs the technical specification for an SVC model.
    #            It recursively calls `build_model` on the inner spatial component (e.g., BYM2, ICAR)
    #            to obtain its precision matrix template (`Q_template`). This template is then
    #            passed up to the main configuration, ensuring that the code generator for the
    #            SVC has the correct structural information to model the spatially varying coefficient.
    # v1.0.0 (2026-07-20)
    
    # The inner model (e.g., BYM2) determines the structure.
    # We call its builder to get the Q_template.
    
    spatial_model_spec_node = get(module_metadata[:params], :spatial_model_spec, nothing)
    if isnothing(spatial_model_spec_node); error("SVC builder is missing the inner spatial model specification."); end
    
    spatial_mod_data = Dict(:type => spatial_model_spec_node.module_type, :params => spatial_model_spec_node.args, :variables => get(spatial_model_spec_node.args, :positional_args, []))
    
    # Call the builder for the inner spatial model
    inner_spec = build_model(m.model, data_inputs, spatial_mod_data)
    
    # The SVC component itself doesn't have hyperparameters, but we pass them
    # from the inner model for the code generator to use.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Pass the inner spec's hyper-parameters up.
    hyper_dict[:inner_hyper] = inner_spec.hyper
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:svc, hyper=NamedTuple(hyper_dict))
end


function _generate_component_code_fragments(m::GP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_obs = size(spec.Q_template, 1) # For GP, Q_template holds the coordinates

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_obs)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    update = """
    begin
        # Full Gaussian Process (GP) Logic for $(key_str)
        local X_coords = spec_registry["$(key_str)"].Q_template
        local kernel_type = Symbol("$(m.kernel)")

        local K_mat = evaluate_kernel_matrix(X_coords, $(v.sigma), $(v.ls), kernel_type, noise)
        local F_gp = cholesky(Symmetric(K_mat))
        $(v.latent) = F_gp.L * $(v.raw)
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= $(v.latent)
    end
    """
    return (priors=priors, update=update)
end



"""
    build_model(m::Spherical, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Spherical` component when used as a full Gaussian Process.

# Rationale for Update
This is a new function that enables the `Spherical` component to be treated as a
continuous-space Gaussian Process, consistent with its definition which includes
priors for `sigma` and `range`. It ensures that the coordinate data from a `smooth()`
call is correctly captured and passed to the code generator.
"""
function build_model(m::Spherical, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords); error("Spherical component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`."); end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    
    return (Q_template=nothing, scaling_factor=1.0, model_type=:spherical, hyper=NamedTuple(hyper_dict))
end




"""
    generate_rff_params(...)

Generates random projection weights (W) and biases (b) for RFF approximation.
This version is included for completeness and supports ARD.
"""
function generate_rff_params(in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector}, kernel_name::String)
    b = rand(Uniform(0, 2 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    k_name = lowercase(kernel_name)

    if k_name in ["se", "gaussian", "rbf"]
        if lengthscale isa Real
            W .= rand(Normal(0, 1.0 / lengthscale), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = rand(Normal(0, 1.0 / lengthscale[d]), n_features); end
        end
    elseif occursin("matern", k_name)
        nu = if k_name == "matern12"; 0.5; elseif k_name == "matern32"; 1.5; else 2.5; end
        df = 2 * nu
        if lengthscale isa Real
            W .= (sqrt(df) / lengthscale) .* rand(TDist(df), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = (sqrt(df) / lengthscale[d]) .* rand(TDist(df), n_features); end
        end
    else
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        return generate_rff_params(in_dims, n_features, lengthscale, "se")
    end
    return W, b
end



"""
    build_model(m::RFF, data_inputs::Dict, module_metadata::Dict)

Updated model builder for the `RFF` component to handle anisotropic lengthscales.

# Rationale for Update
This version correctly handles the case where the `lengthscale` prior is a vector.
It computes a vector of initial lengthscale values by taking the mean of each prior
distribution. This initial vector is then passed to `generate_rff_params`, which
already supports ARD and will generate the fixed projection weights `W_fixed` from
the corresponding anisotropic spectral density.
"""
function build_model(m::RFF, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords); error("RFF component requires coordinates."); end
    
    ls_prior = m.lengthscale
    local ls_initial
    if ls_prior isa Vector
        ls_initial = [mean(p isa Truncated ? untruncated(p) : p) for p in ls_prior]
    else
        ls_initial = mean(ls_prior isa Truncated ? untruncated(ls_prior) : ls_prior)
    end
    
    in_dims = size(coords, 2)
    W_fixed, b_fixed = generate_rff_params(in_dims, m.n_features, ls_initial, m.kernel)

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    hyper_dict[:W_fixed] = W_fixed
    hyper_dict[:b_fixed] = b_fixed
    return (Q_template=nothing, scaling_factor=1.0, model_type=:rff, hyper=NamedTuple(hyper_dict))
end




"""
    build_model(m::FITC, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `FITC` (Fully Independent Training Conditional) component.

# Rationale for Update
This version cleans up the internal implementation by removing redundant code. Its primary
role remains to configure the `FITC` sparse Gaussian Process model by:
1.  Storing the observation coordinates (`coords`) in the `Q_template` field.
2.  Storing the pre-computed inducing point locations (`Z_inducing`) in the component's
    `hyper` registry for use by the code generator.
"""
function build_model(m::FITC, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("FITC component requires coordinates, but none were not found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("FITC component requires inducing points, but they were not found. This is an internal error in the `smooth` or `spatial` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:Z_inducing] = Z_inducing
    
    # The `Q_template` field is used by convention to pass the observation coordinates.
    return (Q_template=coords, scaling_factor=1.0, model_type=:fitc, hyper=NamedTuple(hyper_dict))
end


"""
    build_model(m::Moran, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Moran` spatial component.

# Rationale for Update
This is a new function to correctly implement the Moran eigenvector spectral model.
It computes the Moran operator `M = (I - 11'/n)W(I - 11'/n)`, calculates its
eigenvectors, and stores them in the component's hyperparameter registry. These
eigenvectors serve as the basis functions for the spatial effect, aligning the
implementation with the documented behavior of `Moran's I Basis Component`.
"""
function build_model(m::Moran, data_inputs::Dict, module_metadata::Dict)
    W = get(data_inputs, :W, nothing)
    if isnothing(W)
        error("The `moran` component requires an adjacency matrix `W`, but it was not found in the model configuration.")
    end
    
    n = size(W, 1)
    
    # Create the centering matrix H = I - (1/n) * 1*1'
    H = I - (1/n) * ones(n, n)
    
    # Compute the Moran operator M = HWH
    # Ensure W is a concrete matrix for computation
    W_mat = Matrix(W)
    moran_operator = H * W_mat * H
    
    # Compute the eigenvectors of the symmetric Moran operator
    eig_result = eigen(Symmetric(moran_operator))
    moran_eigenvectors = eig_result.vectors
    
    # Store the eigenvectors in the hyperparameter registry for the code generator
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:moran_eigenvectors] = moran_eigenvectors
    
    # Q_template is not used for this spectral model, but a placeholder is returned for API consistency.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:moran, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::Mosaic, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `Mosaic` component.

# Rationale
This new builder method ensures that the pre-computed mosaic centers from the
`process_mosaic_module!` are correctly passed into the component's hyperparameter
registry. This makes the centers accessible to the code generator, which needs them
to calculate the soft-weighting for the mixture of experts.
"""
function build_model(m::Mosaic, data_inputs::Dict, module_metadata::Dict)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    mosaic_centers = get(module_metadata[:params], :mosaic_centers, nothing)
    if isnothing(mosaic_centers); error("Mosaic centers not found in module metadata. This indicates an issue in `process_mosaic_module!`."); end
    hyper_dict[:mosaic_centers] = mosaic_centers
    return (Q_template=nothing, scaling_factor=1.0, model_type=:mosaic, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::Mosaic, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `Mosaic` component.

# Rationale
This new builder method ensures that the pre-computed mosaic centers from the
`process_mosaic_module!` are correctly passed into the component's hyperparameter
registry. This makes the centers accessible to the code generator, which needs them
to calculate the soft-weighting for the mixture of experts.
"""
function build_model(m::Mosaic, data_inputs::Dict, module_metadata::Dict)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    
    mosaic_centers = get(module_metadata[:params], :mosaic_centers, nothing)
    if isnothing(mosaic_centers)
        error("Mosaic centers not found in module metadata. This indicates an issue in `process_mosaic_module!`.")
    end
    hyper_dict[:mosaic_centers] = mosaic_centers
    
    # Q_template is not used for this type of model.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:mosaic, hyper=NamedTuple(hyper_dict))
end



function _generate_component_code_fragments(m::Mosaic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing code fragments for the `Mosaic` spatial model.
    # Rationale: This function implements the "mixture of experts" or "soft clustering" approach
    #            for the mosaic model. The key steps are:
    #            1. Priors for Local Experts: It defines priors for the means of the local
    #               spatial effects (`mu_local`), one for each of the `n_regions`.
    #            2. Softmax Weighting: For each observation, it calculates the distance to every
    #               mosaic center and uses a softmax function to produce a vector of weights,
    #               implementing "soft boundary stitching".
    #            3. Weighted Combination: The final latent effect is computed as the weighted
    #               sum of the local expert means.
    #            4. Scaling: The combined effect is scaled by the overall `sigma` hyperparameter.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_regions = m.n_regions
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    mu_local_name = v.raw # Use 'raw' to store the local means
    sigma_name = v.sigma
    latent_name = v.latent

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(sigma_name))")
    end
    push!(priors_acc, "$(mu_local_name) ~ NamedDist(MvNormal(zeros(T, $(n_regions)), I), :$(mu_local_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Mosaic mixture-of-experts model for $(key_str)
        local centers = spec_registry["$(key_str)"].hyper.mosaic_centers
        local coords = hcat(M.data.s_x, M.data.s_y)
        
        # Calculate squared distances from each observation to each mosaic center
        local dist_sq = pairwise(SqEuclidean(), coords, centers', dims=1)
        
        # Use softmax on negative distances to get weights (responsibilities)
        local weights = softmax(-dist_sq, dims=2)
        
        # The latent effect is the weighted sum of the local expert means, scaled by sigma.
        local $(latent_name) = (weights * $(mu_local_name)) .* $(sigma_name)
        
        $(eta_update_target) .+= $(latent_name)
    end
    """
    
    return (priors=priors_str, update=update_str)
end



"""
    evaluate_cross_kernel_matrix(coords1, coords2, param_val, ls, kernel_type)

Computes the cross-covariance kernel matrix between two sets of coordinates.

# Rationale
This function is essential for sparse GP methods like FITC, which require the
computation of the covariance between the data points (X) and the inducing points (Z).
It supports both isotropic and Automatic Relevance Determination (ARD) kernels.
"""
function evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol)
    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords1, 2) != length(ls) || size(coords2, 2) != length(ls)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions does not match number of lengthscales.")
        end
        # Calculate weighted squared Euclidean distance
        dist_sq = pairwise(SqEuclidean(), coords1 ./ ls', coords2 ./ ls', dims=1)
    else # Isotropic case
        dist_sq = pairwise(SqEuclidean(), coords1, coords2, dims=1) ./ ls^2
    end

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return (param_val^2) .* exp.(-0.5 .* dist_sq)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return (param_val^2) .* exp.(-d)
    
    # Matern 3/2
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(3.0) .* d
        return (param_val^2) .* (1.0 .+ val) .* exp.(-val)
    
    # Matern 5/2
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(5.0) .* d
        return (param_val^2) .* (1.0 .+ val .+ (val.^2 ./ 3.0)) .* exp.(-val)

    # Fallback Dispatch
    else
        return (param_val^2) .* exp.(-0.5 .* dist_sq)
    end
end


function build_model(m::Warp, data_inputs::Dict, module_metadata::Dict)
    # For Warp, we need the raw coordinates to apply the warping function to.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Warp component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or a spatial module with available coordinates.")
    end
    
    # The warping function's parameters are fully learned, so we don't
    # pre-generate fixed features like in RFF. We just need to pass the coordinates.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:coords] = coords

    # The Q_template is not used, but we provide a placeholder for consistency
    # with the rest of the framework's data structures.
    Q_template = sparse(I, m.n_features, m.n_features)
    return (Q_template=Q_template, scaling_factor=1.0, model_type=:warp, hyper=NamedTuple(hyper_dict))
end

function build_model(m::SVGP, data_inputs::Dict, module_metadata::Dict)
    # For SVGP, we need both the observation coordinates and the inducing point coordinates.
    # The "template" will store the observation coordinates.
    # The inducing points will be stored in the `hyper` registry.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("SVGP component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("SVGP component requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}(); for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:Z_inducing] = Z_inducing
    return (Q_template=coords, scaling_factor=1.0, model_type=:svgp, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Nystrom, data_inputs::Dict, module_metadata::Dict)
    # For Nystrom, we need both the observation coordinates and the inducing point coordinates.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Nystrom component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom component requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    hyper_dict[:Z_inducing] = Z_inducing
    return (Q_template=nothing, scaling_factor=1.0, model_type=:nystrom, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Nystrom, data_inputs::Dict, module_metadata::Dict)
    # For Nystrom, we need both the observation coordinates and the inducing point coordinates.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Nystrom component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom component requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    hyper_dict[:Z_inducing] = Z_inducing
    return (Q_template=nothing, scaling_factor=1.0, model_type=:nystrom, hyper=NamedTuple(hyper_dict))
end

function build_model(m::GP, data_inputs::Dict, module_metadata::Dict)
    # For GP, the "template" is the coordinate matrix itself, not the distance matrix.
    # This allows the kernel evaluation to handle ARD kernels correctly.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("GP component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Store the raw coordinates in the template field.
    return (Q_template=coords, scaling_factor=1.0, model_type=:gp, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Hyperbolic, data_inputs::Dict, module_metadata::Dict)
    # For Hyperbolic GP, we need the raw coordinates to compute hyperbolic distances.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Hyperbolic component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    return (Q_template=nothing, scaling_factor=1.0, model_type=:hyperbolic, hyper=NamedTuple(hyper_dict))
end

function build_model(m::ExponentialDecay, data_inputs::Dict, module_metadata::Dict)
    # For Exponential Decay GP, we need the raw coordinates to compute distances.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("ExponentialDecay component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or `spatial(lon, lat, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    return (Q_template=nothing, scaling_factor=1.0, model_type=:exponentialdecay, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Harmonic, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for continuous, spectral, and other advanced components.
    # Rationale: These models do not rely on pre-computed templates in the same way as GMRFs, so they use a pass-through builder.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_pass_through_model(m, data_inputs, module_metadata)
end


"""
    build_model(m::NetworkFlow, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `NetworkFlow` component.

# Rationale
This function dispatches the `NetworkFlow` component to the template-based builder
with a `:spatial` context. This ensures that the adjacency matrix `W` from the main
model configuration is correctly identified and passed as the structural template
for the network model.

# Arguments
- `m::NetworkFlow`: The NetworkFlow component object.
- `data_inputs::Dict`: The main model configuration dictionary.
- `module_metadata::Dict`: The parsed dictionary for the module.

# Returns
- A `NamedTuple` containing the component's technical specification.
"""
function build_model(m::NetworkFlow, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata)
end


function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for spline-based smoothers.
    # Rationale: Determines the appropriate underlying GMRF template (RW1 or RW2) based on the spline type and penalty order.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    n = m.nbins
    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, module_metadata; Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Cyclic, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the `Cyclic` component.
    # Rationale: Creates a circulant precision matrix for smooth periodic effects.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, module_metadata; model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::BCGN, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the BCGN (Bipartite Graph Convolutional Network) component.
    # Rationale: Constructs the precision matrix template from the provided bipartite adjacency matrix.
    #            The precision is based on the graph Laplacian of the one-mode projection of the
    #            bipartite graph. This induces a GMRF structure on one set of nodes, where two
    #            nodes are considered "neighbors" if they share a common neighbor in the other partition.
    # v1.0.0 (2026-07-19)

    B = m.bipartite_adj
    if isempty(B) || all(iszero, B)
        error("BCGN component requires a non-empty `bipartite_adj` matrix, but it was not provided or is all zeros.")
    end

    # The latent effect is defined on the first set of nodes (rows of B).
    # We create the precision matrix from the one-mode projection onto this set.
    W_proj = B * B'
    
    # For a standard graph Laplacian, self-loops (diagonal elements) are set to zero.
    W_proj[diagind(W_proj)] .= 0
    W_proj = dropzeros(W_proj)

    # Build the graph Laplacian from the projected adjacency matrix: L = D - W
    D_proj = spdiagm(0 => vec(sum(W_proj, dims=2)))
    Q_template = D_proj - W_proj

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); if fn != :bipartite_adj; hyper_dict[fn] = getfield(m, fn); end; end
    
    return (Q_template=Q_template, scaling_factor=1.0, model_type=:bcgn, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::Eigen, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `Eigen` component.

# Rationale for Update
This new builder method ensures that the pre-processed data matrix required for the
Bayesian PCA is correctly passed from the module processor into the component's
hyperparameter registry. This makes the data accessible to the code generator.
"""
function build_model(m::Eigen, data_inputs::Dict, module_metadata::Dict)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    
    # Retrieve the data matrix from the parameters populated by `process_eigen_module!`.
    eigen_data = get(module_metadata[:params], :eigen_data, nothing)
    if isnothing(eigen_data)
        error("Eigen data matrix not found in module metadata. This indicates an issue in `process_eigen_module!`.")
    end
    hyper_dict[:eigen_data] = eigen_data
    
    # Q_template is not used for the Eigen component, but a placeholder is returned for API consistency.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:eigen, hyper=NamedTuple(hyper_dict))
end


function build_model(m::DynamicsComponent, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the `DynamicsComponent`.
    # Rationale: This version is updated to handle `effort_col` for models that use it,
    #            such as the reformulated `delay_difference` model.
    # v1.0.6 (2026-07-31)

    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    if isnothing(W)
        error("DynamicsComponent requires an adjacency matrix W, but it was not found in the model configuration.")
    end

    # 1. Build the second-order diffusion operator (Graph Laplacian).
    L_template = build_structure_template(:besag, n; W=W).matrix

    # 2. Build a first-order advection operator (if needed).
    A_template = if m.model in ["advection", "advection_diffusion"]
        W_dir = tril(W, -1)
        out_degree = sum(W_dir, dims=2)[:]
        D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
        D_inv * W_dir
    else
        spzeros(Float64, n, n)
    end

    # 3. Store operators and `grid_areas` in the component's hyperparameter registry.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:L_template] = L_template
    hyper_dict[:A_template] = A_template
    hyper_dict[:areas] = get(data_inputs, :grid_areas, ones(n)) # Store grid_areas

    # 4. Process covariates passed as parameters (e.g., effort, catch_data).
    s_N = get(data_inputs, :s_N, 1)
    t_N = get(data_inputs, :t_N, 1)
    y_N = data_inputs[:y_N]
    s_idx = get(data_inputs, :s_idx, ones(Int, y_N))
    t_idx = get(data_inputs, :t_idx, ones(Int, y_N))
    data = data_inputs[:data]

    processed_params = Dict{Symbol, Any}()
    for (key, val) in m.params
        is_data_col_symbol = (val isa Symbol && hasproperty(data, val))
        covariate_vector = (val isa AbstractVector && length(val) == y_N) ? val : (is_data_col_symbol ? data[!, val] : nothing)
        
        if !isnothing(covariate_vector) && covariate_vector isa AbstractVector && length(covariate_vector) == y_N
            cov_matrix = zeros(s_N, t_N)
            counts = zeros(Int, s_N, t_N)
            for i in 1:y_N; si, ti = s_idx[i], t_idx[i]; cov_matrix[si, ti] += covariate_vector[i]; counts[si, ti] += 1; end
            cov_matrix ./= max.(1, counts) # Average if multiple obs per cell
            
            storage_key = is_data_col_symbol ? val : key
            processed_params[storage_key] = cov_matrix
        else
            processed_params[key] = val
        end
    end
    hyper_dict[:processed_params] = processed_params

    # Explicitly store symbols for catch or effort columns if present.
    if haskey(m.params, :catch_data_col)
        hyper_dict[:catch_data_col_sym] = m.params[:catch_data_col]
    end
    if haskey(m.params, :effort_col)
        hyper_dict[:effort_col_sym] = m.params[:effort_col]
    end

    # The Q_template field is still required by the generic component pathway.
    return (Q_template=L_template, scaling_factor=1.0, model_type=:dynamics, hyper=NamedTuple(hyper_dict))
end


 
function build_model(m::TensorProductSmooth, data_inputs::Dict, module_metadata::Dict)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        if !(fn in [:Q_template])
            hyper_dict[fn] = getfield(m, fn)
        end
    end
    return (Q_template=m.Q_template, scaling_factor=1.0, model_type=:tensorproductsmooth, hyper=NamedTuple(hyper_dict))
end
 

function build_model(m::ComposedComponent, data_inputs::Dict, module_metadata::Dict)
    if m.operator == :pipe
        # Handles spatially/temporally varying curves, e.g., `random(model=pspline) |> random(model=icar)`
        if length(m.components) != 2; error("Pipe operator requires exactly two components: dynamic |> state."); end

        dynamic_component = m.components[1] # The curve (e.g., PSpline)
        state_component = m.components[2]   # The field (e.g., ICAR)

        # Reconstruct metadata for the state component to pass to its builder.
        # This information is nested within the original formula parse tree.
        state_node = module_metadata[:params][:components][2]
        state_mod_data = Dict(
            :type => get(state_node.args, :structure, :spatial), # Infer structure
            :params => state_node.args,
            :variables => get(state_node.args, :positional_args, [])
        )
        state_spec = build_model(state_component, data_inputs, state_mod_data)

        # Find the variable associated with the dynamic component to identify its basis matrix.
        dynamic_node = module_metadata[:params][:components][1]
        dynamic_vars = get(dynamic_node.args, :positional_args, [])
        if isempty(dynamic_vars); error("The dynamic part of a pipe operator (e.g., pspline) must have a variable."); end
        dynamic_basis_key = Symbol(join(dynamic_vars, "_"))

        # Package all necessary info for the code generator into the hyper registry.
        hyper_dict = Dict{Symbol, Any}(
            :state_spec => state_spec,
            :dynamic_component_obj => dynamic_component,
            :dynamic_basis_key => dynamic_basis_key
        )

        # The ComposedComponent itself does not have a Q_template.
        return (Q_template=nothing, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    elseif m.operator == :composition
        base_component = get(m.components, 1, nothing)
        if isnothing(base_component); error("Composition component is missing its base component."); end 
        
        base_spec = build_model(base_component, data_inputs, module_metadata)
        hyper_dict = Dict(:base_spec => base_spec)
        return (Q_template=base_spec.Q_template, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    else
        # For other operators like ⊗, no special template is needed at this stage,
        # as they are either handled by other processors or are not supported generically.
        return _build_pass_through_model(m, data_inputs, module_metadata)
    end
end





function _build_from_template(m::ComponentModel, data_inputs::Dict, structure::Symbol, module_metadata::Dict)
    # Purpose: A generic builder for components that use a pre-defined template.
    # Rationale: This version is updated to correctly resolve the adjacency matrix `W`. It now
    #            searches for `W` first in the local module's parameters (highest precedence),
    #            then falls back to the global model configuration. This ensures that `W`
    #            provided inside a nested `random()` call (e.g., in an SVC) is correctly found.
    # v1.0.1 (2026-07-28)
    # Inputs:
    #   - m: The ComponentModel object.
    #   - data_inputs: The main model configuration dictionary (`M`).
    #   - structure: The structure of the component (:spatial, :temporal, :mixed).
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the component's technical specification.
    model_sym = Symbol(lowercase(string(typeof(m))))
    
    local n, W_mat
    if structure == :spatial
        # --- W Resolution Logic ---
        # 1. Check for W in the local module's parameters first.
        local W_from_local = nothing
        if haskey(module_metadata[:params], :W)
            w_val = module_metadata[:params][:W]
            if w_val isa Expr || w_val isa Symbol
                calling_mod = get(data_inputs, :calling_module, Main)
                try
                    W_from_local = Core.eval(calling_mod, w_val)
                catch e
                    error("Could not evaluate `W` argument `$(w_val)` in module '$(get(module_metadata, :type, "unknown"))'. Error: $e")
                end
            else
                W_from_local = w_val
            end
        end

        # 2. Fallback to the global W from the main configuration.
        W_from_main = get(data_inputs, :W, nothing)

        # 3. Prioritize the locally provided W.
        W_mat = isnothing(W_from_local) ? W_from_main : W_from_local
        
        # Determine the number of spatial units from W if available.
        n = isnothing(W_mat) ? get(data_inputs, :s_N, 1) : size(W_mat, 1)

    elseif structure == :temporal
        n = get(data_inputs, :t_N, 10)
        W_mat = nothing
    elseif structure == :mixed
        n_levels = get(get(module_metadata, :params, Dict()), :n_cat, 0)
        if n_levels == 0
            error("Could not determine number of levels for mixed effect. `n_cat` not found in module parameters.")
        end
        n = n_levels
        W_mat = nothing
    else
        @warn "Unrecognized structure '$structure'. Defaulting to spatial context."
        n = get(data_inputs, :s_N, 1)
        W_mat = get(data_inputs, :W, nothing)
    end

    template = build_structure_template(model_sym, n; W=W_mat)
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        if !(fn in [:Q_template])
            hyper_dict[fn] = getfield(m, fn)
        end
    end
    
    return (Q_template = template.matrix, scaling_factor = template.scaling_factor, model_type = model_sym, hyper = NamedTuple(hyper_dict))
end


function _build_pass_through_model(m::ComponentModel, data_inputs::Dict, module_metadata::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # Purpose: A generic builder for components that do not require complex template generation.
    # Rationale: Used for models where the structure is defined by parameters (e.g., splines) or handled dynamically.
    # v1.0.0 (2026-07-16)
    #            This version ensures a default identity Q_template is created for basis-like models.
    # Assumptions: None.
    # Inputs:
    #   - m, data_inputs, and optional overrides.
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the component's technical specification.
    model_sym = isnothing(model_type_sym) ? Symbol(lowercase(string(typeof(m)))) : model_type_sym

    # If Q_template is not provided, create a default identity matrix based on n_features, nbins, or n_inducing.
    # This is crucial for allowing these models to be used in compositions.
    if isnothing(Q_template_val)
        n_units = 0
        if hasproperty(m, :n_features); n_units = m.n_features;
        elseif hasproperty(m, :nbins); n_units = m.nbins;
        elseif hasproperty(m, :n_inducing); n_units = m.n_inducing;
        end
        
        if n_units > 0; Q_template_val = sparse(I(n_units)); end
    end

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    if haskey(hyper_dict, :Q_template); delete!(hyper_dict, :Q_template); end
    return (Q_template=Q_template_val, scaling_factor=sf_val, model_type=model_sym, hyper=NamedTuple(hyper_dict))
end

# ==============================================================================
# SECTION 7: DYNAMIC MODEL ASSEMBLER
# ==============================================================================

function observation_volatility(M::NamedTuple)
    # Purpose: Generates code fragments for the observation error variance.
    # Rationale: Handles both constant variance and a spatiotemporal stochastic volatility (SV) model
    # v1.0.0 (2026-07-16)
    #            using Random Fourier Features (RFF), activated by a flag in the model configuration.
    # Inputs:
    #   - M: The model configuration NamedTuple.
    # Outputs: A NamedTuple with code strings for priors and calculations.

    if get(M, :volatility, false)
        # Stochastic Volatility Model using RFF
        required_keys = [:M_rff_sigma, :W_sigma_fixed, :b_sigma_fixed, :coords_st]
        if !all(k -> haskey(M, k), required_keys)
            error("Stochastic volatility is enabled, but one or more required keys are missing from the model configuration: $required_keys. Ensure 'volatility=true' is set in the likelihood() module and necessary data is provided.")
        end

        priors_str = """
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol ~ NamedDist(MvNormal(zeros(T, M.M_rff_sigma), sigma_log_var^2 * I), :beta_vol)
        """

        # The calculation projects spatiotemporal coordinates through the RFF basis
        # to generate a latent log-variance field, which is then transformed to
        # the standard deviation `y_sigma`.
        calc_str = """
        local vol_proj = (M.coords_st * M.W_sigma_fixed) .+ M.b_sigma_fixed'
        local log_var_latent = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj) * beta_vol
        y_sigma = exp.(log_var_latent ./ 2.0)
        """
    else
        # Default behavior: constant observation variance.
        # The y_sigma_const prior is defined in the assembler. This just uses it.
        priors_str = ""
        calc_str = "y_sigma = fill(y_sigma_const, N)"
    end

    return (priors=priors_str, calculation=calc_str)
end


# ==============================================================================
# SECTION 7.5: COMPONENT CODE GENERATORS
# ==============================================================================

function _generate_component_code_fragments(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing})
    # Purpose: Dispatches code generation to a specific method based on the component object type.
    # Rationale: This is the entry point for converting a high-level component specification
    # v1.0.0 (2026-07-16)
    #            into low-level Turing model code strings.
    return _generate_component_code_fragments(spec.component_obj, spec, arch, outcome_idx)
end
 


"""
    _generate_component_code_fragments(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `Harmonic` component, now supporting multiple,
independently estimated periods.

# Rationale
This version is refactored to support a vector of priors for the `period` parameter.
- If `m.period` is a `Vector`, it generates code to sample each period independently.
- If `m.period` is a single `Distribution`, it samples one period shared by all harmonics.
- If `m.period` is a `Real`, it is treated as a fixed value.
The update logic now uses the specific period for each harmonic component, `period_vals[m]`,
transforming the model into a flexible sum-of-sines.
"""
function _generate_component_code_fragments(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    nharmonics = m.nharmonics

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    
    amplitude_raw_name = Symbol(string(v.amplitude) * "_raw")

    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        # Priors for amplitude and phase are vectors.
        push!(priors_acc, "$(amplitude_raw_name) ~ NamedDist(filldist($(_distribution_to_string(m.amplitude)), $(nharmonics)), :$(amplitude_raw_name))")
        push!(priors_acc, "$(v.phase) ~ NamedDist(filldist($(_distribution_to_string(m.phase)), $(nharmonics)), :$(v.phase))")
        
        # Handle period sampling based on its type.
        if m.period isa Vector
            # Sample each period from its specified prior.
            for i in 1:nharmonics
                period_var_i = Symbol(string(v.period, "_", i))
                push!(priors_acc, "$(period_var_i) ~ NamedDist($(_distribution_to_string(m.period[i])), :$(period_var_i))")
            end
        elseif m.period isa UnivariateDistribution
            # Sample a single period value to be used for all harmonics.
            push!(priors_acc, "$(v.period) ~ NamedDist($(_distribution_to_string(m.period)), :$(v.period))")
        end
    end
    
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "u_idx" 
    
    # Logic to construct the vector of period values for the update loop.
    local period_access_logic
    if m.period isa Real
        period_access_logic = "local period_vals = fill($(m.period), $(nharmonics))"
    elseif m.period isa UnivariateDistribution
        period_access_logic = "local period_vals = fill($(v.period), $(nharmonics))"
    elseif m.period isa Vector
        period_vars = [string(v.period, "_", i) for i in 1:nharmonics]
        period_access_logic = "local period_vals = [$(join(period_vars, ", "))]"
    end

    update_str = """
    begin
        # Harmonic model for $(spec.key) with $(nharmonics) independent harmonic(s).
        local harmonic_effect = zeros(T, length(M.$(index_var)))
        local amplitudes = abs.($(amplitude_raw_name))
        local phases = $(v.phase)
        local time_points = M.$(index_var)
        $(period_access_logic)
        
        for m in 1:$(nharmonics)
            local phase_rad = 2.0 * pi * phases[m]
            # The angle for the m-th harmonic uses its own independent period.
            local angle = (2.0 * pi / period_vals[m]) .* time_points
            
            local beta_cos = amplitudes[m] * cos(phase_rad)
            local beta_sin = amplitudes[m] * sin(phase_rad)
            
            harmonic_effect .+= beta_cos .* cos.(angle) .+ beta_sin .* sin.(angle)
        end
        
        $(eta_update_target) .+= harmonic_effect
    end
    """
    
    return (priors=priors_str, update=update_str)
end





function _generate_component_code_fragments(m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        push!(priors_acc, "$(v.rho1) ~ NamedDist($(_distribution_to_string(m.rho1)), :$(v.rho1))")
        push!(priors_acc, "$(v.rho2) ~ NamedDist($(_distribution_to_string(m.rho2)), :$(v.rho2))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # AR2 state-space evolution for $(spec.key)
        local $(v.latent) = Vector{T}(undef, $(n_latent))
        
        # NOTE: The initialization of the first two states is an approximation.
        # A full stationary initialization would require solving the Yule-Walker equations,
        # which is more complex. This approach is generally sufficient if the time series is long.
        if $(n_latent) > 0; $(v.latent)[1] = $(v.innov)[1]; end
        if $(n_latent) > 1; $(v.latent)[2] = $(v.rho1) * $(v.latent)[1] + $(v.innov)[2]; end
        for t in 3:$(n_latent)
            $(v.latent)[t] = $(v.rho1) * $(v.latent)[t-1] + $(v.rho2) * $(v.latent)[t-2] + $(v.innov)[t]
        end
        $(v.latent) .*= $(v.sigma)
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    return (priors=priors_str, update=update_str)
end



function _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # Purpose: Generates code for the Householder reflection (spectral orientation) feature.
    # Rationale: This allows for rotating the latent space in multivariate models to better
    # v1.0.0 (2026-07-16)
    #            align signals, which can be useful for processes with directional dependencies.
    #            This is controlled by the `spectral_orientation=true` keyword argument.
    # Inputs:
    #   - M: The model configuration NamedTuple.
    #   - is_multivariate: A boolean indicating if the model is multivariate.
    #   - eta_name: The name of the latent predictor matrix (e.g., "eta_latent").
    # Outputs: A tuple of strings (priors_str, update_str).

    if !is_multivariate || !M.spectral_orientation
        return "", ""
    end

    K = M[:outcomes_N]
    
    priors_str = """
    # Householder reflection for spectral orientation
    v_raw_reflection ~ NamedDist(MvNormal(zeros(T, $(K)), I), :v_raw_reflection)
    """

    update_str = """
    begin
        v_reflection = v_raw_reflection / (norm(v_raw_reflection) + 1e-9)
        H_reflection = I - 2.0 * v_reflection * v_reflection'
        $(eta_name) = $(eta_name) * H_reflection
    end
    """
    return priors_str, update_str
end



function bstm_text_assembler(M::NamedTuple, model_func_name::Symbol)
    # Purpose: Assembles the full Turing model string from all component fragments.
    # Rationale: This version adds a special handling block for multivariate dynamics models
    #            (like `leslie_matrix`). It calls a dedicated code generator for these models
    #            and ensures they are not processed again in the main component loop.
    # v1.0.1 (2026-07-31)
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"

    eta_name = is_multivariate ? "eta_latent" : "eta"
    eta_init = is_multivariate ? "zeros(T, N, K)" : "zeros(T, N)"
    outcomes_N = get(M, :outcomes_N, 1)

    spec_registry = Dict{String, Any}()
    priors_acc = String[]
    updates_acc = String[]

    main_spatial_spec = nothing
    main_temporal_spec = nothing
    
    has_custom_likelihood_from_component = any(spec -> any(T -> spec.component_obj isa T, [LGCP, LogGammaCoxProcess, ShotNoiseCoxProcess]), M.components)
    has_custom_likelihood_from_family = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    has_custom_likelihood = has_custom_likelihood_from_component || has_custom_likelihood_from_family

    # --- Handle multivariate dynamics separately ---
    if get(M, :is_multivariate_dynamics, false)
        mv_dyn_key = M[:multivariate_dynamics_key]
        spec_idx = findfirst(s -> string(s.key) == mv_dyn_key, M.components)
        if !isnothing(spec_idx)
            spec = M.components[spec_idx]
            spec_registry[string(spec.key)] = spec
            frags = _generate_multivariate_dynamics_code(spec.component_obj, spec, M)
            push!(priors_acc, frags.priors)
            push!(updates_acc, frags.update)
        end
    end

    for spec in M.components
        if get(M, :is_multivariate_dynamics, false) && string(spec.key) == M[:multivariate_dynamics_key]
            continue
        end
        spec_registry[string(spec.key)] = spec
        for k in 1:outcomes_N
            outcome_idx = is_multivariate ? k : nothing            
            frag = _generate_component_code_fragments(spec.component_obj, spec, arch, outcome_idx)
            if !isempty(Base.strip(frag.priors)); push!(priors_acc, frag.priors); end
            if !isempty(Base.strip(frag.update)); push!(updates_acc, frag.update); end
        end

        if spec.structure == :spatial && isnothing(main_spatial_spec)
            main_spatial_spec = spec
        end
        if spec.structure == :temporal && isnothing(main_temporal_spec)
            main_temporal_spec = spec
        end
    end

    function _indent_block(text::String, level=1)
        if isempty(Base.strip(text)) return "" end
        indent_str = "    " ^ level
        return indent_str * replace(Base.strip(text), "\n" => "\n" * indent_str)
    end

    likelihood_section = _generate_likelihood_section(M, is_multivariate)
    intercept_priors, intercept_update = _generate_intercept_block(M, is_multivariate, eta_name)
    offset_block = _generate_offset_block(M, is_multivariate, eta_name)
    fixed_effects_priors, fixed_effects_update = _generate_fixed_effects_block(M, is_multivariate, eta_name)
    st_interaction_block = _generate_st_interaction_block(M, main_spatial_spec, main_temporal_spec, is_multivariate, eta_name)
    householder_priors, householder_update = _generate_householder_reflection_block(M, is_multivariate, eta_name)
    
    final_likelihood = if has_custom_likelihood
        if has_custom_likelihood_from_family
            _generate_final_likelihood_block(M, is_multivariate)
        else
            ""
        end
    else
        _generate_final_likelihood_block(M, is_multivariate)
    end
    
    nested_priors, nested_updates, nested_likelihoods = _generate_nested_model_block(M, is_multivariate, eta_name)

    if !isempty(Base.strip(intercept_priors)); push!(priors_acc, intercept_priors); end
    if !isempty(Base.strip(fixed_effects_priors)); push!(priors_acc, fixed_effects_priors); end

    priors_code = join([p for p in priors_acc if !isempty(Base.strip(p))], "\n\n")
    updates_code = join([u for u in updates_acc if !isempty(Base.strip(u))], "\n\n")
 
    model_string = """
@model function $(model_func_name)(M, spec_registry; T::Type=Float64)
    noise = M.noise
    N = M.y_N
    K = $(outcomes_N)
$(_indent_block(likelihood_section))
$(_indent_block(priors_code))
$(_indent_block(householder_priors))
$(_indent_block(nested_priors))
    $(eta_name) = $(eta_init)
$(_indent_block(intercept_update))
$(_indent_block(offset_block))
$(_indent_block(fixed_effects_update))
$(_indent_block(updates_code))
$(_indent_block(householder_update))
$(_indent_block(nested_updates))
$(_indent_block(st_interaction_block))
$(_indent_block(final_likelihood))
$(_indent_block(nested_likelihoods))
end
"""
    
    model_string = join(filter(l -> !all(isspace, l), Base.split(model_string, '\n')), '\n')

    try
        return model_string, Meta.parse(model_string), spec_registry
    catch e
        println("BSTM Assembler Error: Failed to parse the generated model string.")
        println(model_string)
        rethrow(e)
    end
end


 

function bstm_codegen(config::NamedTuple)
    # Purpose: Generates the necessary components to define and instantiate a Turing model.
    # Rationale: Decouples code generation from evaluation to better handle Julia's world-age issues.
    # v1.0.0 (2026-07-18)
    # Assumptions: `config` is a valid model configuration.
    # Inputs:
    #   - config: The model configuration NamedTuple.
    # Outputs: A tuple containing the model function name, the model definition expression,
    #          the updated configuration, and the specification registry.
    # Generate a unique name for the model function to avoid world age issues
    # when interactively redefining models.
    random_suffix = rand(10000:99999)
    model_func_name = Symbol("bstm_dynamic_model_$(random_suffix)")

    model_string, expr, registry = bstm_text_assembler(config, model_func_name)

    config_dict = Dict(pairs(config))
    config_dict[:generated_model_code] = model_string
    new_config = NamedTuple(config_dict)
    
    return model_func_name, expr, new_config, registry
end



# ==============================================================================
# SECTION 8: LIKELIHOOD IMPLEMENTATION
# ==============================================================================

function bstm_Likelihood(family_input::Union{String, Symbol}, y_obs;
    zi_state=nothing, censoring_state=nothing, weight=1.0,
    phi_zi=-Inf, phi_hurdle=-Inf, r_nb=1.0, sigma_y=1.0, trial=1, 
    censor_lower=-Inf, censor_upper=Inf, hurdle=-Inf, extra_params=nothing
)
    # Purpose: Constructor for the unified likelihood structure.
    # Rationale: Provides a single, flexible constructor that uses traits to handle different likelihood modifications.
    # v1.0.0 (2026-07-16)
    # Assumptions: `family_input` is a valid key in `BSTM_FAMILY_REGISTRY`.
    # Inputs:
    #   - family_input: String or Symbol for the likelihood family.
    #   - y_obs: The observed data point(s).
    #   - kwargs: Optional parameters for likelihood modifications (censoring, ZI, etc.).
    # Outputs: An instance of the `bstm_Likelihood` struct with appropriate traits.
    f_trait = get_model_family(string(family_input))
    
    h_val = isnothing(hurdle) ? -Inf : hurdle
    
    zi_trait = phi_zi > -Inf ? ZeroInflated() : NonZeroInflated()

    yL_val = isnothing(censor_lower) ? -Inf : censor_lower
    yU_val = isnothing(censor_upper) ? Inf : censor_upper

    censor_trait = if !isfinite(yL_val) && !isfinite(yU_val); Uncensored()
        elseif isfinite(yL_val) && !isfinite(yU_val); RightCensored()
        elseif !isfinite(yL_val) && isfinite(yU_val); LeftCensored()
        else IntervalCensored() end

    y_vec = y_obs isa AbstractVector ? y_obs : [y_obs]
    
    return bstm_Likelihood(f_trait, y_vec, zi_trait, censor_trait, weight, phi_zi, phi_hurdle, r_nb, sigma_y, trial, yL_val, yU_val, h_val, extra_params)
end

Base.length(d::bstm_Likelihood) = length(d.y_obs)
Base.size(d::bstm_Likelihood) = (length(d.y_obs),)

function Distributions._logpdf(d::bstm_Likelihood, eta::AbstractVector{<:Real})
    # Purpose: Internal logpdf implementation for vector-based observations.
    # Rationale: Required for `ContinuousMultivariateDistribution` compliance.
    # v1.0.0 (2026-07-16)
    # Assumptions: `eta` has the same length as `d.y_obs`.
    # Inputs:
    #   - d: The `bstm_Likelihood` instance.
    #   - eta: A vector of linear predictors.
    # Outputs: The total log-probability.
    logp = 0.0
    if d.family isa DirichletMultinomialFamily
        # For a single multivariate observation, eta is a vector of predictors for each category.
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        # d.y_obs is already the vector of counts for this observation.
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs) * w
    else
        # For independent univariate observations, loop through each one.
        for i in 1:length(eta)
            sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            w = d.weight isa AbstractVector ? d.weight[i] : d.weight
            logp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
        end
        return logp
    end
end
 

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    # This method is for a single scalar observation. It is not used by the DirichletMultinomial path.
    # It is preserved for backward compatibility with existing univariate likelihoods.
    if d.family isa DirichletMultinomialFamily
        error("DirichletMultinomial likelihood requires a vector of linear predictors, but received a scalar.")
    end

    # Purpose: Public scalar overload for `logpdf`.
    # Rationale: Provides a convenient interface for single-observation likelihood evaluation.
    # v1.0.0 (2026-07-16)
    # Assumptions: `d.y_obs` contains a single observation.
    # Inputs:
    #   - d: The `bstm_Likelihood` instance.
    #   - eta: A scalar linear predictor.
    # Outputs: The log-probability.
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

function Distributions.logpdf(d::bstm_Likelihood, y::AbstractVector{<:Real})
    # Purpose: Public vector overload to maintain `MultivariateDistribution` compliance.
    # Rationale: Delegates to the internal `_logpdf` implementation.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - d: The `bstm_Likelihood` instance.
    #   - y: A vector of linear predictors (matches `eta` in `_logpdf`).
    # Outputs: The total log-probability.
    return Distributions._logpdf(d, y)
end

function get_model_family(model_family::String)
    # Purpose: Maps a string identifier to its corresponding concrete `AbstractBSTM_Family` type.
    # Rationale: Centralizes the mapping from string names to type instances.
    # v1.0.0 (2026-07-16)
    # Assumptions: `model_family` is a valid key.
    # Inputs:
    #   - model_family: The string name of the family.
    # Outputs: An instance of a concrete subtype of `AbstractBSTM_Family`.
    family_key = lowercase(model_family)
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$model_family'. Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end

function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(clamp(exp(eta), 0.0, 1e9)); end
function get_dist_ref(::DirichletFamily, d, eta, sig); error("The Dirichlet likelihood is for compositional outcomes and is not supported in the current univariate response framework."); end
function get_dist_ref(::InverseWishartFamily, d, eta, sig); error("The Inverse-Wishart likelihood is for covariance matrix outcomes and is not supported in the current univariate response framework."); end
function get_dist_ref(::GaussianFamily, d, eta, sig); return Normal(eta, max(sig, 1e-9)); end
function get_dist_ref(::LogNormalFamily, d, eta, sig)
    # For a log-normal GLM, the standard log-link models the mean of the response, E[y] = exp(eta).
    # The LogNormal(μ, σ) distribution has a mean of exp(μ + σ²/2).
    # To match these, we must set μ = eta - σ²/2. This ensures the model's linear
    # predictor `eta` correctly corresponds to the log of the expected value of the response.
    # This also corrects the Jacobian for the transformation from the log-scale predictor
    # to the response-scale data.
    μ = eta - (sig^2) / 2.0
    return LogNormal(μ, max(sig, 1e-9))
end
function get_dist_ref(::NegativeBinomialFamily, d, eta, sig)
    # The mean of the Negative Binomial is modeled on the log scale: μ = exp(eta).
    # The distribution can be parameterized by its mean (μ) and a dispersion parameter (ϕ),
    # where the variance is μ + μ²/ϕ. In our framework, `d.r_nb` is this dispersion parameter.
    # Using the (μ, ϕ) constructor is more direct and numerically stable than converting to the (r, p) parameterization.
    μ = clamp(exp(eta), 1e-9, 1e9)
    return NegativeBinomial(μ, d.r_nb)
end
function get_dist_ref(::BinomialFamily, d, eta, sig); n = d.trial isa AbstractVector ? d.trial[1] : d.trial; return Binomial(Int(n), logistic(eta)); end
function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, clamp(exp(eta), 1e-9, 1e9)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(clamp(mu*phi, 1e-9, Inf), clamp((1.0-mu)*phi, 1e-9, Inf)); end
function get_dist_ref(::InverseGaussianFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return InverseGaussian(mu, lambda); end
function get_dist_ref(::StudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return LocationScale(eta, max(sig, 1e-9), TDist(nu)); end
function get_dist_ref(::HalfNormalFamily, d, eta, sig); return truncated(Normal(0.0, max(sig, 1e-9)), 0.0, Inf); end
function get_dist_ref(::HalfStudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return truncated(LocationScale(0.0, max(sig, 1e-9), TDist(nu)), 0.0, Inf); end
function get_dist_ref(::LaplaceFamily, d, eta, sig); return Laplace(eta, max(sig, 1e-9)); end
function get_dist_ref(::ParetoFamily, d, eta, sig)
    shape = d.extra_params isa Number && d.extra_params > 1.0 ? d.extra_params : 1.1
    mean_val = clamp(exp(eta), 1e-9, 1e9)
    scale = mean_val * (shape - 1.0) / shape
    return Pareto(shape, scale)
end
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)
    alpha_0 = max(sig, 1e-6)
    mean_probs = softmax(eta_vec)
    alpha_params = alpha_0 .* mean_probs
    n_total = sum(d.y_obs)
    return DirichletMultinomial(Int(n_total), alpha_params)
end

# # Add trait for Dirichlet family

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end
function is_discrete_family(::AbstractBSTM_Family)
    return false
end

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zero_inflated::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for an uncensored observation.
    # Rationale: Handles the three cases: standard, zero-inflated, and hurdle models.
    # v1.0.0 (2026-07-16)
    # Assumptions: `d` is a valid `bstm_Likelihood` object.
    # Inputs:
    #   - fam, censoring_state, zi_state: Traits for dispatch.
    #   - d: The likelihood struct.
    #   - eta, sig, y: Linear predictor, scale, and observation.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    if zero_inflated isa ZeroInflated
        # Numerically stable log-probabilities for the ZI mixture.
        log_phi = log(d.phi_zi)
        log_one_minus_phi = log1p(-d.phi_zi)
        
        if y == 0.0
            if is_discrete_family(fam)
                logp_base_zero = logpdf(dist, 0.0)
                # log( P(zero) + (1-P(zero)) * P_base(y=0) )
                return logsumexp(log_phi, log_one_minus_phi + logp_base_zero)
            else
                # For continuous distributions, P(y=0) is 0, so the probability of observing 0
                # comes entirely from the zero-inflation component.
                return log_phi
            end
        else
            # log( (1-P(zero)) * P_base(y) )
            return log_one_minus_phi + logpdf(dist, y)
        end
    elseif d.phi_hurdle > -Inf
        # Numerically stable log-probabilities for the hurdle mixture.
        log_phi = log(d.phi_hurdle)
        log_one_minus_phi = log1p(-d.phi_hurdle)
        
        if y <= d.hurdle
            # Probability of being at or below the hurdle.
            return log_one_minus_phi
        else
            # Probability of being above the hurdle, from the truncated distribution.
            # log( P(above) * P_base(y | y > hurdle) )
            logp_truncated = logpdf(dist, y) - logccdf(dist, d.hurdle)
            return log_phi + logp_truncated
        end
    else
        # Standard non-modified likelihood.
        return logpdf(dist, y)
    end
end

function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zero_inflated::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for a left-censored observation.
    # Rationale: Correctly calculates the cumulative probability for standard, ZI, and hurdle models.
    # v1.0.0 (2026-07-16)
    # Assumptions: `d.censor_upper` is finite.
    # Inputs/Outputs: See `bstm_kernel` for uncensored.
    dist = get_dist_ref(fam, d, eta, sig)
    upper_bound = d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper

    if zero_inflated isa ZeroInflated
        log_phi = log(d.phi_zi)
        log_one_minus_phi = log1p(-d.phi_zi)
        lp_base = logcdf(dist, upper_bound)
        if upper_bound >= 0.0
            # P(y <= U) = P(y=0) + P(0 < y <= U) = phi + (1-phi)*P_base(y <= U)
            return logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            # If U < 0, the zero component is not included in the interval.
            return log_one_minus_phi + lp_base
        end
    elseif d.phi_hurdle > -Inf
        log_phi = log(d.phi_hurdle)
        log_one_minus_phi = log1p(-d.phi_hurdle)
        if upper_bound <= d.hurdle
            return log_one_minus_phi
        end
        # P(y <= upper_bound) = P(y <= hurdle) + P(hurdle < y <= upper_bound)
        # log P(y <= upper_bound) = logsumexp( log(1-phi), log(phi) + log P(y <= upper_bound | y > hurdle) )
        log_prob_in_interval_given_hurdle = _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, d.hurdle)) - logccdf(dist, d.hurdle)
        return logsumexp(log_one_minus_phi, log_phi + log_prob_in_interval_given_hurdle)
    else
        return logcdf(dist, upper_bound)
    end
end

function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zero_inflated::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for a right-censored observation.
    # Rationale: Correctly calculates the complementary cumulative probability for all model types.
    # v1.0.0 (2026-07-16)
    # Assumptions: `d.censor_lower` is finite.
    # Inputs/Outputs: See `bstm_kernel` for uncensored.
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower
    adj_L = is_discrete_family(fam) ? lower_bound - 1.0 : lower_bound

    if zero_inflated isa ZeroInflated
        log_phi = log(d.phi_zi)
        log_one_minus_phi = log1p(-d.phi_zi)
        
        log_p_le_L = if lower_bound < 0.0
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        # P(y > L) = 1 - P(y <= L)
        return log1mexp(log_p_le_L)

    elseif d.phi_hurdle > -Inf
        log_phi = log(d.phi_hurdle)
        adj_L = is_discrete_family(fam) ? lower_bound - 1.0 : lower_bound
        adj_hurdle = is_discrete_family(fam) ? d.hurdle - 1.0 : d.hurdle

        if lower_bound > d.hurdle
            # P(Y > lower_bound) = phi * P_trunc(Y > lower_bound)
            return log_phi + logccdf(dist, adj_L) - logccdf(dist, adj_hurdle)
        else # lower_bound <= hurdle
            # P(Y > lower_bound) = P(Y > hurdle) = phi
            return log_phi
        end
    else
        return logccdf(dist, adj_L)
    end
end

function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zero_inflated::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for an interval-censored observation.
    # Rationale: Calculates the probability mass within the interval [censor_lower, censor_upper].
    # v1.0.0 (2026-07-16)
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower
    upper_bound = d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper
    adj_L = is_discrete_family(fam) ? lower_bound - 1.0 : lower_bound

    if zero_inflated isa ZeroInflated
        log_phi = log(d.phi_zi)
        log_one_minus_phi = log1p(-d.phi_zi)

        # P(y <= U) for ZI model
        log_p_le_U = if upper_bound < 0.0; log_one_minus_phi + logcdf(dist, upper_bound);
        else; logsumexp(log_phi, log_one_minus_phi + logcdf(dist, upper_bound)); end

        # P(y <= L) for ZI model
        log_p_le_L = if lower_bound < 0.0; log_one_minus_phi + logcdf(dist, lower_bound);
        else; logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound)); end

        # P(L < y <= U) = P(y <= U) - P(y <= L)
        return _stable_logsubexp(log_p_le_U, log_p_le_L)

    elseif d.phi_hurdle > -Inf
        log_phi = log(d.phi_hurdle)
        adj_L = is_discrete_family(fam) ? lower_bound - 1.0 : lower_bound
        adj_hurdle = is_discrete_family(fam) ? d.hurdle - 1.0 : d.hurdle

        if upper_bound <= d.hurdle
            return -Inf
        end

        effective_lower = max(adj_L, adj_hurdle)
        log_prob_in_interval = _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, effective_lower))
        log_normalizer = logccdf(dist, adj_hurdle)
        return log_phi + log_prob_in_interval - log_normalizer
    else
        return _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, adj_L))
    end
end

 
function bstm_kernel(fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec)
    dist = get_dist_ref(fam, d, eta_vec, sig)
    return logpdf(dist, y_vec)
end

 
# ==============================================================================
# SECTION 9: UTILITY AND HELPER FUNCTIONS
# ==============================================================================
### Version 1.9.8 - 2025-05-22 17:00:00
### Technical Descriptor: Consolidated Inducing Point Selection with KDTree Optimization

function generate_inducing_points(coords::AbstractMatrix, n_inducing::Int; method::String="kmeans", seed::Int=42)
    # Purpose: Selects a representative subset of coordinates to serve as inducing points for sparse GPs.
    # Rationale: Inducing points must be actual or representative locations in the input space. 
    # Systematic methods (quantile/regular) generate ideal targets that are then mapped to the 
    # nearest available data points using an efficient KDTree search to ensure spatial fidelity.

    n_obs, n_dims = size(coords)

    if n_inducing >= n_obs
        return coords
    end

    Random.seed!(seed)

    if method == "random"
        # Simple stochastic selection
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]

    elseif method == "kmeans"
        # Centroid-based selection via Clustering.jl
        # kmeans expects observations in columns: [dims x obs]
        kmeans_res = Clustering.kmeans(coords', n_inducing; maxiter=200, display=:none)
        return kmeans_res.centers'

    elseif method == "quantile" || method == "regular"
        # Systematic mapping methods requiring KDTree for efficiency
        
        target_pts = zeros(Float64, n_inducing, n_dims)
        
        if method == "quantile"
            # Density-aware target generation using marginal quantiles
            probs = range(0.0, stop=1.0, length=n_inducing)
            for d in 1:n_dims
                target_pts[:, d] = Statistics.quantile(coords[:, d], probs)
            end
        else 
            # method == "regular"
            # Grid-like target generation across marginal ranges
            for d in 1:n_dims
                v_min, v_max = extrema(coords[:, d])
                target_pts[:, d] = range(v_min, stop=v_max, length=n_inducing)
            end
        end

        # Efficient Nearest Neighbor Search
        # Build KDTree from the data points
        tree = KDTree(coords')
        
        # Find the single nearest observation for each target coordinate
        # knn returns (indices, distances)
        nn_indices_vec, _ = knn(tree, target_pts', 1, true)
        
        # Extract the scalar index from each neighbor search result and deduplicate
        unique_nn_indices = unique([idx_list[1] for idx_list in nn_indices_vec])
        
        return coords[unique_nn_indices, :]

    else
        @warn "Inducing point method '$method' not recognized. Falling back to random selection."
        selected_idx = StatsBase.sample(1:n_obs, n_inducing, replace=false)
        return coords[selected_idx, :]
    end
end
 


function _stable_logsubexp(a::Real, b::Real)
    # Purpose: Numerically stable computation of log(exp(a) - exp(b)).
    # Rationale: Avoids overflow and underflow by factoring out the larger term.
    # v1.0.0 (2026-07-16)
    #            This is equivalent to LogExpFunctions.logsubexp.
    # Inputs:
    #   - a, b: Real numbers.
    # Outputs: log(exp(a) - exp(b)).
    if a <= b
        return -Inf
    end
    return a + log1mexp(b - a)
end

function ar1_statespace(rho, sigma, innov, T, n_latent, noise)
    # Helper function for AR(1) state-space evolution.
    latent = Vector{T}(undef, n_latent)
    if n_latent > 0
        latent[1] = innov[1] / sqrt(1.0 - rho^2 + noise)
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innov[t]
        end
        latent .*= sigma
    end
    return latent
end

function create_pc_prior(param_name::Symbol, constraint::Tuple)
    # Purpose: Creates a Penalized Complexity (PC) prior distribution from a user-specified quantile constraint.
    # Rationale: Translates an intuitive belief (e.g., "P(sigma > 1.0) = 0.05") into a formal prior distribution.
    #            This version is updated to use the base parameter name directly, as the `_prior` suffix
    #            is no longer part of the API for component structs.
    # v1.0.0 (2026-07-20)
    # Assumptions: `param_name` is one of the recognized types (:sigma, :rho, etc.).
    # Inputs:
    #   - param_name: The base name of the parameter.
    #   - constraint: A tuple `(U, α)` or `(U, α, direction)`.
    # Outputs: A `Distribution` object.

    
    direction = :upper
    if length(constraint) == 2; U, α = constraint; elseif length(constraint) == 3; U, α, direction = constraint; else; error("PC prior constraint must be a tuple of (U, α) or (U, α, direction)."); end
    
    if param_name == :sigma || endswith(string(param_name), "_sigma")
        direction != :upper && error("PC prior for sigma only supports upper tail constraints.")
        λ = -log(α) / U
        return Exponential(λ)
    elseif param_name == :rho || endswith(string(param_name), "_rho")
        direction != :upper && error("PC prior for 'rho' only supports upper tail constraints.")
        λ = log(α) / log(1.0 - U)
        return Exponential(λ)
    elseif param_name == :lengthscale || endswith(string(param_name), "_lengthscale")
        direction != :lower && error("PC prior for 'lengthscale' only supports lower tail constraints.")
        λ = -U * log(α)
        return Exponential(λ)
    elseif param_name == :kappa || endswith(string(param_name), "_kappa")
        direction != :upper && error("PC prior for kappa only supports upper tail constraints.")
        λ = -log(α) / U
        return Exponential(λ)
    else
        sigma = -U / quantile(Normal(0, 1), α / 2)
        return Normal(0, sigma)
    end
end


function get_optimal_sampler(
    model_obj::DynamicPPL.Model;
    sampler_choice=:auto,
    sampler_map::Dict{Symbol, <:AbstractMCMC.AbstractSampler}=Dict{Symbol, AbstractMCMC.AbstractSampler}(),
    target_acceptance=0.8,
    adaptation_steps=1000,
    group_components::Bool=true,
    n_particles=20,
    hmc_leapfrog_steps=10
)
    # Purpose: Automatically constructs an efficient composite Gibbs sampler for a `bstm` model.
    # Rationale: This function is updated to use the modern Turing.jl API for Gibbs sampling.
    #            The deprecated API of passing parameter spaces directly to sub-sampler constructors
    #            (e.g., `NUTS(..., space)`) caused a persistent `MethodError`. The corrected
    #            implementation now creates samplers without space arguments and assigns them to
    #            parameter groups using `space => sampler` pairs within the `Gibbs` constructor.
    # v1.0.5 (2026-07-30)
    # Assumptions: The model has been instantiated.
    # Inputs:
    #    - model_obj: The instantiated Turing.jl model object.
    #    - sampler_choice: If a specific sampler algorithm is provided, it is used directly.
    #    - sampler_map: A dictionary to manually assign specific samplers to parameter symbols.
    #    - target_acceptance: The target acceptance rate for `NUTS`.
    #    - adaptation_steps: The number of adaptation steps for `NUTS`.
    #    - group_components: If `true`, groups all parameters of a single component into a single `NUTS` block.
    #    - n_particles: The number of particles for the `PG` sampler.
    #    - hmc_leapfrog_steps: Unused, kept for API consistency.
    # Outputs: A Turing.jl sampler object (e.g., `Gibbs`, `NUTS`).

    if sampler_choice isa AbstractMCMC.AbstractSampler
        @info "Using user-specified sampler: $(typeof(sampler_choice))"
        return sampler_choice
    end

    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)

    sampler_assignments = [] # This will be a vector of `space => sampler` Pairs.
    all_processed_vns = Set{VarName}()

    # 1. Handle user-provided sampler map first.
    for (param_sym, sampler) in sampler_map
        sym_vns = filter(vn -> DynamicPPL.getsym(vn) == param_sym, vns)
        if !isempty(sym_vns)
            # Group all VarNames for the given symbol into a single assignment.
            push!(sampler_assignments, Tuple(sym_vns) => sampler)
            union!(all_processed_vns, sym_vns)
            @info "Applying user-defined sampler $(typeof(sampler)) for parameter group: $(param_sym)"
        else
            @warn "Parameter :$(param_sym) in sampler_map not found in model."
        end
    end

    # 2. Handle component grouping if enabled.
    if group_components
        @info "Component grouping enabled. Grouping hyperparameters and latent fields for joint sampling."
        component_groups = Dict{String, Set{VarName}}()

        param_suffixes = [
            "sigma", "rho", "rho1", "rho2", "rho_field", "kappa", "ls", "range", "period",
            "amplitude", "phase", "raw", "innov", "latent", "struct", "iid",
            "velocity", "diffusion", "pca_sd", "pdef_sd", "beta_cos", "beta_sin",
            "L_corr", "sigma_effects"
        ]
        regex_str = "^(.+?)_(" * join(param_suffixes, "|") * ")\$"
        component_regex = Regex(regex_str)

        for vn in vns
            if vn in all_processed_vns; continue; end

            m = match(component_regex, string(DynamicPPL.getsym(vn)))

            if !isnothing(m)
                component_key = m.captures[1]
                if !haskey(component_groups, component_key); component_groups[component_key] = Set{VarName}(); end
                push!(component_groups[component_key], vn)
            end
        end

        for (key, params_vns) in component_groups
            if !isempty(params_vns)
                # FIX: Create sampler without space, assign space in a Pair.
                sampler = NUTS(adaptation_steps, target_acceptance)
                push!(sampler_assignments, Tuple(params_vns) => sampler)
                union!(all_processed_vns, params_vns)
                param_syms = Set(DynamicPPL.getsym.(params_vns))
                @info "Created NUTS block for component '$(key)' with parameters: $(param_syms)"
            end
        end
    end

    # 3. Default grouping for all remaining parameters.
    remaining_vns = filter(vn -> !(vn in all_processed_vns), vns)
    if !isempty(remaining_vns)
        param_groups = Dict(:discrete => Set{VarName}(), :gaussian => Set{VarName}(), :bounded => Set{VarName}(), :other_continuous => Set{VarName}())

        for vn in remaining_vns
            try
                dist = DynamicPPL.getdist(vi, vn)
                support = Distributions.value_support(typeof(dist))
                if support isa Distributions.Discrete; push!(param_groups[:discrete], vn);
                elseif support isa Distributions.Continuous
                    if dist isa Union{Normal, MvNormal, Truncated{<:Normal}}; push!(param_groups[:gaussian], vn);
                    elseif isfinite(minimum(dist)) || isfinite(maximum(dist)); push!(param_groups[:bounded], vn);
                    else; push!(param_groups[:other_continuous], vn); end
                end
            catch e; push!(param_groups[:other_continuous], vn); end
        end

        # FIX: Create samplers without space, assign space in Pairs.
        if !isempty(param_groups[:discrete])
            params = Tuple(param_groups[:discrete])
            push!(sampler_assignments, params => PG(n_particles))
            @info "Using Particle Gibbs (PG) for: $(DynamicPPL.getsym.(params))"
        end
        if !isempty(param_groups[:gaussian])
            params = Tuple(param_groups[:gaussian])
            push!(sampler_assignments, params => ESS())
            @info "Using Elliptical Slice Sampling (ESS) for: $(DynamicPPL.getsym.(params))"
        end
        if !isempty(param_groups[:bounded])
            params = Tuple(param_groups[:bounded])
            push!(sampler_assignments, params => Slice())
            @info "Using Slice sampling for: $(DynamicPPL.getsym.(params))"
        end
        if !isempty(param_groups[:other_continuous])
            params = Tuple(param_groups[:other_continuous])
            push!(sampler_assignments, params => NUTS(adaptation_steps, target_acceptance))
            @info "Using NUTS for remaining continuous parameters: $(DynamicPPL.getsym.(params))"
        end
    end

    # 4. Construct the final sampler.
    if isempty(sampler_assignments)
        @warn "Could not identify any parameters to sample. Defaulting to NUTS for all."
        return NUTS(adaptation_steps, target_acceptance)
    elseif length(sampler_assignments) == 1
        # Gibbs is still the correct constructor even for a single assignment.
        return Gibbs(sampler_assignments[1])
    else
        return Gibbs(sampler_assignments...)
    end
end



function create_fixed_design(formula_rhs::AbstractString, data::DataFrame; contrasts=Dict{Symbol, Any}())
    # Purpose: Creates the fixed-effects design matrix (`X`) from a formula string.
    # Rationale: A wrapper around `StatsModels.jl` to handle formula parsing and contrast coding.
    # v1.0.0 (2026-07-16)
    # Assumptions: `formula_rhs` contains only fixed effects terms.
    # Inputs:
    #   - formula_rhs: The RHS of the formula string.
    #   - data: The input DataFrame.
    #   - contrasts: A dictionary specifying contrast coding for categorical variables.
    # Outputs: A NamedArray containing the design matrix and the applied formula object.
    df_internal = copy(data)
    final_rhs_string = Base.strip(formula_rhs)

    if isempty(final_rhs_string)
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[])), nothing
    end

    if final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept])), nothing
    end

    try
        placeholder_name = :__y_placeholder
        df_internal[!, placeholder_name] = zeros(size(df_internal, 1))

        formula_expression = Meta.parse("@formula($placeholder_name ~ $final_rhs_string)")
        dynamic_formula = Main.eval(formula_expression)

        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema, StatsModels.RegressionModel)

        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)

        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector)), applied_formula

    catch design_error
        @warn "BSTM Registry: create_fixed_design expansion failed for: $final_rhs_string. Error: $design_error"
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[])), nothing
    end
end


function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Purpose: Constructs an orthonormal loadings matrix (eigenvectors) from a matrix of Householder reflector vectors.
    # Rationale: Provides a differentiable and numerically stable way to parameterize an orthonormal matrix for Bayesian PCA.
    # v1.0.0 (2026-07-16)
    # Assumptions: `v_mat` contains the reflector vectors.
    # Inputs:
    #   - v_mat: Matrix of reflector vectors.
    #   - nU: Number of variables.
    #   - n_factors: Number of factors.
    # Outputs: An orthonormal loadings matrix `[nU x n_factors]`.
    U = Matrix{T}(I, nU, nU)

    for k in 1:n_factors
        vk = v_mat[:, k]
        norm_v = LinearAlgebra.norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    return U[:, 1:n_factors]
end

function show_model(m::DynamicPPL.Model)
    # Purpose: Displays a comprehensive summary of the `bstm` model configuration and a pseudo-code representation.
    # Rationale: Provides a user-friendly way to inspect the model structure before and after fitting.
    # v1.0.0 (2026-07-16)
    # Assumptions: `m` is a Turing model generated by the `bstm` framework.
    # Inputs:
    #   - m: The Turing model instance.
    # Outputs: None (prints to console).
    println("\n--- Model Summary ---\n")
    config = m.args.M
    println("Model Name: ", get(config, :model_name, nameof(m.f)))
    println("Model Architecture: ", get(config, :model_arch, "N/A"))
    println("Likelihood Family: ", get(config.likelihood_specs[1], :family, "N/A"))
    println("Number of observations: ", get(config, :y_N, "N/A"))
    println("Number of spatial units: ", get(config, :s_N, "N/A"))
    println("Number of time units: ", get(config, :t_N, "N/A"))
    println("\nFixed Effects:")
    if get(config, :Xfixed_N, 0) > 0
        println("  Variables: ", join(string.(get(config, :Xfixed_names, ["N/A"])), ", "))
    else
        println("  None")
    end
    println("\nComponents:\n")
    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            println("  - Key: ", spec.key)
            println("    Structure: ", spec.structure)
            println("    Variable: ", spec.var)
            println("    Component Type: ", typeof(spec.component_obj))
            println("    Parameters:")
            for (p_key, p_val) in pairs(spec.params)
                println("      ", p_key, ": ", p_val)
            end
        end
    else
        println("  None")
    end
    if haskey(config, :generated_model_code)
        println("\n--- Generated Model Source ---\n")
        println(config.generated_model_code)
        println("\n--- End Generated Model Source ---")
    else
        println("\n--- Reconstructed Model Source (Pseudo-code) ---\n")
        println(_generate_model_pseudocode(m))
        println("\n--- End Reconstructed Model Source ---")
    end
    println("\n--- End Model Summary ---")
    return nothing
end


function _generate_model_pseudocode(m::DynamicPPL.Model)
    # Purpose: Reconstructs a pseudo-code representation of the Turing model definition.
    # Rationale: For inspection and clarity. This version is updated to use the simplified
    #            hyperparameter names (e.g., `sigma` instead of `sigma_prior`) and to be
    #            more comprehensive in the hyperpriors it checks for.
    # v1.0.0 (2026-07-20)
    # Assumptions: `m` is a Turing model generated by the `bstm` framework.
    # Inputs:
    #   - m: The Turing model instance.
    # Outputs: A string containing the pseudo-code.
    config = m.args.M
    model_name = get(config, :model_name, nameof(m.f))
    
    lines = ["@model function $model_name(M)"]
    push!(lines, "    # --- Priors & Hyperparameters ---")

    family = get(config.likelihood_specs[1], :family, "gaussian")
    if family == "negbin"
        push!(lines, "    r_nb ~ Exponential(1.0)")
    end
    if get(config, :use_zi, false)
        push!(lines, "    phi_zi ~ Beta(1, 1)")
    end
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"] && !get(config, :use_sv, false)
        push!(lines, "    y_sigma ~ Exponential(1.0)")
    end

    if haskey(config, :components) && !isempty(config.components)
        for spec in config.components
            m_obj = spec.component_obj
            m_type_str = string(typeof(m_obj))
            key = spec.key
            
            push!(lines, "\n    # Priors for component: $(key) ($(m_type_str))")
            
            possible_hyperpriors = [
                (:sigma, "sigma_$(key)"),
                (:rho, "rho_$(key)"),
                (:lengthscale, "ls_$(key)"),
                (:kappa, "kappa_$(key)"),
                (:amplitude, "amp_$(key)"),
                (:phase, "phase_$(key)"),
                (:range, "range_$(key)"),
                (:pca_sd, "pca_sd_$(key)"),
                (:pdef_sd, "pdef_sd_$(key)")
            ]

            for (field_sym, name_str) in possible_hyperpriors
                if hasproperty(m_obj, field_sym)
                    prior_dist = getfield(m_obj, field_sym)
                    if !isnothing(prior_dist)
                        push!(lines, "    $(name_str) ~ $(prior_dist)")
                    end
                end
            end
        end
    end

    if get(config, :Xfixed_N, 0) > 0
        push!(lines, "\n    # Prior for fixed effects")
        push!(lines, "    Xfixed_beta ~ MvNormal(0, 5.0 * I)")
    end

    push!(lines, "\n    # --- Latent Field Definitions & Linear Predictor Assembly ---")
    eta_parts = haskey(config, :log_offset) && !all(iszero, get(config, :log_offset, [])) ? ["M.log_offset"] : []

    if get(config, :add_intercept, false)
        push!(eta_parts, "intercept")
    end
    
    model_st = get(config, :model_st, "none")
    if model_st != "none"
        push!(eta_parts, "spacetime_interaction")
    end

    if haskey(config, :components)
        for spec in config.components
            push!(eta_parts, string(spec.key))
        end
    end
    if get(config, :Xfixed_N, 0) > 0
        push!(eta_parts, "M.Xfixed * Xfixed_beta")
    end
    
    push!(lines, "    eta = " * (isempty(eta_parts) ? "zeros(M.y_N)" : join(eta_parts, " .+ ")))

    push!(lines, "\n    # --- Likelihood ---")
    push!(lines, "    y_obs ~ bstm_Likelihood(\"$family\", eta, ...)")
    push!(lines, "end")

    return join(lines, "\n")
end


# ==============================================================================
# SECTION 10: ADVANCED MODELING UTILITIES 
# ==============================================================================

"""
    bstm_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; knot_method::Symbol=:quantile, custom_knots::Union{AbstractVector, Nothing}=nothing)

Generates a B-spline basis matrix of a specified degree. This function implements the
De Boor-Cox recursive formula in an iterative manner to create basis functions.

# Arguments
- `x`: The vector of data points for which the basis is evaluated.
- `n_basis`: The number of basis functions to generate.
- `degree`: The polynomial degree of the B-spline (e.g., 1 for linear, 3 for cubic).
- `knot_method`: The method for placing interior knots. Can be `:quantile` (default) for knots placed at data quantiles, or `:range` for knots spaced evenly over the data range.
- `custom_knots`: An optional vector of pre-defined interior knots.

# Returns
- A tuple `(Matrix, Int)` containing the basis matrix of size `(length(x), n_basis)` and the final number of basis functions used.
"""
function bstm_bspline_basis(x::AbstractVector, n_basis::Int, degree::Int; knot_method::Symbol=:quantile, custom_knots::Union{AbstractVector, Nothing}=nothing)
    p = degree
    if n_basis <= p
        error("Number of basis functions (nbins) must be greater than the spline degree. Got n_basis=$n_basis, degree=$p.")
    end

    # The number of interior knots is determined by the number of basis functions and the degree.
    n_interior_knots = n_basis - p

    local knots
    if !isnothing(custom_knots)
        knots = custom_knots
    else
        # Place interior knots based on the chosen method.
        if n_interior_knots > 0
            if knot_method == :quantile
                # Ensure that quantile probabilities are unique to avoid issues with discrete data
                probs = range(0, 1, length=n_interior_knots + 2)[2:end-1]
                knots = quantile(x, probs)
            else # :range
                knots = range(minimum(x), maximum(x), length=n_interior_knots + 2)[2:end-1]
            end
        else
            knots = Float64[]
        end
    end

    # Define the full knot vector by adding boundary knots.
    boundary_knots = [minimum(x), maximum(x)]
    all_knots = sort(unique(vcat(boundary_knots, knots)))

    # --- Robustness Check for Low-Variability Data ---
    # The number of basis functions that can be constructed depends on the number of unique knots.
    # If the data has low variability, quantile-based knots can be identical, reducing the
    # number of unique knots and thus the possible number of basis functions.
    n_basis_possible = length(all_knots) + p - 1
    
    if n_basis > n_basis_possible
        @warn "Requested n_basis ($n_basis) is too high for the number of unique knots ($(length(all_knots))) and degree ($p) supported by the data's variability. Reducing n_basis to the maximum possible value: $n_basis_possible."
        n_basis = n_basis_possible
    end
    # --- End Robustness Check ---

    # Augment the knot vector by repeating the boundary knots `p` times at each end.
    # This is required for the De Boor-Cox recursion.
    t = vcat(fill(all_knots[1], p), all_knots, fill(all_knots[end], p))
    
    N = length(x)
    # The number of basis functions is length of augmented knots - degree - 1.
    num_total_basis = length(t) - p - 1
    B = zeros(N, num_total_basis)

    # Base case: degree 0 splines are piecewise constant.
    for j in 1:num_total_basis
        B[:, j] = (t[j] .<= x .< t[j+1])
    end
    # Ensure the last point is included in the final basis function.
    if !isempty(x) && t[end] == maximum(x)
        B[x .== t[end], num_total_basis] .= 1.0
    end

    # Iteratively compute higher-degree splines from lower-degree ones.
    for d in 1:p
        for j in 1:(num_total_basis - d)
            w1 = zeros(N)
            denom1 = t[j+d] - t[j]
            if denom1 > 1e-9 # Avoid division by zero
                w1 = (x .- t[j]) ./ denom1
            end
            
            w2 = zeros(N)
            denom2 = t[j+d+1] - t[j+1]
            if denom2 > 1e-9 # Avoid division by zero
                w2 = (t[j+d+1] .- x) ./ denom2
            end
            
            B[:, j] = w1 .* B[:, j] + w2 .* B[:, j+1]
        end
    end

    # Return the final basis matrix and the actual number of basis functions.
    return (B[:, 1:n_basis], n_basis)
end

"""
    bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; ...)

Generates a tensor product B-spline basis matrix for multidimensional data.

# Rationale for Update
This version fixes a `MethodError` that occurred when unsupported keyword arguments
were passed down to the `bstm_bspline_basis` function. The `kwargs...` splat has been
removed from the call to `bstm_bspline_basis`, and only the relevant keyword arguments
(`knot_method`, `custom_knots`) are explicitly passed. It also correctly handles the
tuple `(matrix, n_basis)` returned by `bstm_bspline_basis` to ensure the tensor
product is computed on the basis matrices.
"""
function bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; knot_method::Symbol=:quantile, kwargs...)
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims || length(degrees_per_dim) != n_dims
        error("Number of dimensions in coords must match length of nbins_per_dim and degrees_per_dim.")
    end

    # Filter kwargs to only include those accepted by bstm_bspline_basis.
    # This prevents passing unsupported arguments like `structure`, `model`, etc.
    bspline_kwargs = Dict{Symbol, Any}()
    if haskey(kwargs, :custom_knots)
        bspline_kwargs[:custom_knots] = kwargs[:custom_knots]
    end

    # Generate 1D B-spline basis matrices for each dimension.
    basis_matrices_1D = Vector{Matrix{Float64}}(undef, n_dims)
    for i in 1:n_dims
        # For multi-dimensional custom knots, we need to pass the correct slice.
        local_bspline_kwargs = copy(bspline_kwargs)
        if haskey(local_bspline_kwargs, :custom_knots) && local_bspline_kwargs[:custom_knots] isa Tuple
            local_bspline_kwargs[:custom_knots] = local_bspline_kwargs[:custom_knots][i]
        end
        
        # bstm_bspline_basis returns a tuple (matrix, n_basis). We only need the matrix here.
        basis_mat, _ = bstm_bspline_basis(
            coords[:, i], 
            nbins_per_dim[i], 
            degrees_per_dim[i]; 
            knot_method=knot_method, 
            local_bspline_kwargs...
        ) # FIX: Pass custom_knots explicitly
        basis_matrices_1D[i] = basis_mat
    end

    if isempty(basis_matrices_1D)
        return zeros(size(coords, 1), 0)
    end

    # Initialize with the first basis matrix.
    B_final = basis_matrices_1D[1]

    # Iteratively compute the tensor product with the remaining matrices.
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        
        # Reshape for broadcasting to compute row-wise outer products.
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        
        tensor_prod = B_final_reshaped .* B_next_reshaped
        
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    
    return B_final
end



function _reconstruct_wavelet_function_from_filters(h::Vector{Float64}, g::Vector{Float64}, n_iterations::Int)
    # Dynamically load Interpolations to ensure it's available in the execution scope.
    Interpolations = Base.require(Base.Main, :Interpolations)

    L = length(h) 
    x_min_support = 0.0
    x_max_support = L > 1 ? L - 1.0 : 1.0

    num_points_final_grid = max(2, (2^n_iterations) * max(1, L - 1) + 1)
    x_grid_final = collect(range(x_min_support, stop=x_max_support, length=num_points_final_grid))

    phi_current_vals = zeros(length(x_grid_final))
    for i in eachindex(x_grid_final)
        if 0.0 <= x_grid_final[i] < 1.0; phi_current_vals[i] = 1.0; end
    end
    # FIX: Qualify call with the dynamically loaded module.
    phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_current_vals, extrapolation_bc=Interpolations.Flat())

    psi_next_vals = zeros(length(x_grid_final))

    for iter in 1:n_iterations
        phi_next_vals = zeros(length(x_grid_final))
        for idx in eachindex(x_grid_final)
            x_val = x_grid_final[idx]
            phi_sum = 0.0
            psi_sum = 0.0
            for k_filter in 0:(L-1)
                phi_sum += h[k_filter+1] * phi_itp(2.0 * x_val - k_filter)
                psi_sum += g[k_filter+1] * phi_itp(2.0 * x_val - k_filter)
            end
            phi_next_vals[idx] = sqrt(2.0) * phi_sum
            psi_next_vals[idx] = sqrt(2.0) * psi_sum
        end
        # FIX: Qualify call with the dynamically loaded module.
        phi_itp = Interpolations.linear_interpolation(x_grid_final, phi_next_vals, extrapolation_bc=Interpolations.Flat())
    end
    
    return x_grid_final, psi_next_vals
end



function bstm_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, lengthscale::Float64)
    # Dynamically load Interpolations to ensure it's available in the execution scope.
    Interpolations = Base.require(Base.Main, :Interpolations)

    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    v_min, v_max = minimum(vals), maximum(vals)
    v_range = v_max - v_min
    if v_range < 1e-9; v_range = 1.0; end

    local wt_type
    try
        wt_type = getfield(Wavelets.WT, family)
    catch e
        @error "Could not resolve wavelet family ':$family'. Error: $e. Defaulting to db4."
        wt_type = Wavelets.WT.db4
    end

    local wt_instance
    try
        wt_instance = Wavelets.wavelet(wt_type)
    catch e
        error("Failed to instantiate wavelet object from type '$wt_type'. Error: $e")
    end

    h_filter = wt_instance.qmf
    
    L = length(h_filter)
    g_filter = similar(h_filter)
    for i in 1:L
        g_filter[i] = (-1.0)^(i-1) * h_filter[L - (i-1)]
    end

    n_reconstruction_iterations = 8
    x_psi_grid, psi_vals = _reconstruct_wavelet_function_from_filters(h_filter, g_filter, n_reconstruction_iterations)

    # FIX: Qualify call with the dynamically loaded module.
    itp = Interpolations.linear_interpolation(x_psi_grid, psi_vals, extrapolation_bc=Interpolations.Flat())

    n_scales = max(1, floor(Int, log2(nbins/4)))
    bins_per_scale = div(nbins, n_scales)
    
    current_bin = 1
    for j in 1:n_scales
        scale_factor = lengthscale * (2.0^(j-1))
        
        n_translations = (j == n_scales) ? (nbins - current_bin + 1) : bins_per_scale
        if n_translations <= 0; continue; end

        probs = n_translations == 1 ? [0.5] : range(0, 1, length=n_translations)
        centers = quantile(vals, probs)
        
        for k in 1:n_translations
            if current_bin > nbins; break; end
            
            transformed_vals = (vals .- centers[k]) ./ (scale_factor * v_range)
            B[:, current_bin] = itp.(transformed_vals)
            current_bin += 1
        end
    end
    return B
end



# Dependent function for tensor product wavelet basis (no changes needed)
function bstm_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, family::Symbol, lengthscale::Union{Real, AbstractVector})
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims; error("Length of `nbins_per_dim` must match coordinate dimensions."); end
    
    ls_vec = if lengthscale isa Real
        fill(Float64(lengthscale), n_dims)
    else
        if length(lengthscale) != n_dims; error("Length of lengthscale vector must match coordinate dimensions."); end
        lengthscale
    end

    basis_matrices_1D = [bstm_wavelet_basis_1D(coords[:, i], nbins_per_dim[i], family, ls_vec[i]) for i in 1:n_dims]
    
    if isempty(basis_matrices_1D); return zeros(size(coords, 1), 0); end

    B_final = basis_matrices_1D[1]
    for i in 2:n_dims
        B_next = basis_matrices_1D[i]
        n_obs, n_cols_final = size(B_final)
        _, n_cols_next = size(B_next)
        B_final_reshaped = reshape(B_final, n_obs, n_cols_final, 1)
        B_next_reshaped = reshape(B_next, n_obs, 1, n_cols_next)
        tensor_prod = B_final_reshaped .* B_next_reshaped
        B_final = reshape(tensor_prod, n_obs, n_cols_final * n_cols_next)
    end
    return B_final
end


"""
    bstm_barycentric_basis_4D(coords::AbstractMatrix, knots::Vector{Point4D}, n_marginal::Int)

Generates a 4D basis matrix using multilinear interpolation on a regular grid of knots.

# Rationale
A true barycentric interpolation in 4D would require a Delaunay triangulation of the
knot points to form a mesh of 4-simplices (pentatopes). This is computationally
prohibitive and not supported by standard Julia libraries.

This function provides a practical and efficient approximation by constructing a basis
from the tensor product of 1D linear "tent" functions centered at the provided knot
points. This is equivalent to multilinear interpolation within the 4D hyper-rectangles
defined by the grid of knots.

# Arguments
- `coords`: An `N x 4` matrix of data points.
- `knots`: A vector of `Point4D` knot points, assumed to form a regular grid.
- `n_marginal`: The number of knots along each of the 4 dimensions.

# Returns
- A basis matrix of size `(N, length(knots))`.
"""
function bstm_barycentric_basis_4D(coords::AbstractMatrix, knots::Vector{Point4D}, n_marginal::Int)
    n_obs = size(coords, 1)
    n_knots = length(knots)
    B = zeros(Float64, n_obs, n_knots)

    if n_knots != n_marginal^4
        error("Number of knots must equal n_marginal^4 for 4D tensor product basis. Got n_knots=$n_knots and n_marginal^4=$(n_marginal^4).")
    end

    # Extract unique knot coordinates along each dimension and sort them
    k1 = sort(unique([k.x for k in knots]))
    k2 = sort(unique([k.y for k in knots]))
    k3 = sort(unique([k.z for k in knots]))
    k4 = sort(unique([k.t for k in knots]))

    # Calculate grid spacing (h) for each dimension
    h1 = (maximum(k1) - minimum(k1)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h1 = h1 > 0 ? h1 : 1.0
    h2 = (maximum(k2) - minimum(k2)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h2 = h2 > 0 ? h2 : 1.0
    h3 = (maximum(k3) - minimum(k3)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h3 = h3 > 0 ? h3 : 1.0
    h4 = (maximum(k4) - minimum(k4)) / (n_marginal > 1 ? (n_marginal - 1) : 1); h4 = h4 > 0 ? h4 : 1.0

    idx = 1
    # The loop order must match the order in which the knot_points vector was created.
    # The standard is x-fastest, then y, then z, etc.
    for l in 1:n_marginal
        for k in 1:n_marginal
            for j in 1:n_marginal
                for i in 1:n_marginal
                    b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i]) ./ h1)
                    b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j]) ./ h2)
                    b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k]) ./ h3)
                    b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l]) ./ h4)
                    B[:, idx] .= b1 .* b2 .* b3 .* b4
                    idx += 1
                end
            end
        end
    end
    
    return B
end


 
"""
    bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; ...)

Generates a 1D basis matrix for various smoothers. This is an updated version.

# Rationale for Update
The implementation for `pspline` and `bspline` now calls `bstm_bspline_basis` to generate
a proper B-spline basis of the specified `degree`. The previous implementation was
hardcoded to a linear spline. The original linear "tent function" behavior is retained
for the aliases `smooth`, `barycentric`, and `linear` for backward compatibility.
"""
function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{AbstractVector, Nothing} = nothing, kwargs...)
    n_obs = length(vals)
    
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]
    local knots
    if knot_method == :custom && !isnothing(custom_knots)
        knots = custom_knots
    elseif knot_method == :range || use_regular_grid
        knots = collect(range(v_min, stop=v_max, length=nbins))
    else # :quantile or any other default
        knots = quantile(vals, range(0, 1, length=nbins))
    end
    local B_out
    local actual_nbins_generated = nbins # Default to nbins, will be updated for splines
    if type in ["pspline", "bspline"]
        # bstm_bspline_basis now returns a tuple (basis_matrix, actual_nbins).
        B_out, actual_nbins_generated = bstm_bspline_basis(vals, nbins, degree; knot_method=knot_method, custom_knots=custom_knots)
        # The slicing is already done inside bstm_bspline_basis, so no further action is needed here.
    else # For other types, generate B and assume nbins is the actual count
        B_out = zeros(Float64, n_obs, nbins)
        if type in ["smooth", "barycentric", "linear"]
            # Retain the original linear tent function implementation for these aliases.
            h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1)
            h = h > 0 ? h : 1.0
            for m in 1:nbins
                dist = abs.(vals .- knots[m]) ./ h
                mask = dist .< 1.0
                B_out[mask, m] .= 1.0 .- dist[mask]
            end
        elseif type == "tps"
            # The radial basis function for 1D TPS (m=2, d=1) is r^3.
            for m in 1:nbins
                r = abs.(vals .- knots[m])
                B_out[:, m] .= r.^3
            end
        elseif type == "rff"
            m_rff = nbins
            ls = get(kwargs, :lengthscale, v_std)
            Omega = randn(1, m_rff) ./ ls
            Phi_phases = rand(m_rff) .* (2.0 * pi)
            B_out .= sqrt(2.0 / m_rff) .* cos.((vals * Omega) .+ Phi_phases')
        elseif type == "fft"
            ls = get(kwargs, :lengthscale, v_std)
            t_coords = vals ./ ls
            for m in 1:div(nbins, 2)
                B_out[:, 2m-1] .= sin.(2.0 * pi * m .* t_coords)
                B_out[:, 2m]   .= cos.(2.0 * pi * m .* t_coords)
            end
        elseif type == "wavelet"
            family = get(kwargs, :family, :db4)
            lengthscale = get(kwargs, :lengthscale, 0.1)
            B_out = bstm_wavelet_basis_1D(vals, nbins, family, lengthscale)
        elseif type == "spherical"
            range_r = get(kwargs, :range, v_std * 2.0)
            for m in 1:nbins
                h = abs.(vals .- knots[m]) ./ range_r
                mask = h .< 1.0
                B_out[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
            end
        elseif type == "decay"
            ls = get(kwargs, :lengthscale, v_std)
            for m in 1:nbins
                B_out[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
            end
        elseif type == "invdist"
            for m in 1:nbins
                dist_sq = (vals .- knots[m]).^2
                B_out[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
            end
        elseif type == "kriging"
            ls = get(kwargs, :lengthscale, v_std)
            for m in 1:nbins
                dist_sq = (vals .- knots[m]).^2
                B_out[:, m] .= exp.(-dist_sq ./ (2 * ls^2))
            end
        else
            B_out = ones(n_obs, 1)
            actual_nbins_generated = 1
        end
    end
    return B_out, actual_nbins_generated
end



"""
    bstm_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})

Generates a 2D barycentric basis matrix based on a Delaunay triangulation of knot points.

# Rationale
This provides a true triangulation-based barycentric interpolation, aligning the
implementation with the documentation's reference to "Delaunay/Voronoi" methods.
It is more flexible for irregularly spaced data than the previous grid-based
bilinear interpolation.

# Arguments
- `coords`: An `N x 2` matrix of data points.
- `knots`: A vector of `Point2D` knot points (vertices for the triangulation).

# Returns
- A sparse basis matrix of size `(N, length(knots))`.
"""
function bstm_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})
    n_obs = size(coords, 1)
    n_knots = length(knots)
    B = spzeros(Float64, n_obs, n_knots)

    # 1. Perform Delaunay triangulation on the knot points
    triangles = _delaunay_triangulation(knots)
    if isempty(triangles)
        @warn "Delaunay triangulation failed or resulted in no triangles. Returning an empty basis."
        return B
    end

    # 2. For each observation, find its enclosing triangle and barycentric coordinates
    for i in 1:n_obs
        obs_point = Point2D(coords[i, 1], coords[i, 2])
        
        for tri in triangles
            v1_idx, v2_idx, v3_idx = tri.v1, tri.v2, tri.v3
            p1, p2, p3 = knots[v1_idx], knots[v2_idx], knots[v3_idx]

            if _is_inside_triangle(obs_point, p1, p2, p3)
                bary_coords = _get_barycentric_coords(obs_point, p1, p2, p3)
                if !isnothing(bary_coords)
                    w1, w2, w3 = bary_coords
                    B[i, v1_idx] = w1
                    B[i, v2_idx] = w2
                    B[i, v3_idx] = w3
                end
                break # Found the enclosing triangle
            end
        end
    end
    return B
end



# Helper to get barycentric coordinates of a point in a triangle
function _get_barycentric_coords(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    # Using the formula based on areas
    area_total = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    if area_total < 1e-9 return nothing end

    # Area of sub-triangles
    area1 = abs((p2.x - p.x) * (p3.y - p.y) - (p3.x - p.x) * (p2.y - p.y)) # for p1
    area2 = abs((p3.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p3.y - p.y)) # for p2
    area3 = abs((p1.x - p.x) * (p2.y - p.y) - (p2.x - p.x) * (p1.y - p.y)) # for p3

    w1 = area1 / area_total
    w2 = area2 / area_total
    w3 = area3 / area_total
    
    # Normalize to ensure sum is 1, accounting for float precision
    w_sum = w1 + w2 + w3
    return (w1/w_sum, w2/w_sum, w3/w_sum)
end



# Helper function to calculate the circumcenter and squared radius of a triangle
function _get_circumcircle(p1::Point2D, p2::Point2D, p3::Point2D)
    D = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(D) < 1e-9
        return nothing, nothing # Collinear points
    end

    p1_sq = p1.x^2 + p1.y^2
    p2_sq = p2.x^2 + p2.y^2
    p3_sq = p3.x^2 + p3.y^2

    center_x = (p1_sq * (p2.y - p3.y) + p2_sq * (p3.y - p1.y) + p3_sq * (p1.y - p2.y)) / D
    center_y = (p1_sq * (p3.x - p2.x) + p2_sq * (p1.x - p3.x) + p3_sq * (p2.x - p1.x)) / D
    
    center = Point2D(center_x, center_y)
    radius_sq = (p1.x - center.x)^2 + (p1.y - center.y)^2
    
    return center, radius_sq
end

# Helper to check if a point is inside the circumcircle of a triangle
function _is_in_circumcircle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    center, radius_sq = _get_circumcircle(p1, p2, p3)
    if isnothing(center)
        return false # Cannot be in the circumcircle of collinear points
    end
    dist_sq = (p.x - center.x)^2 + (p.y - center.y)^2
    return dist_sq < radius_sq
end


# Bowyer-Watson algorithm for Delaunay triangulation
function _delaunay_triangulation(points::Vector{Point2D})
    n = length(points)
    if n < 3
        return []
    end

    # Determine a "super-triangle" that encloses all points
    min_x = minimum(p.x for p in points)
    max_x = maximum(p.x for p in points)
    min_y = minimum(p.y for p in points)
    max_y = maximum(p.y for p in points)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # Define vertices of the super-triangle
    p_super1 = Point2D(mid_x - 20 * delta_max, mid_y - delta_max)
    p_super2 = Point2D(mid_x + 20 * delta_max, mid_y - delta_max)
    p_super3 = Point2D(mid_x, mid_y + 20 * delta_max)
    
    # The indices of the super-triangle vertices will be n+1, n+2, n+3
    super_triangle = Triangle(n + 1, n + 2, n + 3)
    all_points = [points; p_super1; p_super2; p_super3]

    triangulation = [super_triangle]

    for i in 1:n
        point = points[i]
        bad_triangles = []
        
        for tri in triangulation
            p1 = all_points[tri.v1]
            p2 = all_points[tri.v2]
            p3 = all_points[tri.v3]
            if _is_in_circumcircle(point, p1, p2, p3)
                push!(bad_triangles, tri)
            end
        end

        polygon = []
        for tri in bad_triangles
            edges = [(tri.v1, tri.v2), (tri.v2, tri.v3), (tri.v3, tri.v1)]
            for edge in edges
                is_shared = false
                for other_tri in bad_triangles
                    if tri === other_tri continue end
                    other_edges = [(other_tri.v1, other_tri.v2), (other_tri.v2, other_tri.v3), (other_tri.v3, other_tri.v1)]
                    if (edge in other_edges) || ((edge[2], edge[1]) in other_edges)
                        is_shared = true
                        break
                    end
                end
                if !is_shared
                    push!(polygon, edge)
                end
            end
        end

        # Remove bad triangles from triangulation
        filter!(t -> !(t in bad_triangles), triangulation)

        # Form new triangles from the polygon edges to the new point
        for edge in polygon
            push!(triangulation, Triangle(edge[1], edge[2], i))
        end
    end

    # Remove triangles that include vertices of the super-triangle
    filter!(t -> !(t.v1 > n || t.v2 > n || t.v3 > n), triangulation)

    return triangulation
end

# Helper to check if a point is inside a triangle
function _is_inside_triangle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    # Using barycentric coordinates. A point is inside if all coordinates are non-negative.
    coords = _get_barycentric_coords(p, p1, p2, p3)
    return !isnothing(coords) && all(c -> c >= -1e-9, coords) # Allow for small float inaccuracies
end


"""
    bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 2D basis matrix for various smoothers.

# Rationale for Update
This version unifies the `barycentric` smoother with the `smooth` and `linear` types.
The previous implementation for `barycentric` used a complex and likely incorrect
Delaunay triangulation on a regular grid. The corrected version now uses the robust
and standard tensor product of 1D linear "tent" functions (bilinear interpolation),
which is the correct interpretation for a barycentric-style basis on a grid.
"""
function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)
    
    local n_marginal_x, n_marginal_y
    if nbins isa Int
        n_marginal_x = nbins
        n_marginal_y = nbins
    elseif nbins isa Vector{Int} && length(nbins) == 2
        n_marginal_x = nbins[1]
        n_marginal_y = nbins[2]
    else
        error("For a 2D smooth, `nbins` must be an Int or a Vector{Int} of length 2.")
    end
    total_bins = n_marginal_x * n_marginal_y

    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    local kx, ky
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        kx, ky = custom_knots
    elseif knot_method == :quantile && !use_regular_grid
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal_x))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal_y))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal_x))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal_y))
    end
    
    B = zeros(Float64, n_obs, total_bins)

    if type in ["pspline", "bspline"]
        degree_val = get(kwargs, :degree, 3)
        nbins_per_dim = [n_marginal_x, n_marginal_y]
        degrees_per_dim = [degree_val, degree_val]
        return bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method, kwargs...)
    end

    if type in ["smooth", "barycentric", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1); hy = hy > 0 ? hy : 1.0
        idx = 1
        for j in 1:n_marginal_y, i in 1:n_marginal_x
            if idx > total_bins; break; end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            B[:, idx] .= b_x .* b_y
            idx += 1
        end
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal_x, n_marginal_y]
        return bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
    end

    if type == "tps"
        centers = [(x, y) for y in ky for x in kx][:]
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            r = sqrt.(dx.^2 .+ dy.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-9)
        end
        return B
    end

    if type == "rff" || type == "anisotropic"
        Omega = randn(2, total_bins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B = sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    elseif type == "fft"
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y
        idx = 1
        for my in 1:n_marginal_y, mx in 1:n_marginal_x
            if idx + 1 <= total_bins
                arg = mx .* nx .+ my .* ny
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end
    end

    if type == "spherical"
        centers = [(x, y) for x in kx, y in ky][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            h = sqrt.(dx.^2 .+ dy.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
    elseif  type == "invdist"
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:total_bins
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end
    end

    if  type == "kriging"
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:total_bins
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end

"""
    bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; ...)

Generates a 3D basis matrix for various smoothers.

# Rationale for Update
This version fixes a critical bug where the `barycentric` smoother attempted to call
a non-existent function `bstm_barycentric_basis_3D`. The logic has been corrected
to unify `barycentric` with the `smooth` and `linear` types, which correctly implement
a tensor product of 1D linear basis functions (trilinear interpolation).
"""
function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)

    local n_marginal_x, n_marginal_y, n_marginal_z
    if nbins isa Int
        n_marginal_x = nbins
        n_marginal_y = nbins
        n_marginal_z = nbins
    elseif nbins isa Vector{Int} && length(nbins) == 3
        n_marginal_x = nbins[1]
        n_marginal_y = nbins[2]
        n_marginal_z = nbins[3]
    else
        error("For a 3D smooth, `nbins` must be an Int or a Vector{Int} of length 3.")
    end
    total_bins = n_marginal_x * n_marginal_y * n_marginal_z

    c_min = [minimum(coords[:, i]) for i in 1:3]
    c_max = [maximum(coords[:, i]) for i in 1:3]
    c_std = [std(coords[:, i]) for i in 1:3] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])

    local kx, ky, kz
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 3 || length(custom_knots[1]) != n_marginal_x || length(custom_knots[2]) != n_marginal_y || length(custom_knots[3]) != n_marginal_z
            @warn "Custom knots for 3D smoother must be a Tuple of 3 AbstractVectors with lengths matching `nbins`. Falling back to :quantile method."
            kx = quantile(coords[:, 1], range(0, 1, length=n_marginal_x))
            ky = quantile(coords[:, 2], range(0, 1, length=n_marginal_y))
            kz = quantile(coords[:, 3], range(0, 1, length=n_marginal_z))
        else
            kx, ky, kz = custom_knots
        end
    elseif knot_method == :quantile && !use_regular_grid
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal_x))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal_y))
        kz = quantile(coords[:, 3], range(0, 1, length=n_marginal_z))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal_x))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal_y))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal_z))
    end

    B = zeros(Float64, n_obs, total_bins)

    if type in ["pspline", "bspline"]
        nbins_per_dim = [n_marginal_x, n_marginal_y, n_marginal_z]
        degree_val = get(kwargs, :degree, 3)
        degrees_per_dim = [degree_val, degree_val, degree_val]
        full_B = bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal_x, n_marginal_y, n_marginal_z]
        full_B = bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    end

    if type in ["smooth", "barycentric", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal_x > 1 ? (n_marginal_x - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal_y > 1 ? (n_marginal_y - 1) : 1); hy = hy > 0 ? hy : 1.0
        hz = (c_max[3] - c_min[3]) / (n_marginal_z > 1 ? (n_marginal_z - 1) : 1); hz = hz > 0 ? hz : 1.0

        idx = 1
        for k_idx in 1:n_marginal_z, j_idx in 1:n_marginal_y, i_idx in 1:n_marginal_x
            if idx > total_bins; break; end
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i_idx]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j_idx]) ./ hy)
            b_z = max.(0.0, 1.0 .- abs.(coords[:, 3] .- kz[k_idx]) ./ hz)
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end
    elseif type == "tps"
        centers = [(x, y, z) for z in kz for y in ky for x in kx][:]
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            dz = coords[:, 3] .- centers[m][3]
            r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            B[:, m] .= r
        end
    elseif type == "rff"
        Omega = randn(3, total_bins)
        Omega[1, :] ./= ls_x; Omega[2, :] ./= ls_y; Omega[3, :] ./= ls_z
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')
    elseif type == "fft"
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y
        nz = coords[:, 3] ./ ls_z
        idx = 1
        for mz in 1:n_marginal_z, my in 1:n_marginal_y, mx in 1:n_marginal_x
            if idx + 1 <= total_bins
                arg = mx .* nx .+ my .* ny .+ mz .* nz
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end
    elseif type == "spherical"
        centers = [(x, y, z) for z in kz for y in ky for x in kx][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:total_bins
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            dz = coords[:, 3] .- centers[m][3]
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
    elseif type == "invdist"
        centers = [(x, y, z) for z in kz for y in ky for x in kx][:]
        for m in 1:total_bins
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end
    elseif type == "kriging"
        centers = [(x,y,z) for z in kz for y in ky for x in kx][:]
        for m in 1:total_bins
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_z^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end
    else
        B = ones(n_obs, total_bins)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end


function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Union{Int, Vector{Int}}; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)

    local n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4
    if nbins isa Int
        n_marginal_1 = nbins
        n_marginal_2 = nbins
        n_marginal_3 = nbins
        n_marginal_4 = nbins
    elseif nbins isa Vector{Int} && length(nbins) == 4
        n_marginal_1 = nbins[1]
        n_marginal_2 = nbins[2]
        n_marginal_3 = nbins[3]
        n_marginal_4 = nbins[4]
    else
        error("For a 4D smooth, `nbins` must be an Int or a Vector{Int} of length 4.")
    end
    total_bins = n_marginal_1 * n_marginal_2 * n_marginal_3 * n_marginal_4

    c_min = [minimum(coords[:, i]) for i in 1:4]
    c_max = [maximum(coords[:, i]) for i in 1:4]
    c_std = [std(coords[:, i]) for i in 1:4] .+ 1e-9

    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])

    local k1, k2, k3, k4

    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 4 || length(custom_knots[1]) != n_marginal_1 || length(custom_knots[2]) != n_marginal_2 || length(custom_knots[3]) != n_marginal_3 || length(custom_knots[4]) != n_marginal_4
            @warn "Custom knots for 4D smoother must be a Tuple of 4 AbstractVectors with lengths matching `nbins`. Falling back to :quantile method."
            k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal_1))
            k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal_2))
            k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal_3))
            k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal_4))
        else
            k1 = custom_knots[1]
            k2 = custom_knots[2]
            k3 = custom_knots[3]
            k4 = custom_knots[4]
        end
    elseif knot_method == :quantile && !use_regular_grid
        k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal_1)); k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal_2)); k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal_3)); k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal_4))
    else # :range or if regular grid is required
        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal_1)); k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal_2)); k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal_3)); k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal_4))
    end

    B = zeros(Float64, n_obs, total_bins)

    if type in ["pspline", "bspline"]
        nbins_per_dim = [n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4]
        degree_val = get(kwargs, :degree, 3)
        degrees_per_dim = fill(degree_val, 4)
        full_B = bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal_1, n_marginal_2, n_marginal_3, n_marginal_4]
        full_B = bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    end

    if type == "barycentric"
        knot_points = [Point4D(x, y, z, t) for t in k4 for z in k3 for y in k2 for x in k1]
        full_B = bstm_barycentric_basis_4D(coords, knot_points, n_marginal_1) # Assumes cubic grid
        return full_B[:, 1:min(total_bins, size(full_B, 2))]
    end

    if type in [ "smooth", "linear"]
        hx1 = (c_max[1] - c_min[1]) / (n_marginal_1 > 1 ? (n_marginal_1 - 1) : 1); hx1 = hx1 > 0 ? hx1 : 1.0
        hx2 = (c_max[2] - c_min[2]) / (n_marginal_2 > 1 ? (n_marginal_2 - 1) : 1); hx2 = hx2 > 0 ? hx2 : 1.0
        hx3 = (c_max[3] - c_min[3]) / (n_marginal_3 > 1 ? (n_marginal_3 - 1) : 1); hx3 = hx3 > 0 ? hx3 : 1.0
        hx4 = (c_max[4] - c_min[4]) / (n_marginal_4 > 1 ? (n_marginal_4 - 1) : 1); hx4 = hx4 > 0 ? hx4 : 1.0

        idx = 1
        for l_idx in 1:n_marginal_4, k_idx in 1:n_marginal_3, j_idx in 1:n_marginal_2, i_idx in 1:n_marginal_1
            if idx > total_bins; break; end
            b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i_idx]) ./ hx1)
            b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j_idx]) ./ hx2)
            b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k_idx]) ./ hx3)
            b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l_idx]) ./ hx4)
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end

    elseif type == "tps"
        centers = [(k1_i, k2_i, k3_i, k4_i) for k4_i in k4 for k3_i in k3 for k2_i in k2 for k1_i in k1][:]

        for m in 1:total_bins
            d1 = coords[:, 1] .- centers[m][1]
            d2 = coords[:, 2] .- centers[m][2]
            d3 = coords[:, 3] .- centers[m][3]
            d4 = coords[:, 4] .- centers[m][4]
            r = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            B[:, m] .= log.(r .+ 1e-9)
        end

    elseif type == "rff"
        Omega = randn(4, total_bins)
        Omega[1, :] ./= ls_1; Omega[2, :] ./= ls_2; Omega[3, :] ./= ls_3; Omega[4, :] ./= ls_4
        Phi_phases = rand(total_bins) .* (2.0 * pi)
        B .= sqrt(2.0 / total_bins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        nx1 = coords[:, 1] ./ ls_1
        nx2 = coords[:, 2] ./ ls_2
        nx3 = coords[:, 3] ./ ls_3
        nx4 = coords[:, 4] ./ ls_4
        idx = 1
        for m4 in 1:n_marginal_4, m3 in 1:n_marginal_3, m2 in 1:n_marginal_2, m1 in 1:n_marginal_1
            if idx + 1 <= total_bins
                arg = m1 .* nx1 .+ m2 .* nx2 .+ m3 .* nx3 .+ m4 .* nx4
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end
 
    elseif type == "spherical"
        centers = [(k1_i, k2_i, k3_i, k4_i) for k4_i in k4 for k3_i in k3 for k2_i in k2 for k1_i in k1][:]
        range_r = get(kwargs, :range, mean(c_std))

        for m in 1:total_bins
            d1 = coords[:, 1] .- centers[m][1]
            d2 = coords[:, 2] .- centers[m][2]
            d3 = coords[:, 3] .- centers[m][3]
            d4 = coords[:, 4] .- centers[m][4]
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
 
    elseif type == "invdist"
        centers = [(w,x,y,z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:total_bins
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2 .+ (coords[:, 4] .- centers[m][4]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        centers = [(w,x,y,z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:total_bins
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_1^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_2^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_3^2) .+ ((coords[:, 4] .- centers[m][4]).^2 ./ ls_4^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, total_bins)
    end

    return B[:, 1:min(total_bins, size(B, 2))]
end


"""
    evaluate_kernel_matrix(coords, param_val, ls, kernel_type, noise; ...)

Computes the covariance kernel matrix for a given set of coordinates.

# Rationale for Update
This version corrects a `MethodError` that occurred when adding the `noise` term
(nugget) to the kernel matrix. The original implementation used broadcast addition (`.+`),
which is incompatible with the `UniformScaling` type of `LinearAlgebra.I`. The fix
replaces `.+ (noise * I)` with standard matrix addition `+ (noise * I)`, which correctly
materializes the identity matrix and adds the noise to the diagonal, resolving the error.
"""
function evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords, 2) != length(ls)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords, 2))) does not match number of lengthscales ($(length(ls))).")
        end
        # Calculate weighted squared Euclidean distance
        dist_sq = pairwise(SqEuclidean(), coords ./ ls', dims=1)
    else # Isotropic case
        dist_sq = pairwise(SqEuclidean(), coords, dims=1) ./ ls^2
    end

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return (param_val^2) .* exp.(-0.5 .* dist_sq) + (noise * I)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12 
        d = sqrt.(dist_sq) # distance is now scaled (already done by pairwise)
        return (param_val^2) .* exp.(-d) + (noise * I)
    
    # Matern 3/2
    elseif kernel_type == :matern32 
        d = sqrt.(dist_sq) # distance is now scaled (already done by pairwise)
        val = sqrt(3.0) .* d
        return (param_val^2) .* (1.0 .+ val) .* exp.(-val) + (noise * I)
    
    # Matern 5/2
    elseif kernel_type == :matern52 
        d = sqrt.(dist_sq) # distance is now scaled (already done by pairwise)
        val = sqrt(5.0) .* d
        return (param_val^2) .* (1.0 .+ val .+ (val.^2 ./ 3.0)) .* exp.(-val) + (noise * I)

    # Constant Kernel (Identity innovation)
    elseif kernel_type == :constant
        return fill(param_val^2, size(dist_sq))

    # Linear Kernel
    elseif kernel_type == :linear
        return (param_val^2) .* dist_sq

    # Wavelet Multiscale Kernel
    elseif kernel_type == :wavelet
        K_accum = zeros(eltype(dist_sq), size(dist_sq))
        for wv_scale in 1:wavelet_levels
            ls_scale_sq = (ls isa Real ? ls^2 : 1.0) / (4^(wv_scale-1))
            weight_scale = (param_val^2) * exp(-wv_scale / ls)
            K_accum .+= weight_scale .* exp.(-0.5 .* dist_sq ./ ls_scale_sq) # Element-wise addition
        end
        return K_accum + (noise * I) # FIX: Changed .+ to +

    # Fallback Dispatch
    else
        return (param_val^2) .* exp.(-0.5 .* dist_sq) + (noise * I)
    end
end


function recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; extra_param=nothing, noise=1e-4, kwargs...)
    n_s = size(template_s, 1)

    if m_type == :SPDE
        kappa = isnothing(extra_param) ? 1.0 : extra_param
        local Q_kappa
        if kappa isa Real
            Q_kappa = kappa^2 * I(n_s)
        else
            if length(kappa) != n_s; error("Anisotropic kappa vector length must match number of spatial units."); end
            Q_kappa = Diagonal(kappa.^2)
        end
        L_spde = Q_kappa + template_s
        return Symmetric(L_spde' * L_spde)
    end

    if m_type == :NoneComponent || m_type == :FIXED
        return Symmetric(sparse(I(n_s)))
    end

    if m_type == :Besag || m_type == :ICAR || m_type == :Cyclic
        return Symmetric(template_s)
    end

    if m_type == :AR1
        rho = isnothing(extra_param) ? 0.0 : extra_param
        if abs(rho) >= 1.0; rho = sign(rho) * 0.9999; end
        
        Q = spdiagm(0 => fill(1.0 + rho^2, n_s))
        Q[1, 1] = 1.0
        Q[n_s, n_s] = 1.0
        
        # Add off-diagonal elements using the pre-built adjacency template
        Q .+= rho .* template_s
        return Symmetric(Q)
    end

    if m_type == :RW1 || m_type == :RW2
        # This error is correct for standalone models, but the context is now wider.
        # The state-space form is preferred, but precision is needed for Kronecker products.
        # For now, we assume this function is not called for RW models in interactions.
        error("recompose_precision should not be called for $(m_type) models. Use the state-space implementation.")
    end

    if m_type == :Leroux || m_type == :LocalAdaptive
        lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(lambda_val .* template_s + (1.0 - lambda_val) .* sparse(I(n_s)))
    end

    if m_type == :ST_I || m_type == :ST_II || m_type == :ST_III || m_type == :ST_IV
        # This branch is likely incorrect as template_t is not defined.
        # It should be handled by the _generate_st_interaction_block.
        error("recompose_precision should not be called for ST_I/II/III/IV models directly.")
    end

    if m_type == :NetworkFlow
        rho_net = isnothing(extra_param) ? 0.8 : extra_param
        W_net = template_s
        flow_direction = get(kwargs, :flow_direction, :bidirectional)
        
        L_op = if flow_direction == :upstream
            I(n_s) - rho_net .* W_net'
        elseif flow_direction == :downstream
            I(n_s) - rho_net .* W_net
        else # :bidirectional or default
            W_symm = sparse((W_net + W_net') .> 0)
            I(n_s) - rho_net .* W_symm
        end

        return Symmetric(L_op' * L_op)
    end

    if m_type == :SAR || m_type == :DAG
        rho_p = isnothing(extra_param) ? 0.8 : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(L_op' * L_op)
    end

    if m_type == :GP
        ls = isnothing(extra_param) ? 1.0 : extra_param
        K = (param_val^2) .* exp.(-(Matrix(template_s).^2) ./ (2 * ls^2 + noise))
        return inv(Symmetric(K))
    end

    if m_type == :RFF || m_type == :FFT || m_type == :BSpline || m_type == :PSpline || m_type == :TPS
        return Symmetric(template_s)
    end

    return Symmetric(template_s)
end
 
     
function _distribution_to_string(d::Distribution)
    # Purpose: Converts a Distribution object into a string that represents a valid constructor call.
    # Rationale: This version is updated to use accessor functions (e.g., `shape`, `scale`, `mean`) instead of
    #            direct field access (e.g., `d.α`). This makes the function more robust to
    #            internal changes in the `Distributions.jl` package. It also adds explicit handling
    #            for `LogNormal` to ensure positional arguments are used, resolving a `MethodError`.
    # v1.0.1 (2026-07-31)
    # Inputs:
    #   - d: The Distribution object.
    # Outputs: A string representing the constructor call.
    T = eltype(d)
    dist_name = string(typeof(d).name.name)
    
    if d isa Exponential
        # Constructor is Exponential(rate)
        return "$(dist_name){$T}($(rate(d)))"
    elseif d isa Normal
        return "$(dist_name){$T}($(mean(d)), $(std(d)))"
    elseif d isa LogNormal
        # FIX: Explicitly use positional arguments to avoid MethodError with some Distributions.jl versions.
        # params(d) returns (μ, σ) for the log-scale parameters.
        return "$(dist_name){$T}($(params(d)[1]), $(params(d)[2]))"
    elseif d isa Beta
        # params(d) returns (α, β)
        return "$(dist_name){$T}($(params(d)[1]), $(params(d)[2]))"
    elseif d isa InverseGamma
        return "$(dist_name){$T}($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Gamma
        return "$(dist_name){$T}($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Uniform
        return "$(dist_name){$T}($(minimum(d)), $(maximum(d)))"
    else
        # Fallback for less common distributions. This might fail if their
        # string representation uses keywords, but covers many cases.
        return string(d)
    end
end

  

function _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    needs_sigma = any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t", "dirichlet_multinomial"], families)

    zi_prior_block = ""
    hurdle_prior_block = ""
    if get(M, :user_provided_hurdle, false)
        hurdle_prior_block = "lik_phi_hurdle ~ NamedDist(Beta(1,1), :lik_phi_hurdle)"
    elseif get(M, :use_zi, false)
        zi_prior_block = "lik_phi_zi ~ NamedDist(Beta(1,1), :lik_phi_zi)"
    end

    nu_student_t_block = ""
    if any(f -> string(f) == "student_t", families)
        nu_student_t_block = "lik_nu_student_t ~ NamedDist(Exponential(1.0), :lik_nu_student_t)"
    end

    sigma_block = ""
    if needs_sigma
        sigma_y_prior_str = _distribution_to_string(Exponential(1.0))
        if is_multivariate
            sigma_block = "sigma_y ~ NamedDist(filldist($(sigma_y_prior_str), K), :sigma_y)"
        else
            sigma_block = "sigma_y ~ NamedDist($(sigma_y_prior_str), :sigma_y)"
        end
    end
    
    extra_params_block = ""
    if any(f -> string(f) in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
        extra_params_block = "lik_extra_params ~ NamedDist(Exponential(1.0), :lik_extra_params)"
    end

    corr_block = is_multivariate ? "L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)" : ""

    ordinal_priors_block = ""
    ordinal_spec_idx = findfirst(s -> string(get(s, :family, "")) == "ordinal", M.likelihood_specs)
    if !isnothing(ordinal_spec_idx)
        K = M.likelihood_specs[ordinal_spec_idx][:K]
        latent_dist = M.likelihood_specs[ordinal_spec_idx][:latent_dist]

        if K > 1
            for j in 1:(K-1)
                # FIX: Use `\n` instead of `\\n` for correct line breaks
                ordinal_priors_block *= "ordinal_alpha_$(j) ~ NamedDist(Dirac(T(0.0)), :ordinal_alpha_$(j))\n"
            end
            if K > 2
                ordinal_priors_block *= """
                ordinal_alpha_raw_1 ~ NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)
                ordinal_alpha_diffs ~ NamedDist(filldist(Exponential(1.0), $(K - 2)), :ordinal_alpha_diffs)
                """
            elseif K == 2
                # FIX: Use `\n` instead of `\\n`
                ordinal_priors_block *= "ordinal_alpha_raw_1 ~ NamedDist(Normal(0, 5), :ordinal_alpha_raw_1)\n"
            end

            if latent_dist == :student_t
                # FIX: Use `\n` instead of `\\n`
                ordinal_priors_block *= "ordinal_df ~ NamedDist(Exponential(1.0), :ordinal_df)\n"
            end
        end
    end

    return """
    $(corr_block)
    $(sigma_block)
    $(hurdle_prior_block)
    $(zi_prior_block)
    $(nu_student_t_block)
    $(extra_params_block)
    $(ordinal_priors_block)
    """
end



function _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)
    # v1.0.3 (2026-07-31)
    # Rationale: This version is updated to source the list of non-proportional effects
    #            from the main model configuration `M[:non_proportional_effects]`, aligning
    #            it with the new `fixed(..., non_proportional_effects=true)` syntax. This
    #            decouples the likelihood generator from the `likelihood()` module's parameters.
    if is_multivariate
        return _generate_multivariate_likelihood_block(M)
    end

    family = string(M.likelihood_specs[1][:family])

    if family == "ordinal"
        K = M.likelihood_specs[1][:K]
        latent_dist_val = M.likelihood_specs[1][:latent_dist]
        if K < 2; return ""; end

        non_prop_terms = get(M, :non_proportional_effects, Symbol[])
        is_npo = !isempty(non_prop_terms)
        
        npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
        n_npo_vars = length(npo_indices)

        assignment_lines = ""
        for j in 1:(K-1)
            assignment_lines *= "ordinal_alpha_$(j) = alphas_computed[$(j)]\n"
        end

        npo_update_block = ""
        if is_npo && n_npo_vars > 0
            npo_update_block = """
            # Non-proportional effects calculation
            local X_npo = M.Xfixed[:, $(npo_indices)]
            local beta_npo_matrix = reshape(beta_npo, $(n_npo_vars), $(K-1))
            local eta_npo = X_npo * beta_npo_matrix
            """
        end

        return """
        # Proportional Odds Likelihood
        let
            local alphas_computed
            if $(K > 2)
                alphas_computed = cumsum([ordinal_alpha_raw_1; ordinal_alpha_diffs])
            else
                alphas_computed = [ordinal_alpha_raw_1]
            end

            $(assignment_lines)
            local latent_dist_symbol = :$(latent_dist_val)
            $(npo_update_block)

            for i in 1:N
                linear_predictor_prop = eta[i]
                
                local linear_predictor_vec
                if $(is_npo && n_npo_vars > 0)
                    # Combine proportional and non-proportional parts for each cut-point
                    linear_predictor_vec = linear_predictor_prop .+ view(eta_npo, i, :)
                else
                    # If fully proportional, broadcast the single predictor
                    linear_predictor_vec = fill(linear_predictor_prop, $(K-1))
                end
                
                local cumulative_probs
                if latent_dist_symbol == :normal
                    cumulative_probs = Distributions.cdf.(Normal(), alphas_computed .- linear_predictor_vec)
                elseif latent_dist_symbol == :logistic
                    cumulative_probs = logistic.(alphas_computed .- linear_predictor_vec)
                elseif latent_dist_symbol == :student_t
                    cumulative_probs = Distributions.cdf.(TDist(ordinal_df), alphas_computed .- linear_predictor_vec)
                else
                    error("Unsupported latent distribution ':\$(latent_dist_symbol)' for ordinal model.")
                end
                
                probs = Vector{T}(undef, $(K))
                if $(K > 1)
                    probs[1] = cumulative_probs[1]
                    for j in 2:($(K-1))
                        probs[j] = max(0.0, cumulative_probs[j] - cumulative_probs[j-1])
                    end
                    probs[$(K)] = max(0.0, 1.0 - cumulative_probs[$(K-1)])
                else
                    probs[1] = 1.0
                end

                probs ./= (sum(probs) + 1e-9)
                Turing.@addlogprob! logpdf(Categorical(probs), M.y_obs[i])
            end
        end
        """
    else
        return _generate_univariate_likelihood_block(M)
    end
end





function _generate_multivariate_likelihood_block(M::NamedTuple)
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    any_needs_sigma = any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)
    any_needs_nu = any(f -> f == "student_t", families)
    any_needs_extra = any(f -> f in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
    is_multinomial = any(f -> f == "dirichlet_multinomial", families)

    if is_multinomial
        return """
        eta = eta_latent * L_corr.L
        family = M.likelihood_specs[1][:family]
        for i in 1:N
            d_lik = bstm_Likelihood(family, M.y_obs[i, :]; trial=sum(M.y_obs[i, :]))
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, :])
        end
        """
    end

    kwargs_parts = String[]
    if any_needs_sigma; push!(kwargs_parts, "sigma_y=sigma_y[k]"); end
    if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, k])"); end
    if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[i, k]"); end
    if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=M.censor_lower[i, k]"); end
    if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=M.censor_upper[i, k]"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=M.hurdle[i, k]"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle");
    elseif get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end

    extra_param_logic = if any_needs_nu && any_needs_extra
        "local extra_p = family_k == \"student_t\" ? lik_nu_student_t : lik_extra_params"
    elseif any_needs_nu
        "local extra_p = lik_nu_student_t"
    elseif any_needs_extra
        "local extra_p = lik_extra_params"
    else "" end
    if !isempty(extra_param_logic); push!(kwargs_parts, "extra_params=extra_p"); end

    kwargs_str = join(kwargs_parts, ", ")

    return """
    eta = eta_latent * L_corr.L
    for k in 1:K
        family_k = M.likelihood_specs[k][:family]
        $(extra_param_logic)
        for i in 1:N
            d_lik = bstm_Likelihood(family_k, T(M.y_obs[i, k]); $(kwargs_str))
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
        end
    end
    """
end





function _generate_univariate_likelihood_block(M::NamedTuple)
    family = string(M.likelihood_specs[1][:family])
    any_needs_sigma = family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
    any_needs_nu = family == "student_t"
    any_needs_extra = family in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"]

    kwargs_parts = String[]
    if any_needs_sigma; push!(kwargs_parts, "sigma_y=sigma_y"); end
    if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, 1])"); end
    if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[i, 1]"); end
    if get(M, :user_provided_censor_lower, false); push!(kwargs_parts, "censor_lower=M.censor_lower[i, 1]"); end
    if get(M, :user_provided_censor_upper, false); push!(kwargs_parts, "censor_upper=M.censor_upper[i, 1]"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=M.hurdle[i, 1]"); end
    if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle");
    elseif get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end

    extra_param_logic = if any_needs_nu && any_needs_extra
        "local extra_p = family == \"student_t\" ? lik_nu_student_t : lik_extra_params"
    elseif any_needs_nu
        "local extra_p = lik_nu_student_t"
    elseif any_needs_extra
        "local extra_p = lik_extra_params"
    else "" end
    if !isempty(extra_param_logic); push!(kwargs_parts, "extra_params=extra_p"); end

    kwargs_str = join(kwargs_parts, ", ")

    return """
    family = M.likelihood_specs[1][:family]
    $(extra_param_logic)
    for i in 1:N
        d_lik = bstm_Likelihood(family, T(M.y_obs[i]); $(kwargs_str))
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
    """
end

 

function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.0.0 (2026-07-18)
    # Rationale: Refactored to return separate prior and update code blocks.
    if !get(M, :add_intercept, false) return "", "" end
    
    intercept_prior_obj = get(M, :intercept_prior, Normal(0,5))
    local dist_str, update_code, prior_code
    if is_multivariate
        dist_str = "filldist($(_distribution_to_string(intercept_prior_obj)), K)"
        update_code = "for k in 1:K; $(eta_name)[:, k] .+= intercept[k]; end"
    else
        dist_str = _distribution_to_string(intercept_prior_obj)
        update_code = "$(eta_name) .+= intercept"
    end
    
    prior_code = "intercept ~ NamedDist($(dist_str), :intercept)"
    return prior_code, update_code
end



function _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.0.0 (2026-07-16)
    # Corrected in this review to consistently use M[:log_offsets] which is an N x K matrix.
    if !haskey(M, :log_offsets) || all(iszero, M[:log_offsets])
        return ""
    end
    
    if is_multivariate
        # For multivariate models, add the entire N x K matrix to the N x K eta_latent.
        return "$(eta_name) .+= M.log_offsets"
    else
        # For univariate models, add the first column of the N x 1 matrix to the N-element eta.
        return "$(eta_name) .+= M.log_offsets[:, 1]"
    end
end


function _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.0.2 (2026-07-31)
    # Rationale: This version is updated to source the list of non-proportional effects
    #            from the main model configuration `M[:non_proportional_effects]`, which is
    #            populated by the new logic in `bstm_config`. This decouples the code
    #            generator from the `likelihood()` module's parameters and aligns it with
    #            the new `fixed(..., non_proportional_effects=true)` syntax.
    if get(M, :Xfixed_N, 0) == 0; return "", ""; end

    priors_vec = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
    
    # --- Non-Proportional Odds (NPO) Logic for Ordinal Models ---
    is_ordinal = any(spec -> string(get(spec, :family, "")) == "ordinal", M.likelihood_specs)
    non_prop_terms = is_ordinal ? get(M, :non_proportional_effects, Symbol[]) : Symbol[]
    
    prop_indices = collect(1:M.Xfixed_N)
    npo_indices = Int[]

    if is_ordinal && !isempty(non_prop_terms)
        npo_indices = findall(x -> x in non_prop_terms, M.Xfixed_names)
        prop_indices = setdiff(prop_indices, npo_indices)
    end

    n_prop = length(prop_indices)
    n_npo = length(npo_indices)
    K_ordinal = is_ordinal ? get(M.likelihood_specs[1], :K, 0) : 0

    prior_parts = String[]
    update_parts = String[]

    # --- Priors and Updates for Proportional Effects ---
    if n_prop > 0
        priors_prop = priors_vec[prop_indices]
        all_same_prop = !isempty(priors_prop) && all(p -> p == priors_prop[1], priors_prop)
        
        if is_multivariate
            beta_prop_name = "Xfixed_beta_prop_flat"
            push!(update_parts, "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * reshape($(beta_prop_name), $(n_prop), M.outcomes_N)")
            
            if all_same_prop
                prior_str = _distribution_to_string(priors_prop[1])
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(filldist($(prior_str), $(n_prop * M.outcomes_N)), :Xfixed_beta_prop)")
            else
                full_priors_list = vcat([priors_prop for _ in 1:M.outcomes_N]...)
                priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :Xfixed_beta_prop)")
            end
        else # Univariate
            beta_prop_name = "Xfixed_beta_prop"
            push!(update_parts, "$(eta_name) .+= M.Xfixed[:, $(prop_indices)] * $(beta_prop_name)")
            
            if all_same_prop
                prior_str = _distribution_to_string(priors_prop[1])
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(filldist($(prior_str), $(n_prop)), :Xfixed_beta_prop)")
            else
                priors_str_list = [_distribution_to_string(p) for p in priors_prop]
                push!(prior_parts, "$(beta_prop_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :Xfixed_beta_prop)")
            end
        end
    end

    # --- Priors for Non-Proportional Effects (if any) ---
    if n_npo > 0 && K_ordinal > 1
        priors_npo = priors_vec[npo_indices]
        all_same_npo = !isempty(priors_npo) && all(p -> p == priors_npo[1], priors_npo)
        beta_npo_name = "beta_npo"
        n_npo_params = n_npo * (K_ordinal - 1)

        if all_same_npo
            prior_str = _distribution_to_string(priors_npo[1])
            push!(prior_parts, "$(beta_npo_name) ~ NamedDist(filldist($(prior_str), $(n_npo_params)), :beta_npo)")
        else
            full_priors_list = vcat([priors_npo for _ in 1:(K_ordinal-1)]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            push!(prior_parts, "$(beta_npo_name) ~ NamedDist(Product([$(join(priors_str_list, ", "))]), :beta_npo)")
        end
        # The update for NPO effects happens inside the ordinal likelihood block.
    end

    prior_code = join(prior_parts, "\n    ")
    update_code = join(update_parts, "\n    ")
    
    return prior_code, update_code
end

 


function _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool, eta_name::String)
    if get(M, :model_st, "none") == "none" 
        return ""
    end

    if isnothing(s_spec) || isnothing(t_spec)
        @warn "Spatiotemporal interaction requested but marginal specifications are missing."
        return ""
    end

    s_key = string(s_spec.key)
    t_key = string(t_spec.key)
    
    s_chol_access = "cholesky(Symmetric(spec_registry[\"$s_key\"].Q_template + noise * I))"
    
    t_model_type = t_spec.component_obj |> typeof |> Symbol
    t_rho_var_name = "rho_$(t_key)"
    t_chol_access = "cholesky(Symmetric(recompose_precision(:$(t_model_type), spec_registry[\"$t_key\"].Q_template, 1.0; extra_param=$(t_rho_var_name)) + noise * I))"

    K = get(M, :outcomes_N, 1)

    dummy_interaction_spec = (
        key = Symbol("st_interaction"),
        structure = :spacetime,
        var = "$(s_key)_$(t_key)",
        component_obj = NoneComponent(),
        params = Dict{Symbol, Any}(),
        Q_template = nothing,
        scaling_factor = 1.0,
        hyper = nothing
    )
    v_st_interaction = generate_full_variable_names(dummy_interaction_spec, "univariate", nothing)

    sigma_name = string(v_st_interaction.sigma)
    raw_name = string(v_st_interaction.raw)

    st_sigma_prior_dist_str = haskey(M, :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) : "Exponential(1.0)"

    if is_multivariate
        interaction_code = """
        # --- Spatiotemporal Interaction Priors ---
        $(sigma_name) ~ NamedDist(filldist($(st_sigma_prior_dist_str), $K), :$(Symbol(sigma_name)))
        $(raw_name) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $K), I), :$(Symbol(raw_name)))

        let
            C_s = $(s_chol_access)
            C_t = $(t_chol_access)
            
            Z_tensor = reshape($(raw_name), M.s_N, M.t_N, $K)
            
            for k in 1:$K
                Z_k = view(Z_tensor, :, :, k)
                
                tmp_spatial = C_s.U \\ Z_k
                
                # Materialize the transpose to avoid dispatch issues with FactorComponent
                tmp_spatial_T = Matrix(transpose(tmp_spatial))
                st_field_k_unscaled = transpose(C_t.U \\ tmp_spatial_T)
                
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_k_unscaled)) # Soft sum-to-zero constraint
                
                st_field_k = st_field_k_unscaled .* $(sigma_name)[k]

                for i in 1:N
                    $(eta_name)[i, k] += st_field_k[M.s_idx[i], M.t_idx[i]]
                end
            end
        end
        """
    else
        interaction_code = """
        # --- Spatiotemporal Interaction Priors ---
        $(sigma_name) ~ NamedDist($(st_sigma_prior_dist_str), :$(Symbol(sigma_name)))
        $(raw_name) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(Symbol(raw_name)))

        let
            C_s = $(s_chol_access)
            C_t = $(t_chol_access)
            
            Z_matrix = reshape($(raw_name), M.s_N, M.t_N)
            
            tmp_spatial = C_s.U \\ Z_matrix
            
            # Materialize the transpose to avoid dispatch issues with FactorComponent
            tmp_spatial_T = Matrix(transpose(tmp_spatial))
            st_field_unscaled = transpose(C_t.U \\ tmp_spatial_T)
            
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled)) # Soft sum-to-zero constraint
            
            st_field = st_field_unscaled .* $(sigma_name)

            for i in 1:N
                $(eta_name)[i] += st_field[M.s_idx[i], M.t_idx[i]]
            end
        end
        """
    end
    
    return interaction_code
end


function _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)
    if !haskey(M, :nested_components) || isempty(M.nested_components)
        return "", "", ""
    end

    all_nested_priors = String[]
    all_nested_updates = String[]
    all_nested_likelihoods = String[]

    for (var_key, sub_config) in pairs(M.nested_components)
        prefix = string(var_key)
        sub_eta_name = "eta_$(prefix)"

        # --- Generate Priors for Sub-Model ---
        # Component priors
        for spec in sub_config.components
            frag = _generate_component_code_fragments(spec.component_obj, spec, "univariate", nothing; prefix=prefix)
            push!(all_nested_priors, frag.priors)
        end
        
        # Fixed effects priors
        if get(sub_config, :Xfixed_N, 0) > 0
            priors_vec = get(sub_config, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:sub_config.Xfixed_N])
            all_same = isempty(priors_vec) ? false : all(p -> p == priors_vec[1], priors_vec)

            local prior_block
            if all_same && !isempty(priors_vec)
                prior_str = _distribution_to_string(priors_vec[1])
                prior_block = "$(prefix)_Xfixed_beta ~ NamedDist(filldist($(prior_str), $(sub_config.Xfixed_N)), :$(Symbol(prefix, "_Xfixed_beta")))"
            else
                priors_str_list = [_distribution_to_string(p) for p in priors_vec]
                priors_product_str = "Product([$(join(priors_str_list, ", "))])"
                prior_block = "$(prefix)_Xfixed_beta ~ NamedDist($(priors_product_str), :$(Symbol(prefix, "_Xfixed_beta")))"
            end
            push!(all_nested_priors, prior_block)
        end

        # Intercept prior
        if get(sub_config, :add_intercept, false)
            prior_obj = get(sub_config, :intercept_prior, Normal(0,5))
            push!(all_nested_priors, "$(prefix)_intercept ~ NamedDist($(_distribution_to_string(prior_obj)), :$(Symbol(prefix, "_intercept")))")
        end

        # Observation sigma prior (if needed)
        sub_sigma_name = "$(prefix)_y_sigma"
        sub_lik_spec = sub_config.likelihood_specs[1]
        sub_family_str = string(get(sub_lik_spec, :family, "gaussian"))
        sub_needs_sigma = sub_family_str in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        if sub_needs_sigma
            push!(all_nested_priors, "$(sub_sigma_name) ~ NamedDist(Exponential(1.0), :$(sub_sigma_name))")
        end

        # --- Generate Updates for Sub-Model Linear Predictor ---
        update_block = """
        begin 
            # Assemble predictor for nested model: $(prefix)
            $(sub_eta_name) = zeros(T, sub_config.y_N)
        """
        if get(sub_config, :add_intercept, false); update_block *= "\n    $(sub_eta_name) .+= $(prefix)_intercept"; end # Add intercept
        if haskey(sub_config, :log_offset); update_block *= "\n    $(sub_eta_name) .+= M.nested_components[:$(var_key)].log_offset"; end
        if get(sub_config, :Xfixed_N, 0) > 0; update_block *= "\n    $(sub_eta_name) .+= M.nested_components[:$(var_key)].Xfixed * $(prefix)_Xfixed_beta"; end
        
        # Component updates
        for spec in sub_config.components
            frag = _generate_component_code_fragments(spec.component_obj, spec, "univariate", nothing; prefix=prefix)
            update_block *= "\n" * frag.update
        end
        update_block *= "\nend"
        push!(all_nested_updates, update_block)

        # --- Generate Linking Code ---
        rho_name = "rho_nested_$(prefix)"
        if is_multivariate
            K = M.outcomes_N
            push!(all_nested_priors, "$(rho_name) ~ NamedDist(filldist(Normal(1.0, 0.5), $(K)), :$(rho_name))")
            push!(all_nested_updates, "for k in 1:$(K); $(main_eta_name)[:, k] .+= $(rho_name)[k] .* $(sub_eta_name); end")
        else
            push!(all_nested_priors, "$(rho_name) ~ NamedDist(Normal(1.0, 0.5), :$(rho_name))")
            push!(all_nested_updates, "$(main_eta_name) .+= $(rho_name) .* $(sub_eta_name)")
        end

        # --- Generate Likelihood for Sub-Model ---
        kwargs_parts = String[]
        if sub_needs_sigma; push!(kwargs_parts, "sigma_y=$(sub_sigma_name)"); end
        
        param_keys = [:trials, :weights, :censor_lower, :censor_upper, :hurdle]
        for key in param_keys
            if get(sub_config, Symbol("user_provided_", key), false)
                param_val = get(sub_config, key, nothing) # Get the value from sub_config
                val_code = param_val isa AbstractVector ? "sub_M.$(key)[i]" : "sub_M.$(key)"
                if key == :trials; val_code = "Int(" * val_code * ")"; end
                push!(kwargs_parts, "$(key)=$(val_code)")
            end
        end
        kwargs_str = join(kwargs_parts, ", ")

        lik_loop = """
        # Likelihood for nested model: $(prefix)
        let sub_M = M.nested_components[:$(var_key)]
            sub_family = sub_M.likelihood_specs[1][:family] # Get family from sub-model config
            for i in 1:sub_M.y_N
                d_lik_sub = bstm_Likelihood(sub_family, [T(sub_M.y_obs[i])]; $(kwargs_str))
                Turing.@addlogprob! Distributions.logpdf(d_lik_sub, $(sub_eta_name)[i])
            end
        end
        """
        push!(all_nested_likelihoods, lik_loop)
    end

    return join(all_nested_priors, "\n\n"), join(all_nested_updates, "\n\n"), join(all_nested_likelihoods, "\n\n")
end

 
 
function bstm_dynamic_model(config::NamedTuple)
    # Purpose: A unified entry point for compiling and instantiating any dynamically generated model.
    # Rationale: Decouples model generation from execution.
    # v1.0.0 (2026-07-16)
    # Assumptions: `config` is a valid model configuration.
    # Inputs:
    #   - config: The model configuration NamedTuple.
    # Outputs: An instantiated Turing.jl model object.
    model_string, expr, registry = bstm_text_assembler(config)

    println("\n--- Dynamically Generated Model Code ---")
    println(model_string)
    println("----------------------------------------\n")

    config_dict = Dict(pairs(config))
    config_dict[:generated_model_code] = model_string
    new_config = NamedTuple(config_dict)

    Base.invokelatest(eval, expr)

    arch = get(config, :model_arch, "univariate")
    model_func_name = if arch == "multivariate"
        :bstm_text_generated_multivariate
    elseif arch == "multifidelity"
        :bstm_text_generated_multifidelity
    else
        :bstm_text_generated_univariate
    end
    model_func = getfield(Main, model_func_name)
    return Base.invokelatest(model_func, new_config, registry)
end






function build_model(m::SVAR, data_inputs::Dict, module_metadata::Dict)
    s_N = get(data_inputs, :s_N, 1)
    t_N = get(data_inputs, :t_N, 1)
    
    # The inner spatial model object is already correctly stored in m.rho_spatial.
    # We need to build its specification to store it in the hyper registry for the code generator.
    
    # To call build_model on the inner component, we need to reconstruct its module_metadata.
    inner_model_name = Symbol(lowercase(string(typeof(m.rho_spatial))))
    
    inner_mod_data = Dict(
        :type => :spatial, # The inner model of an SVAR is always spatial.
        :params => module_metadata[:params], # Pass the original params down.
        :variables => get(module_metadata, :variables, [])
    )
    # Ensure the 'model' parameter in the metadata reflects the actual inner model type.
    inner_mod_data[:params][:model] = inner_model_name

    # Recursively call build_model on the inner spatial component.
    rho_spatial_spec = build_model(m.rho_spatial, data_inputs, inner_mod_data)
    
    # Store the resulting specification in the hyper registry.
    hyper_dict = Dict(
        :rho_spatial_spec => rho_spatial_spec,
        :s_N => s_N,
        :t_N => t_N
    )
    
    # The SVAR component itself does not have a Q_template.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:svar, hyper=NamedTuple(hyper_dict))
end


function build_model(m::AdaptiveSmooth, data_inputs::Dict, module_metadata::Dict)
    # Resolve coordinates for the non-linear transformation
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("AdaptiveSmooth requires coordinates in the module parameters.")
    end

    hyper_dict = Dict(
        :coords => Float64.(coords),
        :in_dim => size(coords, 2),
        :hidden_dim => m.hidden_dim,
        :nbins => m.nbins
    )

    # AdaptiveSmooth relies on learned weights rather than a fixed precision matrix template
    return (Q_template=nothing, scaling_factor=1.0, model_type=:adaptivesmooth, hyper=NamedTuple(hyper_dict))
end


function _generate_multivariate_dynamics_code(m::DynamicsComponent, spec::NamedTuple, M::NamedTuple)
    # Purpose: Generates Turing code for multivariate dynamics models.
    # Rationale: This version adds a new branch to handle the multivariate `delay_difference`
    #            model, which treats population and recruitment as two separate, linked outcomes.
    #            It also refactors the `leslie_matrix` logic to correctly handle the `haskey`
    #            check for the optional carrying capacity parameter `K`.
    # v1.0.5 (2026-07-31)
    key_str = string(spec.key)
    params = m.params
    model_type = m.model
    prefixed_key = key_str

    if model_type == "leslie_matrix"
        n_age_classes = get(params, :n_age_classes, M.outcomes_N)
        spatially_varying_K = get(params, :spatially_varying_K, false)
        spatially_varying_rates = get(params, :spatially_varying_rates, false)

        if n_age_classes != M.outcomes_N
            error("Number of age classes ($n_age_classes) in dynamics(model=leslie_matrix) must match number of outcomes ($(M.outcomes_N)).")
        end

        priors_acc = []
        if spatially_varying_rates
            push!(priors_acc, "log_fecundity_mean_$(key_str) ~ NamedDist(filldist(Normal(0, 1), $(n_age_classes)), :log_fecundity_mean_$(key_str))")
            push!(priors_acc, "sigma_fecundity_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_age_classes)), :sigma_fecundity_$(key_str))")
            push!(priors_acc, "fecundity_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * $(n_age_classes)), I), :fecundity_raw_$(key_str))")
            push!(priors_acc, "logit_survival_mean_$(key_str) ~ NamedDist(filldist(Normal(1.5, 1), $(n_age_classes-1)), :logit_survival_mean_$(key_str))")
            push!(priors_acc, "sigma_survival_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_age_classes-1)), :sigma_survival_$(key_str))")
            push!(priors_acc, "survival_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * ($(n_age_classes)-1)), I), :survival_raw_$(key_str))")
        else
            push!(priors_acc, "survival_rates_$(key_str) ~ NamedDist(filldist(Beta(9, 1), $(n_age_classes - 1)), :survival_rates_$(key_str))")
            push!(priors_acc, "fecundity_rates_$(key_str) ~ NamedDist(filldist(LogNormal(0, 1), $(n_age_classes)), :fecundity_rates_$(key_str))")
        end

        if haskey(params, :K) || spatially_varying_K
            if spatially_varying_K
                sigma_K_prior = get(params, :sigma_K, Exponential(1.0))
                log_K_mean_prior = get(params, :log_K_mean, haskey(params, :K) && params[:K] isa LogNormal ? Normal(Distributions.params(params[:K])...) : Normal(log(100.0), 0.5))
                push!(priors_acc, "sigma_K_$(prefixed_key) ~ NamedDist($(_distribution_to_string(sigma_K_prior)), :sigma_K_$(prefixed_key))")
                push!(priors_acc, "log_K_mean_$(prefixed_key) ~ NamedDist($(_distribution_to_string(log_K_mean_prior)), :log_K_mean_$(prefixed_key))")
                push!(priors_acc, "K_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :K_raw_$(prefixed_key))")
            else
                K_prior = get(params, :K, LogNormal(log(100.0), 1.0))
                push!(priors_acc, "K_$(key_str) ~ NamedDist($(_distribution_to_string(K_prior)), :K_$(key_str))")
            end
        end

        push!(priors_acc, "sigma_process_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_age_classes)), :sigma_process_$(key_str))")
        push!(priors_acc, "innov_process_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(n_age_classes)), I), :innov_process_$(key_str))")
        priors_str = join(priors_acc, "\n    ")

        update_str = """
        begin
            # Multivariate Leslie Matrix Dynamics for $(key_str)
            local Q_spatial = spec_registry["$(key_str)"].hyper.L_template
            local F_spatial = cholesky(Symmetric(Q_spatial + noise * I))
            local areas = spec_registry["$(key_str)"].hyper.areas

            # Construct spatially varying vital rates if specified
            local survival_rates_spatial, fecundity_rates_spatial
            if $(spatially_varying_rates)
                fecundity_raw_matrix = reshape(fecundity_raw_$(key_str), M.s_N, $(n_age_classes))
                fecundity_field = F_spatial.U \\ fecundity_raw_matrix
                fecundity_rates_spatial = exp.(log_fecundity_mean_$(key_str)' .+ fecundity_field .* sigma_fecundity_$(key_str)')

                survival_raw_matrix = reshape(survival_raw_$(key_str), M.s_N, $(n_age_classes-1))
                survival_field = F_spatial.U \\ survival_raw_matrix
                survival_rates_spatial = logistic.(logit_survival_mean_$(key_str)' .+ survival_field .* sigma_survival_$(key_str)')
            end

            # Construct K field (if spatially varying)
            local K_values_$(key_str)
            if $(spatially_varying_K)
                K_field_raw = F_spatial.U \\ K_raw_$(prefixed_key)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(K_field_raw))
                K_values_$(key_str) = exp.(log_K_mean_$(prefixed_key) .+ K_field_raw .* sigma_K_$(prefixed_key))
            elseif haskey(spec_registry["$(key_str)"].component_obj.params, :K)
                K_values_$(key_str) = fill(K_$(key_str), M.s_N)
            end
            
            local innov_tensor_$(key_str) = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_age_classes))
            local population_field_$(key_str) = zeros(T, M.s_N, M.t_N, $(n_age_classes))

            for a in 1:$(n_age_classes)
                population_field_$(key_str)[:, 1, a] = max.(0.0, innov_tensor_$(key_str)[:, 1, a] .* sigma_process_$(key_str)[a])
            end

            for s in 1:M.s_N
                local L_s = zeros(T, $(n_age_classes), $(n_age_classes))
                if $(spatially_varying_rates)
                    for i in 1:($(n_age_classes)-1); L_s[i+1, i] = survival_rates_spatial[s, i]; end
                    L_s[1, :] = fecundity_rates_spatial[s, :]
                else
                    for i in 1:($(n_age_classes)-1); L_s[i+1, i] = survival_rates_$(key_str)[i]; end
                    L_s[1, :] = fecundity_rates_$(key_str)
                end

                for t in 2:M.t_N
                    local N_prev = view(population_field_$(key_str), s, t-1, :)
                    local L_effective = copy(L_s)
                    
                    if haskey(spec_registry["$(key_str)"].component_obj.params, :K)
                        local total_pop_prev = sum(N_prev)
                        local K_density = K_values_$(key_str)[s] / areas[s]
                        local dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density)
                        L_effective[1, :] .*= dd_factor
                    end

                    local N_projected = L_effective * N_prev
                    local current_innov = view(innov_tensor_$(key_str), s, t, :) .* sigma_process_$(key_str)
                    population_field_$(key_str)[s, t, :] = max.(0.0, N_projected .+ current_innov)
                end
            end

            for k in 1:$(n_age_classes)
                for i in 1:N
                    eta_latent[i, k] += log(population_field_$(key_str)[M.s_idx[i], M.t_idx[i], k] + 1e-6)
                end
            end
        end
        """
        return (priors=priors_str, update=update_str)

    elseif model_type == "delay_difference"
        if M.outcomes_N != 2
            error("The multivariate `delay_difference` model requires exactly two outcomes: population and recruitment.")
        end
        
        priors_acc = []
        push!(priors_acc, "r_$(key_str) ~ NamedDist(LogNormal(0, 1), :r_$(key_str))")
        push!(priors_acc, "K_$(key_str) ~ NamedDist(LogNormal(log(100.0), 1.0), :K_$(key_str))")
        push!(priors_acc, "M_nat_$(key_str) ~ NamedDist(LogNormal(-1, 0.5), :M_nat_$(key_str))")
        push!(priors_acc, "sigma_recruitment_$(key_str) ~ NamedDist(Exponential(1.0), :sigma_recruitment_$(key_str))")
        push!(priors_acc, "sigma_population_$(key_str) ~ NamedDist(Exponential(1.0), :sigma_population_$(key_str))")
        push!(priors_acc, "innov_recruitment_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :innov_recruitment_$(key_str))")
        push!(priors_acc, "innov_population_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :innov_population_$(key_str))")
        
        if haskey(params, :effort_col)
            push!(priors_acc, "q_$(key_str) ~ NamedDist(LogNormal(-2, 1), :q_$(key_str))")
        elseif !haskey(params, :catch_data_col)
            error("`delay_difference` model requires either `effort_col` or `catch_data_col` parameter.")
        end
        priors_str = join(priors_acc, "\n    ")

        local catch_logic_block
        if haskey(params, :effort_col)
            effort_col_sym = params[:effort_col]
            catch_logic_block = """
            local effort_matrix = spec_registry["$(key_str)"].hyper.processed_params[:$(effort_col_sym)]
            local C_prev = q_$(key_str) .* effort_matrix[:, t-1] .* N_prev
            """
        else
            catch_data_col_sym = params[:catch_data_col]
            catch_logic_block = """
            local catch_data_matrix = spec_registry["$(key_str)"].hyper.processed_params[:$(catch_data_col_sym)]
            local C_prev = catch_data_matrix[:, t-1]
            """
        end

        update_str = """
        begin
            # Multivariate Delay-Difference Dynamics for $(key_str)
            local areas = spec_registry["$(key_str)"].hyper.areas
            
            local population_field = zeros(T, M.s_N, M.t_N)
            local recruitment_field = zeros(T, M.s_N, M.t_N)
            
            local innov_recruitment_matrix = reshape(innov_recruitment_$(key_str), M.s_N, M.t_N)
            local innov_population_matrix = reshape(innov_population_$(key_str), M.s_N, M.t_N)

            population_field[:, 1] = max.(0.0, innov_population_matrix[:, 1] .* sigma_population_$(key_str))
            recruitment_field[:, 1] = max.(0.0, innov_recruitment_matrix[:, 1] .* sigma_recruitment_$(key_str))

            for t in 2:M.t_N
                local N_prev = population_field[:, t-1]
                local D_prev = N_prev ./ areas
                local K_density = K_$(key_str) ./ areas
                
                local mean_recruitment = r_$(key_str) .* D_prev .* (1.0 .- D_prev ./ K_density) .* areas
                recruitment_field[:, t] = exp.(log.(mean_recruitment .+ 1e-6) .+ innov_recruitment_matrix[:, t] .* sigma_recruitment_$(key_str))
                
                $(catch_logic_block)
                local N_survived = (N_prev .- C_prev) .* exp.(-M_nat_$(key_str))
                population_field[:, t] = max.(0.0, N_survived .+ recruitment_field[:, t] .+ innov_population_matrix[:, t] .* sigma_population_$(key_str))
            end

            for i in 1:N
                s_i, t_i = M.s_idx[i], M.t_idx[i]
                eta_latent[i, 1] += log(population_field[s_i, t_i] + 1e-6)
                eta_latent[i, 2] += log(recruitment_field[s_i, t_i] + 1e-6)
            end
        end
        """
        return (priors=priors_str, update=update_str)

     elseif model_type == "generalized_lotka_volterra"
        spatially_varying_K = get(params, :spatially_varying_K, false)
        priors_acc = []

        # Priors for growth rates, interaction matrix, and K
        push!(priors_acc, "r_$(key_str) ~ NamedDist(filldist(LogNormal(0, 1), $(n_species)), :r_$(key_str))")
        
        n_off_diag = n_species * (n_species - 1)
        push!(priors_acc, "alpha_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, $(n_off_diag)), I), :alpha_raw_$(key_str))")

        if spatially_varying_K
            push!(priors_acc, "log_K_mean_$(key_str) ~ NamedDist(filldist(Normal(log(100.0), 1.0), $(n_species)), :log_K_mean_$(key_str))")
            push!(priors_acc, "sigma_K_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_species)), :sigma_K_$(key_str))")
            push!(priors_acc, "K_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * $(n_species)), I), :K_raw_$(key_str))")
        else
            push!(priors_acc, "K_$(key_str) ~ NamedDist(filldist(LogNormal(log(100.0), 1.0), $(n_species)), :K_$(key_str))")
        end

        # Priors for process noise
        push!(priors_acc, "sigma_process_$(key_str) ~ NamedDist(filldist(Exponential(1.0), $(n_species)), :sigma_process_$(key_str))")
        push!(priors_acc, "innov_process_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(n_species)), I), :innov_process_$(key_str))")
        priors_str = join(priors_acc, "\n    ")

        update_str = """
        begin
            # Generalized Lotka-Volterra Dynamics for $(key_str)
            local areas = spec_registry["$(key_str)"].hyper.areas
            
            # Construct interaction matrix alpha
            local alpha_$(key_str) = diagm(0 => ones(T, $(n_species)))
            local off_diag_indices = [i for i in 1:($(n_species)^2) if mod(i-1, $(n_species)+1) != 0]
            alpha_$(key_str)[off_diag_indices] = alpha_raw_$(key_str)

            # Construct K field
            local K_values_$(key_str)
            if $(spatially_varying_K)
                local Q_spatial = spec_registry["$(key_str)"].hyper.L_template
                local F_spatial = cholesky(Symmetric(Q_spatial + noise * I))
                local K_raw_matrix = reshape(K_raw_$(key_str), M.s_N, $(n_species))
                local K_field = F_spatial.U \\ K_raw_matrix
                K_values_$(key_str) = exp.(log_K_mean_$(key_str)' .+ K_field .* sigma_K_$(key_str)')
            else
                K_values_$(key_str) = repeat(K_$(key_str)', M.s_N, 1)
            end

            local innov_tensor = reshape(innov_process_$(key_str), M.s_N, M.t_N, $(n_species))
            local population_field = zeros(T, M.s_N, M.t_N, $(n_species))
            population_field[:, 1, :] = max.(0.0, innov_tensor[:, 1, :] .* sigma_process_$(key_str)')

            for s in 1:M.s_N, t in 2:M.t_N
                local N_prev = view(population_field, s, t-1, :)
                local D_prev = N_prev ./ areas[s]
                local K_density = K_values_$(key_str)[s, :] ./ areas[s]
                
                local N_intermediate = zeros(T, $(n_species))
                for i in 1:$(n_species)
                    local interaction_sum_density = dot(alpha_$(key_str)[i, :], D_prev)
                    local growth_density = r_$(key_str)[i] * D_prev[i] * (1.0 - interaction_sum_density / K_density[i])
                    N_intermediate[i] = N_prev[i] + growth_density * areas[s]
                end
                
                local current_innov = view(innov_tensor, s, t, :) .* sigma_process_$(key_str)
                population_field[s, t, :] = max.(0.0, N_intermediate .+ current_innov)
            end

            for k in 1:$(n_species)
                for i in 1:N
                    eta_latent[i, k] += log(population_field[M.s_idx[i], M.t_idx[i], k] + 1e-6)
                end
            end
        end
        """
        return (priors=priors_str, update=update_str)
            
    end
    
    return (priors="", update="")
end


function _generate_component_code_fragments(m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="", generate_eta_update::Bool=true)
    # Purpose: Generates Turing code fragments for the Spatially Varying Autoregressive (SVAR) model.
    # Rationale: This version corrects an `UndefVarError` by changing how the inner spatial
    #            model's configuration is accessed. Instead of using the `spec` variable, which
    #            is not available at model runtime, it now correctly uses the `spec_registry`
    #            dictionary. This ensures that the precision matrix for the spatially varying
    #            rho field is correctly retrieved, resolving the error.
    # v1.0.1 (2026-07-31)
    # Inputs: Standard code generation arguments.
    # Outputs: A NamedTuple containing `priors` and `update` code strings.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    # Generate variable names for the main SVAR component and its inner spatial model
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    rho_spatial_prefix = prefixed_key * "_rho_spatial"
    inner_spec_for_codegen = (
        key = Symbol(rho_spatial_prefix),
        structure = :spatial,
        var = spec.var,
        component_obj = m.rho_spatial,
        params = spec.params,
        Q_template = spec.hyper.rho_spatial_spec.Q_template,
        scaling_factor = spec.hyper.rho_spatial_spec.scaling_factor,
        hyper = spec.hyper.rho_spatial_spec.hyper
    )
    v_rho_spatial = generate_full_variable_names(inner_spec_for_codegen, arch, outcome_idx, prefix="")

    # Get inner model spec for parameter access
    inner_model = m.rho_spatial
    n_latent_inner = size(inner_spec_for_codegen.Q_template, 1)
    n_latent_svar = spec.hyper.s_N * spec.hyper.t_N

    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]

    # Priors for the inner spatial model (rho_spatial)
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(inner_model, :sigma)
            push!(priors_acc, "$(v_rho_spatial.sigma) ~ NamedDist($(_distribution_to_string(inner_model.sigma)), :$(v_rho_spatial.sigma))")
        end
    end
    push!(priors_acc, "$(v_rho_spatial.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_inner)), I), :$(v_rho_spatial.raw))")

    # Priors for the main SVAR component
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end
    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent_svar)), I), :$(v.innov))")

    priors_str = join(priors_acc, "\n")
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update_str = if generate_eta_update
        """
        begin
            # --- SVAR Model: $(key_str) ---
            
            # 1. Compute the spatially varying rho field (inner model logic)
            begin
                # --- Intrinsic Component Solve: $(inner_spec_for_codegen.key) ---
                # FIX: Access the inner model's spec via the spec_registry, not the undefined 'spec' variable.
                local Q_template_inner = spec_registry["$(key_str)"].hyper.rho_spatial_spec.Q_template
                local F_inner = cholesky(Symmetric(Q_template_inner + noise * I))
                local latent_field_raw_inner = F_inner.U \\ $(v_rho_spatial.raw)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent_inner)), sum(latent_field_raw_inner))
                $(v_rho_spatial.latent) = latent_field_raw_inner .* $(v_rho_spatial.sigma)
            end
            
            # 2. The latent field from the inner model is the untransformed rho.
            #    We apply tanh to constrain it between -1 and 1.
            local $(v.rho_field) = tanh.($(v_rho_spatial.latent))
            
            # 3. Evolve the state-space model for each spatial location.
            local $(v.latent) = zeros(T, M.s_N, M.t_N)
            local innov_matrix = reshape($(v.innov), M.s_N, M.t_N)
            
            for s in 1:M.s_N
                $(v.latent)[s, :] = ar1_statespace($(v.rho_field)[s], 1.0, innov_matrix[s, :], T, M.t_N, noise)
            end
            $(v.latent) .*= $(v.sigma)
     
            # 4. Apply the effect to the linear predictor.
            for i in 1:M.y_N
                $(eta_update_target)[i] += $(v.latent)[M.s_idx[i], M.t_idx[i]]
            end
        end
        """
    else
        ""
    end

    return (priors=priors_str, update=update_str)
end





function ar1_statespace(rho, sigma, innov, T, n_latent, noise)
    # Helper function for AR(1) state-space evolution.
    latent = Vector{T}(undef, n_latent)
    if n_latent > 0
        latent[1] = innov[1] / sqrt(1.0 - rho^2 + noise)
        for t in 2:n_latent
            latent[t] = rho * latent[t-1] + innov[t]
        end
        latent .*= sigma
    end
    return latent
end

 

# Adaptive Basis Functions: Learns a non-linear warping of coordinates using a hidden layer before kernel application, facilitating the discovery of complex spatial/temporal deformations.
 
function build_model(m::AdaptiveSmooth, data_inputs::Dict, module_metadata::Dict)
    # Resolve coordinates for the transformation
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("AdaptiveSmooth requires coordinates in the module parameters.")
    end

    hyper_dict = Dict(
        :coords => Float64.(coords),
        :in_dim => size(coords, 2),
        :hidden_dim => m.hidden_dim,
        :nbins => m.nbins
    )

    return (Q_template=nothing, scaling_factor=1.0, model_type=:adaptive_smooth, hyper=NamedTuple(hyper_dict))
end


# Code Generators for Advanced Components

"""
    _generate_component_code_fragments(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `AdaptiveSmooth` component.

# Rationale for Correction
This version corrects the logic for the adaptive smoother. The original implementation
incorrectly summed the generated basis functions. The corrected version introduces a
set of random coefficients (`coeffs`) for the adaptive basis. The neural network now
generates the basis matrix `B`, and the final effect is correctly computed as the
matrix-vector product `B * coeffs`, which is the standard formulation for a basis
function smoother.
"""
function _generate_component_code_fragments(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    h_dim = m.hidden_dim
    in_dim = spec.hyper.in_dim
    n_bins = m.nbins

    # Define variable names for the MLP weights and the basis coefficients
    W1_name = Symbol("$(v.raw)_W1")
    b1_name = Symbol("$(v.raw)_b1")
    W2_name = Symbol("$(v.raw)_W2")
    coeffs_name = v.innov # Use 'innov' for the basis coefficients' raw innovations

    priors = """
    # Adaptive Basis Priors (Neural Transformation Weights & Coefficients)
    $(W1_name) ~ NamedDist(MvNormal(zeros(T, $(in_dim * h_dim)), I), :$(W1_name))
    $(b1_name) ~ NamedDist(MvNormal(zeros(T, $(h_dim)), I), :$(b1_name))
    $(W2_name) ~ NamedDist(MvNormal(zeros(T, $(h_dim * n_bins)), I), :$(W2_name))
    
    # Priors for the coefficients of the adaptive basis functions
    $(coeffs_name) ~ NamedDist(MvNormal(zeros(T, $(n_bins)), I), :$(coeffs_name))
    
    # Prior for the overall scale of the smooth effect
    $(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))
    """

    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    update = """
    begin
        # Adaptive Basis transformation and effect application
        local X_orig = spec_registry["$(key_str)"].hyper.coords
        
        # 1. Reconstruct the MLP weights
        local W1 = reshape($(W1_name), $(in_dim), $(h_dim))
        local b1 = $(b1_name)
        local W2 = reshape($(W2_name), $(h_dim), $(n_bins))

        # 2. Compute the adaptive basis matrix B
        # Layer 1: Learnable coordinate transformation
        local H = tanh.(X_orig * W1 .+ b1')
        # Layer 2: Projection to the final basis functions
        local B_adaptive = H * W2

        # 3. Scale the basis coefficients
        local scaled_coeffs = $(coeffs_name) .* $(v.sigma)
        
        # 4. Compute the final effect by multiplying the basis matrix by the coefficients
        local adaptive_effect = B_adaptive * scaled_coeffs
        
        # 5. Add the effect to the linear predictor
        $(eta_target) .+= adaptive_effect
    end
    """

    return (priors=priors, update=update)
end



"""
    bstm_sample_nowarn(model, sampler, n_samples; kwargs...)

A wrapper around `Turing.sample` that suppresses world-age warnings by temporarily
redirecting `stderr` to `devnull`.

# Rationale
Dynamically generated models from `@bstm` can trigger non-fatal world-age warnings
when interacting with pre-compiled library functions inside Turing.jl. This wrapper
provides a clean way to run the sampler without printing these warnings to the console.

# Arguments
- `model`: The Turing model object.
- `sampler`: The MCMC sampler to use.
- `n_samples`: The number of samples to draw.
- `kwargs...`: Additional keyword arguments passed directly to `Turing.sample`.

# Returns
- The MCMC chain object returned by `Turing.sample`.

# Example
```julia
m = @bstm(likelihood(y) ~ 1, data)
chn = bstm_sample_nowarn(m, NUTS(), 1000)
```
"""
function bstm_sample_nowarn(model, sampler, n_samples; kwargs...)
    local chain
    redirect_stderr(devnull) do
        chain = sample(model, sampler, n_samples; kwargs...)
    end
    return chain
end



function build_model(m::TAR, data_inputs::Dict, module_metadata::Dict)
    t_N = get(data_inputs, :t_N, 1)
    data = data_inputs[:data]
    threshold_var_sym = m.threshold_var
    
    if !hasproperty(data, threshold_var_sym)
        error("TAR model's threshold variable ':$threshold_var_sym' not found in the provided data.")
    end
    
    threshold_data = data[!, threshold_var_sym]

    hyper_dict = Dict(
        :threshold_var => threshold_var_sym,
        :threshold_data => Float64.(threshold_data),
        :t_N => t_N
    )

    return (Q_template=nothing, scaling_factor=1.0, model_type=:tar, hyper=NamedTuple(hyper_dict))
end






function _generate_component_code_fragments(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    rho1_name = Symbol("$(v.rho)_1")
    rho2_name = Symbol("$(v.rho)_2")
    sigma1_name = Symbol("$(v.sigma)_1")
    sigma2_name = Symbol("$(v.sigma)_2")
    thresh_raw_name = Symbol("$(v.raw)_thresh")

    priors = """
    # TAR Regime-Switching Priors
    $(rho1_name) ~ NamedDist($(_distribution_to_string(m.rho_regimes[1])), :$(rho1_name))
    $(rho2_name) ~ NamedDist($(_distribution_to_string(m.rho_regimes[2])), :$(rho2_name))
    $(sigma1_name) ~ NamedDist($(_distribution_to_string(m.sigma_regimes[1])), :$(sigma1_name))
    $(sigma2_name) ~ NamedDist($(_distribution_to_string(m.sigma_regimes[2])), :$(sigma2_name))
    $(thresh_raw_name) ~ NamedDist(Normal(0, 1), :$(thresh_raw_name))
    $(v.innov) ~ NamedDist(MvNormal(zeros(T, M.t_N), I), :$(v.innov))
    """

    update = """
    begin
        # Threshold resolution logic: Learned threshold level relative to covariate mean
        threshold_level = mean(spec_registry["$(key_str)"].hyper.threshold_data) + $(thresh_raw_name)
        
        # Temporal state reconstruction for TAR model
        $(v.latent) = zeros(T, M.t_N)
        innovations = $(v.innov)
        
        for t in 1:M.t_N
            # Determine regime for current time step
            regime_indicator = spec_registry["$(key_str)"].hyper.threshold_data[t] > threshold_level
            
            curr_rho = regime_indicator ? $(rho2_name) : $(rho1_name)
            curr_sigma = regime_indicator ? $(sigma2_name) : $(sigma1_name)
            
            if t == 1
                # Initialize state from its stationary distribution within the regime
                $(v.latent)[t] = (innovations[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + noise)
            else
                # Evolve state using the regime-specific AR(1) process
                $(v.latent)[t] = curr_rho * $(v.latent)[t-1] + innovations[t] * curr_sigma
            end
        end
        
        # Apply the effect to the linear predictor
        $(eta_target) .+= view($(v.latent), M.t_idx)
    end
    """

    return (priors=priors, update=update)
end

 
function build_model(m::LGCP, data_inputs::Dict, module_metadata::Dict)
    # The LGCP struct 'm' already contains the resolved inner model (m.model)
    # and the original inner model node (m.inner_model_node).
    
    # Reconstruct the module_metadata for the inner model from m.inner_model_node
    inner_mod_data = Dict(
        :type => m.inner_model_node.module_type, # e.g., :random
        :params => m.inner_model_node.args,
        :variables => get(m.inner_model_node.args, :positional_args, [])
    )
    # Ensure structure is inferred if not explicit in the inner node's args
    if !haskey(inner_mod_data[:params], :structure)
        inner_mod_data[:params][:structure] = _infer_structure_from_args(inner_mod_data[:params])
    end

    # Now, call build_model on the inner model (m.model) with its reconstructed metadata.
    # This will correctly build its Q_template and hyper.
    inner_spec = build_model(m.model, data_inputs, inner_mod_data)
    
    temporal_spec_idx = findfirst(s -> s.structure == :temporal, data_inputs[:components])
    areas = get(data_inputs, :grid_areas, ones(data_inputs[:s_N]))

    hyper_dict = Dict(
        :inner_spec => inner_spec, 
        :areas => Float64.(areas), 
        :s_N => data_inputs[:s_N], 
        :t_N => get(data_inputs, :t_N, 1)
    )
    if !isnothing(temporal_spec_idx)
        hyper_dict[:temporal_spec] = data_inputs[:components][temporal_spec_idx]
    end

    return (Q_template=inner_spec.Q_template, scaling_factor=1.0, model_type=:lgcp, hyper=NamedTuple(hyper_dict))
end


"""
    build_model(m::Kriging, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Kriging` component.

# Rationale for Update
This version is updated to be consistent with the `GP` builder. It now stores the
coordinate matrix in the `Q_template` field, ensuring that the code generator can
correctly access the coordinates to build the dense covariance matrix.
"""
function build_model(m::Kriging, data_inputs::Dict, module_metadata::Dict)
    # Kriging requires continuous coordinates (e.g., s_x, s_y)
    coords = get(module_metadata[:params], :coords, nothing)
    
    if isnothing(coords)
        # Fallback: check if coordinates exist in the top-level data inputs
        if haskey(data_inputs, :coords)
            coords = data_inputs[:coords]
        else
            error("Kriging component requires coordinate data. Ensure s_x and s_y are provided.")
        end
    end

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Store the raw coordinates in the Q_template field for consistency with the code generator.
    return (Q_template=coords, scaling_factor=1.0, model_type=:kriging, hyper=NamedTuple(hyper_dict))
end


"""
    _generate_component_code_fragments(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for a Kriging (full Gaussian Process) component.

# Rationale
This new function provides the specific implementation for `Kriging`, which is functionally
equivalent to a full Gaussian Process. It correctly computes the dense covariance matrix
based on the chosen kernel and samples the latent field, ensuring `kriging` models are properly assembled.
"""
function _generate_component_code_fragments(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_obs = size(spec.Q_template, 1) # For Kriging/GP, Q_template holds the coordinates

    priors_acc = String[]
    push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    
    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors_acc, "$(v.ls) ~ NamedDist(Product([$(ls_priors_str)]), :$(v.ls))")
    else
        push!(priors_acc, "$(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))")
    end
    
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_obs)), I), :$(v.raw))")
    priors = join(priors_acc, "\n")

    update = """
    begin
        local X_coords = spec_registry["$(key_str)"].Q_template
        local kernel_type = Symbol("$(m.kernel)")
        local K_mat = evaluate_kernel_matrix(X_coords, $(v.sigma), $(v.ls), kernel_type, noise)
        local F_krig = cholesky(Symmetric(K_mat))
        $(v.latent) = F_krig.L * $(v.raw)
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= $(v.latent)
    end
    """
    return (priors=priors, update=update)
end




# ==============================================================================
# SECTION 1: CORE UTILITIES FOR PARAMETER EXTRACTION
# ==============================================================================

function get_kernel_from_string(kernel_name::String)
    # Purpose: Maps a string identifier to a `KernelFunctions.jl` kernel object.
    # Rationale: Centralizes kernel selection for GP-based models.
    # Inputs:
    #   - kernel_name: The string name of the kernel.
    # Outputs: A `Kernel` object.
    k_name = lowercase(kernel_name)
    if k_name == "constant"; return ConstantKernel();
    elseif k_name == "linear"; return LinearKernel();
    elseif k_name == "matern12" || k_name == "exponential"; return Matern12Kernel();
    elseif k_name == "matern32"; return Matern32Kernel();
    elseif k_name == "matern52"; return Matern52Kernel();
    elseif k_name == "spherical"; return SphericalKernel();
    elseif k_name == "squared_exponential" || k_name == "se" || k_name == "gaussian" || k_name == "rbf"; return SqExponentialKernel();
    elseif k_name == "periodic"; return PeriodicKernel();
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end




function _find_parameter_new(p_names, var, param, k=nothing)
    # First, check for the new hyperparameter naming convention: param_var
    base_name_new = "$(param)_$(var)"
    
    if !isnothing(k)
        specific_name_new = "$(base_name_new)_$(k)"
        if specific_name_new in p_names
            return specific_name_new
        end
    end

    if base_name_new in p_names
        return base_name_new
    end

    # Fallback to the old convention and the convention for latent fields: var_param
    base_name_old = "$(var)_$(param)"
    
    if !isnothing(k)
        specific_name_old = "$(base_name_old)_$(k)"
        if specific_name_old in p_names
            return specific_name_old
        end
    end

    if base_name_old in p_names
        return base_name_old
    end
    
    # Check for indexed versions (e.g., var_param[1])
    re_indexed = Regex("^" * escape_string(base_name_old) * "\\[")
    indexed_match = findfirst(n -> occursin(re_indexed, n), p_names)
    if !isnothing(indexed_match)
        return base_name_old
    end

    return ""
end



function get_params_vector(chain, base_name::String, expected_len::Int)
    # Purpose: Extracts all posterior samples for a given parameter into a matrix.
    # Rationale: Handles both scalar and vector parameters, correctly parsing indexed names.
    # Inputs:
    #   - chain: The MCMC chain object.
    #   - base_name: The base name of the parameter (e.g., "latent_spatial").
    #   - expected_len: The expected number of elements for this parameter.
    # Outputs: A matrix of size `[n_samples x expected_len]`.
    local N_samples = size(chain, 1)
    local all_names = string.(FlexiChains.parameters(chain))

    local regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    local matched_names = filter(n -> occursin(regex, n), all_names)

    if !isempty(matched_names)
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))
        local res_mat = zeros(Float64, N_samples, length(matched_names))
        for (idx, n) in enumerate(matched_names)
            local val_obj = chain[Symbol(n)]
            local raw = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)
            for s in 1:N_samples
                local v = raw[s]
                res_mat[s, idx] = (v isa AbstractVector) ? Float64(v[1]) : Float64(v)
            end
        end
        if size(res_mat, 2) == 1 && expected_len > 1
            return repeat(res_mat, 1, expected_len)
        end
        return res_mat
    end

    if base_name in all_names
        local val_obj = chain[Symbol(base_name)]
        local raw_data = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)
        local mat_data = if eltype(raw_data) <: AbstractVector
             reduce(hcat, [vec(collect(v)) for v in raw_data])'
        else
             Matrix{Float64}(reshape(collect(raw_data), N_samples, :))
        end
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 2) == 1 && expected_len > 1
            return repeat(mat_data, 1, expected_len)
        else
            @warn "Parameter '$base_name' was found, but its length ($(size(mat_data, 2))) does not match expected length ($expected_len). Returning as is."
            return mat_data
        end
    end

    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end


# ==============================================================================
# SECTION 2: COMPONENT-SPECIFIC EXTRACTION
# ==============================================================================
function extract_component(m_obj::Union{ICAR, Besag, Cyclic, IID}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    structure = spec.structure
    for k in 1:outcomes_N
        var = string(spec.key)
        
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        latent_name = string(v.raw)
        
        n_units = if structure == :spatial; M.s_N; elseif structure == :temporal; M.t_N; else M.u_N; end
        
        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "raw", k))
            @warn "Parameters for component $(spec.key) (structure $(structure), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_effects, zeros(Float64, n_units, n_samples))
            continue
        end

        Q_template = spec.Q_template
        # Use sparse matrix for Cholesky factorization for efficiency and consistency.
        F = cholesky(Symmetric(sparse(Q_template) + M.noise * I))

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        raw_samples = get_params_vector(chain, latent_name, n_units)
        
        effect = zeros(Float64, n_units, n_samples)
        for s in 1:n_samples
            # Apply non-centered parameterization transformation
            latent_field_raw = F.U \ raw_samples[s, :]
            # Apply sum-to-zero constraint for intrinsic models
            if m_obj isa Union{ICAR, Besag, Cyclic}
                latent_field_raw .-= mean(latent_field_raw)
            end
            effect[:, s] = latent_field_raw .* sigma_samples[s, 1]
        end
        push!(structured_effects, effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end


"""
    extract_component(m_obj::Union{Leroux, SAR}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for `Leroux` and `SAR` models.

# Rationale for New Implementation
This is a new, specialized function to correctly reconstruct `Leroux` and `SAR` models.
Unlike `ICAR`, the precision matrices for these models are dynamic and depend on the `rho`
hyperparameter. This function iterates through each posterior sample, reconstructs the
correct precision matrix for that sample's `rho` value using `recompose_precision`,
and then transforms the raw innovations. This ensures the reconstructed posterior
accurately reflects the fitted model.
"""
function extract_component(m_obj::Union{Leroux, SAR}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    structure = spec.structure
    for k in 1:outcomes_N
        var = string(spec.key)
        
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        rho_name = string(v.rho)
        latent_name = string(v.raw)
        
        n_units = if structure == :spatial; M.s_N; elseif structure == :temporal; M.t_N; else M.u_N; end
        
        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "rho", k)) || isempty(_find_parameter_new(p_names, var, "raw", k))
            @warn "Parameters for component $(spec.key) (structure $(structure), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_effects, zeros(Float64, n_units, n_samples))
            continue
        end

        Q_template = spec.Q_template
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        raw_samples = get_params_vector(chain, latent_name, n_units)
        
        effect = zeros(Float64, n_units, n_samples)
        m_type = Symbol(lowercase(string(typeof(m_obj))))

        for s in 1:n_samples
            # Recompose the precision matrix for each sample using the sampled rho
            Q_final = recompose_precision(m_type, Q_template, 1.0; extra_param=rho_samples[s, 1], noise=M.noise)
            F = cholesky(Symmetric(Q_final))
            
            # Apply non-centered parameterization transformation
            effect[:, s] = (F.U \ raw_samples[s, :]) .* sigma_samples[s, 1]
        end
        push!(structured_effects, effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end




"""
    extract_component(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for a BYM2 component.

# Rationale for Correction
This version is updated to match the corrected generative logic in the code generator.
It now scales the reconstructed structured component by its standard deviation at each
posterior sample, ensuring the `struct_effect` and `unstruct_effect` components
correctly represent their contribution to the total variance as defined by the
Riebler parameterization.
"""
function extract_component(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    noisy_effects = Vector{Matrix{Float64}}()

    structure = spec.structure
    n_latent = size(spec.Q_template, 1)
    noise = get(M, :noise, 1e-6)

    # Use sparse matrix for Cholesky factorization for efficiency.
    F = cholesky(Symmetric(sparse(spec.Q_template) + noise * I))

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)

        sigma_name = string(v.sigma)
        rho_name = string(v.rho)
        struct_raw_name = string(v.struct)
        iid_name = string(v.iid)

        if isempty(_find_parameter_new(p_names, var, "sigma", k))
            @warn "Parameters for BYM2 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_latent, n_samples))
            push!(unstructured_effects, zeros(Float64, n_latent, n_samples))
            push!(noisy_effects, zeros(Float64, n_latent, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]
        struct_raw_samples = get_params_vector(chain, struct_raw_name, n_latent)
        iid_samples = get_params_vector(chain, iid_name, n_latent)

        struct_effect = zeros(Float64, n_latent, n_samples)
        unstruct_effect = zeros(Float64, n_latent, n_samples)

        for s in 1:n_samples
            # Apply the non-centered parameterization transformation.
            struct_latent = F.U \ struct_raw_samples[s, :]
            
            # Enforce sum-to-zero constraint and scale to unit variance.
            struct_latent_centered = struct_latent .- mean(struct_latent)
            struct_latent_scaled = struct_latent_centered ./ (std(struct_latent_centered) + 1e-9)

            # Combine components using the Riebler parameterization.
            struct_effect[:, s] = (sqrt(rho_samples[s]) .* struct_latent_scaled) .* sigma_samples[s]
            unstruct_effect[:, s] = (sqrt(1.0 - rho_samples[s]) .* iid_samples[s, :]) .* sigma_samples[s]
        end
        
        noisy_effect = struct_effect .+ unstruct_effect

        push!(structured_effects, struct_effect)
        push!(unstructured_effects, unstruct_effect)
        push!(noisy_effects, noisy_effect)
    end
    return (structured=structured_effects, unstructured=unstructured_effects, noisy=noisy_effects)
end


function extract_component(m_obj::RW1, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Specialized state-space reconstruction for RW1.
    structured_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        innov_name = string(v.innov)

        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "innov", k))
            @warn "Parameters for RW1 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, M.t_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            latent_field_raw = cumsum(innovations_samples[j, :])
            temporal_effect_k[:, j] = (latent_field_raw .- mean(latent_field_raw)) .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_component(m_obj::RW2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Specialized state-space reconstruction for RW2.
    structured_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        innov_name = string(v.innov)
        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "innov", k)); @warn "Parameters for RW2 component $(spec.key) not found."; continue; end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            latent_field_raw = Vector{Float64}(undef, M.t_N)
            if M.t_N > 0; latent_field_raw[1] = innovations_samples[j, 1]; end
            if M.t_N > 1; latent_field_raw[2] = 2*latent_field_raw[1] + innovations_samples[j, 2]; end
            for i in 3:M.t_N; latent_field_raw[i] = 2*latent_field_raw[i-1] - latent_field_raw[i-2] + innovations_samples[j, i]; end 
            temporal_effect_k[:, j] = (latent_field_raw .- mean(latent_field_raw)) .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_component(m_obj::AR1, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # This is the existing, correct state-space reconstruction for AR1.
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        rho_name = string(v.rho)
        innov_name = string(v.innov)

        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "rho", k)) || isempty(_find_parameter_new(p_names, var, "innov", k))
            @warn "Parameters for AR1 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, M.t_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            temporal_field_j = Vector{Float64}(undef, M.t_N)
            temporal_field_j[1] = innovations_samples[j, 1] / sqrt(1.0 - rho_samples[j]^2 + noise_val)
            for i in 2:M.t_N
                temporal_field_j[i] = rho_samples[j] * temporal_field_j[i-1] + innovations_samples[j, i]
            end
            temporal_effect_k[:, j] = temporal_field_j .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end 



function extract_component(m_obj::SVAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot; prefix::String="")
    structured_effects = Vector{Matrix{Float64}}()
    s_N = M.s_N
    t_N = M.t_N
    noise_val = get(M, :noise, 1e-6)
    s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
    t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx
    N_tot = length(s_idx_full)

    for k in 1:outcomes_N
        var = string(spec.key)
        prefixed_key = isempty(prefix) ? var : "$(prefix)_$(var)"
        
        v = generate_full_variable_names(spec, "univariate", k, prefix=prefix)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        innov_samples = get_params_vector(chain, string(v.innov), s_N * t_N)
        
        rho_spatial_spec = spec.hyper.rho_spatial_spec
        rho_spatial_prefix = prefixed_key * "_rho_spatial"
        
        rho_field_results = extract_component(rho_spatial_spec.component_obj, chain, M, n_samples, 1, p_names, rho_spatial_spec, nothing, M.s_N; prefix=rho_spatial_prefix)
        rho_field_untransformed_samples = rho_field_results.structured[1]
        
        rho_field_samples = tanh.(rho_field_untransformed_samples)
        
        latent_field_samples = Array{Float64, 2}(undef, s_N * t_N, n_samples)
        
        for j in 1:n_samples
            rho_s_j = rho_field_samples[:, j]
            sigma_j = sigma_samples[j]
            innov_matrix_j = reshape(innov_samples[j, :], s_N, t_N)
            
            latent_field_j = zeros(Float64, s_N, t_N)
            for s in 1:s_N
                latent_field_j[s, :] = ar1_statespace(rho_s_j[s], sigma_j, innov_matrix_j[s, :], Float64, t_N, noise_val)
            end
            latent_field_samples[:, j] = vec(latent_field_j)
        end
        
        effect_k = zeros(Float64, N_tot, n_samples)
        st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full
        for i in 1:N_tot
            effect_k[i, :] = latent_field_samples[st_idx_full[i], :]
        end
        push!(structured_effects, effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end




function extract_component(m_obj::Union{Spherical, ExponentialDecay}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # New reconstruction for specific full GP models.
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = spec.hyper.coords
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        Matrix{Float64}(coords_train)
    end
    n_obs_full = size(coords_full, 1)
    noise = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        param_name = m_obj isa Spherical ? string(v.range) : string(v.ls)

        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, m_obj isa Spherical ? "range" : "ls", k))
            @warn "Parameters for $(typeof(m_obj)) component $(spec.key) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        param_samples = get_params_vector(chain, param_name, 1)[:, 1]
        raw_samples = get_params_vector(chain, string(v.raw), n_obs_full)

        effect_k = zeros(Float64, n_obs_full, n_samples)
        dist_matrix = pairwise(Euclidean(), coords_full, dims=1)

        for j in 1:n_samples
            local K_mat
            if m_obj isa Spherical
                h = dist_matrix ./ param_samples[j]
                K_mat = zeros(Float64, size(h))
                mask = h .< 1.0
                K_mat[mask] = (sigma_samples[j]^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
            else # ExponentialDecay
                K_mat = (sigma_samples[j]^2) .* exp.(-dist_matrix ./ param_samples[j])
            end
            K_mat += (noise * I)
            
            F = cholesky(Symmetric(K_mat))
            effect_k[:, j] = F.L * raw_samples[j, :]
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end



function extract_component(m_obj::Moran, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    eigenvectors = spec.hyper.moran_eigenvectors
    n_basis = size(eigenvectors, 2)
    s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        coeffs_name = string(v.raw)

        if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "raw", k))
            @warn "Parameters for Moran component $(spec.key) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_tot, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        coeffs_samples = get_params_vector(chain, coeffs_name, n_basis)

        effect_k = zeros(Float64, N_tot, n_samples)
        for j in 1:n_samples
            scaled_coeffs = coeffs_samples[j, :] .* sigma_samples[j]
            spatial_field = eigenvectors * scaled_coeffs
            effect_k[:, j] = view(spatial_field, s_idx_full)
        end
        push!(structured_effects, effect_k)

    end
    return (structured=structured_effects, noisy=structured_effects)
end




function extract_component(m_obj::Mosaic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot; prefix::String="")
    # Purpose: Reconstructs the posterior samples for a `Mosaic` component.
    # Rationale: This function mirrors the generative logic from the code generator. It calculates
    #            the softmax-weighted combination of the posterior samples for the local expert
    #            means to reconstruct the full spatial field for both training and prediction data.
    structured_effects = Vector{Matrix{Float64}}()
    
    coords_train = hcat(M.data.s_x, M.data.s_y)
    coords_full = if !isnothing(PS) && hasproperty(PS.data, :s_x) && hasproperty(PS.data, :s_y)
        vcat(coords_train, hcat(PS.data.s_x, PS.data.s_y))
    else
        coords_train
    end
    
    centers = spec.hyper.mosaic_centers
    dist_sq = pairwise(SqEuclidean(), coords_full, centers', dims=1)
    weights = softmax(-dist_sq, dims=2)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k, prefix=prefix)
        
        mu_local_samples = get_params_vector(chain, string(v.raw), m_obj.n_regions)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        
        # Effect is weights * mu_local, scaled by sigma for each sample
        effect_k = (weights * mu_local_samples') .* sigma_samples'
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end



function extract_component(m_obj::Nystrom, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the posterior samples for a Nystrom component.
    # Rationale: This function mirrors the generative logic from the Turing model to reconstruct
    #            the full latent field from the raw MCMC samples.
    structured_effects = Vector{Matrix{Float64}}()
    
    coords_train = spec.hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end

    Z_inducing = spec.hyper.Z_inducing
    kernel_str = m_obj.kernel
    noise = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = string(spec.key)
        
        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        ls_name = _find_parameter_new(p_names, var, "ls", k)
        v_latent_name = _find_parameter_new(p_names, var, "v_latent", k)

        if isempty(sigma_name) || isempty(ls_name) || isempty(v_latent_name)
            @warn "Parameters for Nystrom component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, size(coords_full, 1), n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_samples = get_params_vector(chain, ls_name, 1)[:, 1]
        v_latent_samples = get_params_vector(chain, v_latent_name, m_obj.n_inducing)

        effect_k = zeros(Float64, size(coords_full, 1), n_samples)
        for j in 1:n_samples
            K_UU = evaluate_kernel_matrix(Z_inducing, sigma_samples[j], ls_samples[j], Symbol(kernel_str), noise)
            K_XU = evaluate_cross_kernel_matrix(coords_full, Z_inducing, sigma_samples[j], ls_samples[j], Symbol(kernel_str))
            L_UU = cholesky(Symmetric(K_UU)).L
            
            effect_k[:, j] = K_XU * (L_UU' \ v_latent_samples[j, :])
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end


function extract_component(m_obj::LGCP, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    log_intensity_surfaces = Vector{Array{Float64, 3}}() # [s_N, t_N, n_samples]

    key = string(spec.key)
    is_spatiotemporal = hasproperty(spec.hyper, :temporal_spec)
    noise = get(M, :noise, 1e-6)

    eta_base = zeros(Float64, N_tot, n_samples)
    if haskey(M, :fixed_effects_posterior)
        eta_base .+= M.fixed_effects_posterior
    end
    if haskey(M, :intercept_posterior)
        eta_base .+= M.intercept_posterior
    end

    for k in 1:outcomes_N
        var = key
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        
        n_latent_dims = is_spatiotemporal ? M.s_N * M.t_N : M.s_N
        raw_samples = get_params_vector(chain, string(v.raw), n_latent_dims)

        log_intensity_k = zeros(Float64, M.s_N, M.t_N, n_samples)

        for j in 1:n_samples
            local latent_field_st
            if is_spatiotemporal
                s_spec = spec.hyper.inner_spec
                t_spec = spec.hyper.temporal_spec
                C_s = cholesky(Symmetric(s_spec.Q_template + noise * I))
                C_t = cholesky(Symmetric(t_spec.Q_template + noise * I))
                Z_matrix = reshape(raw_samples[j, :], M.s_N, M.t_N)
                tmp_spatial = C_s.U \ Z_matrix
                latent_field_st = (transpose(C_t.U \ transpose(tmp_spatial))) .* sigma_samples[j]
            else
                Q_lgcp = spec.hyper.inner_spec.Q_template
                F_lgcp = cholesky(Symmetric(Q_lgcp + noise * I))
                spatial_component = (F_lgcp.U \ raw_samples[j, :]) .* sigma_samples[j]
                latent_field_st = repeat(spatial_component, 1, M.t_N)
            end

            log_intensity_surface = zeros(Float64, M.s_N, M.t_N)
            for t in 1:M.t_N, s in 1:M.s_N
                obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:M.y_N)
                base_contribution = isempty(obs_indices) ? 0.0 : mean(view(eta_base, obs_indices, j))
                log_intensity_surface[s, t] = base_contribution + latent_field_st[s, t]
            end
            log_intensity_k[:, :, j] = log_intensity_surface
        end
        push!(log_intensity_surfaces, log_intensity_k)
    end
    
    return (structured=log_intensity_surfaces, noisy=log_intensity_surfaces)
end




function extract_component(m_obj::AR2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Specialized state-space reconstruction for AR2, now handling prediction sets.
    structured_effects = Vector{Matrix{Float64}}()
    t_N_train = M.t_N
    t_N_full = if isnothing(PS); t_N_train; else; max(maximum(M.t_idx), maximum(PS.t_idx)); end

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_name = string(v.sigma)
        rho1_name = string(v.rho1)
        rho2_name = string(v.rho2)
        innov_name = string(v.innov)

        if isempty(_find_parameter_new(p_names, var, "sigma", k))
            @warn "Parameters for AR2 component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, t_N_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho1_samples = get_params_vector(chain, rho1_name, 1)[:, 1]
        rho2_samples = get_params_vector(chain, rho2_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, t_N_train)

        temporal_effect_k = zeros(Float64, t_N_full, n_samples)
        for j in 1:n_samples
            temporal_field_j = Vector{Float64}(undef, t_N_full)
            
            # Reconstruct training part
            if t_N_train > 0; temporal_field_j[1] = innovations_samples[j, 1]; end
            if t_N_train > 1; temporal_field_j[2] = rho1_samples[j] * temporal_field_j[1] + innovations_samples[j, 2]; end
            for i in 3:t_N_train
                temporal_field_j[i] = rho1_samples[j] * temporal_field_j[i-1] + rho2_samples[j] * temporal_field_j[i-2] + innovations_samples[j, i]
            end

            # Predict future part by sampling new innovations
            if t_N_full > t_N_train
                new_innovs = randn(t_N_full - t_N_train)
                for i in (t_N_train + 1):t_N_full
                    temporal_field_j[i] = rho1_samples[j] * temporal_field_j[i-1] + rho2_samples[j] * temporal_field_j[i-2] + new_innovs[i - t_N_train]
                end
            end
            
            temporal_effect_k[:, j] = temporal_field_j .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end



function extract_component(m_obj::Union{PSpline, BSpline, TPS, FFT, Wavelet, Barycentric}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    basis_key = Symbol(spec.var)
    
    if !haskey(M.basis_matrices, basis_key)
        @warn "Basis matrix for smooth component $(basis_key) not found. Returning zero-matrices."
        return (structured=[zeros(Float64, N_tot, n_samples)], noisy=[zeros(Float64, N_tot, n_samples)], coefficients=[zeros(Float64, 1, n_samples)])
    end

    B_mat_train = M.basis_matrices[basis_key]
    B_mat_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, basis_key)
        vcat(B_mat_train, PS.basis_matrices[basis_key])
    else
        B_mat_train
    end
    n_basis_cols = size(B_mat_full, 2)

    structured_effects = Vector{Matrix{Float64}}()
    coefficient_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var = string(spec.key)
        
        # The coefficients are stored in the 'latent' parameter for basis models.
        coeffs_name = _find_parameter_new(p_names, var, "latent", k)
        
        if isempty(coeffs_name)
            push!(structured_effects, zeros(Float64, size(B_mat_full, 1), n_samples))
            push!(coefficient_effects, zeros(Float64, n_basis_cols, n_samples))
            continue
        end

        coeffs = get_params_vector(chain, coeffs_name, n_basis_cols)
        
        push!(coefficient_effects, coeffs')

        effect = B_mat_full * coeffs'
        push!(structured_effects, effect)
    end

    return (structured=structured_effects, noisy=structured_effects, coefficients=coefficient_effects)
end



"""
    extract_component(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for a `Harmonic` component, now supporting multiple,
independently estimated periods.

# Rationale
This version is updated to align with the refactored `Harmonic` component. It now
fetches the posterior samples for the vector of `period` parameters and reconstructs
the total harmonic effect by summing the contributions of each harmonic component,
each with its own period.
"""
function extract_component(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    nharmonics = m_obj.nharmonics
    
    # Reconstruct the matrix of period samples.
    local period_samples_matrix
    if m_obj.period isa Real
        period_samples_matrix = fill(m_obj.period, n_samples, nharmonics)
    elseif m_obj.period isa UnivariateDistribution
        period_name = _find_parameter_new(p_names, string(spec.key), "period", nothing)
        period_samps_vec = isempty(period_name) ? fill(mean(m_obj.period), n_samples) : get_params_vector(chain, period_name, 1)[:, 1]
        period_samples_matrix = repeat(period_samps_vec, 1, nharmonics)
    elseif m_obj.period isa Vector
        period_samples_matrix = zeros(Float64, n_samples, nharmonics)
        for i in 1:nharmonics
            period_name_i = _find_parameter_new(p_names, string(spec.key), "period_$(i)", nothing)
            if !isempty(period_name_i)
                period_samples_matrix[:, i] = get_params_vector(chain, period_name_i, 1)[:, 1]
            else
                @warn "Period parameter for harmonic $(i) not found. Using prior mean."
                period_samples_matrix[:, i] .= mean(m_obj.period[i])
            end
        end
    end

    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        
        amplitude_raw_name = string(v.amplitude) * "_raw"
        phase_name = string(v.phase)
        
        if isempty(_find_parameter_new(p_names, var, "amplitude_raw", k)) || isempty(_find_parameter_new(p_names, var, "phase", k))
            @warn "Parameters for Harmonic component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_tot, n_samples))
            continue
        end

        amplitude_raw_samples = get_params_vector(chain, amplitude_raw_name, nharmonics)
        phase_samples = get_params_vector(chain, phase_name, nharmonics)

        total_effect = zeros(Float64, N_tot, n_samples)
        for s in 1:n_samples
            amplitudes_s = abs.(amplitude_raw_samples[s, :])
            phases_s = phase_samples[s, :]
            periods_s = period_samples_matrix[s, :]
            
            harmonic_effect_s = zeros(Float64, N_tot)
            for m in 1:nharmonics
                phase_rad = 2.0 * pi * phases_s[m]
                angle = (2.0 * pi / periods_s[m]) .* u_idx_full
                
                beta_cos = amplitudes_s[m] * cos(phase_rad)
                beta_sin = amplitudes_s[m] * sin(phase_rad)
                
                harmonic_effect_s .+= beta_cos .* cos.(angle) .+ beta_sin .* sin.(angle)
            end
            total_effect[:, s] = harmonic_effect_s
        end
        push!(structured_effects, total_effect)
    end

    return (structured=structured_effects, noisy=structured_effects)
end



"""
    extract_component(m_obj::RFF, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for an `RFF` component.

# Rationale
This function provides the necessary posterior reconstruction logic for the `RFF` model,
mirroring the generative process defined in its code generator. It ensures that results
from `rff` models can be correctly analyzed and visualized.
"""
function extract_component(m_obj::RFF, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    coords_train = spec.hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)
    in_dims = size(coords_full, 2)
    n_features = m_obj.n_features

    # Shared parameters
    var = string(spec.key)
    W_name = "$(var)_W"
    b_name = "$(var)_b"
    
    if isempty(_find_parameter_new(p_names, var, "W", nothing))
        @warn "Shared parameters for RFF component $(spec.key) not found. Returning zero-matrix."
        for k in 1:outcomes_N
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
        end
        return (structured=structured_effects, noisy=structured_effects)
    end

    W_samples = get_params_vector(chain, W_name, in_dims * n_features)
    b_samples = get_params_vector(chain, b_name, n_features)

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, "univariate", k)
        
        beta_name = string(v.innov)
        sigma_name = string(v.sigma)
        ls_name = string(v.ls)

        if isempty(_find_parameter_new(p_names, var, "innov", k))
             @warn "Parameters for RFF component $(spec.key) outcome $k not found. Returning zero-matrix."
             push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
             continue
        end

        beta_samples = get_params_vector(chain, beta_name, n_features)
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        ls_samples = get_params_vector(chain, ls_name, 1)

        effect_samples = Array{Float64, 2}(undef, n_obs_full, n_samples)
        
        for i in 1:n_samples
            # Reconstruct RFF parameters for this sample
            W_matrix = reshape(W_samples[i, :], in_dims, n_features) ./ ls_samples[i, 1]
            b_vec = b_samples[i, :]
            
            # Compute the RFF feature matrix Phi
            Phi = sqrt(2.0 / n_features) .* cos.((coords_full * W_matrix) .+ b_vec')
            
            # Scale the coefficients
            scaled_beta = beta_samples[i, :] .* sigma_samples[i, 1]
            
            # Compute the final effect
            effect_samples[:, i] = Phi * scaled_beta
        end
        push!(structured_effects, effect_samples)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end



function extract_component(m_obj::SVCComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot) 
    cov_var = m_obj.covariate
    is_intercept = (string(cov_var) == "1" || string(cov_var) == "intercept()")

    local x_svc_full
    if is_intercept
        x_svc_full = ones(Float64, N_tot)
    else
        if !hasproperty(M.data, cov_var)
            @warn "Covariate $(cov_var) for SVCComponent not found. Returning zero-matrices."
            return (structured=[zeros(Float64, N_tot, n_samples)], noisy=[zeros(Float64, N_tot, n_samples)])
        end
        x_svc_train = M.data[!, cov_var]
        x_svc_full = if !isnothing(PS) && hasproperty(PS.data, cov_var)
            vcat(x_svc_train, PS.data[!, cov_var])
        else
            x_svc_train
        end
    end

    s_idx_full = if !isnothing(PS)
        vcat(M.s_idx, PS.s_idx)
    else
        M.s_idx
    end
    
    inner_model = m_obj.model
    inner_spec = (key=spec.key, structure=:spatial, var=spec.var, component_obj=inner_model)
    inner_effects = extract_component(inner_model, chain, M, n_samples, outcomes_N, p_names, inner_spec, PS, N_tot)

    structured_effects = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        spatial_field_k = inner_effects.structured[k] # This is [s_N x n_samples]
        
        effect_k = zeros(Float64, N_tot, n_samples)
        for j in 1:n_samples
            spatial_field_j = view(spatial_field_k, :, j)
            effect_k[:, j] = view(spatial_field_j, s_idx_full) .* x_svc_full
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end



"""
    extract_component(m_obj::Union{GP, Kriging}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for a full Gaussian Process (`GP`) or `Kriging` component.
This function handles both in-sample and out-of-sample prediction by reconstructing the dense
covariance matrix for the combined set of coordinates and applying the Cholesky decomposition
to the raw innovations from the MCMC chain.
"""
function extract_component(m_obj::Union{GP, Kriging}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = spec.Q_template
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        Matrix{Float64}(coords_train)
    end
    n_obs_full = size(coords_full, 1)
    noise = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        ls_samples = get_params_vector(chain, string(v.ls), size(coords_full, 2))
        raw_samples = get_params_vector(chain, string(v.raw), n_obs_full)

        effect_k = zeros(Float64, n_obs_full, n_samples)
        for j in 1:n_samples
            K_mat = evaluate_kernel_matrix(coords_full, sigma_samples[j], ls_samples[j, :], Symbol(m_obj.kernel), noise)
            F_gp = cholesky(Symmetric(K_mat))
            effect_k[:, j] = F_gp.L * raw_samples[j, :]
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 



function extract_component(m_obj::DAG, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
    n_latent = size(spec.Q_template, 1)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        rho_samples = get_params_vector(chain, string(v.rho), 1)[:, 1]
        innov_samples = get_params_vector(chain, string(v.innov), n_latent)

        effect_k = zeros(Float64, N_tot, n_samples)
        W_dag = spec.Q_template

        for j in 1:n_samples
            latent_field = zeros(Float64, n_latent)
            for i in 1:n_latent
                parent_effect = 0.0
                for j_ptr in nzrange(W_dag, i)
                    parent_idx = W_dag.rowval[j_ptr]
                    parent_effect += W_dag.nzval[j_ptr] * latent_field[parent_idx]
                end
                latent_field[i] = rho_samples[j] * parent_effect + innov_samples[j, i]
            end
            latent_field .*= sigma_samples[j]
            effect_k[:, j] = view(latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end



"""
    extract_component(m_obj::LocalAdaptive, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for a `LocalAdaptive` spatial component. This function
reconstructs the cluster-specific means and the non-zero mean Leroux GMRF field.
"""
function extract_component(m_obj::LocalAdaptive, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
    n_latent = size(spec.Q_template, 1)
    n_clusters = spec.hyper.n_clusters
    cluster_assignments = M.cluster_assignments
    noise = M.noise

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        rho_samples = get_params_vector(chain, string(v.rho), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(v.raw), n_latent)
        mu_clusters_raw_samples = get_params_vector(chain, string(v.innov), n_clusters)

        effect_k = zeros(Float64, N_tot, n_samples)
        
        for j in 1:n_samples
            # Re-center cluster means
            mu_clusters_centered = mu_clusters_raw_samples[j, :] .- mean(mu_clusters_raw_samples[j, :])
            mean_vector = mu_clusters_centered[cluster_assignments]
            
            # Recompose precision and solve
            Q_final = recompose_precision(:localadaptive, spec.Q_template, 1.0; extra_param=rho_samples[j])
            F = cholesky(Symmetric(Q_final + noise * I))
            latent_field_centered_part = F.U \ raw_samples[j, :]
            
            latent_field = (mean_vector .+ latent_field_centered_part) .* sigma_samples[j]
            
            effect_k[:, j] = view(latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end



"""
    extract_component(m_obj::NetworkFlow, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for a `NetworkFlow` component. This function mirrors
the SAR-like logic from the model generator, handling directed or bidirectional flows.
"""
function extract_component(m_obj::NetworkFlow, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
    n_latent = size(spec.Q_template, 1)
    noise = M.noise

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
        rho_samples = get_params_vector(chain, string(v.rho), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(v.raw), n_latent)

        effect_k = zeros(Float64, N_tot, n_samples)
        W_net = spec.Q_template

        for j in 1:n_samples
            L_op = if m_obj.flow_direction == :upstream
                I(n_latent) - rho_samples[j] .* W_net'
            elseif m_obj.flow_direction == :downstream
                I(n_latent) - rho_samples[j] .* W_net
            else # :bidirectional
                W_symm = sparse((W_net + W_net') .> 0)
                I(n_latent) - rho_samples[j] .* W_symm
            end
            
            Q_final = Symmetric(L_op' * L_op + noise * I)
            F = cholesky(Q_final)
            
            latent_field = (F.U \ raw_samples[j, :]) .* sigma_samples[j]
            effect_k[:, j] = view(latent_field, s_idx_full)
        end
        push!(structured_effects, effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end


function extract_component(m_obj::Union{FITC, SVGP}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    coord_vars = get(spec.params, :positional_args, [])
    coords_train = haskey(spec.params, :coords) ? spec.params.coords : Matrix{Float64}(M.data[!, Symbol.(coord_vars)])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        Matrix{Float64}(coords_train)
    end
    n_obs_full = size(coords_full, 1)

    for k in 1:outcomes_N
        var = string(spec.key)
        
        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        ls_name = _find_parameter_new(p_names, var, "ls", k)
        u_raw_name = _find_parameter_new(p_names, var, "u_raw", k)
        f_innov_name = _find_parameter_new(p_names, var, "f_raw", k)
        
        n_inducing = m_obj.n_inducing
        Z_inducing = spec.params.Z_inducing

        kernel_str = m_obj.kernel
        kernel = get_kernel_from_string(kernel_str)
        noise = get(M, :noise, 1e-6)

        if isempty(sigma_name) || isempty(ls_name) || isempty(u_raw_name) || isempty(f_innov_name)
            @warn "Parameters for GP component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        ls_samples = get_params_vector(chain, ls_name, 1)
        u_raw_samples = get_params_vector(chain, u_raw_name, n_inducing)
        f_innov_samples = get_params_vector(chain, f_innov_name, n_obs_full)

        gp_effect_k = zeros(Float64, n_obs_full, n_samples)

        for j in 1:n_samples
            sigma_j = sigma_samples[j, 1]
            ls_j = ls_samples[j, 1]
            u_raw_j = u_raw_samples[j, :]
            f_innov_j = f_innov_samples[j, 1:n_obs_full]

            kernel_scaled = sigma_j^2 * (kernel ∘ ScaleTransform(1.0 / ls_j))
            
            K_uu = kernelmatrix(kernel_scaled, RowVecs(Z_inducing)) + noise * I
            K_uf = kernelmatrix(kernel_scaled, RowVecs(Z_inducing), RowVecs(coords_full))
            k_ff_diag = diag(kernelmatrix(kernel_scaled, RowVecs(coords_full)))

            L_uu = cholesky(Symmetric(K_uu)).L
            u_latent = L_uu * u_raw_j

            A = (L_uu') \ K_uf
            mean_f = A' * u_latent
            var_f = k_ff_diag - vec(sum(A.^2, dims=1))

            gp_effect_k[:, j] = mean_f + sqrt.(max.(var_f, 0.0) .+ noise) .* f_innov_j
        end
        push!(structured_effects, gp_effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end



function extract_component(m_obj::DynamicsComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the DynamicsComponent.
    # Rationale: This version adds a new branch to handle the multivariate `delay_difference` model,
    #            ensuring its joint dynamics are correctly reconstructed during posterior simulation.
    #            It also restores the full logic for other biological and physical models.
    # v1.1.8 (2026-07-31)
    model_type = m_obj.model
    key = string(spec.key)
     
    if model_type == "leslie_matrix"
        if outcomes_N <= 1; error("Leslie matrix reconstruction requires a multivariate model."); end

        n_age_classes = get(m_obj.params, :n_age_classes, outcomes_N)
        
        sigma_samples = get_params_vector(chain, "sigma_process_$(key)", n_age_classes)
        innov_samples = get_params_vector(chain, "innov_process_$(key)", M.s_N * M.t_N * n_age_classes)
        areas = spec.hyper.areas
        noise = get(M, :noise, 1e-6)
        L = spec.hyper.L_template
        F_spatial = cholesky(Symmetric(L + noise * I))

        local K_samples, survival_samples, fecundity_samples
        if spatially_varying_K; sigma_K_samples = get_params_vector(chain, "sigma_K_$(key)", 1)[:, 1]; log_K_mean_samples = get_params_vector(chain, "log_K_mean_$(key)", 1)[:, 1]; K_raw_samples = get_params_vector(chain, "K_raw_$(key)", M.s_N); else K_samples = get_params_vector(chain, "K_$(key)", 1)[:, 1]; end
        if spatially_varying_rates; log_fecundity_mean_samples = get_params_vector(chain, "log_fecundity_mean_$(key)", n_age_classes); sigma_fecundity_samples = get_params_vector(chain, "sigma_fecundity_$(key)", n_age_classes); fecundity_raw_samples = get_params_vector(chain, "fecundity_raw_$(key)", M.s_N * n_age_classes); logit_survival_mean_samples = get_params_vector(chain, "logit_survival_mean_$(key)", n_age_classes - 1); sigma_survival_samples = get_params_vector(chain, "sigma_survival_$(key)", n_age_classes - 1); survival_raw_samples = get_params_vector(chain, "survival_raw_$(key)", M.s_N * (n_age_classes - 1)); else survival_samples = get_params_vector(chain, "survival_rates_$(key)", n_age_classes - 1); fecundity_samples = get_params_vector(chain, "fecundity_rates_$(key)", n_age_classes); end

        age_class_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:n_age_classes]
        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

        for j in 1:n_samples
            local K_values, survival_rates_spatial, fecundity_rates_spatial
            if spatially_varying_K; K_field_raw = F_spatial.U \ K_raw_samples[j, :]; K_field_raw .-= mean(K_field_raw); K_values = exp.(log_K_mean_samples[j] .+ K_field_raw .* sigma_K_samples[j]); else K_values = fill(K_samples[j], M.s_N); end
            if spatially_varying_rates
                fecundity_raw_matrix = reshape(fecundity_raw_samples[j, :], M.s_N, n_age_classes)
                fecundity_field = F_spatial.U \ fecundity_raw_matrix
                fecundity_rates_spatial = exp.(log_fecundity_mean_samples[j, :]' .+ fecundity_field .* sigma_fecundity_samples[j, :]')
                
                survival_raw_matrix = reshape(survival_raw_samples[j, :], M.s_N, n_age_classes - 1)
                survival_field = F_spatial.U \ survival_raw_matrix
                survival_rates_spatial = logistic.(logit_survival_mean_samples[j, :]' .+ survival_field .* sigma_survival_samples[j, :]')
            end

            innov_tensor_j = reshape(innov_samples[j, :], M.s_N, M.t_N, n_age_classes)
            sigma_j = sigma_samples[j, :]
            population_field_j = zeros(Float64, M.s_N, M.t_N, n_age_classes)
            for a in 1:n_age_classes; population_field_j[:, 1, a] = max.(0.0, innov_tensor_j[:, 1, a] .* sigma_j[a]); end

            for s in 1:M.s_N
                L_s = zeros(Float64, n_age_classes, n_age_classes)
                if spatially_varying_rates
                    for i in 1:(n_age_classes-1); L_s[i+1, i] = survival_rates_spatial[s, i]; end
                    L_s[1, :] = fecundity_rates_spatial[s, :]
                else
                    for i in 1:(n_age_classes-1); L_s[i+1, i] = survival_samples[j, i]; end
                    L_s[1, :] = fecundity_samples[j, :]
                end

                for t in 2:M.t_N
                    N_prev = view(population_field_j, s, t-1, :)
                    L_effective = copy(L_s)
                    if spatially_varying_K || haskey(m_obj.params, :K)
                        total_pop_prev = sum(N_prev); K_density = K_values[s] / areas[s]; dd_factor = max(0.0, 1.0 - (total_pop_prev / areas[s]) / K_density); L_effective[1, :] .*= dd_factor
                    end
                    N_projected = L_effective * N_prev
                    current_innov = view(innov_tensor_j, s, t, :) .* sigma_j
                    population_field_j[s, t, :] = max.(0.0, N_projected .+ current_innov)
                end
            end
            
            for a in 1:n_age_classes; log_pop_field = log.(view(population_field_j, :, :, a) .+ 1e-6); for i in 1:N_tot; age_class_effects[a][i, j] = log_pop_field[s_idx_full[i], t_idx_full[i]]; end; end
        end
        return (structured=age_class_effects, noisy=age_class_effects)

    elseif model_type == "delay_difference" && M.outcomes_N > 1
        # Multivariate delay_difference reconstruction
        r_samples = get_params_vector(chain, "r_$(key)", 1)[:, 1]
        K_samples = get_params_vector(chain, "K_$(key)", 1)[:, 1]
        M_nat_samples = get_params_vector(chain, "M_nat_$(key)", 1)[:, 1]
        sigma_rec_samples = get_params_vector(chain, "sigma_recruitment_$(key)", 1)[:, 1]
        sigma_pop_samples = get_params_vector(chain, "sigma_population_$(key)", 1)[:, 1]
        innov_rec_samples = get_params_vector(chain, "innov_recruitment_$(key)", M.s_N * M.t_N)
        innov_pop_samples = get_params_vector(chain, "innov_population_$(key)", M.s_N * M.t_N)
        
        areas = spec.hyper.areas
        
        local effort_matrix, q_samples, catch_data_matrix
        if haskey(m_obj.params, :effort_col)
            effort_col_sym = m_obj.params[:effort_col]
            effort_matrix = spec.hyper.processed_params[effort_col_sym]
            q_samples = get_params_vector(chain, "q_$(key)", 1)[:, 1]
        else
            catch_data_col_sym = m_obj.params[:catch_data_col]
            catch_data_matrix = spec.hyper.processed_params[catch_data_col_sym]
        end

        pop_effects = zeros(Float64, N_tot, n_samples)
        rec_effects = zeros(Float64, N_tot, n_samples)
        
        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

        for j in 1:n_samples
            innov_rec_matrix = reshape(innov_rec_samples[j, :], M.s_N, M.t_N)
            innov_pop_matrix = reshape(innov_pop_samples[j, :], M.s_N, M.t_N)
            
            population_field = zeros(M.s_N, M.t_N)
            recruitment_field = zeros(M.s_N, M.t_N)
            population_field[:, 1] = max.(0.0, innov_pop_matrix[:, 1] .* sigma_pop_samples[j])
            recruitment_field[:, 1] = max.(0.0, innov_rec_matrix[:, 1] .* sigma_rec_samples[j])

            for t in 2:M.t_N
                N_prev = population_field[:, t-1]
                D_prev = N_prev ./ areas
                K_density = K_samples[j] ./ areas
                
                mean_recruitment = r_samples[j] .* D_prev .* (1.0 .- D_prev ./ K_density) .* areas
                recruitment_field[:, t] = exp.(log.(mean_recruitment .+ 1e-6) .+ innov_rec_matrix[:, t] .* sigma_rec_samples[j])
                
                C_prev = haskey(m_obj.params, :effort_col) ? (q_samples[j] .* effort_matrix[:, t-1] .* N_prev) : catch_data_matrix[:, t-1]
                N_survived = (N_prev .- C_prev) .* exp.(-M_nat_samples[j])
                population_field[:, t] = max.(0.0, N_survived .+ recruitment_field[:, t] .+ innov_pop_matrix[:, t] .* sigma_pop_samples[j])
            end

            for i in 1:N_tot
                s_i, t_i = s_idx_full[i], t_idx_full[i]
                pop_effects[i, j] = log(population_field[s_i, t_i] + 1e-6)
                rec_effects[i, j] = log(recruitment_field[s_i, t_i] + 1e-6)
            end
        end
        return (structured=[pop_effects, rec_effects], noisy=[pop_effects, rec_effects])
            elseif model_type == "generalized_lotka_volterra"
        n_species = M.outcomes_N
        spatially_varying_K = get(m_obj.params, :spatially_varying_K, false)
        
        r_samples = get_params_vector(chain, "r_$(key)", n_species)
        alpha_raw_samples = get_params_vector(chain, "alpha_raw_$(key)", n_species * (n_species - 1))
        sigma_process_samples = get_params_vector(chain, "sigma_process_$(key)", n_species)
        innov_process_samples = get_params_vector(chain, "innov_process_$(key)", M.s_N * M.t_N * n_species)
        
        areas = spec.hyper.areas
        noise = get(M, :noise, 1e-6)
        L = spec.hyper.L_template
        F_spatial = cholesky(Symmetric(L + noise * I))

        local K_samples
        if spatially_varying_K
            log_K_mean_samples = get_params_vector(chain, "log_K_mean_$(key)", n_species)
            sigma_K_samples = get_params_vector(chain, "sigma_K_$(key)", n_species)
            K_raw_samples = get_params_vector(chain, "K_raw_$(key)", M.s_N * n_species)
        else
            K_samples = get_params_vector(chain, "K_$(key)", n_species)
        end

        species_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:n_species]
        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

        for j in 1:n_samples
            alpha_j = diagm(0 => ones(n_species))
            off_diag_indices = [i for i in 1:(n_species^2) if mod(i-1, n_species+1) != 0]
            alpha_j[off_diag_indices] = alpha_raw_samples[j, :]

            local K_values_j
            if spatially_varying_K
                K_raw_matrix = reshape(K_raw_samples[j, :], M.s_N, n_species)
                K_field = F_spatial.U \ K_raw_matrix
                K_values_j = exp.(log_K_mean_samples[j, :]' .+ K_field .* sigma_K_samples[j, :]')
            else
                K_values_j = repeat(K_samples[j, :]', M.s_N, 1)
            end

            innov_tensor_j = reshape(innov_process_samples[j, :], M.s_N, M.t_N, n_species)
            population_field_j = zeros(M.s_N, M.t_N, n_species)
            population_field_j[:, 1, :] = max.(0.0, innov_tensor_j[:, 1, :] .* sigma_process_samples[j, :]')

            for s in 1:M.s_N, t in 2:M.t_N
                N_prev = view(population_field_j, s, t-1, :)
                D_prev = N_prev ./ areas[s]
                K_density = K_values_j[s, :] ./ areas[s]
                
                N_intermediate = zeros(n_species)
                for i in 1:n_species
                    interaction_sum_density = dot(alpha_j[i, :], D_prev)
                    growth_density = r_samples[j, i] * D_prev[i] * (1.0 - interaction_sum_density / K_density[i])
                    N_intermediate[i] = N_prev[i] + growth_density * areas[s]
                end
                
                current_innov = view(innov_tensor_j, s, t, :) .* sigma_process_samples[j, :]
                population_field_j[s, t, :] = max.(0.0, N_intermediate .+ current_innov)
            end

            for i in 1:n_species
                log_pop_field = log.(view(population_field_j, :, :, i) .+ 1e-6)
                for obs_idx in 1:N_tot
                    species_effects[i][obs_idx, j] = log_pop_field[s_idx_full[obs_idx], t_idx_full[obs_idx]]
                end
            end
        end
        return (structured=species_effects, noisy=species_effects)

    end

    # Fallback to existing logic for other dynamics models
    structured_effects = Vector{Matrix{Float64}}()
    L = spec.hyper.L_template
    A = spec.hyper.A_template
    spatially_varying_K = get(m_obj.params, :spatially_varying_K, false)
    spatially_varying_r = get(m_obj.params, :spatially_varying_r, false)
    has_propagator = model_type in ["advection", "diffusion", "advection_diffusion"]
    noise = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = key
        v = generate_full_variable_names(spec, "univariate", k)
        
        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx
        effect_k = zeros(Float64, N_tot, n_samples)

        if has_propagator
            sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
            innov_samples = get_params_vector(chain, string(v.innov), M.s_N * M.t_N)
            vel_samples = model_type in ["advection", "advection_diffusion"] ? get_params_vector(chain, string(v.velocity), 1)[:, 1] : nothing
            diff_samples = model_type in ["diffusion", "advection_diffusion"] ? get_params_vector(chain, string(v.diffusion), 1)[:, 1] : nothing

            for j in 1:n_samples
                dyn_field = zeros(Float64, M.s_N, M.t_N)
                innov_matrix = reshape(innov_samples[j, :], M.s_N, M.t_N)
                
                local propagator
                if model_type == "diffusion"; op_matrix = Symmetric(I(M.s_N) - diff_samples[j] * L + noise * I); propagator = cholesky(op_matrix);
                elseif model_type == "advection"; op_matrix = I(M.s_N) - vel_samples[j] * A + noise * I; propagator = lu(op_matrix);
                else; op_matrix = I(M.s_N) - vel_samples[j] * A - diff_samples[j] * L + noise * I; propagator = lu(op_matrix); end

                dyn_field[:, 1] = innov_matrix[:, 1]
                for t in 2:M.t_N; dyn_field[:, t] = (propagator \ dyn_field[:, t-1]) + innov_matrix[:, t]; end
                dyn_field .*= sigma_samples[j]
                for i in 1:N_tot; effect_k[i, j] = dyn_field[s_idx_full[i], t_idx_full[i]]; end
            end
            push!(structured_effects, effect_k)

        elseif model_type in ["logistic_basic", "logistic_exploitation", "delay_difference", "leslie_logistic"]
            sigma_samples = get_params_vector(chain, string(v.sigma), 1)[:, 1]
            innov_samples = get_params_vector(chain, string(v.innov), M.s_N * M.t_N)
            areas = spec.hyper.areas

            local K_samples, r_samples
            if spatially_varying_K; sigma_K_samples = get_params_vector(chain, _find_parameter_new(p_names, key, "sigma_K", k), 1)[:, 1]; log_K_mean_samples = get_params_vector(chain, _find_parameter_new(p_names, key, "log_K_mean", k), 1)[:, 1]; K_raw_samples = get_params_vector(chain, _find_parameter_new(p_names, key, "K_raw", k), M.s_N); F_K = cholesky(Symmetric(L + noise * I)); else K_samples = get_params_vector(chain, string(v.K), 1)[:, 1]; end
            if spatially_varying_r; sigma_r_samples = get_params_vector(chain, _find_parameter_new(p_names, key, "sigma_r", k), 1)[:, 1]; log_r_mean_samples = get_params_vector(chain, _find_parameter_new(p_names, key, "log_r_mean", k), 1)[:, 1]; r_raw_samples = get_params_vector(chain, _find_parameter_new(p_names, key, "r_raw", k), M.s_N); F_r = cholesky(Symmetric(L + noise * I)); elseif model_type != "leslie_logistic"; r_samples = get_params_vector(chain, string(v.r), 1)[:, 1]; end

            q_samples = if model_type in ["logistic_exploitation", "delay_difference"] && haskey(m_obj.params, :effort_col); get_params_vector(chain, string(v.q), 1)[:, 1]; else nothing; end
            effort_col_sym = get(m_obj.params, :effort_col, nothing)
            effort_matrix = if !isnothing(effort_col_sym); spec.hyper.processed_params[effort_col_sym]; else nothing; end
            catch_data_col_sym = get(m_obj.params, :catch_data_col, nothing)
            catch_data_matrix = if !isnothing(catch_data_col_sym); spec.hyper.processed_params[catch_data_col_sym]; else nothing; end
            M_nat_samples = model_type == "delay_difference" ? get_params_vector(chain, string(v.M_nat), 1)[:, 1] : nothing
            survival_samples = model_type == "leslie_logistic" ? get_params_vector(chain, "survival_rates", m_obj.params[:n_age_classes] - 1) : nothing
            fecundity_samples = model_type == "leslie_logistic" ? get_params_vector(chain, "fecundity_rates", m_obj.params[:n_age_classes]) : nothing
            n_age_classes = model_type == "leslie_logistic" ? m_obj.params[:n_age_classes] : 0

            for j in 1:n_samples
                local K_values, r_values
                if spatially_varying_K; K_field_raw = F_K.U \ K_raw_samples[j, :]; K_field_raw .-= mean(K_field_raw); K_values = exp.(log_K_mean_samples[j] .+ K_field_raw .* sigma_K_samples[j]); else K_values = K_samples[j]; end
                if spatially_varying_r; r_field_raw = F_r.U \ r_raw_samples[j, :]; r_field_raw .-= mean(r_field_raw); r_values = exp.(log_r_mean_samples[j] .+ r_field_raw .* sigma_r_samples[j]); elseif model_type != "leslie_logistic"; r_values = r_samples[j]; end

                dyn_field = zeros(Float64, M.s_N, M.t_N)
                innov_matrix = reshape(innov_samples[j, :], M.s_N, M.t_N)
                dyn_field[:, 1] = innov_matrix[:, 1]

                for t in 2:M.t_N
                    N_prev = dyn_field[:, t-1]
                    D_prev = N_prev ./ areas
                    K_density = K_values ./ areas
                    
                    local growth
                    if model_type == "leslie_logistic"
                        L_mat = zeros(n_age_classes, n_age_classes); for i in 1:(n_age_classes-1); L_mat[i+1, i] = survival_samples[j, i]; end; L_mat[1, :] = fecundity_samples[j, :]; r_leslie = log(maximum(abs.(eigen(L_mat).values))); growth = r_leslie .* D_prev .* (1.0 .- D_prev ./ K_density)
                    else; growth = r_values .* D_prev .* (1.0 .- D_prev ./ K_density); end
                    
                    local N_intermediate
                    if model_type == "logistic_basic" || model_type == "leslie_logistic"; N_intermediate = N_prev .+ (growth .* areas)
                    elseif model_type == "logistic_exploitation"; exploitation = q_samples[j] .* effort_matrix[:, t] .* N_prev; N_intermediate = N_prev .+ (growth .* areas) .- exploitation
                    elseif model_type == "delay_difference"; C_prev = !isnothing(effort_matrix) ? (q_samples[j] .* effort_matrix[:, t-1] .* N_prev) : catch_data_matrix[:, t-1]; N_intermediate = (N_prev .+ (growth .* areas) .- C_prev) .* exp(-M_nat_samples[j]); end
                    
                    dyn_field[:, t] = max.(0.0, N_intermediate .+ innov_matrix[:, t])
                end
                dyn_field .*= sigma_samples[j]
                for i in 1:N_tot; effect_k[i, j] = dyn_field[s_idx_full[i], t_idx_full[i]]; end
            end
            push!(structured_effects, effect_k)
        end
    end
    return (structured=structured_effects, noisy=structured_effects)
end



"""
    extract_component(m_obj::MixedComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for a `MixedComponent`, with updated logic for
structured out-of-sample prediction.

# Rationale for Update
This function has been significantly updated to address a key limitation in out-of-sample
prediction for mixed effects models. The previous implementation sampled new group levels
from an IID prior (`Normal(0, σ²)`), ignoring any spatial or temporal structure defined
by the inner model (e.g., ICAR, RW2).

The new implementation performs principled conditional prediction for GMRF-based inner
models. When a prediction set (`PS`) with a full adjacency matrix (`W_full`) is provided,
the function calculates the conditional distribution of the random effects for the new
levels, given the posterior samples of the effects at the training levels.

The logic is as follows:
1.  For each posterior sample, the full precision matrix `Q_full` for all group levels
    (training and prediction) is reconstructed using the sampled hyperparameters (e.g., `rho`).
2.  This matrix is partitioned into blocks corresponding to training-training (`Q_tt`),
    training-new (`Q_tn`), and new-new (`Q_nn`) interactions.
3.  The conditional mean and covariance for the new levels are calculated using standard
    GMRF properties:
    - `μ_cond = -inv(Q_nn) * Q_nt * effect_train`
    - `Σ_cond = inv(Q_nn) * σ²`
4.  New effects are then sampled from `MvNormal(μ_cond, Σ_cond)`.

This ensures that predictions for new levels correctly "borrow strength" from their
neighbors in the training set, respecting the learned spatial or temporal correlation
structure. If the full adjacency matrix is not provided, or if the inner model is IID,
the function gracefully falls back to the previous IID sampling method. This change
makes the model's predictive capabilities far more robust and statistically sound.
"""
function extract_component(m_obj::MixedComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    lhs_effects = m_obj.lhs
    n_terms = length(lhs_effects)
    var = string(spec.key)
    group_var_sym = m_obj.group_var
    inner_m = m_obj.model
    noise_val = get(M, :noise, 1e-6)

    # Determine all unique levels across training and prediction sets
    train_levels = unique(M.data[!, group_var_sym])
    n_groups_train = length(train_levels)
    
    all_levels = train_levels
    has_new_levels = false
    if !isnothing(PS) && hasproperty(PS.data, group_var_sym)
        pred_levels = unique(PS.data[!, group_var_sym])
        if !isempty(setdiff(pred_levels, train_levels))
            has_new_levels = true
        end
        all_levels = unique(vcat(train_levels, pred_levels))
    end
    n_all_groups = length(all_levels)
    
    # Create a map from level value to index
    level_map = Dict(level => i for (i, level) in enumerate(all_levels))
    
    # Get indices for the full dataset (train + pred)
    full_indices = if !isnothing(PS) && hasproperty(PS.data, group_var_sym)
        [level_map[v] for v in vcat(M.data[!, group_var_sym], PS.data[!, group_var_sym])]
    else
        [level_map[v] for v in M.data[!, group_var_sym]]
    end

    # --- Case 1: Simple (uncorrelated) random effects ---
    if n_terms == 1
        structured_effects = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            v = generate_full_variable_names(spec, "univariate", k)
            sigma_name = string(v.sigma)
            latent_name = string(v.raw) # Raw innovations for the training set

            if isempty(_find_parameter_new(p_names, var, "sigma", k)) || isempty(_find_parameter_new(p_names, var, "raw", k))
                @warn "Parameters for simple MixedComponent $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, n_all_groups, n_samples))
                continue
            end

            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            raw_samples_train = get_params_vector(chain, latent_name, n_groups_train)

            effect_k = zeros(Float64, n_all_groups, n_samples)
            train_level_indices = [level_map[level] for level in train_levels]
            
            # Reconstruct the latent field for training levels
            Q_train_template = spec.Q_template
            F_train = cholesky(Symmetric(Q_train_template + noise_val * I))
            latent_samples_train = (F_train.U \ raw_samples_train')' .* sigma_samples'

            effect_k[train_level_indices, :] = latent_samples_train'

            # --- Out-of-sample prediction logic ---
            if has_new_levels
                new_level_indices = setdiff(1:n_all_groups, train_level_indices)
                W_full = get(PS, :W, nothing)
                
                # Use structured prediction if possible, otherwise fall back to IID.
                if !isnothing(W_full) && inner_m isa Union{ICAR, Besag, Leroux, SAR}
                    model_type_sym = Symbol(lowercase(string(typeof(inner_m))))
                    Q_full_template = build_structure_template(model_type_sym, n_all_groups; W=W_full).matrix
                    
                    rho_samples = hasproperty(inner_m, :rho) ? get_params_vector(chain, string(v.rho), 1)[:, 1] : nothing

                    for s in 1:n_samples
                        rho_s = isnothing(rho_samples) ? nothing : rho_samples[s]
                        Q_full = recompose_precision(model_type_sym, Q_full_template, 1.0; extra_param=rho_s)
                        
                        Q_nn = Q_full[new_level_indices, new_level_indices]
                        Q_nt = Q_full[new_level_indices, train_level_indices]
                        
                        Q_nn_inv = inv(Symmetric(Matrix(Q_nn)) + noise_val * I)
                        
                        mu_cond = -Q_nn_inv * Q_nt * latent_samples_train[s, :]
                        Sigma_cond = Symmetric(Q_nn_inv) # Variance is 1 before scaling by sigma
                        
                        # Sample new effects from the conditional distribution
                        new_effects_raw = rand(MvNormal(mu_cond, Sigma_cond))
                        effect_k[new_level_indices, s] = new_effects_raw .* sigma_samples[s]
                    end
                else
                    # Fallback to IID sampling
                    if inner_m isa Union{ICAR, Besag, Leroux, SAR}; @warn "Structured prediction for MixedComponent '$(spec.key)' requires a full adjacency matrix `W` in the prediction set. Falling back to IID sampling."; end
                    for s in 1:n_samples
                        new_effects = rand(Normal(0, sigma_samples[s]), length(new_level_indices))
                        effect_k[new_level_indices, s] = new_effects
                    end
                end
            end
            push!(structured_effects, effect_k)
        end
        return (type=:simple, effects=structured_effects, lhs=lhs_effects[1], indices=full_indices)
    
    # --- Case 2: Correlated random effects ---
    else
        correlated_effects = Dict{Symbol, Vector{Matrix{Float64}}}()
        for k in 1:outcomes_N
            v = generate_full_variable_names(spec, "univariate", k)
            l_corr_name = string(v.L_corr)
            sigma_effects_name = string(v.sigma_effects)
            raw_name = string(v.raw)

            if isempty(_find_parameter_new(p_names, var, "L_corr", k)) || isempty(_find_parameter_new(p_names, var, "sigma_effects", k)) || isempty(_find_parameter_new(p_names, var, "raw", k))
                continue
            end

            l_corr_samps = get_params_vector(chain, l_corr_name, n_terms * n_terms)
            sigma_eff_samps = get_params_vector(chain, sigma_effects_name, n_terms)
            raw_samps = get_params_vector(chain, raw_name, n_groups_train * n_terms)

            recon_matrix_k_full = zeros(n_all_groups, n_terms, n_samples)

            for s in 1:n_samples
                L_corr_s = reshape(l_corr_samps[s, :], n_terms, n_terms)
                L_eff_t = (Diagonal(sigma_eff_samps[s, :]) * L_corr_s)'
                
                L_grp_inv_t = if inner_m isa IID
                    sparse(I, n_groups_train, n_groups_train)
                else
                    cholesky(Symmetric(spec.Q_template + noise_val * I)).U \ I
                end
                
                innov_mat_train = reshape(raw_samps[s, :], n_groups_train, n_terms)
                gamma_train = L_grp_inv_t * innov_mat_train
                recon_matrix_train = gamma_train * L_eff_t
                
                train_level_indices = [level_map[level] for level in train_levels]
                recon_matrix_k_full[train_level_indices, :, s] = recon_matrix_train

                # --- Out-of-sample prediction logic ---
                if has_new_levels
                    new_level_indices = setdiff(1:n_all_groups, train_level_indices)
                    n_new = length(new_level_indices)
                    gamma_new = zeros(n_new, n_terms)
                    
                    W_full = get(PS, :W, nothing)
                    if !isnothing(W_full) && inner_m isa Union{ICAR, Besag, Leroux, SAR}
                        model_type_sym = Symbol(lowercase(string(typeof(inner_m))))
                        Q_full_template = build_structure_template(model_type_sym, n_all_groups; W=W_full).matrix
                        Q_full = recompose_precision(model_type_sym, Q_full_template, 1.0; extra_param=nothing) # Assuming no rho for simplicity

                        Q_nn = Q_full[new_level_indices, new_level_indices]
                        Q_nt = Q_full[new_level_indices, train_level_indices]
                        Q_nn_inv = inv(Symmetric(Matrix(Q_nn)) + noise_val * I)
                        Sigma_cond = Symmetric(Q_nn_inv)

                        for j in 1:n_terms
                            mu_cond_j = -Q_nn_inv * Q_nt * gamma_train[:, j]
                            gamma_new[:, j] = rand(MvNormal(mu_cond_j, Sigma_cond))
                        end
                    else
                        if inner_m isa Union{ICAR, Besag, Leroux, SAR}; @warn "Structured prediction for correlated MixedComponent '$(spec.key)' requires a full adjacency matrix `W` in the prediction set. Falling back to IID sampling."; end
                        gamma_new = randn(n_new, n_terms)
                    end
                    
                    recon_matrix_new = gamma_new * L_eff_t
                    recon_matrix_k_full[new_level_indices, :, s] = recon_matrix_new
                end
            end

            for (i, term) in enumerate(lhs_effects)
                is_intercept_term = (term == "1" || term == "intercept()")
                term_key = is_intercept_term ? :intercept : Symbol("slope_$(term)")
                if !haskey(correlated_effects, term_key)
                    correlated_effects[term_key] = [zeros(0,0) for _ in 1:outcomes_N]
                end
                correlated_effects[term_key][k] = recon_matrix_k_full[:, i, :]
            end
        end
        return (type=:correlated, effects=correlated_effects, lhs=lhs_effects, indices=full_indices)
    end
end


function extract_component(m_obj::Eigen, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    key = string(spec.key)
    var = string(spec.key)
    
    n_factors = m_obj.n_factors
    n_obs_train = M.y_N
    
    factors_name = _find_parameter_new(p_names, var, "factors_flat", nothing)

    if isempty(factors_name)
        @warn "Latent factors for Eigen component $(key) not found in chain. Returning zero-matrix."
        push!(structured_effects, zeros(Float64, N_tot, n_samples))
        return (structured=structured_effects, noisy=structured_effects)
    end

    factors_samples = get_params_vector(chain, factors_name, n_obs_train * n_factors)
    
    # Reconstruct the effect that was added to eta, which is the sum of the factor scores.
    total_effect = zeros(Float64, n_obs_train, n_samples)
    for j in 1:n_samples
        F_matrix_j = reshape(factors_samples[j, :], n_obs_train, n_factors)
        total_effect[:, j] = sum(F_matrix_j, dims=2)
    end

    # Handle prediction set (PS) if provided
    if N_tot > n_obs_train
        # For out-of-sample prediction, the latent factors are unknown.
        # A common approach is to use the mean effect (which is zero for standard normal factors).
        # Here, we will pad with zeros, assuming the effect is centered.
        pred_effect = zeros(Float64, N_tot - n_obs_train, n_samples)
        total_effect = vcat(total_effect, pred_effect)
    end

    # The Eigen effect is univariate; it applies the same effect to all outcomes.
    for k in 1:outcomes_N
        push!(structured_effects, total_effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end




function extract_component(m_obj::TAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    t_N_train = M.t_N
    t_N_full = isnothing(PS) ? t_N_train : max(maximum(M.t_idx), maximum(PS.t_idx))
    
    threshold_data_train = spec.hyper.threshold_data
    threshold_data_full = if !isnothing(PS) && hasproperty(PS.data, m_obj.threshold_var)
        vcat(threshold_data_train, PS.data[!, m_obj.threshold_var])
    else
        threshold_data_train
    end
    if length(threshold_data_full) < t_N_full
        @warn "Threshold data for TAR model is shorter than the full time series. Predictions may be unreliable."
        threshold_data_full = vcat(threshold_data_full, fill(mean(threshold_data_full), t_N_full - length(threshold_data_full)))
    end

    v = generate_full_variable_names(spec, "univariate", 1)
    rho1_name = string(v.rho) * "_1"
    rho2_name = string(v.rho) * "_2"
    sigma1_name = string(v.sigma) * "_1"
    sigma2_name = string(v.sigma) * "_2"
    thresh_raw_name = string(v.raw) * "_thresh"
    innov_name = string(v.innov)

    rho1_samples = get_params_vector(chain, rho1_name, 1)[:,1]
    rho2_samples = get_params_vector(chain, rho2_name, 1)[:,1]
    sigma1_samples = get_params_vector(chain, sigma1_name, 1)[:,1]
    sigma2_samples = get_params_vector(chain, sigma2_name, 1)[:,1]
    thresh_raw_samples = get_params_vector(chain, thresh_raw_name, 1)[:,1]
    innov_samples_train = get_params_vector(chain, innov_name, t_N_train)

    mean_threshold_data = mean(threshold_data_train)
    noise = M.noise

    latent_field_samples = Array{Float64, 2}(undef, t_N_full, n_samples)
    
    for i in 1:n_samples
        threshold_level = mean_threshold_data + thresh_raw_samples[i]
        
        innov_full = vcat(innov_samples_train[i, :], randn(t_N_full - t_N_train))

        for t in 1:t_N_full
            regime_indicator = threshold_data_full[t] > threshold_level
            
            curr_rho = regime_indicator ? rho2_samples[i] : rho1_samples[i]
            curr_sigma = regime_indicator ? sigma2_samples[i] : sigma1_samples[i]
            
            if t == 1
                latent_field_samples[t, i] = (innov_full[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + noise)
            else
                latent_field_samples[t, i] = curr_rho * latent_field_samples[t-1, i] + innov_full[t] * curr_sigma
            end
        end
    end
    
    return (structured=[latent_field_samples], noisy=[latent_field_samples])
end


"""
    extract_component(m_obj::AdaptiveSmooth, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs the posterior samples for an `AdaptiveSmooth` component.

# Rationale for Correction
This version is updated to match the corrected generative logic. It now reconstructs
the adaptive basis matrix `B` for each posterior sample of the MLP weights and multiplies
it by the posterior samples of the basis coefficients `β` to compute the final smooth
effect. This ensures the reconstruction is consistent with the model's definition.
"""
function extract_component(m_obj::AdaptiveSmooth, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    coords_train = spec.hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    for k in 1:outcomes_N
        var = string(spec.key)
        v = generate_full_variable_names(spec, "univariate", k)
        
        W1_name = string(v.raw) * "_W1"
        b1_name = string(v.raw) * "_b1"
        W2_name = string(v.raw) * "_W2"
        coeffs_name = string(v.innov)
        sigma_name = string(v.sigma)

        if isempty(_find_parameter_new(p_names, var, "raw_W1", k))
             @warn "Parameters for AdaptiveSmooth component $(spec.key) not found. Returning zero-matrix."
             push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
             continue
        end

        W1_samples = get_params_vector(chain, W1_name, spec.hyper.in_dim * spec.hyper.hidden_dim)
        b1_samples = get_params_vector(chain, b1_name, spec.hyper.hidden_dim)
        W2_samples = get_params_vector(chain, W2_name, spec.hyper.hidden_dim * m_obj.nbins)
        coeffs_samples = get_params_vector(chain, coeffs_name, m_obj.nbins)
        sigma_samples = get_params_vector(chain, sigma_name, 1)

        effect_samples = Array{Float64, 2}(undef, n_obs_full, n_samples)
        
        for i in 1:n_samples
            # Reconstruct MLP weights for this sample
            W1 = reshape(W1_samples[i, :], spec.hyper.in_dim, spec.hyper.hidden_dim)
            b1 = b1_samples[i, :]
            W2 = reshape(W2_samples[i, :], spec.hyper.hidden_dim, m_obj.nbins)
            
            # Reconstruct basis coefficients for this sample
            scaled_coeffs = coeffs_samples[i, :] .* sigma_samples[i, 1]
            
            # Compute the adaptive basis matrix B for the full coordinate set
            H = tanh.(coords_full * W1 .+ b1')
            B_adaptive = H * W2
            
            # Compute the final effect
            effect_samples[:, i] = B_adaptive * scaled_coeffs
        end
        push!(structured_effects, effect_samples)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end


"""
    extract_component(m_obj::ComposedComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

Reconstructs effects from composed components, including state-space models (`|>`),
non-stationary variance models (`∘`), and spatiotemporal interactions (`⊗`).
"""
function extract_component(m_obj::ComposedComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs effects from composed components (`|>`, `∘`, `⊗`).
    # Rationale: This function was incomplete. This version correctly implements reconstruction
    #            for all supported composition types, mirroring their generative logic.
    op = m_obj.operator 
    key = string(spec.key)
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N

    if op == :composition
        # Handles non-stationary variance: base_effect * modifier_effect
        base_component_obj = m_obj.components[1]
        modifier_component_obj = m_obj.components[2]
        base_spec = spec.hyper.base_spec
        
        # Find the modifier spec in the main component list using a heuristic key match.
        modifier_key_str = replace(string(key), "composition" => "smooth")
        modifier_spec_idx = findfirst(s -> string(s.key) == modifier_key_str, M.components)
        
        if isnothing(modifier_spec_idx)
            @warn "Could not find modifier spec for composition component '$(key)'. Returning zero effect."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end
        modifier_spec = M.components[modifier_spec_idx]

        # Recursively reconstruct both components.
        base_effects_result = extract_component(base_component_obj, chain, M, n_samples, outcomes_N, p_names, base_spec, PS, N_tot)
        modifier_effects_result = extract_component(modifier_component_obj, chain, M, n_samples, outcomes_N, p_names, modifier_spec, PS, N_tot)

        all_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]
        for k in 1:outcomes_N
            # The modifier effect is the log-sigma field. We exponentiate it.
            log_sigma_field = modifier_effects_result.structured[k]
            spatially_varying_sigma = exp.(log_sigma_field)
            
            # The base effect is the underlying spatial field.
            base_field = base_effects_result.structured[k]
            
            # The final effect is the product.
            all_effects[k] = base_field .* spatially_varying_sigma
        end
        return (structured=all_effects, noisy=all_effects)
    
    elseif op == :kronecker_product && haskey(M, :model_st) && M.model_st != "none"
        # Handles spatiotemporal interactions.
        spatial_comp_obj = m_obj.components[1]
        temporal_comp_obj = m_obj.components[2]

        spatial_spec_idx = findfirst(s -> s.component_obj === spatial_comp_obj, M.components)
        temporal_spec_idx = findfirst(s -> s.component_obj === temporal_comp_obj, M.components)

        if isnothing(spatial_spec_idx) || isnothing(temporal_spec_idx)
            @warn "Could not resolve components for Kronecker product '$(key)'. ST effect reconstruction skipped."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

        s_Q = M.components[spatial_spec_idx].Q_template
        t_Q = M.components[temporal_spec_idx].Q_template
        noise = get(M, :noise, 1e-6)
        
        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
        C_t = cholesky(Symmetric(Matrix(t_Q) + noise * I))

        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

        all_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]

        for k in 1:outcomes_N
            st_sigma_samples = get_params_vector(chain, "st_interaction_sigma", outcomes_N)[:, k]
            st_raw_samples = if outcomes_N > 1
                st_raw_flat = get_params_vector(chain, "st_interaction_raw", M.s_N * M.t_N * outcomes_N)
                st_raw_flat[:, (k-1)*M.s_N*M.t_N+1 : k*M.s_N*M.t_N]
            else
                get_params_vector(chain, "st_interaction_raw", M.s_N * M.t_N)
            end

            st_effect_k = zeros(Float64, N_tot, n_samples)

            for j in 1:n_samples
                st_innov_matrix = reshape(st_raw_samples[j, :], M.s_N, M.t_N)
                tmp_spatial = C_s.U \ st_innov_matrix
                st_inter = (transpose(C_t.U \ transpose(tmp_spatial))) .* st_sigma_samples[j]
                for i in 1:N_tot; st_effect_k[i, j] = st_inter[s_idx_full[i], t_idx_full[i]]; end
            end
            all_effects[k] = st_effect_k
        end
        return (structured=all_effects, noisy=all_effects)

    elseif op == :pipe
        # Handles spatially varying curves.
        state_component_obj = m_obj.components[1] 
        dynamic_component_obj = get(spec.params, :dynamic_component_obj, nothing)

        if isnothing(dynamic_component_obj); @warn "Could not resolve dynamic component for piped component '$(key)'."; return (structured=[], noisy=[]); end

        basis_key = get(spec.params, :dynamic_basis_key, nothing)
        if isnothing(basis_key) || !haskey(M.basis_matrices, basis_key); @warn "Could not find basis matrix for dynamic component of piped component '$(key)'."; return (structured=[], noisy=[]); end

        B_dynamic_train = M.basis_matrices[basis_key]
        B_dynamic_full = if !isnothing(PS) && haskey(PS, :basis_matrices) && haskey(PS.basis_matrices, basis_key)
            vcat(B_dynamic_train, PS.basis_matrices[basis_key])
        else
            B_dynamic_train
        end
        n_basis = size(B_dynamic_full, 2)
        n_spatial = M.s_N

        all_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]

        for k in 1:outcomes_N
            v = generate_full_variable_names(spec, "univariate", k)
            sigma_name = string(v.sigma)
            rho_name = string(v.rho)
            coeffs_raw_name = string(v.raw)

            if isempty(_find_parameter_new(p_names, string(spec.key), "sigma", k)) || isempty(_find_parameter_new(p_names, string(spec.key), "raw", k)); continue; end
     
            sigma_samples = get_params_vector(chain, sigma_name, 1)
            rho_samples = hasproperty(state_component_obj, :rho) ? get_params_vector(chain, rho_name, 1) : nothing
            coeffs_raw_samples = get_params_vector(chain, coeffs_raw_name, n_spatial * n_basis)

            Q_spatial_template = spec.hyper.state_spec.Q_template
            state_m_type = spec.hyper.state_spec.model_type
            s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx

            for j in 1:n_samples
                rho_val = isnothing(rho_samples) ? nothing : rho_samples[j, 1]
                Q_spatial = recompose_precision(state_m_type, Q_spatial_template, 1.0; extra_param=rho_val)
                F_spatial = cholesky(Symmetric(Q_spatial + 1e-6 * I))

                coeffs_raw_matrix = reshape(coeffs_raw_samples[j, :], n_spatial, n_basis)
                spatial_coeffs = sigma_samples[j, 1] .* (F_spatial.U \ coeffs_raw_matrix)

                all_effects[k][:, j] = sum(B_dynamic_full .* spatial_coeffs[s_idx_full, :], dims=2)
            end
        end
        return (structured=all_effects, noisy=all_effects)
    end
    
    @warn "Reconstruction for ComposedComponent with operator ':$op' is not implemented. Returning zero-effect for '$(key)'."
    return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
end




function extract_component(m_obj::CustomComponent, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    reconstruct_func = get(m_obj.params, :reconstruct_func, nothing)

    if !isnothing(reconstruct_func) && isa(reconstruct_func, Function)
        try
            return reconstruct_func(chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
        catch e
            @error "The custom reconstruction function for component '$(spec.key)' failed."
            rethrow(e)
        end
    else
        @warn "Reconstruction for custom component '$(spec.key)' is not defined. Returning a zero-effect. Please provide a `reconstruct_func` to the `custom()` module to enable posterior reconstruction."
        structured_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]
        return (structured=structured_effects, noisy=structured_effects)
    end
end

function extract_component(m_obj::Component, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    @warn "No specific reconstruction logic for component type $(typeof(m_obj)). Returning zero effects."
    n_units = if spec.structure == :spatial; M.s_N; elseif spec.structure == :temporal; M.t_N; else 1; end
    return (structured=[zeros(Float64, n_units, n_samples)], noisy=[zeros(Float64, n_units, n_samples)])
end


function _apply_multivariate_correlation(eta_latent, chain, outcomes_N)
    # Purpose: Applies the estimated correlation structure to independent latent fields.
    # Rationale: Centralizes the core logic of multivariate models, where independent
    #            latent effects are combined via a learned correlation matrix.
    # Inputs:
    #   - eta_latent: A 3D array of un-correlated effects [n_obs, n_samples, n_outcomes].
    #   - chain: The MCMC chain, to extract the correlation matrix.
    #   - outcomes_N: The number of outcomes.
    # Outputs: A 3D array of correlated effects.
    if outcomes_N == 1
        return eta_latent
    end
    N_tot, n_samples, _ = size(eta_latent)
    L_corr_samples = get_params_vector(chain, "L_corr", outcomes_N * outcomes_N)
    eta_final = zeros(N_tot, n_samples, outcomes_N)
    for s in 1:n_samples
        L_s = reshape(L_corr_samples[s, :], outcomes_N, outcomes_N)
        eta_final[:, s, :] = eta_latent[:, s, :] * L_s'
    end
    return eta_final
end

function _summarize_effects_registry(registry, M, outcomes_N, alpha)
    # Purpose: Summarizes the posterior samples for all discovered component effects.
    # Rationale: Consolidates the logic for summarizing simple, mixed, and multivariate
    #            effects into a single, reusable function.
    # Inputs:
    #   - registry: The dictionary of raw posterior effects.
    #   - M: The model configuration object. 
    #   - outcomes_N: The number of outcomes.
    #   - alpha: The significance level for credible intervals.
    # Outputs: A NamedTuple containing summarized effects.
    summarized_registry = Dict{Symbol, Any}()
    mixed_effects_summaries = Dict{Symbol, Any}()

    for (key, effects) in pairs(registry)
        if key in [:intercept, :fixed]; continue; end

        spec_idx = findfirst(s -> s.key == key, M.components)
        if !isnothing(spec_idx) && M.components[spec_idx].component_obj isa MixedComponent
            summaries_per_outcome = [Dict{Symbol, Any}() for _ in 1:outcomes_N]
            if effects.type == :simple
                for k in 1:outcomes_N
                    summaries_per_outcome[k][Symbol(effects.lhs)] = summarize_array(effects.effects[k], alpha=alpha)
                end
            elseif effects.type == :correlated
                for (term_name, term_effects) in pairs(effects.effects)
                    for k in 1:outcomes_N
                        summaries_per_outcome[k][term_name] = summarize_array(term_effects[k], alpha=alpha)
                    end
                end
            end
            
            summaries_final = outcomes_N > 1 ? [NamedTuple(s) for s in summaries_per_outcome] : NamedTuple(summaries_per_outcome[1])
            mixed_effects_summaries[key] = (group_var=M.components[spec_idx].var, summaries=summaries_final)
        else
            effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
            if outcomes_N > 1
                summarized_registry[key] = [summarize_array(effect_set[k], alpha=alpha) for k in 1:outcomes_N]
            else
                summarized_registry[key] = summarize_array(effect_set[1], alpha=alpha)
            end
        end
    end
    if !isempty(mixed_effects_summaries); summarized_registry[:mixed_effects] = NamedTuple(mixed_effects_summaries); end
    
    return NamedTuple(summarized_registry)
end

# ==============================================================================
# SECTION 3: CORE RECONSTRUCTION WORKFLOW
# ==============================================================================

function _reconstruct(arch::UnivariateArchitecture, mode::String, chain, M, PS, alpha)
    # Purpose: Main reconstruction entry point for univariate models.
    # Rationale: Orchestrates the discovery, assembly, and summarization of all model effects.
    # Inputs: Standard reconstruction arguments for a univariate model.
    # Outputs: A comprehensive NamedTuple with all summarized posterior statistics.
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N

    registry = _discover_component_realizations(chain, M, PS, n_samples, p_names, 1, N_tot)
    eta_latent = _modular_eta_assembly(registry, M, PS, n_samples, 1)
    eta_final = eta_latent[:,:,1] # Drop the third dimension

    pred_results = _process_ll_and_predictions(eta_final, chain, M, PS, 1, 1)
    
    summarized_effects = _summarize_effects_registry(registry, M, 1, alpha)
    
    p_denoised_summary = summarize_array(pred_results.p_denoised, alpha=alpha)
    p_noisy_summary = summarize_array(pred_results.p_noisy, alpha=alpha)
    waic = _compute_waic(pred_results.log_lik)

    return (
        predictions_denoised = p_denoised_summary,
        predictions_noisy = p_noisy_summary,
        raw_predictions_denoised = pred_results.p_denoised,
        raw_predictions_noisy = pred_results.p_noisy,
        log_likelihood = pred_results.log_lik,
        waic = waic,
        effects = summarized_effects,
        arch = arch
    )
end

function _discover_component_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    # Purpose: Extracts all latent effects from the MCMC chain.
    # Rationale: Iterates through all specified components and fixed effects, calling the appropriate
    #            extraction function for each to populate a central registry of posterior samples.
    # Inputs: Standard reconstruction arguments.
    # Outputs: A NamedTuple registry containing posterior samples for each model component.
    registry = Dict{Symbol, Any}()

    # Fixed effects
    if M.Xfixed_N > 0
        Xfixed_train = M.Xfixed
        Xfixed_pred = if isnothing(PS) || !haskey(PS, :Xfixed) || isempty(PS.Xfixed)
            zeros(0, M.Xfixed_N)
        else
            PS.Xfixed
        end
        Xfixed_full = vcat(Xfixed_train, Xfixed_pred)
        
        if outcomes_N > 1
            beta_samples_flat = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)
            fixed_effects_all = zeros(Float64, N_tot, n_samples, outcomes_N)
            for k in 1:outcomes_N
                beta_k = beta_samples_flat[:, (k-1)*M.Xfixed_N+1 : k*M.Xfixed_N]
                fixed_effects_all[:, :, k] = Xfixed_full * beta_k'
            end
            registry[:fixed] = fixed_effects_all
        else
            beta_samples = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)
            registry[:fixed] = Xfixed_full * beta_samples'
        end
    else
        registry[:fixed] = zeros(Float64, N_tot, n_samples, outcomes_N)
    end

    # Intercept
    if M.add_intercept
        intercept_samples = get_params_vector(chain, "intercept", outcomes_N)
        intercept_effects = zeros(Float64, N_tot, n_samples, outcomes_N)
        for k in 1:outcomes_N
            intercept_effects[:, :, k] .= intercept_samples[:, k]'
        end
        registry[:intercept] = intercept_effects
    else
        registry[:intercept] = zeros(Float64, N_tot, n_samples, outcomes_N)
    end

    # Components
    for spec in M.components
        effects = extract_component(spec.component_obj, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
        registry[spec.key] = effects
    end

    return NamedTuple(registry)
end



"""
    _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)

Assembles the full linear predictor (`eta`) from all discovered latent effects.

# Rationale for Update
This version includes more detailed comments to clarify the assembly logic for different
component structures. A key clarification is in the handling of `MixedComponent`. The `extract_component`
method for `MixedComponent` now returns the full index vector (`effects.indices`) that maps
each observation (both training and prediction) to its corresponding group level. This
simplifies the assembly logic here, as this function no longer needs to compute these
indices itself. It simply uses the provided indices to apply the random effects correctly.
"""
function _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    eta_latent = zeros(Float64, N_tot, n_samples, outcomes_N)

    # Add intercept and fixed effects first.
    eta_latent .+= registry.intercept
    eta_latent .+= registry.fixed

    # Pre-compute full index vectors for spatial, temporal, and seasonal structures.
    s_idx_full = haskey(M, :s_idx) ? (isnothing(PS) || !haskey(PS, :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? (isnothing(PS) || !haskey(PS, :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)
    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    for spec in M.components
        key = spec.key
        if !haskey(registry, key); continue; end
        
        effects = registry[key]
        effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
        if isempty(effect_set); continue; end
        
        for k in 1:outcomes_N
            if spec.structure in [:spatial, :temporal]
                # For standard GMRFs, the effect matrix is [n_units x n_samples].
                # We use the appropriate index vector to map the effect to each observation.
                effect_to_add = effect_set[k]
                idx_vec = spec.structure == :spatial ? s_idx_full : t_idx_full
                eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
            elseif spec.structure == :seasonal
                effect_to_add = effect_set[k]
                if spec.component_obj isa Harmonic # Harmonic basis is already expanded to N_tot
                    eta_latent[:, :, k] .+= effect_to_add
                else # Assumes Cyclic or other GMRF-like seasonal model
                    idx_vec = u_idx_full
                    eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
                end
            elseif spec.structure == :smooth || spec.structure == :interact
                # For smoothers and interactions, the effect matrix is already [N_tot x n_samples].
                eta_latent[:, :, k] .+= effect_set[k]
            elseif spec.structure == :mixed
                # For mixed effects, the `extract_component` function returns the full index vector.
                idx_full = effects.indices
                
                if effects.type == :simple
                    # Uncorrelated random intercept or slope.
                    effect_to_add = effects.effects[k]
                    if effects.lhs == "1"; eta_latent[:, :, k] .+= effect_to_add[idx_full, :]; else; cov_vec = isnothing(PS) ? M.data[!, Symbol(effects.lhs)] : vcat(M.data[!, Symbol(effects.lhs)], PS.data[!, Symbol(effects.lhs)]); eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec; end
                elseif effects.type == :correlated
                    # Correlated random effects.
                    for (term_name, term_effects) in pairs(effects.effects)
                        effect_to_add = term_effects[k]
                        if term_name == :intercept; eta_latent[:, :, k] .+= effect_to_add[idx_full, :]; else; cov_name = Symbol(replace(string(term_name), "slope_" => "")); cov_vec = isnothing(PS) ? M.data[!, cov_name] : vcat(M.data[!, cov_name], PS.data[!, cov_name]); eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec; end
                    end
                end
            else
                # Fallback for other component types.
                if size(effect_set[k], 1) == N_tot
                    eta_latent[:, :, k] .+= effect_set[k]
                end
            end
        end
    end

    # Add log_offsets at the very end.
    if haskey(M, :log_offsets)
        offset_full = isnothing(PS) ? M.log_offsets : vcat(M.log_offsets, get(PS, :log_offsets, zeros(PS.y_N)))
        for k in 1:outcomes_N
            eta_latent[:, :, k] .+= offset_full
        end
    end

    return eta_latent
end



function _process_ll_and_predictions(eta_samples, chain, M, PS, outcomes_N, k)
    # Purpose: Generates denoised predictions, noisy predictions, and log-likelihood values from eta.
    # Rationale: This function applies the inverse link function and samples from the predictive distribution.
    # Inputs: Eta samples and model configuration.
    # Outputs: A NamedTuple with denoised predictions, noisy predictions, and log-likelihood matrix.
    n_samples = size(eta_samples, 2)
    N_train = M.y_N
    N_pred = isnothing(PS) ? 0 : PS.y_N
    N_tot = N_train + N_pred

    y_obs_k = outcomes_N > 1 ? M.y_obs[:, k] : M.y_obs
    
    lik_spec = M.likelihood_specs[k]
    family = string(get(lik_spec, :family, "gaussian"))
    use_zi = get(M, :use_zi, false)
    phi_zi_samples = use_zi ? get_params_vector(chain, "lik_phi_zi", 1)[:,1] : zeros(n_samples)
    
    # Denoised predictions (on response scale)
    p_denoised_samples = similar(eta_samples)
    for s in 1:n_samples
        p_denoised_samples[:, s] = _apply_link_and_lik(family, eta_samples[:, s], use_zi, phi_zi_samples[s])
    end

    p_noisy_samples = similar(eta_samples)
    log_lik_samples = zeros(Float64, N_train, n_samples)

    # Get likelihood-specific parameters
    y_sigma_samples = get_params_vector(chain, "y_sigma", outcomes_N)
    r_nb_samples = get_params_vector(chain, "r_nb", outcomes_N)

    trials_full = haskey(M, :trials) ? (isnothing(PS) ? M.trials : vcat(M.trials, get(PS, :trials, ones(Int, PS.y_N)))) : ones(Int, N_tot)
    
    family_trait = get_model_family(family)

    for s in 1:n_samples
        phi_zi_s = phi_zi_samples[s]
        y_sigma_s = y_sigma_samples[s, k]
        r_nb_s = r_nb_samples[s, k]
        
        for i in 1:N_tot
            eta_is = eta_samples[i, s]
            
            # For sampling, y_obs in lik_obj doesn't matter.
            lik_obj = bstm_Likelihood(family, [0.0]; phi_zi=phi_zi_s, r_nb=r_nb_s, sigma_y=y_sigma_s, trial=trials_full[i])
            dist = get_dist_ref(lik_obj.family, lik_obj, eta_is, y_sigma_s)
            
            p_noisy_samples[i, s] = rand(dist) 

            if i <= N_train
                log_lik_samples[i, s] = logpdf(dist, y_obs_k[i])
            end
        end
    end

    return (p_denoised = p_denoised_samples, p_noisy = p_noisy_samples, log_lik = log_lik_samples)
end

function _process_multinomial_predictions(eta_samples, chain, M, PS)
    # Purpose: Generates predictions and log-likelihood for multinomial models.
    # Rationale: This specialized function handles the vector nature of multinomial outcomes.
    n_samples = size(eta_samples, 2)
    N_train = M.y_N
    N_pred = isnothing(PS) ? 0 : PS.y_N
    N_tot = N_train + N_pred
    K = M.outcomes_N

    y_obs_train = M.y_obs # [N_train, K]

    # Denoised predictions (proportions)
    p_denoised_samples = zeros(Float64, N_tot, K, n_samples)
    for s in 1:n_samples 
        for i in 1:N_tot
            p_denoised_samples[i, :, s] = softmax(eta_samples[i, s, :])
        end
    end

    # Noisy predictions (counts)
    p_noisy_samples = zeros(Int, N_tot, K, n_samples)
    log_lik_samples = zeros(Float64, N_train, n_samples)

    # Get total trials for each observation
    trials_train = sum(y_obs_train, dims=2)
    # For prediction, we might need to assume a total count, or it could be in PS.
    # Assuming 1 for simplicity if not provided.
    trials_pred = haskey(PS, :trials) ? sum(PS.trials, dims=2) : ones(Int, N_pred)
    trials_full = vcat(vec(trials_train), vec(trials_pred))

    for s in 1:n_samples 
        for i in 1:N_tot
            probs = p_denoised_samples[i, :, s]
            dist = Multinomial(Int(trials_full[i]), probs)
            p_noisy_samples[i, :, s] = rand(dist)
            if i <= N_train; log_lik_samples[i, s] = logpdf(dist, y_obs_train[i, :]); end
        end
    end
    return (p_denoised=p_denoised_samples, p_noisy=p_noisy_samples, log_lik=log_lik_samples)
end

function _reconstruct(arch::MultivariateArchitecture, mode::String, chain, M, PS, alpha)
    # Purpose: Main reconstruction entry point for multivariate models.
    # Rationale: Handles the additional complexity of multiple outcomes and their correlations.
    # Inputs: Standard reconstruction arguments for a multivariate model.
    # Outputs: A comprehensive NamedTuple with all summarized posterior statistics.
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    outcomes_N = M.outcomes_N
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N

    registry = _discover_component_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    eta_latent = _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    eta_final = _apply_multivariate_correlation(eta_latent, chain, outcomes_N)

    summarized_effects = _summarize_effects_registry(registry, M, outcomes_N, alpha)

    local p_denoised_summaries, p_noisy_summaries, raw_denoised, raw_noisy, all_log_lik

    is_multinomial = any(s -> get(s, :family, "gaussian") == "dirichlet_multinomial", M.likelihood_specs)

    if is_multinomial
        pred_results = _process_multinomial_predictions(eta_final, chain, M, PS)
        # For multinomial, we summarize the proportions for each category.
        p_denoised_summaries = [summarize_array(pred_results.p_denoised[:, k, :], alpha=alpha) for k in 1:outcomes_N]
        p_noisy_summaries = [summarize_array(pred_results.p_noisy[:, k, :], alpha=alpha) for k in 1:outcomes_N]
        raw_denoised = [pred_results.p_denoised[:, k, :] for k in 1:outcomes_N]
        raw_noisy = [pred_results.p_noisy[:, k, :] for k in 1:outcomes_N]
        all_log_lik = pred_results.log_lik
    else
        all_pred_results = [_process_ll_and_predictions(eta_final[:,:,k], chain, M, PS, outcomes_N, k) for k in 1:outcomes_N]
        p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
        p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
        raw_denoised = [res.p_denoised for res in all_pred_results]
        raw_noisy = [res.p_noisy for res in all_pred_results]
        all_log_lik = hcat([res.log_lik for res in all_pred_results]...)
    end

    waic = _compute_waic(all_log_lik)

    return (
        predictions_denoised = p_denoised_summaries,
        predictions_noisy = p_noisy_summaries,
        raw_predictions_denoised = raw_denoised,
        raw_predictions_noisy = raw_noisy,
        log_likelihood = all_log_lik,
        waic = waic,
        effects = NamedTuple(summarized_effects),
        arch = arch
    )
end


"""
    _reconstruct(arch::MultifidelityArchitecture, mode::String, chain, M, PS, alpha)

Main reconstruction entry point for multi-fidelity models.

# Rationale for Update
This function orchestrates the hierarchical reconstruction of a main model and its
nested sub-models. This updated version includes more detailed comments to clarify
the recursive nature of the process and how the sub-model effects are calibrated
and integrated into the main model's linear predictor.

The process is as follows:
1.  Reconstruct the main model's base effects (fixed, spatial, temporal, etc.).
2.  Iterate through each nested sub-model defined in `M[:nested_components]`.
3.  For each sub-model, recursively call `_reconstruct` to get its full posterior summary.
4.  The sub-model's linear predictor (`eta_sub`) is then scaled by its learned `rho_nested`
    coefficient and added to the main model's linear predictor (`eta_main`).
5.  Finally, the main model's predictions and likelihood are computed based on the
    completed `eta_main`.
"""
function _reconstruct(arch::MultifidelityArchitecture, mode::String, chain, M, PS, alpha)
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N = M.outcomes_N

    # 1. Reconstruct the main model's components (excluding nested effects)
    main_registry = _discover_component_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    
    # 2. Assemble the main model's base eta
    eta_main = _modular_eta_assembly(main_registry, M, PS, n_samples, outcomes_N)

    # 3. Reconstruct sub-models' etas and add them to the main eta
    nested_results = Dict{Symbol, Any}()
    if haskey(M, :nested_components)
        for (key, sub_M) in M.nested_components
            # Determine if a corresponding prediction set exists for the sub-model
            sub_PS = if !isnothing(PS) && haskey(PS, :nested_prediction_sets)
                get(PS.nested_prediction_sets, key, nothing)
            else
                nothing
            end

            # Recursively call _reconstruct for the sub-model. This is the core of the
            # hierarchical reconstruction. The sub-model is treated as a complete, standalone model.
            sub_outcomes_N = get(sub_M, :outcomes_N, 1)
            sub_N_tot = isnothing(sub_PS) ? sub_M.y_N : sub_M.y_N + sub_PS.y_N
            sub_registry = _discover_component_realizations(chain, sub_M, sub_PS, n_samples, p_names, sub_outcomes_N, sub_N_tot)
            eta_sub = _modular_eta_assembly(sub_registry, sub_M, sub_PS, n_samples, sub_outcomes_N)

            # Get the scaling parameter `rho` that links the sub-model to the main model.
            rho_name = "rho_nested_$(key)"
            rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]

            if size(eta_sub, 1) != N_tot
                @warn "Size mismatch between main model observations ($N_tot) and nested model '$(key)' observations ($(size(eta_sub, 1))). Cannot apply nested effect."
                continue
            end
            
            if outcomes_N > 1 || sub_outcomes_N > 1
                @warn "Multi-fidelity connection between multivariate models is not fully supported. Assuming a 1-to-1 outcome mapping." 
            end
            
            # Add the calibrated sub-model effect to the main linear predictor.
            eta_main .+= reshape(rho_samples, 1, n_samples, 1) .* eta_sub

            # Store the full results of the sub-model reconstruction.
            sub_arch_raw = get(sub_M, :model_arch, "univariate")
            sub_arch_type = sub_arch_raw == "multivariate" ? MultivariateArchitecture() : UnivariateArchitecture()
            nested_results[key] = _reconstruct(sub_arch_type, mode, chain, sub_M, sub_PS, alpha)
        end
    end

    # 4. Apply correlation and generate predictions for the final main model
    eta_final = _apply_multivariate_correlation(eta_main, chain, outcomes_N)
    
    if outcomes_N > 1
        all_pred_results = [_process_ll_and_predictions(eta_final[:,:,k], chain, M, PS, outcomes_N, k) for k in 1:outcomes_N]
        p_denoised_summaries = [summarize_array(res.p_denoised, alpha=alpha) for res in all_pred_results]
        p_noisy_summaries = [summarize_array(res.p_noisy, alpha=alpha) for res in all_pred_results]
        raw_denoised = [res.p_denoised for res in all_pred_results]
        raw_noisy = [res.p_noisy for res in all_pred_results]
        all_log_lik = hcat([res.log_lik for res in all_pred_results]...)
    else
        pred_results = _process_ll_and_predictions(eta_final[:,:,1], chain, M, PS, 1, 1)
        p_denoised_summaries = summarize_array(pred_results.p_denoised, alpha=alpha)
        p_noisy_summaries = summarize_array(pred_results.p_noisy, alpha=alpha)
        raw_denoised = pred_results.p_denoised
        raw_noisy = pred_results.p_noisy
        all_log_lik = pred_results.log_lik
    end

    summarized_effects = _summarize_effects_registry(main_registry, M, outcomes_N, alpha)
    waic = _compute_waic(all_log_lik)

    return (
        predictions_denoised = p_denoised_summaries, 
        predictions_noisy = p_noisy_summaries,
        raw_predictions_denoised = raw_denoised,
        raw_predictions_noisy = raw_noisy,
        log_likelihood = all_log_lik, 
        waic = waic, 
        effects = summarized_effects, 
        nested_results = nested_results, 
        arch = arch
    )
end


# ==============================================================================
# SECTION 3: POSTERIOR ASSEMBLY AND SUMMARIZATION
# ==============================================================================

function _quantile_along_last_dim(A::AbstractArray, q::Real; sample_dim=ndims(A))
    other_dims = size(A)[1:end-1]
    out = Array{Float64}(undef, other_dims)
    
    for I in CartesianIndices(out) 
        slice_view = view(A, I.I..., :)
        out[I] = quantile(slice_view, q)
    end
    return out
end

function summarize_array(samples::AbstractArray; alpha=0.05)
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end 

    sample_dim = ndims(samples)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    low_bound = _quantile_along_last_dim(samples, low_prob; sample_dim=sample_dim)
    high_bound = _quantile_along_last_dim(samples, high_prob; sample_dim=sample_dim)

    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end

function _compute_waic(log_lik)
    nobs, nsamples = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    local mu
    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        mu = exp.(eta)
    elseif family in ["bernoulli", "binomial", "beta"]
        mu = logistic.(eta)
    else
        mu = eta 
    end
    if use_zi
        mu = (1.0 .- phi) .* mu
    end
    return mu
end


# ==============================================================================
# SECTION 4: MAIN RECONSTRUCTION AND PREDICTION API
# ==============================================================================

function model_results_comprehensive(model::DynamicPPL.Model, chain; au=nothing, data=nothing, alpha=0.05)
    # Purpose: The primary post-processing engine that generates comprehensive summaries,
    #          diagnostics, and plots from a fitted bstm model and MCMC chain.
    # Rationale: This function orchestrates the entire reconstruction workflow, from latent
    #            field discovery to metric calculation and visualization, providing a unified
    #            and standardized output for model assessment.
    # Inputs:
    #   - model: The fitted Turing model object. 
    #   - chain: The MCMC chain result.
    #   - au: (Optional) Areal unit object containing spatial geometries for plotting.
    #   - data: (Optional) The original DataFrame, used for plotting covariate effects.
    #   - alpha: The significance level for credible intervals.
    # Outputs: A comprehensive NamedTuple with `:metrics`, `:pstats` (posterior stats), and `:plots`.

    # #
    # 1. Metadata and Architecture Extraction
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")

    arch_type = if raw_arch == "multivariate"; MultivariateArchitecture()
    elseif raw_arch == "multifidelity"; MultifidelityArchitecture()
    else; UnivariateArchitecture(); end

    # #
    # 2. Core Reconstruction
    # This calls the appropriate _reconstruct method based on the model architecture.
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # #
    # 2.5 Post-Stratification Weight Calculation
    # This is done here because we need the raw denoised prediction samples, which are
    # returned by _reconstruct but not typically stored in the final summary.
    post_strat_weights = nothing 
    if hasproperty(res, :raw_predictions_denoised)
        samples_denoised = res.arch isa MultivariateArchitecture ? res.raw_predictions_denoised[1] : res.raw_predictions_denoised
        post_strat_weights = post_stratification_weights(res, M, nothing, samples_denoised)
    end

    # #
    # 3. Performance Metric Calculation
    # Handles both univariate and multivariate cases for RMSE and Pearson R.
    pred_summary = res.predictions_denoised
    y_pred = pred_summary isa AbstractVector ? vcat([ps.mean for ps in pred_summary]...) : (hasproperty(pred_summary, :mean) ? pred_summary.mean : [])
    y_obs_flat = vec(collect(y_obs)) 
    y_pred_flat = vec(collect(y_pred)) 
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0
    if !isempty(valid_idx) 
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]
        rmse_val = sqrt(mean((obs_v .- pred_v).^2))
        try; r_pearson = cor(obs_v, pred_v); catch; r_pearson = 0.0; end
    end

    # #
    # 4. MCMC Diagnostics
    mean_rhat, min_ess, sampling_time = 1.0, 0.0, 0.0 
    try
        chains_obj = MCMCChains.Chains(chain)
        df_stats = DataFrame(MCMCChains.summarize(chains_obj))
        if hasproperty(df_stats, :rhat); r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat); mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals); end
        e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats, :ess) ? :ess : nothing)
        if !isnothing(e_col); e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col]); min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals); end
        if hasproperty(chain, :info) && haskey(chain.info, :stop_time); sampling_time = (chain.info.stop_time - chain.info.start_time); end
    catch e; @warn "MCMC diagnostic extraction failed: $e. Using default values."; end
    
    # #
    # 5. Plot Generation
    data_for_plots = isnothing(data) ? get(M, :data, nothing) : data
    plots = _generate_plots(res, M; au=au, data=data_for_plots)

    return (
        metrics = (rmse = rmse_val, r_pearson = r_pearson, ess = min_ess, rhat = mean_rhat, waic = get(res, :waic, 0.0), time = sampling_time),
        pstats = res,
        plots = plots,
        post_strat_weights = post_strat_weights
    )
end

function _generate_plots(res, M; au=nothing, data=nothing, outcome=1)
    # Purpose: Generates a standard set of diagnostic and summary plots from the
    #          reconstructed posterior results.
    # Rationale: Centralizes visualization logic, providing a consistent visual output
    #            for different model architectures and components.
    # Inputs:
    #   - res: The main results object from `_reconstruct`.
    #   - M: The model configuration object.
    #   - au: (Optional) Areal unit object with geometries.
    #   - data: (Optional) The original DataFrame for covariate plots.
    #   - outcome: The index of the outcome to plot for multivariate models.
    # Outputs: A dictionary of Plots.jl plot objects.

    plots = Dict{Symbol, Any}()
    effects = res.effects
    
    y_obs = get(M, :y_obs, nothing)
    polygons = isnothing(au) ? nothing : get(au, :polygons, nothing)
    centroids = isnothing(au) ? nothing : get(au, :centroids, nothing)

    if hasproperty(res, :predictions_denoised)
        if isnothing(y_obs); @info "Skipping PPC plot: Observation data not found.";
        else
            is_mv = res.arch isa MultivariateArchitecture
            pred_summary = is_mv ? res.predictions_denoised[outcome] : res.predictions_denoised 
            if !isnothing(pred_summary) && hasproperty(pred_summary, :mean)
                y_p, y_o = vec(pred_summary.mean), is_mv ? vec(y_obs[:, outcome]) : vec(y_obs)
                if length(y_p) == length(y_o)
                    p_ppc = scatter(y_p, y_o, title="Posterior Predictive Check", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
                    clean_p, clean_o = filter(!isnan, y_p), filter(!isnan, y_o)
                    if !isempty(clean_p) && !isempty(clean_o); min_val, max_val = min(minimum(clean_p), minimum(clean_o)), max(maximum(clean_p), maximum(clean_o)); plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5); end
                    plots[:ppc] = p_ppc
                end
            end
        end
    end

    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if isnothing(field_data) || !hasproperty(field_data, :mean); @info "Skipping spatial plot '$title_str': Data missing."; return nothing; end 
        if isnothing(polygons) && isnothing(centroids); @info "Skipping spatial plot '$title_str': No geometry provided."; return nothing; end
        s_mean = vec(collect(field_data.mean))
        if all(iszero, s_mean); @info "Skipping spatial plot '$title_str': Mean effect is zero."; return nothing; end
        if !isnothing(polygons) && length(polygons) >= length(s_mean); return plot_choropleth(s_mean, polygons; title=title_str);
        elseif !isnothing(centroids); return scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal); end
        return nothing
    end

    if hasproperty(effects, :spatial_denoised); s_field = (res.arch isa MultivariateArchitecture) ? effects.spatial_denoised[outcome] : effects.spatial_denoised; p = _create_choropleth_plot(s_field, "Spatial Denoised Effect", polygons, centroids); if !isnothing(p); plots[:spatial_denoised] = p; end; end 
    if hasproperty(effects, :spatial_noisy); s_field = (res.arch isa MultivariateArchitecture) ? effects.spatial_noisy[outcome] : effects.spatial_noisy; p = _create_choropleth_plot(s_field, "Total Spatial Effect", polygons, centroids); if !isnothing(p); plots[:spatial_noisy] = p; end; end 

    if hasproperty(effects, :temporal); t_field = (res.arch isa MultivariateArchitecture) ? effects.temporal[outcome] : effects.temporal; if !isnothing(t_field) && hasproperty(t_field, :mean) && !all(iszero, t_field.mean); tm, tl, tu = vec(t_field.mean), vec(t_field.lower), vec(t_field.upper); plots[:temporal] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Time Index"); end; end 
    if hasproperty(effects, :seasonal) && !isnothing(effects.seasonal) && hasproperty(effects.seasonal, :mean) && !all(iszero, effects.seasonal.mean); um, ul, uu = vec(effects.seasonal.mean), vec(effects.seasonal.lower), vec(effects.seasonal.upper); plots[:seasonal] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Component", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Period"); end 

    if hasproperty(effects, :smooth_effects) && effects.smooth_effects isa NamedTuple
        if isnothing(data); @info "Skipping smooth effects plots: `data` not provided.";
        else
            smooth_plots = Dict{Symbol, Any}()
            for (var_sym, smooth_summary) in pairs(effects.smooth_effects)
                if hasproperty(smooth_summary, :mean) && !all(iszero, smooth_summary.mean) && hasproperty(data, var_sym)
                    cov_data = data[!, var_sym]; p_order = sortperm(cov_data); sm, sl, su = vec(smooth_summary.mean), vec(smooth_summary.lower), vec(smooth_summary.upper)
                    smooth_plots[var_sym] = plot(cov_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]), title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Latent Effect", legend=false, color=:darkorange, fillalpha=0.2)
                end
            end
            if !isempty(smooth_plots); plots[:smooth_effects] = smooth_plots; end
        end
    end

    if hasproperty(effects, :fixed_effects) && !isnothing(effects.fixed_effects)
        fe_summary = (res.arch isa MultivariateArchitecture) ? effects.fixed_effects[outcome] : effects.fixed_effects
        if hasproperty(fe_summary, :mean) && !all(iszero, fe_summary.mean) 
            fm, fl, fu = vec(fe_summary.mean), vec(fe_summary.lower), vec(fe_summary.upper)
            if !isempty(fm); coef_names = haskey(M, :Xfixed_names) ? string.(M.Xfixed_names) : ["Coef_$i" for i in 1:length(fm)]; p_forest = scatter(fm, 1:length(fm), xerror=(fm .- fl, fu .- fm), yticks=(1:length(fm), coef_names), title="Fixed Effects Coefficients", xlabel="Estimate", markersize=4, color=:black, legend=false); vline!(p_forest, [0], color=:red, ls=:dash, lw=1); plots[:fixed_effects] = p_forest; end
        end
    end

    if hasproperty(effects, :mixed_effects) && !isnothing(effects.mixed_effects)
        mixed_plots = Dict{Symbol, Any}()
        is_mv = res.arch isa MultivariateArchitecture
        for (key, effect_summary) in pairs(effects.mixed_effects)
            group_var = Symbol(effect_summary.group_var)
            group_levels = hasproperty(data, group_var) ? string.(levels(data[!, group_var])) : nothing 

            summaries_to_plot = is_mv ? effect_summary.summaries[outcome] : effect_summary.summaries

            for (term_name, summary) in pairs(summaries_to_plot)
                if hasproperty(summary, :mean) && !all(iszero, summary.mean)
                    means = vec(summary.mean)
                    lowers = vec(summary.lower)
                    uppers = vec(summary.upper)
                    n_levels = length(means) 
                    y_ticks_labels = isnothing(group_levels) || length(group_levels) != n_levels ? ["Level $i" for i in 1:n_levels] : group_levels
                    p_title = "Mixed Effect: $(term_name) | $(group_var)"
                    p_forest = scatter(means, 1:n_levels, xerror=(means .- lowers, uppers .- means), yticks=(1:n_levels, y_ticks_labels), title=p_title, xlabel="Effect Size", markersize=4, color=:black, legend=false, yflip=true)
                    vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                    mixed_plots[Symbol("$(key)_$(term_name)")] = p_forest
                end
            end
        end
        if !isempty(mixed_plots); plots[:mixed_effects] = mixed_plots; end
    end

    return (
        NamedTuple(plots)
    )
end

function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Purpose: The primary engine for projecting a fitted model onto new data.
    # Rationale: This function constructs a "prediction set" configuration (PS) that mirrors the training configuration (M)
    #            but is adapted for the `new_data`. It correctly handles the projection of fixed effects, smooth basis functions,
    #            and nested models.
    # v1.0.0 (2026-07-17)
    # Inputs:
    #   - model_obj: The fitted Turing model object.
    #   - chain: The MCMC chain result.
    #   - new_data: A DataFrame with the same column names as the training data.
    #   - n_samples: The number of posterior samples to use for prediction.
    #   - alpha: The significance level for credible intervals.
    # Outputs: A NamedTuple containing denoised and noisy predictions, posterior stats, and the PS object.
    M_train = model_obj.args.M
    n_samps = min(size(chain, 1), n_samples)

    PS_dict = Dict(pairs(M_train))
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data)) # Placeholder
    PS_dict[:y_N] = nrow(new_data)

    # Re-create fixed effects design matrix for the new data
    if haskey(M_train, :formula)
        decomposed_formula = decompose_bstm_formula(M_train.formula)
        fixed_effects_vars = String[]
        append!(fixed_effects_vars, decomposed_formula.fixed_effects)
        for (_, mod_data_nt) in decomposed_formula.modules
            if mod_data_nt.module_type == :fixed && haskey(mod_data_nt.args, :positional_args)
                append!(fixed_effects_vars, string.(mod_data_nt.args[:positional_args]))
            end
        end
        fixed_effects_vars = unique(fixed_effects_vars)
        
        if !isempty(fixed_effects_vars)
            rhs = "0 + " * join(fixed_effects_vars, " + ")
            Xfixed_pred, _ = create_fixed_design(rhs, new_data; contrasts=get(M_train, :contrasts, Dict()))
            PS_dict[:Xfixed] = Matrix(Xfixed_pred)
            PS_dict[:Xfixed_N] = size(Xfixed_pred, 2)
            PS_dict[:Xfixed_names] = names(Xfixed_pred, 2)
        end
    end

    # Update indices from new_data
    if haskey(M_train, :s_idx_var) && hasproperty(new_data, M_train.s_idx_var); PS_dict[:s_idx] = new_data[!, M_train.s_idx_var]; end 
    if haskey(M_train, :t_idx_var) && hasproperty(new_data, M_train.t_idx_var); PS_dict[:t_idx] = new_data[!, M_train.t_idx_var]; end 
    if haskey(M_train, :u_idx_var) && hasproperty(new_data, M_train.u_idx_var); PS_dict[:u_idx] = new_data[!, M_train.u_idx_var]; end 

    # Re-create basis matrices for smoothers on the new data
    if haskey(M_train, :components)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.structure == :smooth, M_train.components)
        
        for spec in smooth_specs
            key_sym = Symbol(spec.var)
            vars = get(spec.params, :positional_args, [])
            n_vars = length(vars)
            if haskey(M_train.basis_matrices, key_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                m_obj = spec.component_obj
                model_type_str = lowercase(string(typeof(m_obj)))
                nb = size(M_train.basis_matrices[key_sym], 2)
                if n_vars == 1 
                    ps_basis_registry[key_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, Symbol(vars[1])], nb; spec.params...)
                elseif n_vars > 1
                    coords_new = Matrix{Float64}(new_data[!, Symbol.(vars)])
                    if n_vars == 2; ps_basis_registry[key_sym] = bstm_smooth_basis_2D(model_type_str, coords_new, nb; spec.params...);
                    elseif n_vars == 3; ps_basis_registry[key_sym] = bstm_smooth_basis_3D(model_type_str, coords_new, nb; spec.params...);
                    elseif n_vars == 4; ps_basis_registry[key_sym] = bstm_smooth_basis_4D(model_type_str, coords_new, nb; spec.params...);
                    end
                end
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    # Create prediction sets for nested sub-models
    if haskey(M_train, :nested_components) && !isempty(M_train.nested_components)
        PS_dict[:nested_prediction_sets] = Dict{Symbol, Any}()
        for (key, sub_M) in M_train.nested_components
            sub_PS_dict = Dict(pairs(sub_M))
            sub_PS_dict[:data] = new_data
            sub_PS_dict[:y_obs] = zeros(nrow(new_data)) # Placeholder
            sub_PS_dict[:y_N] = nrow(new_data)

            if haskey(sub_M, :formula)
                sub_decomposed = decompose_bstm_formula(sub_M.formula) 
                
                sub_fixed_effects_vars = String[]
                append!(sub_fixed_effects_vars, sub_decomposed.fixed_effects)
                for (_, mod_data_nt) in sub_decomposed.modules
                    if mod_data_nt.module_type == :fixed && haskey(mod_data_nt.args, :positional_args)
                        append!(sub_fixed_effects_vars, string.(mod_data_nt.args[:positional_args]))
                    end
                end
                sub_fixed_effects_vars = unique(sub_fixed_effects_vars)
                
                if !isempty(sub_fixed_effects_vars)
                    rhs = "0 + " * join(sub_fixed_effects_vars, " + ")
                    Xfixed_sub, _ = create_fixed_design(rhs, new_data; contrasts=get(sub_M, :contrasts, Dict()))
                    sub_PS_dict[:Xfixed] = Matrix(Xfixed_sub)
                    sub_PS_dict[:Xfixed_N] = size(Xfixed_sub, 2)
                    sub_PS_dict[:Xfixed_names] = names(Xfixed_sub, 2)
                else
                    sub_PS_dict[:Xfixed] = zeros(nrow(new_data), 0)
                    sub_PS_dict[:Xfixed_N] = 0
                    sub_PS_dict[:Xfixed_names] = Symbol[]
                end
            end

            if haskey(sub_M, :components)
                sub_ps_basis_registry = Dict{Symbol, Any}()
                sub_smooth_specs = filter(s -> s.structure == :smooth, sub_M.components)
                for spec in sub_smooth_specs
                    v_sym = Symbol(spec.var)
                    vars = get(spec.params, :positional_args, [])
                    n_vars = length(vars)
                    if haskey(sub_M.basis_matrices, v_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                        m_obj = spec.component_obj
                        model_type_str = lowercase(string(typeof(m_obj)))
                        nb = size(sub_M.basis_matrices[v_sym], 2)
                        if n_vars == 1 
                            sub_ps_basis_registry[v_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, Symbol(vars[1])], nb; spec.params...)
                        elseif n_vars > 1
                            coords_new = Matrix{Float64}(new_data[!, Symbol.(vars)])
                            if n_vars == 2; sub_ps_basis_registry[v_sym] = bstm_smooth_basis_2D(model_type_str, coords_new, nb; spec.params...);
                            elseif n_vars == 3; sub_ps_basis_registry[v_sym] = bstm_smooth_basis_3D(model_type_str, coords_new, nb; spec.params...);
                            elseif n_vars == 4; sub_ps_basis_registry[v_sym] = bstm_smooth_basis_4D(model_type_str, coords_new, nb; spec.params...);
                            end
                        end
                    end
                end
                sub_PS_dict[:basis_matrices] = sub_ps_basis_registry
            end

            if haskey(sub_M, :likelihood_specs) && !isempty(sub_M.likelihood_specs) 
                sub_lik_params = sub_M.likelihood_specs[1]
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:log_offsets], :log_offsets)
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:weights], :weights)
                _resolve_obs_param!(sub_PS_dict, sub_lik_params, new_data, [:trials], :trials)
            end
            _precompute_likelihood_params!(sub_PS_dict)

            PS_dict[:nested_prediction_sets][key] = NamedTuple(sub_PS_dict)
        end
    end

    PS = NamedTuple(PS_dict)

    raw_arch = get(M_train, :model_arch, "univariate")
    arch_type = if raw_arch == "multivariate"; MultivariateArchitecture()
    elseif raw_arch == "multifidelity"; MultifidelityArchitecture()
    else; UnivariateArchitecture(); end 

    chain_sub = chain[1:min(n_samps, end), :, :]

    res = _reconstruct(arch_type, "prediction", chain_sub, M_train, PS, alpha)

    # Slice the prediction part from the full summary.
    N_train = M_train.y_N
    
    function slice_summary(summary)
        if summary isa AbstractVector # Multivariate case 
            return [(mean=s.mean[(N_train+1):end], median=s.median[(N_train+1):end], std=s.std[(N_train+1):end], lower=s.lower[(N_train+1):end], upper=s.upper[(N_train+1):end]) for s in summary]
        else # Univariate case
            return (mean=summary.mean[(N_train+1):end], median=summary.median[(N_train+1):end], std=summary.std[(N_train+1):end], lower=summary.lower[(N_train+1):end], upper=summary.upper[(N_train+1):end])
        end
    end

    return (
        predictions_denoised = slice_summary(res.predictions_denoised),
        predictions_noisy = slice_summary(res.predictions_noisy),
        pstats = res,
        PS = PS
    )
end

function post_stratification_weights(res, M, PS, samples_denoised)
    # Purpose: Computes post-stratification weights to scale sample-level predictions to population-level estimates.
    # Rationale: This is essential for generating total abundance or biomass indices from survey data.
    #            The weight for an observation `i` in stratum `j` is calculated as `Area(j) / n_obs_in_stratum(j)`.
    #            Multiplying the predicted density at `i` by this weight gives its contribution to the total stratified estimate.
    # Assumptions:
    #   1. `M` contains a `:strata_info` DataFrame with `stratum_id` and `stratum_area` columns.
    #   2. The data (`M.data` and optionally `PS.data`) contains a `stratum_id` column.
    # Inputs:
    #   - res: The main results object (not used in this implementation but kept for API consistency).
    #   - M: The model configuration object for the training data.
    #   - PS: The prediction set configuration object (can be `nothing`).
    #   - samples_denoised: A matrix of posterior predictions [n_obs x n_samples].
    # Outputs: A matrix of weights of the same size as `samples_denoised`.

    # #
    # Input validation
    if !haskey(M, :strata_info) || !("stratum_id" in names(M.strata_info)) || !("stratum_area" in names(M.strata_info))
        @warn "Post-stratification requires `:strata_info` in the model configuration with `stratum_id` and `stratum_area` columns. Returning ones." 
        return ones(Float64, size(samples_denoised))
    end
    if !hasproperty(M.data, :stratum_id)
        @warn "Post-stratification requires a `stratum_id` column in the training data. Returning ones."
        return ones(Float64, size(samples_denoised))
    end

    # #
    # Combine stratum IDs from training and prediction sets
    strata_info = M.strata_info
    strata_ids_train = M.data.stratum_id

    strata_ids_full = if !isnothing(PS)
        if !hasproperty(PS.data, :stratum_id)
            @warn "Prediction set provided but is missing `stratum_id` column. Post-stratification weights will only be calculated for training data."
            strata_ids_train
        else
            vcat(strata_ids_train, PS.data.stratum_id)
        end
    else
        strata_ids_train
    end

    n_obs_total = length(strata_ids_full)
    n_samples = size(samples_denoised, 2)

    # #
    # Calculate the weight for each stratum (Area / N_obs)
    unique_strata = unique(strata_info.stratum_id)
    stratum_area_map = Dict(row.stratum_id => row.stratum_area for row in eachrow(strata_info))
    obs_counts = StatsBase.countmap(strata_ids_full)
    
    stratum_weight_map = Dict{eltype(unique_strata), Float64}()
    for stratum in unique_strata
        area = get(stratum_area_map, stratum, 0.0)
        count = get(obs_counts, stratum, 0)
        stratum_weight_map[stratum] = count > 0 ? area / count : 0.0
    end

    # #
    # Map stratum weights to each observation
    obs_weights = [get(stratum_weight_map, id, 0.0) for id in strata_ids_full]

    # #
    # Return weights matrix, broadcasted across all posterior samples
    return repeat(obs_weights, 1, n_samples)
end

function model_results_plots(res)
    # Purpose: Displays all plots generated by `model_results_comprehensive`.
    # Rationale: A simple convenience function to iterate through and display the
    #            contents of the `plots` object returned by the main results function.
    if !hasproperty(res, :plots) || isempty(res.plots)
        println("No plots found in the results object.") 
        return
    end

    println("--- Displaying Generated Plots ---")
    for (plot_name, plot_obj) in pairs(res.plots)
        if plot_obj isa Dict # Handle nested plot dictionaries like for smooth_effects
            for (sub_name, sub_plot) in plot_obj
                println("--- Plot: $plot_name -> $sub_name ---")
                display(sub_plot)
            end
        else
            println("--- Plot: $plot_name ---")
            display(plot_obj)
        end
    end
    println("--- End of Plots ---")
end

function plot_choropleth(values::AbstractVector, polygons::Vector; title="Spatial Distribution", cmap=:viridis)
    # Purpose: A simple choropleth plotting utility.
    # Rationale: Provides a basic visualization for spatial fields on polygonal units.
    plt = plot(aspect_ratio=:equal, title=title, legend=false, grid=false, showaxis=false, xticks=false, yticks=false)

    # Determine the color range for normalization
    min_val, max_val = extrema(values)
    
    for i in 1:min(length(polygons), length(values))
        poly_coords = polygons[i]
        
        # A valid polygon requires at least 3 vertices
        if length(poly_coords) > 2 
            # Extract x and y coordinates, filtering out any NaN values
            px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
            py = [pt[2] for pt in poly_coords if !isnan(pt[2])]
            
            # Proceed only if there are valid coordinates 
            if !isempty(px)
                # Ensure the polygon is closed for plotting
                if (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end
                
                plot!(plt, px, py, seriestype=:shape, fill_z=values[i], c=cmap, linecolor=:black, lw=0.5, fillalpha=0.8, label=nothing) 
            end
        end
    end
    return plt
end 


function bstm_cv_orchestrator(
    formula::String, 
    data::DataFrame; 
    method::Symbol = :kfold, 
    cv_var::Symbol = :s_idx, 
    n_folds::Int = 5, 
    n_samples::Int = 500, 
    sampler = NUTS(500, 0.65), 
    alpha = 0.05, 
    cv_space_vars::Vector{Symbol} = [:s_x, :s_y],
    kwargs...
)    
    # Purpose: An orchestration utility for performing cross-validation. It supports standard 
    #          k-fold, Leave-One-Location-Out (LOLO), spatial blocking, and temporal blocking/forward-chaining
    #          strategies to assess model performance on held-out data.
    # Rationale: Provides a standardized and flexible way to evaluate model predictive performance
    #            while accounting for spatial and temporal data structures.
    # Inputs:
    #   - formula: The bstm model formula.
    #   - data: The input DataFrame.
    #   - method: The CV method. One of `:kfold`, `:lolo`, `:spatial_block`, `:temporal_block`, `:temporal_forward_chain`.
    #   - cv_var: The column name to use for grouping/blocking (for `:lolo`, `:temporal_block`, `:temporal_forward_chain`).
    #   - n_folds: The number of folds for k-fold or blocking methods.
    #   - sampler: The Turing sampler to use.
    #   - cv_space_vars: Columns for spatial coordinates for `:spatial_block`.
    #   - kwargs: Additional arguments passed to `bstm_config`.
    # Outputs: A NamedTuple containing fold-level results and summary metrics.
    
    meta_discovery = decompose_bstm_formula(formula)
    response_name = Symbol(meta_discovery.outcomes[1][:var])

    folds_indices = Vector{Vector{Int}}()
    is_forward_chain = false

    if method == :lolo
        if !hasproperty(data, cv_var); error("LOLO cross-validation requires the specified `cv_var` column ':$cv_var' in the data."); end
        unique_locs = unique(data[!, cv_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, cv_var]))
        end
    elseif method == :spatial_block
        if !all(hasproperty(data, v) for v in cv_space_vars); error("Spatial block cross-validation requires coordinate columns specified in `cv_space_vars`: $cv_space_vars."); end
        coords = Matrix(data[!, cv_space_vars])' # kmeans expects features in rows
        R = Clustering.kmeans(coords, n_folds; maxiter=200, display=:none)
        assignments = R.assignments
        for k in 1:n_folds
            fold_k_indices = findall(x -> x == k, assignments)
            if !isempty(fold_k_indices); push!(folds_indices, fold_k_indices); end 
        end
    elseif method == :temporal_block
        if !hasproperty(data, cv_var); error("Temporal block cross-validation requires the specified `cv_var` column ':$cv_var' in the data."); end
        unique_times = sort(unique(data[!, cv_var]))
        fold_size = cld(length(unique_times), n_folds) # ceiling division
        for i in 1:n_folds
            start_idx = (i - 1) * fold_size + 1
            end_idx = min(i * fold_size, length(unique_times))
            if start_idx > length(unique_times); continue; end 
            time_block = unique_times[start_idx:end_idx]
            push!(folds_indices, findall(t -> t in time_block, data[!, cv_var]))
        end
    elseif method == :temporal_forward_chain
        if !hasproperty(data, cv_var); error("Forward-chaining cross-validation requires the specified `cv_var` column ':$cv_var' in the data."); end
        is_forward_chain = true
        unique_times = sort(unique(data[!, cv_var]))
        if length(unique_times) <= n_folds; @warn "Number of unique time points ($(length(unique_times))) is less than or equal to `n_folds` ($n_folds). Consider reducing `n_folds` for forward-chaining."; end
        test_times = unique_times[end-n_folds+1:end]
        for t in test_times
            push!(folds_indices, findall(x -> x == t, data[!, cv_var]))
        end
    else # Default to k-fold
        n_obs = size(data, 1)
        row_indices = Random.randperm(n_obs)
        fold_size = cld(n_obs, n_folds)
        for i in 1:n_folds
            idx_start = (i - 1) * fold_size + 1
            idx_end = min(i * fold_size, n_obs)
            if idx_start > n_obs; continue; end 
            push!(folds_indices, row_indices[idx_start:idx_end])
        end
    end

    fold_results = []
    n_actual_folds = length(folds_indices)

    for (f_idx, test_idx) in enumerate(folds_indices)
        test_data = data[test_idx, :]
        
        train_data = if is_forward_chain
            min_test_time = minimum(test_data[!, cv_var])
            train_idx = findall(t -> t < min_test_time, data[!, cv_var])
            data[train_idx, :]
        else
            train_mask = trues(size(data, 1))
            train_mask[test_idx] .= false
            data[train_mask, :]
        end

        if nrow(train_data) == 0; @warn "Fold $f_idx created an empty training set. Skipping."; continue; end 

        opt_train = bstm_config(formula, train_data; kwargs...)
        model_train = bstm(opt_train)
        chain_train = sample(model_train, sampler, n_samples; progress=false)
        res_pred = predict(model_train, chain_train, test_data; n_samples=div(n_samples, 2), alpha=alpha)

        y_test_obs = test_data[!, response_name]
        y_test_pred = res_pred.predictions_denoised.mean

        if length(y_test_obs) == length(y_test_pred)
            residuals = y_test_obs .- y_test_pred
            rmse = sqrt(Statistics.mean(residuals.^2))
            ss_res = sum(residuals.^2)
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2) # This can be zero if all test obs are the same. 
            r2 = 1.0 - (ss_res / (ss_tot + 1e-15))
            push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
        else
            @warn "Fold $f_idx: Prediction length mismatch. Observed: $(length(y_test_obs)), Predicted: $(length(y_test_pred))"
        end
    end

    mean_rmse = Statistics.mean([r.rmse for r in fold_results])
    mean_r2 = Statistics.mean([r.r2 for r in fold_results])

    return (
        folds = fold_results,
        mean_rmse = mean_rmse,
        mean_r2 = mean_r2,
        response_var = response_name,
        method = method,
        n_folds = n_actual_folds
    )
end

# ==============================================================================
# SECTION 5: MODEL SELECTION AND COMPARISON
# ==============================================================================

function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)    
    # Purpose: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance 
    #          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
    # Inputs: model_obj, chain, alpha.
    # Outputs: A NamedTuple containing the LOO object, metrics, log-likelihood matrix, and Pareto k values.
    
    # #
    # 1. Metadata and Architecture Extraction
    # Rationale: M contains the configuration and technical registry required for reconstruction.
    M = model_obj.args.M
    raw_arch = get(M, :model_arch, "univariate")

    # #
    # 2. Technical Dispatch Resolution
    # Mapping the configuration string to the architectural dispatch types.
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture() 
    end

    # #
    # 3. Latent Component Reconstruction for Likelihood Registry
    # Rationale: _reconstruct generates the [Samples x Observations] log-likelihood matrix.
    # We utilize alpha for consistent summarization during the recovery phase.
    println("Audit: Recovering pointwise log-likelihood registry...")
    res = _reconstruct(arch_type, "loo_recovery", chain, M, nothing, alpha)

    # #
    # 4. Matrix Extraction and Validation
    # Rationale: Ensuring the log_likelihood matches the observation grid dimensions.
    log_lik = res.log_likelihood 
    n_samples, n_obs = size(log_lik)

    println("Audit: Processing ", n_samples, " samples for ", n_obs, " observations.")

    # #
    # 5. PSIS-LOO Calculation via PosteriorStats
    # Rationale: LOO-CV provides a reliable estimate of out-of-sample predictive performance.
    loo_result = nothing
    try
        loo_result = loo(log_lik)
    catch e
        @error "BSTM Selection Error: PSIS-LOO calculation failed. Error: " * string(e)
        return nothing
    end

    println("\n--- BSTM Model Selection Report ---")
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
        log_likelihood = log_lik,
        pareto_k = pareto_k
    )
end

function compare_components(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses 
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise 
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.

    println("--- Starting BSTM Component Comparison ---")

    # #
    # 1. LOO Object Extraction
    loo_a = loo_a_report.loo_obj
    loo_b = loo_b_report.loo_obj

    # #
    # 2. Formal Selection Metric Calculation
    comparison_stats = nothing
    try
        comparison_stats = compare([loo_a, loo_b])
    catch e
        @error "BSTM Comparison Error: Selection suite failed. Error: " * string(e)
        return nothing
    end

    # #
    # 3. Parameter and Diagnostic Extraction
    p_loo_a = loo_a_report.metrics.p_loo
    p_loo_b = loo_b_report.metrics.p_loo
    elpd_a = loo_a_report.metrics.elpd
    elpd_b = loo_b_report.metrics.elpd

    # #
    # 4. Report Generation
    println("\n--- BSTM Component Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2] 
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing component structures provide indistinguishable predictive density.")
    end

    # #
    # 5. Table Construction
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



 
const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :fixed         => process_fixed_module!,
    :mixed         => process_mixed_module!,
    :nested        => process_nested_module!,
    :eigen         => process_eigen_module!,
    :dynamics      => process_dynamics_module!,
    :custom        => process_custom_module!,
    :interact      => process_interact_module!,
    :random        => process_random_module!,
    :lgcp          => process_lgcp_module! # Add the new LGCP processor
);