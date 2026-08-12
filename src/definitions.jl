

# ==============================================================================
# SECTION 1: CORE COMPONENT STRUCTURES AND TYPE DEFINITIONS
# ==============================================================================


abstract type Component end

abstract type ComponentOperator <: Component end

# All component models must inherit from ComponentModel
abstract type ComponentModel <: Component end

struct None <: ComponentModel end # Placeholder for empty components


# these constructors will be added upon by each component file
const COMPONENT_CONSTRUCTORS = Dict{Symbol, Function}(
    :none => (p, params) -> None()
)

# these structures will be added to where required in the component file
const MODEL_TO_STRUCTURE_MAP = Dict{Symbol, Symbol}(
    :none => :none  # dummy value to initiate the map 
)



abstract type AbstractModelArchitecture end
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end


const BSTM_MODULE_KEYWORDS = Set([ 
    :intercept, :fixed, :mixed, :random, :nested, :eigen, :dynamics, :pointprocess, :custom,
    :zscore, :log, :center, :scale, :sciml
]);
  
const TRANSFORMATION_FUNCTIONS = Set([:zscore, :log, :center, :scale])


 
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
 

const COMPONENT_TYPE_REGISTRY = Dict{Symbol, Type{<:ComponentModel}}(
    :none => None
)

  
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

# Purpose: Defines a mapping for models whose structure is unambiguous.
# Rationale: This constant provides a central registry for models that always correspond
#            to a specific structure (e.g., :leroux is always :spatial). It is used by
#            `_infer_structure_from_args` to automatically determine the structure,
#            reducing the need for explicit `structure=` specification by the user.
#            This map acts as a fallback if a component file has not yet registered its
#            model type in `MODEL_TO_STRUCTURE_MAP`.
const KNOWN_UNAMBIGUOUS_MODELS = Dict{Symbol, Symbol}(
    :icar => :spatial,
    :besag => :spatial,
    :bym2 => :spatial,
    :leroux => :spatial,
    :sar => :spatial, 
    :dag => :spatial,
    :spde => :spatial,
    :localadaptive => :spatial, :mosaic => :spatial, :networkflow => :spatial,
    :pointprocess => :spatial, # The consolidated point process component is spatial
    :bcgn => :spatial, # Bipartite Graph Convolutional Network

    :ar1 => :temporal,
    :ar2 => :temporal,
    :rw1 => :temporal,
    :rw2 => :temporal,
    :cyclic => :temporal,
    :harmonic => :temporal,
    :tar => :temporal,

    :pspline => :smooth,
    :bspline => :smooth,
    :tps => :smooth, 
    :wavelet => :smooth,
    :waveletgp => :smooth,
    :spectralgp => :smooth,
    
    :eigen => :smooth,
    :moran => :smooth,
    :barycentric => :smooth, 

    :svar => :spacetime,
    :dynamics => :spacetime,
    :composed => :spacetime, # Default, can be overridden
    :nonstationaryvariance => :spacetime,
    :adaptivesmooth => :smooth
)

# Version 1.0.3 (2026-08-08)
# Purpose: Defines a set of model names whose structure is truly ambiguous and must be inferred from context.
# Rationale: These models can represent different structures (spatial, temporal, or smooth) depending
#            on the number and type of input variables. This set is intentionally small, as most models
#            have a clear, unambiguous structure.
const AMBIGUOUS_MODELS = Set([
    :iid, # Can be spatial (with W), temporal (with time var), or smooth (generic var)
    :gp,  # Can be spatial (coords), temporal (time var), or smooth (generic var)
    :rff, # Same as gp
    :fft, # Can be spatial (coords) or temporal (time var)
])



# This new constant maps legacy module names to their corresponding structure type.
const LEGACY_MODULES = Dict(
    :spatial => :spatial,
    :temporal => :temporal,
    :smooth => :smooth,
    :seasonal => :seasonal,
    :spacetime => :spacetime
);

 

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
    :tar => Dict(), :pointprocess => Dict(:model => :lgcp, :inner_model => :icar, :grid_areas => "unit"),
    :localadaptive => Dict(:n_clusters => 5),
    :eigen => Dict(:n_factors => 1)
)



# interface definitions
 
"""
    get_datastructures!(m_type::Type{<:ComponentModel}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup and validation for a component type.

This method is dispatched on the component's *type* (e.g., `IID`, `AR1`) and runs
early in the `bstm_config` pipeline. It is responsible for:
1.  Preparing necessary data structures (e.g., adjacency matrix `W`, basis functions, indices).
2.  Modifying the global model configuration `M` (e.g., setting `M[:s_N]`, `M[:s_idx]`).

# Arguments
- `m_type`: The `Type` of the `ComponentModel` (e.g., `IID`, `AR1`).
- `M`: The main model configuration dictionary, which can be mutated.
- `mod_data`: A dictionary containing parsed module data (e.g., variables, parameters).

# Returns
- `true`: If a component object should be created for this module.
- `false`: If this module only performs setup and does not require a component object
           (e.g., interaction flags that are handled globally).

# Assumptions
- This method is called before the component object itself is instantiated.
- It is responsible for ensuring that all data-related prerequisites for the
  component's `get_precomputes` method are met.
"""
function get_datastructures! end

"""
    get_precomputes(m::ComponentModel, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for a component instance.

This method is dispatched on the `ComponentModel` instance and runs after the
component has been instantiated. It is responsible for:
1.  Building template matrices (e.g., `Q_template` for GMRFs).
2.  Computing spectral decompositions (`U`, `L` for AD-friendly sampling).
3.  Generating fixed basis functions or other static structures.

# Arguments
- `m`: The `ComponentModel` instance.
- `M`: The main model configuration `NamedTuple`.
- `mod_data`: A dictionary containing parsed module data.

# Returns
- A `NamedTuple` containing all precomputed items necessary for code generation
  and posterior reconstruction (e.g., `(Q_template=..., U=..., L=...)`).

# Assumptions
- All data-dependent setup (e.g., `s_N`, `t_N`, `W`) has already been performed
  by `get_datastructures!`.
"""
function get_precomputes end

"""
    get_priors(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the component's priors.

This method is dispatched on the `ComponentModel` instance and is responsible for
producing Julia/Turing code snippets that define:
1.  Priors for the component's hyperparameters (e.g., `sigma`, `rho`).
2.  Priors for the component's latent fields (e.g., `raw ~ MvNormal(...)`).

# Arguments
- `m`: The `ComponentModel` instance.
- `spec`: A `NamedTuple` containing the component's full specification (key, structure, etc.).
- `arch`: The model architecture (`"univariate"`, `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for priors.
"""
function get_priors end

"""
    get_updates(m::ComponentModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the component's effect and adding it to the linear predictor.

This method is dispatched on the `ComponentModel` instance and is responsible for
producing Julia/Turing code snippets that define:
1.  The logic for transforming sampled latent variables into the component's effect.
2.  Adding this effect to the linear predictor (`eta`).

# Arguments
- `m`: The `ComponentModel` instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"`, `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for the component's update logic.
"""
function get_updates end

"""
    get_effects(m::ComponentModel, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the component's effect from the MCMC chain's posterior samples for diagnostics and visualization.

This method is dispatched on the `ComponentModel` instance and is responsible for:
1.  Extracting relevant parameter samples from the MCMC `chain`.
2.  Reconstructing the component's contribution to the linear predictor for each sample.
3.  Returning structured results (e.g., mean, lower, upper credible intervals) of the effect.

# Arguments
- `m`: The `ComponentModel` instance.
- `chain`: The MCMC chain object containing posterior samples.
- `M`: The main model configuration `NamedTuple`.
- `n_samples`: The total number of posterior samples.
- `outcomes_N`: The total number of outcomes in the model.
- `p_names`: A `NamedTuple` of parameter names for easy access.
- `spec`: A `NamedTuple` containing the component's full specification.
- `PS`: A `NamedTuple` containing prediction set data, or `nothing` for in-sample reconstruction.
- `N_total`: The total number of observations (training + prediction).

# Returns
- A `NamedTuple` (e.g., `(structured=..., noisy=...)`) containing the reconstructed effects.
"""
function get_effects end
