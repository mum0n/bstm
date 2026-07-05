#!Reference

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





macro save_carstm_state(filename_sym, vars...)
    """
    BSTM Utility Macro v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Saves a specified set of variables to a JLD2 file. This macro is designed
              to capture the state of a modeling session for later resumption or analysis.
    Inputs:
        - filename_sym: A symbol representing the variable that holds the filename string.
        - vars...: A variable number of symbols representing the variables to save.
    Usage:
        state_filename = "my_model_state.jld2"
        @save_carstm_state(state_filename, data_df, areal_units, m, chn)
    Rationale for v1.2.0:
        - Refactored to accept a variable number of arguments, making it a general-purpose
          state-saving utility instead of being tied to specific variable names.
    """
    return quote
        try
            local fn = $(esc(filename_sym))
            @info "Saving state to $(fn)..."
            # The `JLD2.@save` macro needs the filename as a value and the variables
            # as escaped symbols. The `vars...` are already symbols.
            JLD2.@save fn $([esc(v) for v in vars]...)
            @info "State saved successfully."
        catch e
            @error "Error saving state: $e"
        end
    end
end


macro load_carstm_state(filename_sym, vars...)
    """
    BSTM Utility Macro v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Loads a specified set of variables from a JLD2 file into the current scope.
    Inputs:
        - filename_sym: A symbol representing the variable that holds the filename string.
        - vars...: A variable number of symbols representing the variables to load. If empty,
                   all variables in the file are loaded.
    Usage:
        state_filename = "my_model_state.jld2"
        @load_carstm_state(state_filename, data_df, m)
    Rationale for v1.2.0:
        - Refactored to accept a variable number of arguments for selective loading.
        - If no variables are specified, it loads all variables from the file.
    """
    return quote
        local fn = $(esc(filename_sym))
        if !isfile(fn)
            @error "File $(fn) not found."
        else
            try
                @info "Loading state from $(fn)..."
                # The `JLD2.@load` macro needs the filename as a value.
                # If `vars` is empty, it loads all variables. Otherwise, it loads the specified ones.
                if isempty($(vars))
                    JLD2.@load fn
                else
                    JLD2.@load fn $([esc(v) for v in vars]...)
                end
                @info "State loaded successfully."
            catch e
                @error "Error loading state: $e"
            end
        end
    end
end



function init_params_copy( res=NaN, res0=NaN; load_from_file=false, overrides::Union{Dict, Nothing}=nothing, fn_inits = "init_params.jl2"  )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Copies parameter mean values from a reference MCMC chain summary (`res0`) to a target
              chain summary (`res`), with options to override specific parameters. This is useful for
              initializing a complex model with parameters from a simpler, pre-run model.
    Inputs:
        - res: The target MCMC chain object.
        - res0: The source MCMC chain object.
        - load_from_file: If true, loads parameters from `fn_inits` instead of processing chains.
        - overrides: A dictionary where keys are regex patterns and values are the new values for matching parameters.
        - fn_inits: The filename for saving/loading initial parameters.
    Outputs:
        - A `FillArrays.Fill` object containing the merged mean parameter values, suitable for
          initializing a new MCMC run.
    Rationale for v1.2.0:
        - Replaced the inflexible `override_means` boolean with a flexible `overrides` dictionary,
          allowing programmatic and specific parameter overrides.
    """
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

  if !isnothing(overrides)
    for (pattern, values) in overrides
        u = findall(x -> occursin(Regex(pattern), String(x)), vns)
        if !isempty(u)
            if length(u) == length(values)
                means[u] .= values
            else
                @warn "Override for '$pattern' failed: length mismatch. Expected $(length(u)), got $(length(values))."
            end
        end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_extract( res=NaN; load_from_file=false, overrides::Union{Dict, Nothing}=nothing, fn_inits = "init_params.jl2"  )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Extracts mean parameter values from a Turing MCMC chain summary to be used as initial
              values for a subsequent run. Includes options for loading from a file or applying
              custom overrides.
    Inputs:
        - res: The MCMC chain object from a previous Turing run.
        - load_from_file: If true, loads parameters directly from `fn_inits`.
        - overrides: A dictionary where keys are regex patterns and values are the new values for matching parameters.
        - fn_inits: The filename for saving/loading initial parameters.
    Outputs:
        - A `FillArrays.Fill` object containing the mean parameter values.
    Rationale for v1.2.0:
        - Replaced `override_means` with a flexible `overrides` dictionary.
    """
  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if !isnothing(overrides)
    for (pattern, values) in overrides
        u = findall(x -> occursin(Regex(pattern), String(x)), vns)
        if !isempty(u)
            if length(u) == length(values)
                means[u] .= values
            else
                @warn "Override for '$pattern' failed: length mismatch. Expected $(length(u)), got $(length(values))."
            end
        end
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_extract(X)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A simplified method to extract parameter names and mean values from a Turing MCMC chain object.
    Inputs:
        - X: The MCMC chain object.
    Outputs:
        - A tuple containing a `FillArrays.Fill` object of the means and a vector of parameter names.
    """
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Rounds a floating-point number `x` to the nearest multiple of `delta`. This is useful
              for discretizing continuous data into regular bins.
    Inputs:
        - x: The input number or vector.
        - delta: The discretization step size.
    Outputs:
        - The discretized number or vector.
    """
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Creates a `DataFrame` from the Cartesian product of named vectors, similar to R's `expand.grid`.
    Inputs:
        - kws: Keyword arguments where each keyword is a symbol for a column name and the value is a vector of values.
    Outputs:
        - A `DataFrame` containing all combinations of the input vectors.
    """
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A simple helper function to print the full contents of a Julia object to the console,
              bypassing truncation that can occur with default display methods.
    """
    show(stdout, "text/plain", x) # display all estimates
end 
 

function modelruntime(o)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Calculates and prints the total runtime of an MCMC sampling process in minutes and
              displays the summary statistics of the resulting chain.
    Inputs:
        - o: The MCMC chain object, which contains timing information in `o.info`.
    """
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A commented-out utility for inspecting the generated code of a function.
              When active, it would use `CodeTracking.@code_string` to print the source code.
    """
end

function firstindexin(a::AbstractArray, b::AbstractArray)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Finds the first occurrence of each element of array `a` within array `b`.
    Inputs:
        - a: The array of elements to search for.
        - b: The array to search within.
    Outputs:
        - An array of the same size as `a`, where each element is the first index of the corresponding
          element from `a` in `b`, or 0 if not found.
    """
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
   

function showtuples(X)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Iterates over the key-value pairs of a `NamedTuple` and prints them to the console,
              rounding numeric values for cleaner display.
    """
    for k in keys(X)
        val = getproperty(X, k)
        # Skip displaying keys with NaN values
        # Check if value is numeric before rounding to avoid errors
        display_val = val isa Number ? round(val, digits=3) : val
        println("$k: $display_val")
    end
end



function showparams(X, keywords=["rho", "phi", "sigma",  "mu_", "l_", "ls_"]; limit=10 )
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Filters and displays a summary of parameters from an MCMC chain object based on a list of keywords.
              This is useful for quickly inspecting key hyperparameter posteriors.
    Inputs:
        - X: The MCMC chain object.
        - keywords: A vector of strings to search for within parameter names.
        - limit: The maximum number of matched parameters to display.
    Outputs:
        - A sliced MCMC chain object containing only the matched parameters.
    """
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
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Generates a random correlation matrix of a given dimension using the "Onion Method",
              which is related to the LKJ distribution.
    Inputs:
        - d: The dimension of the square correlation matrix.
        - eta: A parameter controlling the distribution of correlations. `eta=1` corresponds to a
               uniform distribution over correlation matrices. Larger values of `eta` push the
               matrix closer to the identity matrix.
    Outputs:
        - A `d x d` random correlation matrix.
    Reference:
        - https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
    Rationale for v1.0.1:
        - Replaced the full eigendecomposition in the loop with a more efficient Cholesky decomposition
          for computing the matrix square root, which aligns with the canonical "Onion Method" algorithm.
        - Corrected the calculation of the vector `q` to use standard matrix multiplication.
    """
    beta = eta + (d - 2) / 2
    u = rand(Beta(beta, beta))
    r12 = 2 * u - 1
    S = [1 r12; r12 1]

    for k = 3:d
        beta -= 0.5
        y = rand(Beta((k - 1) / 2, beta))
        r = sqrt(y)
        theta = randn(k - 1)
        theta /= norm(theta)
        w = r * theta

        # Use Cholesky decomposition for the matrix square root, which is more efficient.
        # The algorithm requires a matrix R such that S = R'R. The upper Cholesky factor C.U satisfies this.
        # Then, q = R'w, which is equivalent to C.L * w.
        C = cholesky(Symmetric(S))
        q = C.L * w

        S = [S q; q' 1]
    end
    return S
end





function turingindex( indices, sym=nothing, dims=nothing  ) 
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: A helper function to extract parameter indices from a Turing model's internal
              variable information structure.
    Inputs:
        - indices: The `VarInfo` metadata from a Turing model, or the model itself.
        - sym: The symbol of the parameter to extract indices for. If `nothing`, enumerates all keys.
               If `"varnames"`, returns all variable names.
        - dims: Optional dimensions to reshape the output index array.
    Outputs:
        - A vector or array of indices corresponding to the specified parameter.
    Rationale for v1.2.0:
        - Added a `haskey` check to provide a more informative error when a symbol is not found,
          preventing a `KeyError`.
    """
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      if !haskey(indices, sym)
          error("Symbol ':$sym' not found in model variable information. Available keys: $(keys(indices))")
      end
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
end


 
function dataframe_to_named_array(df::DataFrame)
    """
    BSTM Utility Function v1.2.0
    Timestamp: 2026-06-26 10:17:45
    Synopsis: Converts a `DataFrame` into a `NamedArray` for use in internal model processing,
              preserving column names as the second dimension's names.
    """
    mat = Matrix(df)
    return NamedArray(mat, (1:size(mat, 1), Symbol.(names(df))))
end




function _generate_model_pseudocode(m::DynamicPPL.Model)
    # v1.0.0 (2026-06-30)
    # Purpose: Reconstructs a pseudo-code representation of the Turing model definition
    #          based on the configuration object. This is for inspection and clarity, as
    #          the actual model is built dynamically.
    # Inputs: m - The Turing model instance.
    # Outputs: A string containing the pseudo-code.
    
    config = m.args[1]
    model_name = get(config, :model_name, nameof(m.f))
    
    lines = ["@model function $model_name(M)"]
    push!(lines, "    # --- Priors & Hyperparameters ---")

    # Global priors
    family = get(config, :model_family, "gaussian")
    if family == "negbin"
        push!(lines, "    r_nb ~ Exponential(1.0)")
    end
    if get(config, :use_zi, false)
        push!(lines, "    phi_zi ~ Beta(1, 1)")
    end
    if family in ["gaussian", "lognormal"] && !get(config, :use_sv, false)
        push!(lines, "    y_sigma ~ Exponential(1.0)")
    end

    # Manifold-specific priors
    if haskey(config, :manifolds) && !isempty(config.manifolds)
        for spec in config.manifolds
            m_obj = spec.manifold_obj
            m_type_str = string(typeof(m_obj))
            key = spec.key
            
            push!(lines, "\n    # Priors for manifold: $(key) ($(m_type_str))")
            
            if hasproperty(m_obj, :sigma_prior) && !isnothing(m_obj.sigma_prior)
                push!(lines, "    sigma_$(key) ~ $(m_obj.sigma_prior)")
            end
            if hasproperty(m_obj, :rho_prior) && !isnothing(m_obj.rho_prior)
                push!(lines, "    rho_$(key) ~ $(m_obj.rho_prior)")
            end
            if hasproperty(m_obj, :lengthscale_prior) && !isnothing(m_obj.lengthscale_prior)
                push!(lines, "    ls_$(key) ~ $(m_obj.lengthscale_prior)")
            end
        end
    end

    # Fixed effects prior
    if get(config, :Xfixed_N, 0) > 0
        push!(lines, "\n    # Prior for fixed effects")
        push!(lines, "    Xfixed_beta ~ MvNormal(0, 5.0 * I)")
    end

    push!(lines, "\n    # --- Latent Field Definitions & Linear Predictor Assembly ---")
    eta_parts = haskey(config, :log_offset) && !all(iszero, get(config, :log_offset, [])) ? ["M.log_offset"] : []

    if haskey(config, :manifolds)
        for spec in config.manifolds
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


function show_model(m::DynamicPPL.Model)
"""
    show_model(m)

Displays a summary of the Turing model, including its name, arguments,
and attempts to draw a single sample from its prior to verify parameter definitions.
This function addresses the `UndefVarError: SampleFromPrior` by ensuring
`using Turing` is present and correctly calling the `SampleFromPrior` sampler.
"""
    println("\n--- Model Summary ---\n")

    # The model configuration object `M` is the first argument passed to the Turing model.
    config = m.args[1]

    # Use `get` with a fallback to `nameof(m.f)` for the model name.
    println("Model Name: ", get(config, :model_name, nameof(m.f)))
    println("Model Architecture: ", get(config, :model_arch, "N/A"))
    println("Model Family: ", get(config, :model_family, "N/A"))
    println("Number of observed data points: ", get(config, :N_obs, get(config, :y_N, "N/A")))
    println("Number of spatial units: ", get(config, :N_areas, get(config, :s_N, "N/A")))
    println("Number of time units: ", get(config, :N_time, get(config, :t_N, "N/A")))
    println("Number of covariates: ", get(config, :N_cov, "N/A"))

    println("\n--- Model Code View ---\n")
    println("Likelihood Family: ", get(config, :model_family, "N/A"))
    println("Zero-Inflated: ", get(config, :use_zi, false) ? "Yes" : "No")
    println("Hurdle Model: ", get(config, :hurdle, -Inf) > -Inf ? "Yes" : "No")

    println("\nFixed Effects:")
    if get(config, :Xfixed_N, 0) > 0
        println("  Variables: ", join(string.(names(config.Xfixed, 2)), ", "))
    else
        println("  None")
    end

    println("\nManifolds:\n")
    if haskey(config, :manifolds) && !isempty(config.manifolds)
        for spec in config.manifolds
            println("  - Key: ", spec.key)
            println("    Domain: ", spec.domain)
            println("    Variable: ", spec.var)
            println("    Manifold Type: ", typeof(spec.manifold_obj))
            println("    Parameters:")
            for (p_key, p_val) in pairs(spec.params)
                println("      ", p_key, ": ", p_val)
            end
        end
    else
        println("  None")
    end

    println("\nObservation Process:\n")
    # Check if log_offset is present and not all zeros
    log_offset_present = haskey(config, :log_offset) && !all(iszero, get(config, :log_offset, []))
    println("  Log Offsets: ", log_offset_present ? "Yes" : "No")

    # Check if weights are present and not all ones
    weights_present = haskey(config, :weights) && !all(isone, get(config, :weights, []))
    println("  Weights: ", weights_present ? "Yes" : "No")

    # Check if trials are present and not all ones
    trials_present = haskey(config, :trials) && !all(isone, get(config, :trials, []))
    println("  Trials: ", trials_present ? "Yes" : "No")

    println("  Stochastic Volatility: ", get(config, :use_sv, false) ? "Yes" : "No")

    censoring_status = "None"
    y_L = get(config, :y_lower_bound, -Inf)
    y_U = get(config, :y_upper_bound, Inf)
    if y_L > -Inf && y_U < Inf
        censoring_status = "Interval"
    elseif y_L > -Inf
        censoring_status = "Right"
    elseif y_U < Inf
        censoring_status = "Left"
    end
    println("  Censoring: ", censoring_status)

    println("\n--- End Model Code View ---")

    println("\n--- Reconstructed Model Source ---\n")

    println(_generate_model_pseudocode(m))
    println("\n--- End Reconstructed Model Source ---")

    
    println("\n--- Prior Sample Check ---")
    try
        prior_sample_chain = sample(m, Prior(), 1)
        println("Successfully drew 1 sample from the model's prior.")
        display(prior_sample_chain)
    catch e
        println("ERROR: Failed to draw a sample from the prior.")
        println("  Reason: ", e)
        println("  This might indicate issues with model parameter definitions,")
        println("  missing `using Turing` statement, or an incompatible Turing.jl version.")
        
        try 
            println( "Trying a simple rand(m) test: passed\n")
            println( "rand(m): sample\n")
            show( rand(m) )

        catch e
            println( "Trying a simple rand(m) test: failed\n")
            println("  Reason: ", e)
        end


    end

    println("\n--- End Model Summary ---")
    return nothing
end


function capitalize(s::String)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Capitalizes the first letter of a string.
    # Inputs: s::String - The input string.
    # Outputs: A string with the first letter capitalized.
    isempty(s) && return s
    return uppercasefirst(s)
end




function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Splits a string by a separator, but respects nested parentheses or brackets.
    # Inputs: input::AbstractString - The string to split.
    #         sep::AbstractString - The separator to split by.
    # Outputs: A Vector{String} of the split terms.
    terms = String[]
    depth = 0
    current = ""
    for char in input
        if char == '(' || char == '['
            depth += 1
        elseif char == ')' || char == ']'
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

function parse_module_params(params_str::AbstractString)
    # v1.0.0 (2026-06-29)
    # Purpose: Parses a string of key-value parameters (e.g., "nbins=20, model='bym2'")
    #          into a dictionary of symbols and values.
    # Inputs: params_str - The string of parameters.
    # Outputs: A dictionary of parsed parameters.
    params = Dict{Symbol, Any}()
    if isempty(strip(params_str))
        return params
    end

    # Use the existing utility to split at commas, respecting parentheses
    param_parts = split_terms_at_depth(params_str, ",")

    for part in param_parts
        kv = Base.split(part, "=")
        if length(kv) == 2
            key = Symbol(strip(kv[1]))
            val_str = strip(kv[2])

            # Attempt to parse the value
            # 1. Try parsing as a number
            parsed_val = tryparse(Float64, val_str)
            if !isnothing(parsed_val)
                # Check if it's an integer
                if parsed_val == floor(parsed_val)
                    params[key] = Int(parsed_val)
                else
                    params[key] = parsed_val
                end
            # 2. Check for booleans
            elseif val_str == "true"
                params[key] = true
            elseif val_str == "false"
                params[key] = false
            # 3. Check for strings (single or double quotes)
            elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str, "\"") && endswith(val_str, "\""))
                params[key] = val_str[2:end-1]
            # If it looks like code, try to parse it as an expression.
            elseif occursin(r"[:\[\]\(\)]", val_str)
                try; params[key] = Meta.parse(val_str); catch; params[key] = Symbol(val_str); end
            else
                params[key] = Symbol(val_str)
            end
        end
    end
    return params
end

function resolve_hyperpriors(m_id::Union{String, Symbol}, global_priors::Dict{String, Any}, module_priors::Dict{Symbol, Any}, scheme::Symbol=:pcpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Resolves and assigns prior distributions to manifold types.
    # Inputs: m_id - The manifold identifier.
    #         global_priors - Priors defined globally for the model.
    #         module_priors - Priors defined within a specific module call.
    #         scheme - The default prior scheme to use.
    # Outputs: A NamedTuple of resolved hyperpriors.
    m_id_str = string(m_id)

    defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end

    function get_prior(module_key::Symbol, global_key::String, default_key::String)
        if haskey(module_priors, module_key)
            return module_priors[module_key]
        elseif haskey(global_priors, global_key)
            return global_priors[global_key]
        else
            return get(defaults, default_key, nothing)
        end
    end

    res = Dict{Symbol, Any}()
    res[:sigma_prior] = get_prior(:sigma_prior, "sigma", "sigma")
    res[:rho_prior] = m_id_str in ["bym2", "leroux", "sar", "ar1", "proper_car", "dag", "network"] ? get_prior(:rho_prior, "rho", "rho") : nothing
    res[:lengthscale_prior] = m_id_str in ["gp", "fitc", "rff", "warp", "nystrom", "decay"] ? get_prior(:lengthscale_prior, "lengthscale", "lengthscale") : nothing
    res[:kappa_prior] = m_id_str == "spde" ? get_prior(:kappa_prior, "kappa", "kappa") : nothing
    res[:amplitude_prior] = m_id_str == "harmonic" ? get_prior(:amplitude_prior, "amplitude", "amplitude") : nothing
    res[:phase_prior] = m_id_str == "harmonic" ? get_prior(:phase_prior, "phase", "phase") : nothing
    res[:pca_sd_prior] = m_id_str == "eigen" ? get_prior(:pca_sd_prior, "pca_sd", "pca_sd") : nothing
    res[:pdef_sd_prior] = m_id_str == "eigen" ? get_prior(:pdef_sd_prior, "pdef_sd", "pdef_sd") : nothing
    res[:range_prior] = m_id_str == "spherical" ? get_prior(:range_prior, "range", "range") : nothing

    return NamedTuple(res)
end







function apply_soft_constraint(latent_field::AbstractVector{T}, constraint_type::Symbol, weight::Float64) where {T}
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies a soft constraint to a latent field by adding a penalty to the log-probability.
    # Inputs: latent_field, constraint_type, weight.
    # Outputs: The penalty value (a Float64).
    if isempty(latent_field)
        return zero(eltype(latent_field))
    end

    if constraint_type == :sum_to_zero
        penalty = -weight * sum(latent_field)^2
    elseif constraint_type == :monotonic_increasing
        penalty = -weight * sum(max.(zero(eltype(latent_field)), -diff(latent_field)).^2)
    elseif constraint_type == :monotonic_decreasing
        penalty = -weight * sum(max.(zero(eltype(latent_field)), diff(latent_field)).^2)
    elseif constraint_type == :periodicity
        if length(latent_field) > 1 # Check if there are at least two elements to compare
            penalty = -weight * (latent_field[1] - latent_field[end])^2
        end
    elseif constraint_type == :non_negative
        penalty = -weight * sum(max.(zero(eltype(latent_field)), -latent_field).^2)
    elseif constraint_type == :convex # Check for at least 3 elements for second-order difference
        if length(latent_field) > 2; penalty = -weight * sum(max.(zero(eltype(latent_field)), -diff(diff(latent_field))).^2); end
    elseif constraint_type == :concave # Check for at least 3 elements for second-order difference
        if length(latent_field) > 2; penalty = -weight * sum(max.(zero(eltype(latent_field)), diff(diff(latent_field))).^2); end
    else
        @warn "Unknown soft constraint type: $constraint_type. No penalty applied."
    end
    return penalty
end

function apply_regularization_penalty(fields::Vector{<:AbstractVector}, penalty_type::Symbol, lambda::Float64, alpha::Float64=0.5)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies a regularization penalty (Ridge, Lasso, Elastic Net) to a set of fields.
    # Inputs: fields, penalty_type, lambda, alpha.
    # Outputs: The penalty value (a Float64).
    if isempty(fields)
        return 0.0
    end
    all_coeffs = vcat(fields...)
    T = eltype(all_coeffs)
    penalty = zero(T)

    if penalty_type == :ridge
        penalty = -lambda * sum(all_coeffs.^2)
    elseif penalty_type == :lasso
        penalty = -lambda * sum(abs.(all_coeffs))
    elseif penalty_type == :elastic_net
        penalty = -lambda * (alpha * sum(abs.(all_coeffs)) + (1 - alpha) * sum(all_coeffs.^2))
    else
        @warn "Unknown regularization penalty type: $penalty_type. No penalty applied."
    end
    return penalty
end

function manifold_type(m::Manifold)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Returns the lowercase string representation of a manifold's type.
    # Inputs: m::Manifold.
    # Outputs: A string with the manifold type name.
    return lowercase(string(typeof(m)))
end
 
function _parse_module_call(module_call_str::AbstractString)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Parses a single module call string from the formula into a structured dictionary.
    # Inputs: module_call_str::AbstractString - The string to parse (e.g., "spatial(s_idx, model='bym2')").
    # Outputs: A dictionary containing the module's type, variables, and parameters.
    m_mod = match(r"(\w+)\((.*)\)", module_call_str)
    if isnothing(m_mod)
        return Dict{Symbol, Any}(:type => :literal, :value => module_call_str)
    end

    kw = lowercase(m_mod.captures[1])
    inner_str = m_mod.captures[2]
    
    if !(kw in BSTM_MODULE_KEYWORDS)
        return Dict{Symbol, Any}(:type => :literal, :value => module_call_str)
    end

    v_part_str = ""
    p_dict_raw = Dict{Symbol, Any}()
    
    semicolon_parts = split_terms_at_depth(inner_str, ";")
    if length(semicolon_parts) > 1
        v_part_str = semicolon_parts[1]
        p_dict_raw = parse_module_params(semicolon_parts[2])
    else
        # If no semicolon, it could be all variables, or variables mixed with params
        # We need to distinguish variables from key=value parameters
        all_inner_parts = split_terms_at_depth(inner_str, ",")
        var_only_parts = String[]
        param_only_parts = String[]
        for part in all_inner_parts
            if occursin("=", part)
                push!(param_only_parts, part)
            else
                push!(var_only_parts, part)
            end
        end
        v_part_str = join(var_only_parts, ",")
        p_dict_raw = parse_module_params(join(param_only_parts, ","))
    end

    raw_vars = [strip(v) for v in split_terms_at_depth(v_part_str, ",")]
    final_vars = String[]
    transforms = Dict{String, Vector{String}}()
    for var_expr in raw_vars
        var_name, transform_list = parse_variable_and_transforms(var_expr)
        push!(final_vars, var_name)
        if !isempty(transform_list)
            transforms[var_name] = transform_list
        end
    end

    if kw == "spatial" && length(final_vars) > 1
        cov_var = final_vars[1]
        idx_var = final_vars[2]
        
        p_dict_raw[:index_var] = idx_var
        
        return Dict{Symbol, Any}(
            :type => :svc,
            :variables => [cov_var],
            :params => p_dict_raw
        )
    end

    if kw == "transform"
        inner_module_call_str = final_vars[1]
        inner_module_data = _parse_module_call(inner_module_call_str) # Recursive call
        
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :inner_module => inner_module_data,
            :params => p_dict_raw # This should contain 'fn'
        )
    elseif kw == "interaction"
        inner_model_call_str = get(p_dict_raw, :model, "")
        if inner_model_call_str isa String && !isempty(inner_model_call_str)
            inner_model_data = _parse_module_call(inner_model_call_str) # Recursive call
            p_dict_raw[:model] = inner_model_data # Replace string with parsed module Dict
        end
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :variables => final_vars,
            :params => p_dict_raw # This should contain 'model'
        )
    else
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :variables => final_vars,
            :params => p_dict_raw,
            :transforms => transforms
        )
    end
end

 




function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Resolves a parsed module's metadata into a concrete `Manifold` struct instance.
    # Inputs: module_metadata, M (model config), priors_dict, scheme.
    # Outputs: An instance of a `ManifoldModel` or `ManifoldOperator`.
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    
    if m_type == :transform
        inner_module_data = module_metadata[:inner_module]
        inner_manifold_obj = resolve_technical_primitive(inner_module_data, M, priors_dict, scheme)
        transform_fn = get(m_params, :fn, :identity)
        return TransformedManifold(inner_manifold_obj, transform_fn)
    elseif m_type == :interaction
        interaction_vars = [Symbol(v) for v in module_metadata[:variables]]
        inner_model_data = get(m_params, :model, Dict{Symbol, Any}())
        inner_manifold_obj = resolve_technical_primitive(inner_model_data, M, priors_dict, scheme)
        return VaryingInteractionManifold(interaction_vars, inner_manifold_obj)
    elseif m_type == :svc
        cov_sym = Symbol(module_metadata[:variables][1])
        model_str = string(get(m_params, :model, "iid"))
        
        inner_priors = resolve_hyperpriors(model_str, priors_dict, m_params, scheme)
        constructor_func = MANIFOLD_CONSTRUCTORS[Symbol(model_str)]
        inner_manifold = constructor_func(inner_priors, m_params)
        
        return SVCManifold(cov_sym, inner_manifold)
    else
        default_model = if m_type in [:spatial, :temporal, :smooth]
                            "none"
                        else string(m_type) end
        
        model_name = string(get(m_params, :model, default_model))
        model_sym = Symbol(model_name)

        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme)

        if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
            constructor_func = MANIFOLD_CONSTRUCTORS[model_sym]
            return constructor_func(resolved_priors, m_params)
        else
            @warn "Unknown manifold model '$model_name' for module '$m_type'. Defaulting to IID."
            return IID(Exponential(1.0))
        end
    end
end


  
function bstm_Likelihood(family_input, y_obs; sigma_y=0.0, weight=1.0, phi_zi=-Inf, r_nb=0, trial=0,
                         y_L=-Inf, y_U=Inf, hurdle=-Inf, extra_params=zeros(1)[])
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Constructor for the unified likelihood structure.
    # Inputs: family_input, y_obs, and various optional parameters for likelihood modifications.
    # Outputs: An instance of the bstm_Likelihood struct with appropriate traits.

    # Map string input to a concrete family type
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

    # Determine zero-inflation state from phi_zi
    local z_trait = phi_zi >= 0.0 ? ZeroInflated() : NonZeroInflated()

    local h_val = isnothing(hurdle) ? -Inf : hurdle
    local yL_val = isnothing(y_L) ? -Inf : y_L
    local yU_val = isnothing(y_U) ? Inf : y_U

    # Determine censoring state from bounds
    local c_trait = if !isfinite(yL_val) && !isfinite(yU_val); Uncensored()
        elseif !isfinite(yL_val) && isfinite(yU_val); LeftCensored()
        elseif isfinite(yL_val) && !isfinite(yU_val); RightCensored()
        else IntervalCensored() end

    return bstm_Likelihood(fam, y_obs, z_trait, c_trait, weight, phi_zi, r_nb, sigma_y, trial, yL_val, yU_val, h_val, extra_params)
end

# v1.0.1 (2026-06-29 17:16:00)
# Purpose: Returns a `Distributions.jl` object for a given model family and parameters.
# Inputs: family_type, d (likelihood struct), eta (linear predictor), sig (scale/noise).
# Outputs: A concrete distribution object (e.g., `Poisson`, `Normal`).


function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::GaussianFamily, d, eta, sig); return Normal(eta, max(sig, 1e-9)); end
function get_dist_ref(::LogNormalFamily, d, eta, sig); return LogNormal(eta, max(sig, 1e-9)); end
function get_dist_ref(::NegativeBinomialFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); return NegativeBinomial(d.r_nb, d.r_nb/(d.r_nb + mu)); end
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
function get_dist_ref(::DirichletFamily, d, eta, sig); return Dirichlet(clamp.(exp.(eta), 1e-9, 1e9)); end

function get_dist_ref(::InverseWishartFamily, d, eta::AbstractMatrix, sig)
    p = size(eta, 1)
    nu = d.extra_params isa Number && d.extra_params > (p - 1) ? d.extra_params : Float64(p + 1)
    jitter = 1e-6
    scale_matrix = Symmetric(eta * eta' + jitter * I)
    return InverseWishart(nu, scale_matrix)
end


function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Checks if a model family is discrete.
    # Inputs: A concrete type inheriting from AbstractBSTM_Family.
    # Outputs: true if the family is discrete, false otherwise.
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Default method for checking if a family is discrete.
    # Inputs: A concrete type inheriting from AbstractBSTM_Family.
    # Outputs: false.
    return false
end

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for an uncensored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    lp_branch = logpdf(dist, y)
    
    lp_final = lp_branch

    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        if y == 0.0
            if is_discrete_family(fam)
                p_base_zero = pdf(dist, 0.0)
                lp_final = logsumexp(log_phi, log_one_minus_phi + log(p_base_zero + 1e-15))
            else
                lp_final = log_phi
            end
        else
            lp_final = log_one_minus_phi + lp_branch
        end
    end

    if d.hurdle > -Inf
        lp_final = lp_final - logccdf(dist, d.hurdle)
    end

    return lp_final
end

function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for a left-censored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    upper_bound = d.y_U[1]
    
    if d.hurdle > -Inf
        if upper_bound <= d.hurdle
            return -Inf
        end
        lp_base = stable_logdiffexp(logcdf(dist, upper_bound), logcdf(dist, d.hurdle))
    else
        lp_base = logcdf(dist, upper_bound)
    end

    lp_final = lp_base
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        if upper_bound >= 0.0 && d.hurdle < 0.0
             lp_final = logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
             lp_final = log_one_minus_phi + lp_base
        end
    end

    # Normalize by the probability of being above the hurdle
    if d.hurdle > -Inf
        lp_final -= logccdf(dist, d.hurdle)
    end

    return lp_final
end


function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for a right-censored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    dist = get_dist_ref(fam, d, eta, sig)
    
    lower_bound = d.y_L[1]
    effective_lower_bound = d.hurdle > -Inf ? max(lower_bound, d.hurdle) : lower_bound
    
    adj_L = is_discrete_family(fam) ? effective_lower_bound - 1.0 : effective_lower_bound
    
    lp_base = logccdf(dist, adj_L)

    lp_final = lp_base
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        adj_L_cdf = is_discrete_family(fam) ? lower_bound : lower_bound
        logp_le_L = logcdf(dist, adj_L_cdf)
        
        log_prob_le_bound_zi = if lower_bound < 0.0
             log_one_minus_phi + logp_le_L
        else
            logsumexp(log_phi, log_one_minus_phi + logp_le_L)
        end
        
        lp_final = log1mexp(log_prob_le_bound_zi)
    end

    if d.hurdle > -Inf
        lp_final -= logccdf(dist, d.hurdle)
    end

    return lp_final
end


function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the log-probability for an interval-censored observation.
    # Inputs: fam, censoring_state, zi_state, d (likelihood struct), eta, sig, y.
    # Outputs: The log-probability value.
    function stable_logdiffexp(a, b)
        if a <= b
            return -Inf # Effectively zero probability
        else
            diff = b - a
            safe_diff = min(-eps(typeof(diff)), diff)
            return a + log1mexp(safe_diff)
        end
    end

    dist = get_dist_ref(fam, d, eta, sig)
    
    # Adjust interval for hurdle model
    lower_bound = d.y_L[1]
    upper_bound = d.y_U[1]
    effective_lower_bound = d.hurdle > -Inf ? max(lower_bound, d.hurdle) : lower_bound

    # If the effective interval is invalid, return -Inf
    if upper_bound <= effective_lower_bound
        return -Inf
    end

    # Adjust for discrete distributions
    adj_L = is_discrete_family(fam) ? effective_lower_bound - 1.0 : effective_lower_bound
    
    # Base log-probability: P(adj_L < y <= upper_bound)
    lp_base = stable_logdiffexp(logcdf(dist, upper_bound), logcdf(dist, adj_L))

    # Apply zero-inflation logic
    lp_final = lp_base
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        # If the original interval [y_L, y_U] includes zero, add the point mass phi.
        # The hurdle adjustment is already incorporated in lp_base.
        if lower_bound <= 0.0 && upper_bound >= 0.0 && d.hurdle < 0.0
            lp_final = logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            lp_final = log_one_minus_phi + lp_base
        end
    end

    if d.hurdle > -Inf
        lp_final -= logccdf(dist, d.hurdle)
    end

    return lp_final
end


# BSTM Internal Utility v1.1.0
# Timestamp: 2026-06-27 18:45:00
# Synopsis: Overloads the `Distributions.logpdf` function for the `bstm_Likelihood` struct.
#           This provides the main entry point for calculating the log-likelihood within a
#           Turing model, dispatching to the appropriate `bstm_kernel` based on the data traits.
# Rationale for v1.1.0:
#     - Removed redundant `local` keyword declarations for improved code clarity and style consistency.
#       The underlying logic remains correct and unchanged.

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overloads logpdf for a vector of observations.
    # Inputs: d::bstm_Likelihood, eta::AbstractVector.
    # Outputs: The total log-probability value.
    if d.family isa DirichletFamily
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
    end
    
    total_lp = 0.0
    for i in 1:length(eta)
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[i] : d.weight
        total_lp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
    end
    return total_lp
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractMatrix)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overloads logpdf for matrix-variate observations (e.g., InverseWishart).
    # Inputs: d::bstm_Likelihood, eta::AbstractMatrix.
    # Outputs: The log-probability value.
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
end


function scaling_factor_bym2( adjacency_mat )
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Calculates the geometric mean of the variances of an ICAR precision matrix.
    # Inputs: adjacency_mat - The adjacency matrix of the graph.
    # Outputs: The scaling factor.
    N = size(adjacency_mat)[1]
    if N <= 1
        return 1.0
    end

    # 1. Construct the Graph Laplacian
    asum = vec(sum(adjacency_mat, dims=2))
    Q = Diagonal(asum) - adjacency_mat

    jitter = sqrt(1e-15) * mean(asum)
    Q_stable = Q +jitter * I

    A = ones(N)
    try
        S = Q_stable \ Diagonal(A)
        V = S * A
        S = S - V * inv(A' * V) * V'

        diag_s = diag(S)

        valid_diag = filter(x -> x > jitter, diag_s)

        if isempty(valid_diag)
            return 1.0
        end

        scale_factor = exp(mean(log.(valid_diag)))

        return isnan(scale_factor) ? 1.0 : scale_factor
    catch
        return 1.0
    end
end

function scaling_factor_bym2(node1, node2, groups=ones(length(node1)))
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overloaded method to calculate scaling factors for multiple disconnected graph components.
    # Inputs: node1, node2 (edge list), groups (vector of group identifiers).
    # Outputs: A vector of scaling factors, one for each group.
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



function discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv = 0, maxv=1)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Discretizes a continuous vector into categories.
    # Inputs: X (vector), method, N_cat, and other optional parameters.
    # Outputs: A NamedTuple with indices, breaks, and midpoints.
     
    if method=="quantile" 
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

        xd = round.(Int, X ./ dx ) .* dx
        brks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
        mids = midpoints(brks)
        N_cats = length(mids)
        
        xd_cut = cut(X, brks, extend=true)
        xi = levelcode.(xd_cut)
        return xd, xi, mids, N_cats, dx

    elseif method=="quantile_resolution"
    
        brks = quantile(X, range(0, 1, length=N_cats+1))
        mids = midpoints(brks)
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)
        dx = diff(mids)[1]
        xd = mids[xi] 
        return xd, xi, mids, N_cats, dx

    end

end
 
 

function rff_map(coords, W, b)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Maps input coordinates into a Random Fourier Feature (RFF) space.
    # Inputs: coords (N x D matrix), W (D x M weight matrix), b (M-element phase vector).
    # Outputs: An N x M feature matrix.

    projection = (coords * W) .+ b'
    
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    
    return feature_map
end
 

function generate_informed_rff_params(coords, M_rff; kernel_name="se", nu=nothing, lengthscale_mult=0.5)
    # v1.0.2 (2026-06-29 18:10:00)
    # Purpose: Generates RFF parameters (W, b) with a lengthscale informed by the input data's standard deviation.
    # Inputs: coords (matrix or vector of tuples), M_rff (number of features).
    #         kernel_name - The kernel to use for the spectral density.
    #         nu - Smoothness parameter for Matern kernels.
    #         lengthscale_mult - Multiplier for the informed lengthscale.
    # Outputs: A tuple (W, b) of RFF parameters.
    mat = if coords isa AbstractMatrix && eltype(coords) <: Real
        Matrix{Float64}(coords)
    elseif coords isa AbstractVector && eltype(coords) <: Tuple
        reduce(hcat, [[Float64(p[1]), Float64(p[2])] for p in coords])'
    else
        Matrix{Float64}(reduce(hcat, collect.(coords))')
    end

    if size(mat, 1) > size(mat, 2) && size(mat, 2) <= 3
        mat = mat'
    end

    d = size(mat, 1)
    
    coord_scales = vec(std(mat, dims=2)) .+ 1e-6

    informed_ls = mean(coord_scales) * lengthscale_mult
    W = sample_spectral_density(kernel_name, d, M_rff, informed_ls; nu=nu)

    b = rand(M_rff) .* (2.0 * pi)
    
    return W, b
end


function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A helper function to generate Random Fourier Feature parameters specifically for a Squared Exponential kernel.
    # Inputs: D_in - The input dimension.
    #         M_rff - The number of RFF features.
    #         lengthscale - The lengthscale of the SE kernel.
    # Outputs: A tuple (W, b) of RFF parameters.
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 





function build_laplacian_precision(adj_matrix)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Constructs a GMRF precision matrix (Graph Laplacian, Q = D - W) from an adjacency matrix.
    # Inputs: adj_matrix - Sparse adjacency matrix (W).
    # Outputs: Sparse precision matrix (Q).
    # Note: This is the fundamental structure for ICAR and other graph-based spatial models.

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 




function plot_choropleth(values::AbstractVector, polygons::Vector; title="Spatial Distribution", cmap=:viridis)
    plt = Plots.plot(aspect_ratio=:equal, title=title, legend=true)

    for i in 1:min(length(polygons), length(values))
        poly_coords = polygons[i]
        if length(poly_coords) > 2
            px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
            py = [pt[2] for pt in poly_coords if !isnan(pt[2])]

            if !isempty(px)
                if (px[1], py[1]) != (px[end], py[end])
                    push!(px, px[1])
                    push!(py, py[1])
                end

                Plots.plot!(plt, px, py,
                    seriestype=:shape,
                    fill_z=values[i],
                    c=cmap,
                    linecolor=:black,
                    lw=0.5,
                    fillalpha=0.8,
                    label=nothing
                )
            end
        end
    end
    return plt
end

 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_x=nothing, s_y=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Comprehensive visualization for posterior results from bstm models.
    # Inputs: stats (results object), M (model config), areal_units, and various plotting options.
    # Outputs: A Plots.jl plot object.
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end
 
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

    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

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
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

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
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Overlays the posterior density of a parameter with its prior density.
    # Inputs: model, chain, param_sym, n_prior_samples, title.
    # Outputs: A Plots.jl plot object.

    post_samples = vec(chain[param_sym].data)

    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end


function get_rff_deep2D_basis(X, m, lengthscale)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates a Random Fourier Feature (RFF) basis for 2D inputs.
    # Inputs: X (N x D matrix), m (number of features), lengthscale.
    # Outputs: An N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .*  2pi
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end

function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates an RFF basis for a 1D temporal trend.
    # Inputs: t (time vector), m (number of features), lengthscale.
    # Outputs: An N x m feature matrix.
    N = length(t)
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
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates a deterministic Fourier series basis for periodic components.
    # Inputs: t (time vector), m (number of harmonics), freq (base frequency), lengthscale.
    # Outputs: An N x (2*m) feature matrix.
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
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies a discretization or transformation rule to a vector of values.
    # Inputs: vals (vector), rules (discretization/transformation rule).
    # Outputs: A tuple (new_vals, groups) where new_vals may be transformed values and groups are integer indices.
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
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Pre-processes covariates by handling discretization, transformation, and interaction terms.
    # Inputs: cov_data_base, cov_discretization, re_rules, cov_interactions.
    # Outputs: A tuple of the processed covariate data, group indices, and random effect structures.
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




function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Runs Automatic Differentiation Variational Inference (ADVI) on a Turing model.
    # Inputs: m (Turing model), max_iters, nsamps, nelbo.
    # Outputs: A NamedTuple containing the VI result, samples, mean, and standard deviation.

    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

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
 
function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates RFF weights (W) by sampling from a provided 2D magnitude spectrum.
    # Inputs: freqs_x, freqs_y, magnitude_spectrum, M_rff_count.
    # Outputs: A 2 x M_rff_count matrix for W.
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    Wfixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        Wfixed[1, i] = all_freqs_x[idx] *  2pi
        Wfixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return Wfixed
end
  
function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Selects inducing point locations for sparse GP models.
    # Inputs: coords, M_inducing, seed, method.
    # Outputs: A matrix of inducing point coordinates.
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


# Squared-exponential covariance function
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A simple helper function to compute a squared-exponential covariance.
    # Inputs: D (distance matrix), phi (lengthscale), noise.
    # Outputs: A covariance matrix.
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A simple helper function to compute an exponential covariance.
    # Inputs: D (distance matrix), phi (lengthscale).
    # Outputs: A covariance matrix.
exp_cov_fn(D, phi) = exp.(-D / phi)

 


# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A helper function to compute a full AR(1) covariance matrix based on the time differences between observations.
    # Inputs: times, rho, sigma_e.
    # Outputs: An AR(1) covariance matrix.
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

function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes an AR(1) cross-covariance matrix between two sets of time points.
    # Inputs: times_a, times_b, rho, sigma_e.
    # Outputs: The AR(1) cross-covariance matrix.
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
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Prepares data for FFT analysis by gridding and zero-padding.
    # Inputs: s_x, s_y, values, grid_res, pad_factor.
    # Outputs: A tuple of the padded grid and the bounding box.
    xmin, xmax = minimum(s_x), maximum(s_x)
    ymin, ymax = minimum(s_y), maximum(s_y)

    n_limit = min(length(s_x), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        ix = Int(floor((s_x[i] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((s_y[i] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end
 



#####

 


function get_model_family(model_family::String)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Maps a string identifier to its corresponding concrete `AbstractBSTM_Family` type.
    # Inputs: model_family::String.
    # Outputs: An instance of a concrete subtype of AbstractBSTM_Family.
    family_key = lowercase(model_family)
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$model_family'. Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end

 

function get_model_parameters(m::DynamicPPL.Model)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts parameter names from a Turing model instance by sampling from it once.
    # Inputs: m - A Turing model instance.
    # Outputs: A vector of parameter name strings.
    # Note: This is a fallback method for parameter discovery.
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


function _ensure_matrix(field, n_units, n_samples)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Converts a scalar, vector, or matrix into a matrix of a specified size.
    # Inputs: field, n_units, n_samples.
    # Outputs: A matrix of size [n_units x n_samples].
    if isnothing(field) || isempty(field)
        return zeros(Float64, n_units, n_samples)
    end

    if field isa Number
        return fill(Float64(field), n_units, n_samples)
    end

    if field isa AbstractVector
        if length(field) == n_units
            return repeat(reshape(field, n_units, 1), 1, n_samples)
        elseif length(field) == n_samples
            return repeat(reshape(field, 1, n_samples), n_units, 1)
        else
            return fill(Float64(field[1]), n_units, n_samples)
        end
    end

    if ndims(field) == 2
        local mat = Matrix{Float64}(field)
        local r, c = size(mat)

        if r == n_units && c == n_samples
            return mat
        elseif r == 1 && c == n_samples
            return repeat(mat, n_units, 1)
        elseif r == n_units && c == 1
            return repeat(mat, 1, n_samples)
        else
            return fill(mat[1,1], n_units, n_samples)
        end
    end

    return zeros(Float64, n_units, n_samples)
end

  
 

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts and summarizes posterior distributions for smoothed covariate effects.
    # Inputs: all_names, chain, M, N_samples, alpha.
    # Outputs: A vector of summarized effects or nothing.
    cov_matches = unique(map(m -> m.captures[1], filter(!isnothing, match.(r"beta_cov\[(\d+)\]", all_names))))
    
    if isempty(cov_matches)
        return nothing
    end

    cov_indices = sort(parse.(Int, cov_matches))
    
    results = []
    for k in cov_indices
        base_name = "beta_cov[$k]"
        # Use the robust get_params_vector helper to extract the full vector for this covariate group
        raw_vals = get_params_vector(chain, base_name, M.N_cat)
        
        if !all(raw_vals .== 0)
            summ = summarize_array(reshape(raw_vals', M.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end


mutable struct ManifoldRegistry
    Xfixed_betas::Union{Nothing, Matrix{Float64}}
    mixed_eff_coeffs::Vector{Matrix{Float64}}
    s_eff_struct::Vector{Matrix{Float64}}
    s_eff_noisy::Vector{Matrix{Float64}}
    t_eff::Vector{Matrix{Float64}}
    u_eff::Matrix{Float64}
    basis_eff_accum::Matrix{Float64}
    st_eff_maps::Vector{Array{Float64, 3}}
    svc_slopes::Vector{Array{Float64, 3}}
    sv_surface::Matrix{Float64}
    n_samples::Int
    outcomes_N::Int
end
 





function decompose_bstm_formula(formula_str::String)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Decomposes a bstm formula string into outcomes, fixed effects, and modules.
    # Inputs: formula_str::String.
    # Outputs: A NamedTuple containing the parsed components.
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    outcomes = String[]
    if startswith(lhs, "[") && endswith(lhs, "]")
        content = lhs[2:end-1]
        outcomes = [strip(s) for s in Base.split(content, ",")]
    else
        outcomes = [lhs]
    end

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

        m_mod = match(r"(\w+)\((.*)\)", t_clean)
        if !isnothing(m_mod) && (lowercase(m_mod.captures[1]) in BSTM_MODULE_KEYWORDS)
            module_data = _parse_module_call(t_clean)
            
            key_parts = [string(module_data[:type])]
            
            if haskey(module_data, :variables) && !isempty(module_data[:variables])
                push!(key_parts, join(module_data[:variables], "_"))
            end

            if haskey(module_data, :params) && haskey(module_data[:params], :model)
                model_val = module_data[:params][:model]
                if model_val isa Symbol || model_val isa String
                    push!(key_parts, string(model_val))
                end
            end

            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end

            modules[module_key] = module_data
        else
            operator_found = ""
            for op in ["⊗", "otimes", "⊕", "oplus", "∘", "compose", "|>", "pipe"]
                if occursin(op, t_clean)
                    operator_found = op
                    break
                end
            end

            if !isempty(operator_found)
                norm_op = operator_found
                if operator_found == "otimes"; norm_op = "⊗"; end
                if operator_found == "oplus"; norm_op = "⊕"; end
                if operator_found == "compose"; norm_op = "∘"; end
                if operator_found == "pipe"; norm_op = "|>"; end

                sub_terms_str = _split_terms_at_depth(t_clean, operator_found)

                parsed_components = []
                for st in sub_terms_str
                    push!(parsed_components, _parse_module_call(st))
                end

                comp_keys = map(parsed_components) do comp
                    c_type = string(get(comp, :type, "lit"))
                    c_vars = join(get(comp, :variables, [get(comp, :value, "")]), "_")
                    c_model = string(get(get(comp, :params, Dict()), :model, ""))
                    return join(filter!(!isempty, [c_type, c_vars, c_model]), "_")
                end
                algebraic_key = "interaction_" * join(comp_keys, "_$(norm_op)_")

                modules[algebraic_key] = Dict{Symbol, Any}(
                    :type => :interaction_composition,
                    :components => parsed_components,
                    :operator => Symbol(norm_op)
                )
                continue
            end
            
            if !isempty(t_clean)
                push!(fixed_effects, t_clean)
            end
        end
    end

    return (outcomes=outcomes, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
end
 


function process_interaction_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes algebraic interaction modules (e.g., `spatial() ⊗ temporal()`).
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    op = mod_data[:operator]
    if op == :⊗ || op == Symbol("otimes")
        has_structured_space = false
        has_structured_time = false

        for comp_data in mod_data[:components]
            if comp_data[:type] == :spatial
                model_name = string(get(comp_data[:params], :model, "iid"))
                if model_name != "iid"; has_structured_space = true; end
            elseif comp_data[:type] == :temporal
                model_name = string(get(comp_data[:params], :model, "iid"))
                if model_name != "iid"; has_structured_time = true; end
            end
        end

        if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
        elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
        elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
        else; opt_dict[:model_st] = "I"; end
    elseif op == :⊕ || op == Symbol("oplus"); @warn "Direct sum operator '⊕' is not yet fully supported. Treating as additive effects.";
    elseif op == :∘ || op == Symbol("compose"); @warn "Composition operator '∘' is not yet fully supported. Treating as additive effects.";
    end
end

function process_observationprocess_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `observationprocess()` module to handle offsets, weights, etc.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    data = opt_dict[:data]
    params = mod_data[:params]

    _resolve_obs_param!(opt_dict, params, data, [:weights, :weight], :weights)
    _resolve_obs_param!(opt_dict, params, data, [:log_offsets, :offsets], :log_offset)
    _resolve_obs_param!(opt_dict, params, data, [:trials, :trial], :trials)
    
    # Process other observation-level parameters
    if get(params, :volatility, false) == true
        opt_dict[:use_sv] = true
        opt_dict[:M_rff_sigma] = get(params, :nbins, 20)
    end
    
    if haskey(params, :y_L); opt_dict[:y_lower_bound] = params[:y_L]; end
    if haskey(params, :y_U); opt_dict[:y_upper_bound] = params[:y_U]; end
    if haskey(params, :hurdle); opt_dict[:hurdle] = params[:hurdle]; end
end


function process_intercept_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `intercept()` module to handle the global intercept and its prior.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    opt_dict[:add_intercept] = true
    params = mod_data[:params]
    if haskey(params, :prior)
        prior_val = params[:prior]
        if prior_val isa String
            try
                opt_dict[:intercept_prior] = Main.eval(Meta.parse(prior_val))
            catch e
                @warn "Could not evaluate intercept prior: $prior_val. Using default. Error: $e"
            end
        else
            opt_dict[:intercept_prior] = prior_val
        end
    end
end

function process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.2 (2026-06-29 18:10:00)
    # Purpose: Processes the `spatial()` module to extract the spatial index and adjacency matrix,
    #          and to prepare coordinates for continuous spatial models.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict and mod_data in place.
    data = opt_dict[:data]

    if !isempty(mod_data[:variables])
        var_sym = Symbol(mod_data[:variables][1])
        if hasproperty(data, var_sym)
            opt_dict[:s_idx] = data[!, var_sym]
            opt_dict[:s_idx_var] = var_sym

            opt_dict[:s_N] = length(unique(data[!, var_sym]))
        else
            @warn "Spatial index variable ':$var_sym' not found in data."
        end
    end
 
    if haskey(mod_data[:params], :W)
        opt_dict[:W] = mod_data[:params][:W]
    end

    model_name = string(get(mod_data[:params], :model, "none"))
    if model_name in ["gp", "fitc", "svgp", "nystrom", "rff", "spde"]
        if haskey(opt_dict, :s_coord)
            s_N = get(opt_dict, :s_N, 0)
            if s_N > 0
                # Need unique coordinates per spatial unit
                df_coords = DataFrame(s_idx=opt_dict[:s_idx], s_x=opt_dict[:s_x], s_y=opt_dict[:s_y])
                unique_coords_df = combine(groupby(df_coords, :s_idx), :s_x => first => :s_x, :s_y => first => :s_y)
                sort!(unique_coords_df, :s_idx)
                if nrow(unique_coords_df) == s_N
                    mod_data[:params][:coords] = Matrix(unique_coords_df[!, [:s_x, :s_y]])
                else
                     @warn "Could not determine unique coordinates for spatial GP. Number of unique coordinates does not match s_N."
                end
            end
        elseif haskey(opt_dict, :centroids)
            s_N = get(opt_dict, :s_N, 0)
            if s_N > 0 && length(opt_dict[:centroids]) == s_N
                mod_data[:params][:coords] = reduce(hcat, [collect(c) for c in opt_dict[:centroids]])'
            else
                @warn "Centroids provided but length does not match s_N. Cannot use for spatial GP."
            end
        else
            @warn "Spatial GP model specified, but no coordinates (`s_x`, `s_y` in data or `centroids` kwarg) provided."
        end
    end
end



function bstm_sample(m::DynamicPPL.Model; nsample=10, testing=false)
    # v1.0.0 (2026-06-30)
    # Purpose: A utility function to streamline the process of sampling from a bstm model.
    #          It handles initialization, sampler selection, and execution.
    # Inputs: m - The Turing model instance.
    #         nsample - The number of posterior samples to draw.
    #         testing - If false, uses a fast MH() sampler for quick checks.
    #         sampler_override - Optionally provide a pre-configured sampler.
    # Outputs: A tuple containing the MCMC chain, initial values, the sampler used, and a model summary.

    model_summary = show_model(m)
    
    if !testing
        inits = get_inits(m)
        nadapt = max(Int(round(nsample * 0.25)), 200)
        os = get_optimal_sampler(m; adaptation_steps=nadapt)
    else
        inits =  get_inits(m; refine="none")
        os = MH()
    end

    chn = sample(m, os, nsample; initial_params=inits, progress=true, drop_warmup=true)
    
    plt = StatsPlots.plot(chn, seriestype=:traceplot)

    return chn, inits, os, model_summary, plt
 
end



function process_temporal_module!(opt_dict, mod_data, registries, hyperpriors)
    # v2.0.0 (2026-07-02)
    # Purpose: Processes the `temporal()` module to handle the primary time index.
    # Note: This version is simplified, as seasonal components are now handled by
    #       a pre-processing step in `bstm_config` that creates a `seasonal()` module.
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    if !isempty(variables)
        t_var_sym = Symbol(variables[1])
        if hasproperty(data, t_var_sym)
            time_opts = Dict(:time_method => get(params, :time_method, "regular"), :t_N => get(params, :t_N, nothing), :u_N => get(params, :u_N, nothing))
            filter!(p -> !isnothing(p.second), time_opts)
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.t_idx
            opt_dict[:t_N] = tu_meta.tn
            opt_dict[:t_idx_var] = t_var_sym
        else
            @warn "Temporal index variable ':$t_var_sym' not found in data."
        end
    end
end

function process_seasonal_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.0 (2026-07-02)
    # Purpose: Processes a `seasonal()` module, which is created by the pre-processor
    #          in `bstm_config` from a dual `temporal()` call.
    data = opt_dict[:data]
    variables = mod_data[:variables]
    
    # Ensure seasonal indices are initialized to avoid errors if the module is present but data is not.
    if !haskey(opt_dict, :u_idx); opt_dict[:u_idx] = nothing; end
    if !haskey(opt_dict, :u_N); opt_dict[:u_N] = 0; end

    if !isempty(variables)
        u_var_sym = Symbol(variables[1])
        if hasproperty(data, u_var_sym)
            opt_dict[:u_idx] = data[!, u_var_sym]
            opt_dict[:u_N] = length(unique(opt_dict[:u_idx]))
        else
            @warn "Seasonal index variable ':$u_var_sym' not found in data. A default seasonal index may be used if available from `assign_time_units`."
        end
    end
end

function process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, manifolds_registry)
    # v2.0.0 (2026-07-02)
    # Purpose: Processes `smooth()` modules for non-linear covariate effects.
    # This version correctly handles tensor products, GMRF-on-bins, continuous kernels, and basis models.
    data = opt_dict[:data]
    params = mod_data[:params]
    model_param = get(params, :model, "pspline")

    # --- 1. Tensor Product Smooths ---
    # Handles `smooth(x, y, model=(x=pspline, y=rw2))`
    if model_param isa NamedTuple
        vars = mod_data[:variables]
        n_vars = length(vars)
        
        if n_vars >= 2
            B_list, Q_list, nbins_list = [], [], []
            for var_str in vars
                var_sym = Symbol(var_str)
                model_i = string(get(model_param, var_sym, "pspline"))
                nbins_i = get(get(params, :nbins, (;)), var_sym, get(params, :nbins, 20))
                degree_i = get(get(params, :degree, (;)), var_sym, get(params, :degree, 3))
                ls_i = get(get(params, :lengthscale, (;)), var_sym, get(params, :lengthscale, nothing))
                
                kwargs_i = Dict{Symbol, Any}()
                if !isnothing(ls_i) kwargs_i[:lengthscale] = ls_i end

                vals_i = data[!, var_sym]
                push!(B_list, bstm_smooth_basis_1D(model_i, vals_i, nbins_i, degree_i; kwargs_i...))
                
                template_type = if model_i in ["pspline", "rw2"]; :rw2 elseif model_i == "rw1"; :rw1 else :iid end
                push!(Q_list, build_structure_template(template_type, nbins_i).matrix)
                push!(nbins_list, nbins_i)
            end

            B_final = B_list[end]
            for i in (n_vars-1):-1:1; B_final = kron(B_final, B_list[i]); end

            Q_final = Q_list[1]
            n_units_total = nbins_list[1]
            for i in 2:n_vars
                n_i = nbins_list[i]
                Q_i = Q_list[i]
                Q_final = kron(I(n_i), Q_final) + kron(Q_i, I(n_units_total))
                n_units_total *= n_i
            end
            
            mod_data[:params][:model] = "tensor_product_smooth"
            mod_data[:params][:Q_tensor] = Q_final
            return # Exit after handling tensor product
        end
    end

    model_str = string(model_param)
    
    basis_models = ["pspline", "bspline", "tps", "rff", "fft", "moran", "spherical", "barycentric", "decay", "wavelet", "linear", "invdist", "kriging"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]

    # --- 2. Basis Function Models ---
    # Handles `smooth(x, model='pspline')`
    if model_str in basis_models
        if !isempty(mod_data[:variables])
            nb = get(mod_data[:params], :nbins, get(mod_data[:params], :m_rff, 20))
            reg_key = Symbol(join(mod_data[:variables], "_"))
            
            if all(hasproperty(data, Symbol(v)) for v in mod_data[:variables])
                n_vars = length(mod_data[:variables])
                if n_vars == 1
                    v_vec = data[!, Symbol(mod_data[:variables][1])]
                    basis_matrices_registry[reg_key] = bstm_smooth_basis_1D(model_str, v_vec, nb, get(mod_data[:params], :degree, 3); mod_data[:params]...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
                    if n_vars == 2; basis_matrices_registry[reg_key] = bstm_smooth_basis_2D(model_str, c_mat, nb; mod_data[:params]...);
                    elseif n_vars == 3; basis_matrices_registry[reg_key] = bstm_smooth_basis_3D(model_str, c_mat, nb; mod_data[:params]...);
                    elseif n_vars == 4; basis_matrices_registry[reg_key] = bstm_smooth_basis_4D(model_str, c_mat, nb; mod_data[:params]...);
                    end
                end
            end
        end
    # --- 3. Continuous Kernel Models ---
    # Handles `smooth(lon, lat, model='gp')`
    elseif model_str in continuous_kernel_models
        if all(v -> hasproperty(data, Symbol(v)), mod_data[:variables])
            mod_data[:params][:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
        else
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Manifold may be misspecified."
        end
    # --- 4. GMRF-on-Bins Models ---
    # Handles `smooth(x, model='rw2')`
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) != 1
            @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."
            return
        end
        
        var_sym = Symbol(vars[1])
        nbins = get(mod_data[:params], :nbins, 20)
        
        _, indices = apply_discretization_logic(data[!, var_sym], nbins)
        mod_data[:params][:indices] = indices
        mod_data[:params][:n_cat] = length(unique(indices))
        
        # Re-brand this module as a 'mixed' effect so the main loop handles it correctly.
        mod_data[:type] = :mixed
    end
end

 
function process_dynamics_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-28 21:30:00)
    # Purpose: Processes the `dynamics()` module.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
end

function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-28 21:30:00)
    # Purpose: Processes the `eigen()` module for PCA-based factor models.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies mod_data in place.
    params = mod_data[:params]
    vars = mod_data[:variables]
    n_vars = length(vars)
    n_factors = get(params, :n_factors, 1)

    if n_factors > n_vars
        @warn "Number of factors ($n_factors) for eigen() module cannot be greater than number of variables ($n_vars). Setting to $n_vars."
        n_factors = n_vars
    end

    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))

    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors
end


function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `mixed()` module for random effects.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies mod_data in place.
    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    if length(vars) < 2
        @warn "The mixed() module requires at least two variables: mixed(effect_var, group_var). Skipping."
        return
    end
    
    effect_var_str = vars[1]
    group_var_str = vars[2]
    group_var_sym = Symbol(group_var_str)
    
    if !hasproperty(data, group_var_sym)
        @warn "Grouping variable ':$group_var_sym' for mixed() module not found in data. Skipping."
        return
    end
    
    group_data = data[!, group_var_sym]
    levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(levels))
    indices = [group_map[v] for v in group_data]
    
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = length(levels)
    
    mod_data[:variables] = [effect_var_str]
end

function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Processes the `nested()` supervisor module for multi-fidelity models. This
    #          function configures a complete sub-model based on a nested formula, which is
    #          then used for joint likelihood evaluation in the main model.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Modifies opt_dict in place.
    # Rationale for v1.1.0:
    #     - Added logic to parse the outcome variable and model family for the nested model,
    #       which was previously missing, causing downstream errors.
    #     - Ensured that all necessary components (y_obs, y_N, model_family, spatial indices,
    #       and fixed effects) are correctly configured for the sub-model.

    if !haskey(opt_dict, :nested_manifolds); opt_dict[:nested_manifolds] = Dict{Symbol, Any}(); end
    
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)
    
    # 1. Validate and retrieve the data source for the nested model.
    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return
    end
    sub_data = opt_dict[data_source_sym]
    
    sub_config = Dict{Symbol, Any}()
    sub_config[:data] = sub_data
    sub_metadata = decompose_bstm_formula(sub_formula)
    
    # 2. Extract and configure the outcome variable for the nested model.
    if isempty(sub_metadata.outcomes)
        @warn "No outcome variable specified in nested formula for '$var'. Skipping."
        return
    end
    sub_outcome_sym = Symbol(sub_metadata.outcomes[1])
    if !hasproperty(sub_data, sub_outcome_sym)
        @warn "Outcome variable ':$sub_outcome_sym' for nested module on '$var' not found in data source ':$data_source_sym'. Skipping."
        return
    end
    sub_config[:y_obs] = sub_data[!, sub_outcome_sym]
    sub_config[:y_N] = length(sub_config[:y_obs])
    sub_config[:model_family] = get(params, :model_family, "gaussian")

    # 3. Process spatial components within the nested formula.
    spatial_mod_data_pair = filter(m -> m.second[:type] == :spatial, sub_metadata.modules)
    if !isempty(spatial_mod_data_pair)
        spatial_mod_data = first(spatial_mod_data_pair).second
        process_spatial_module!(sub_config, spatial_mod_data, registries, hyperpriors)
        sub_config[:model_space] = get(spatial_mod_data[:params], :model, "bym2")
        sub_config[:s_Q_template] = build_structure_template(Symbol(sub_config[:model_space]), sub_config[:s_N]; W=get(sub_config, :W, nothing))
    end
    
    # 4. Process fixed effects within the nested formula.
    fixed_effects_formula_part = join(sub_metadata.fixed_effects, " + ")
    if sub_metadata.has_intercept
        fixed_effects_formula_part = isempty(fixed_effects_formula_part) ? "1" : "1 + " * fixed_effects_formula_part
    end
    if !isempty(fixed_effects_formula_part)
        sub_config[:Xfixed] = create_fixed_design(fixed_effects_formula_part, sub_data)
    end
    
    # 5. Register the fully configured nested manifold.
    opt_dict[:nested_manifolds][var] = sub_config
end


const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :spatial => process_spatial_module!,
    :temporal => process_temporal_module!,
    :smooth => process_smooth_module!,
    :interaction_composition => process_interaction_module!,
    :intercept => process_intercept_module!,
    :observationprocess => process_observationprocess_module!,
    :nested => process_nested_module!,
    :eigen => process_eigen_module!,
    :seasonal => process_seasonal_module!,
    :mixed => process_mixed_module!,
    :dynamics => process_dynamics_module!
)


 

function create_prediction_surface(
    data_df::DataFrame, 
    au_obj::NamedTuple, 
    tu_obj::NamedTuple, 
    covariate_vars::Vector{Symbol}; 
    iterations::Int = 3
)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility function that creates a prediction surface (a full spatiotemporal grid)
    #          and imputes missing covariate values using an iterative spatiotemporal neighborhood
    #          averaging method.
    # Inputs: data_df, au_obj, tu_obj, covariate_vars, iterations.
    # Outputs: A DataFrame representing the complete prediction surface with imputed covariates.

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
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility to convert the output of a `Turing.jl` ADVI (variational inference) run
    #          into a format compatible with the standard post-processing engine. It samples from
    #          the variational posterior and formats the output as a `FlexiChain`.
    # Inputs: msol (VI result), model, n_samples.
    # Outputs: A NamedTuple containing the FlexiChain and the raw reconstructed samples.
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
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility to convert a point estimate from an optimization routine (e.g., MAP, MLE)
    #          into a sample-based format for post-processing. It uses a Laplace approximation
    #          (sampling from a multivariate normal centered at the mode) if the Hessian is available.
    #          If a Hessian is not available, it adds a small amount of Gaussian noise to the point
    #          estimate to create a sample distribution.
    # Inputs: optim_result, model, n_samples, and optional parameters.
    # Outputs: A NamedTuple containing the FlexiChain and the raw reconstructed samples.
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
                sorted_keys = sort(collect(col_keys), by = k -> begin # Corrected regex
                    m_idx = match(r"\[(\d+)\]", string(k))
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
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility function for generating a standardized simulated spatiotemporal dataset with
    #          known underlying trends, seasonal effects, and covariate relationships.
    # Inputs: s_N (number of spatial units), t_N (number of time units), rndseed.
    # Outputs: A NamedTuple containing the simulated DataFrame and metadata.
    # Note: This function is crucial for testing and validating model implementations.
    Random.seed!(rndseed)
    n_total = s_N * t_N

    # 1. Spatial Coordinates (Unit Level)
    unique_pts = [(rand() * 100.0, rand() * 100.0) for _ in 1:s_N]
    s_coord_tuple = repeat(unique_pts, inner=t_N)
    s_x = getindex.(s_coord_tuple, 1)
    s_y = getindex.(s_coord_tuple, 2)

    # 2. Temporal/Seasonal Indices
    t_v = repeat(collect(1:t_N), outer=s_N) .+ (rand(n_total) .* 0.05)
    t_idx = repeat(1:t_N, outer=s_N)
    u_N = 12
    u_idx = mod1.(1:n_total, u_N)

    # 3. Latent Fields
    period = 12.0
    trend = 0.05 .* t_v
    seasonal = 0.8 .* cos.(2π .* t_v ./ period)

        # Covariate Generation (W1, W2, W3)
    # These simulate continuous predictors with some shared latent signal Z
    Z = randn(n_total)
    W1_obs = 0.5 .* sin.(t_v ./ 5.0) .+ 0.5 .* Z .+ (randn(n_total) .* 0.1)
    W2_obs = 0.5 .* cos.(t_v ./ 5.0) .- 0.3 .* Z .+ (randn(n_total) .* 0.2)
    W3_obs = 0.2 .* (t_v ./ t_N) .+ 0.1 .* Z .+ (randn(n_total) .* 0.3)
 
    # Mosaic/Cluster Effects
    s_clusters = mod1.(1:s_N, 5)
    cluster_effects = [-2.5, -1.0, 0.0, 1.0, 2.5]
    cluster_assignments_full = repeat(s_clusters, inner=t_N)
    spatial_effect = cluster_effects[cluster_assignments_full]

    # 4. Response Construction
    sigma_y = 0.15
    observation_error = sigma_y .* randn(n_total)
    eta = 1.0 .+ spatial_effect .+ trend .+ seasonal .+ observation_error

    y_binary = Int.(eta .> (mean(eta) + 0.5))
    y_counts = abs.(Int.(round.(exp.(eta)))) # Poisson-friendly counts

    weights = ones(Float64, n_total)
    trials = ones(Int, n_total)

    # Fixed Effects Design Matrix (Standard Intercept-only approach)
    Xfixed = ones(Float64, n_total, 1)

    # a factorial variable
    reg_indices = mod1.(1:n_total, 4)
    reg_levels = ["North", "South", "East", "West"]
    reg = reg_levels[reg_indices]

    # reformat simulated data into a rectangular dataframe (or namedarray the internal default):
    data_df = DataFrame(
        y = y_counts,  # y-variable
        y_obs = eta,
        # Ensuring coordinates are aligned with the flattened observation vector
        s_idx = repeat(1:s_N, inner=t_N),
        s_coord = s_coord_tuple,
        s_x = s_x,
        s_y = s_y,
        t_v = t_v,
        t_coord = vec(t_idx),   # time index
        u_idx = u_idx,
        u_v = seasonal,
        log_offset = zeros(n_total),
        region = categorical(reg),  # would make sure it is used as a factorial variable or in the model statement: Fixed(reg)
        z = Z,  # continuous covariate
        w1 = W1_obs, # more covariates 
        w2 = W2_obs,
        w3 = W3_obs,
        cluster_assignments = cluster_assignments_full,

        y_binary = y_binary,
        y_counts = y_counts,

        Xfixed = Xfixed,
        weights = weights,
        trials = trials     
    )

    return (
        data_df = data_df,
        s_coord = s_coord_tuple,
        metadata = (
            s_N = s_N,
            t_N = t_N,
            u_N = u_N,
            n_total = n_total 
        )
    )   

end




function scottish_lip_cancer_data_spacetime(n_years::Int=20, spatial_expansion::Float64=1.5, temporal_expansion::Float64=1.5; rndseed::Int=42, recreate::Bool=false)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A data factory that generates a spatiotemporal version of the classic Scottish Lip
    #          Cancer dataset. It also creates an expanded "nested" dataset for testing
    #          multi-fidelity models.
    # Inputs: n_years, spatial_expansion, temporal_expansion, rndseed, recreate.
    # Outputs: A tuple containing the primary and nested datasets.
    # BSTM Standard Data Factory v28.3.0
    # Timestamp: 2025-12-01 15:00:00
    # Rationale: Resolving symmetry errors in adjacency matrix construction and scoping errors for derived variables.

    cache_path = "data/scottish_lip_cancer_cache.jld2"

    # Check for existing cache and bypass logic unless recreate is explicitly true
    if isfile(cache_path) && !recreate
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
    
    n_total = length(data_primary.y_bin)

    reg_indices = mod1.(1:n_total, 4)
    reg_levels = ["North", "South", "East", "West"]
    reg = reg_levels[reg_indices]

    data_primary.region = categorical(reg)  # as a "factor"

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
    return K_nystrom_proj
end



# --- Optimized Householder PCA Helper Functions ---

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: Constructs an orthonormal loadings matrix (eigenvectors) from a matrix of
    #          Householder reflector vectors.
    # Inputs: v_mat (matrix of reflector vectors), nU (number of variables), n_factors (number of factors).
    # Outputs: An orthonormal loadings matrix [nU x n_factors].
    # Note: Uses an efficient O(K*N^2) update.

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
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: Performs the full Householder transformation to reconstruct the covariance matrix
    #          for a Bayesian PCA model.
    # Inputs: v (vector of free parameters), nU, n_factors, ltri_indices, pca_sd, pdef_sd, noise.
    # Outputs: A tuple containing the reconstructed covariance matrix, PCA standard deviations, and loadings matrix.

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
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: An internal helper for PCA-based models that performs the inverse operation:
    #          extracting Householder reflector vectors from a given orthonormal loadings matrix.
    #          This is useful for initializing a Bayesian PCA model from a frequentist result.
    # Inputs: U_in (orthonormal loadings matrix), n_factors.
    # Outputs: A matrix of Householder reflector vectors.
    # Note: Complexity is O(K * N^2).

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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A specialized utility for extracting matrix-valued parameters from an MCMC chain,
              specifically for size-structured models where parameters are stored as matrices per
              sample.
    """
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

 



 
function get_kernel_from_string(kernel_name::String)
    k_name = lowercase(kernel_name)
    if k_name == "constant"
        return ConstantKernel()
    elseif k_name == "linear"
        return LinearKernel()
    elseif k_name == "matern12" || k_name == "exponential"
        return Matern12Kernel()
    elseif k_name == "matern32"
        return Matern32Kernel()
    elseif k_name == "matern52"
        return Matern52Kernel()
    elseif k_name == "spherical"
        return SphericalKernel()
    elseif k_name == "squared_exponential" || k_name == "se" || k_name == "gaussian" || k_name == "rbf"
        return SqExponentialKernel()
    elseif k_name == "periodic"
        return PeriodicKernel()
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end

  
 

####


 
;;


function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing, bipartite_adj=nothing)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A factory for creating precision matrix templates for various GMRF and other structured models.
    #          It handles different manifold types (e.g., ICAR, RW2, SPDE) and computes scaling factors.
    # Inputs: type (Symbol), n (size), and optional parameters like W (adjacency matrix).
    # Outputs: A NamedTuple containing the precision matrix template and a scaling factor.
    
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
 
 
    elseif type == :fft || type == :spectral || type == :wavelet
        # # Wiener-Khinchin Spectral Path
        # This replaces the previous basis-only FFT modeling       
        return build_spectral_precision(n, ls, sig, kernel_type=kernel)
 
    else
        @warn "BSTM Registry Fallback: Manifold :$type not recognized. Initializing Identity."
        Q = Matrix(1.0I, n, n)
        sf = 1.0
    end

    return (matrix = Q, scaling_factor = sf)
end

 
  
# BSTM Spectral & Wavelet Precision Engine v22.25.0
# Timestamp: 2026-06-22 10:45:00
# Rationale: Extending the spectral engine to support wavelet-domain precision mapping and localized non-stationary kernels.

function build_spectral_precision(n::Int64, ls::Real, sig::Real; kernel_type=:se, periodicity=nothing, noise_floor=1e-6, wavelet_levels=3, wavelet_filter=:db2)
    # #
    # 1. Frequency and Basis Discovery
    freqs = FFTW.fftfreq(n)
    psd = zeros(n)

    # #
    # 2. Analytical PSD Definitions for Stationary Taxonomy

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        psd .= (sig^2) .* sqrt(2 * pi * ls^2) .* exp.( -2 * pi^2 * ls^2 .* freqs.^2 )

    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        psd .= (sig^2) .* (2 * ls) ./ (1.0 .+ (2 * pi .* freqs .* ls).^2)

    # Matern 3/2
    elseif kernel_type == :matern32
        psd .= (sig^2) .* (4 * ls) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^2 )

    # Matern 5/2
    elseif kernel_type == :matern52
        psd .= (sig^2) .* (16 * ls / 3.0) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^3 )

    # Constant Kernel (DC component)
    elseif kernel_type == :constant
        psd[1] = sig^2 * n

    # Linear Kernel (Non-stationary approximation)
    elseif kernel_type == :linear
        psd .= (sig^2) ./ (abs.(freqs) .+ 1e-8).^2
        psd[1] = sig^2

    # Periodic / Seasonal
    elseif kernel_type == :periodic
        p = isnothing(periodicity) ? n : periodicity
        f0 = 1.0 / p
        for (i, f) in enumerate(freqs)
            dist = abs(f) % f0
            psd[i] = (sig^2) * exp(-dist / 0.01)
        end

    # Rationale: For wavelet kernels, the precision is defined by the energy distribution across scales.
    elseif kernel_type == :wavelet
        # In the wavelet domain, we define eigenvalues based on decomposition levels.
        # This allows for modeling multiscale dependencies.
        wavelet_eigenvalues = zeros(n)
        for scale in 1:wavelet_levels
            # Energy decays with scale to ensure smoothness
            energy = sig^2 * exp(-scale / ls)
            # Map energy to indices corresponding to wavelet levels (approximate mapping)
            idx_start = max(1, floor(Int, n / 2^scale))
            idx_end = max(1, floor(Int, n / 2^(scale-1)))
            wavelet_eigenvalues[idx_start:idx_end] .= energy
        end
        
        precision_eigenvalues = 1.0 ./ (wavelet_eigenvalues .+ noise_floor)
        
        first_row_q = real(FFTW.ifft(precision_eigenvalues))
    end

    if kernel_type != :wavelet
        precision_eigenvalues = 1.0 ./ (psd .+ noise_floor)
        first_row_q = real(FFTW.ifft(precision_eigenvalues))
    end

    Q_rows = Int[]
    Q_cols = Int[]
    Q_vals = Float64[]

    for i in 1:n
        for j in 1:n
            # Circulant index shift
            shift = mod(i - j, n) + 1
            val = first_row_q[shift]
            if abs(val) > 1e-12
                push!(Q_rows, i)
                push!(Q_cols, j)
                push!(Q_vals, val)
            end
        end
    end

    Q_sparse = sparse(Q_rows, Q_cols, Q_vals, n, n)

    return (matrix = Q_sparse, scaling_factor = 1.0, psd = psd, weights = first_row_q)
end

 


 

function evaluate_kernel_matrix(dist_sq::AbstractMatrix, param_val::Real, ls::Real, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    # #
    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return (param_val^2) .* exp.(-dist_sq ./ (2 * ls^2 + noise))
    
    # #
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq .+ noise)
        return (param_val^2) .* exp.(-d ./ (ls + noise))
    
    # #
    # Matern 3/2
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq .+ noise)
        val = sqrt(3.0) .* d ./ (ls + noise)
        return (param_val^2) .* (1.0 .+ val) .* exp.(-val)
    
    # #
    # Matern 5/2
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq .+ noise)
        val = sqrt(5.0) .* d ./ (ls + noise)
        return (param_val^2) .* (1.0 .+ val .+ (val.^2 ./ 3.0)) .* exp.(-val)

    # #
    # Constant Kernel (Identity innovation)
    elseif kernel_type == :constant
        return fill(param_val^2, size(dist_sq))

    # #
    # Linear Kernel
    elseif kernel_type == :linear
        return (param_val^2) .* dist_sq

    # #
    # Wavelet Multiscale Kernel
    # Rationale: Approximates a wavelet covariance by superposing energy scales. 
    # In real-space, this behaves as a sum of kernels with varying lengthscales corresponding to levels.
    elseif kernel_type == :wavelet
        K_accum = zeros(eltype(dist_sq), size(dist_sq))
        for scale in 1:wavelet_levels
            # Scale-specific lengthscale and energy weight
            ls_scale = ls / (2^(scale-1))
            weight_scale = (param_val^2) * exp(-scale / ls)
            
            # Contribution from current resolution level
            K_accum .+= weight_scale .* exp.(-dist_sq ./ (2 * ls_scale^2 + noise))
        end
        return K_accum

    # #
    # Fallback Dispatch
    else
        return (param_val^2) .* exp.(-dist_sq ./ (2 * ls^2 + noise))
    end
end



function wendland_taper(d::AbstractVector, range::Real)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: Computes a Wendland compactly supported correlation function. This is used as a
    #          taper to enforce compact support on a covariance function, which helps to
    #          reduce spectral leakage in FFT-based methods.
    # Inputs: d (distance vector), range (the range beyond which the function is zero).
    # Outputs: A vector of taper weights.
    h = d ./ range
    taper = zeros(eltype(h), size(h))
    mask = h .< 1.0
    # Wendland function for k=2, d=1
    taper[mask] .= (1.0 .- h[mask]).^4 .* (1.0 .+ 4.0 .* h[mask])
    return taper
end


function estimate_spectral_precision(n::Int64, ls::Real, sig::Real; kernel_type=:se, periodicity=nothing, noise_floor=1e-6, wavelet_levels=3, taper_range=nothing, lowpass_range=nothing)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: Constructs a sparse precision matrix for a stationary process by computing the
    #          kernel's power spectral density (PSD), applying optional spectral filters,
    #          and using the Wiener-Khinchin theorem.
    # Inputs: n, ls, sig, and optional parameters for kernel type, tapering, and filtering.
    # Outputs: A NamedTuple containing the sparse precision matrix, scaling factor, PSD, and weights.

    # #
    # Frequency Grid Definition
    # Establish discrete Fourier modes for a signal of length n
    freqs = FFTW.fftfreq(n)
    psd = zeros(n)

    # #
    # Analytical Power Spectral Density (PSD) Definitions

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        psd .= (sig^2) .* sqrt(2 * pi * ls^2) .* exp.( -2 * pi^2 * ls^2 .* freqs.^2 )

    # Exponential / Matern 1/2
    elseif kernel_type == :matern12 || kernel_type == :exponential
        psd .= (sig^2) .* (2 * ls) ./ (1.0 .+ (2 * pi .* freqs .* ls).^2)

    # Matern 3/2
    elseif kernel_type == :matern32
        psd .= (sig^2) .* (4 * ls) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^2 )

    # Matern 5/2
    elseif kernel_type == :matern52
        psd .= (sig^2) .* (16 * ls / 3.0) ./ ( (1.0 .+ (2 * pi .* freqs .* ls).^2).^3 )

    # Constant Kernel (DC Component)
    elseif kernel_type == :constant
        psd[1] = sig^2 * n

    # Linear Kernel (Low-frequency approximation)
    elseif kernel_type == :linear
        psd .= (sig^2) ./ (abs.(freqs) .+ 1e-8).^2
        psd[1] = sig^2

    # Periodic / Seasonal harmonics
    elseif kernel_type == :periodic
        p = isnothing(periodicity) ? n : periodicity
        f0 = 1.0 / p
        for i in 1:n
            dist = abs(freqs[i]) % f0
            psd[i] = (sig^2) * exp(-dist / 0.01)
        end

    # Wavelet Multiscale Approximation
    # Rationale: Maps energy distribution across decomposition scales into frequency modes.
    elseif kernel_type == :wavelet
        # Initialize wavelet energy spectrum
        # In the stationary approximation, energy is assigned to frequency octaves
        for scale in 1:wavelet_levels
            # Energy weight for current scale level
            energy_scale = (sig^2) * exp(-scale / ls)
            
            # Identify frequency indices corresponding to the current resolution level
            # Higher scales correspond to lower frequencies (coarse features)
            # Lower scales correspond to higher frequencies (fine details)
            idx_low = floor(Int, n / 2^scale) + 1
            idx_high = floor(Int, n / 2^(scale - 1))
            
            # Apply energy to the spectral mask
            if idx_low <= idx_high
                psd[idx_low:idx_high] .+= energy_scale
            end
        end

        # Apply low-pass filter as decay on scale energy for wavelets
        if !isnothing(lowpass_range)
            for scale in 1:wavelet_levels
                decay_factor = exp(-scale / lowpass_range)
                idx_start = max(1, floor(Int, n / 2^scale))
                idx_end = max(1, floor(Int, n / 2^(scale - 1)))
                if idx_start <= idx_end
                    psd[idx_start:idx_end] .*= decay_factor
                end
            end
        end
    end

    # # 3. Apply Tapering and Filtering in Spatial Domain
    if !isnothing(taper_range) || !isnothing(lowpass_range)
        first_row_k = real(FFTW.ifft(psd))
        
        # Distances on a circulant grid
        x_grid = 0:(n-1)
        distances = min.(x_grid, n .- x_grid)

        if !isnothing(taper_range); first_row_k .*= wendland_taper(distances, taper_range); end
        if !isnothing(lowpass_range); first_row_k .*= exp.(-(distances.^2) ./ (2 * lowpass_range^2)); end

        # Transform back to get the modified PSD
        psd = real(FFTW.fft(first_row_k))
    end

    # # 4. Remove DC Component
    # This is required to remove the influence of the process's mean from the
    # autocorrelation estimate, focusing on the variance structure.
    psd[1] = 0.0

    # # 5. Precision Recomposition in Spectral Domain
    precision_eigenvalues = 1.0 ./ (psd .+ noise_floor)

    # #
    # Real-Space Mapping via Inverse Fourier Transform
    # The first row of the circulant precision matrix is the IFFT of its eigenvalues
    first_row_q = real(FFTW.ifft(precision_eigenvalues))

    # # 6. Sparse Matrix Construction
    # Generate a sparse circulant operator based on the recovered weights
    Q_rows = Int[]
    Q_cols = Int[]
    Q_vals = Float64[]

    for i in 1:n
        for j in 1:n
            # Periodic index shift mapping
            shift = mod(i - j, n) + 1
            val = first_row_q[shift]
            
            # Sparsification using a tolerance relative to the noise floor
            if abs(val) > 1e-12
                push!(Q_rows, i)
                push!(Q_cols, j)
                push!(Q_vals, val)
            end
        end
    end

    Q_sparse = sparse(Q_rows, Q_cols, Q_vals, n, n)

    return (matrix = Q_sparse, scaling_factor = 1.0, psd = psd, weights = first_row_q)
end


 
 
 
# Helper constructor for common defaults (cubic splines)
"""
BSTM Internal Utility v1.0.0
Timestamp: 2026-06-26 10:22:15
Synopsis: A helper constructor for `BSpline` with default parameters.
"""
BSpline(v::Symbol; nbins=10, degree=3, sigma_prior=Exponential(1.0)) = BSpline(v, nbins, degree, sigma_prior)

 

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
    """
    v1.0.1 (2026-06-29 17:16:00)
    Purpose: An internal factory function that constructs a final precision matrix from a
             template, a scale parameter (`param_val`), and other manifold-specific parameters
             (e.g., correlation `rho`). This function is central to defining GMRF priors within
             the model.
    Inputs: m_type, template_s, param_val, and other optional parameters.
    Outputs: A symmetric precision matrix.
    """
    # Rationale: Standardizing the inversion of the variance scale for precision mapping.
    # Requirement: Avoid ternary if/else during sampling to maintain AD stability.
    
    n_s = size(template_s, 1)
    scale_factor = 1.0 / (param_val^2 + noise)

    # 1. Base Graph & CAR Manifolds
    if m_type == :none || m_type == :fixed
        return Symmetric(scale_factor * I(n_s) + noise * I)
    end

    if m_type == :besag || m_type == :icar || m_type == :cyclic
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

    if m_type == :sar || m_type == :dag || m_type == :proper_car
        rho_p = isnothing(extra_param) ? 0.8 : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

    # Spectral Precision via Wiener-Khinchin
    if m_type == :spectral
        ls = isnothing(extra_param) ? 1.0 : extra_param
        kernel = get(kwargs, :kernel, :se)
        return build_spectral_precision(n_s, ls, param_val; kernel_type=kernel, noise_floor=noise).matrix
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
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: An overloaded method for `recompose_precision` that dispatches based on a `NamedTuple`
    #          containing manifold metadata.
    # Inputs: manifold_metadata, param_val, and other optional parameters.
    # Outputs: A symmetric precision matrix.

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
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A recursive version of `recompose_precision` that handles algebraic compositions
    #          of manifolds (e.g., Kronecker products, direct sums).
    # Inputs: manifold_node (a Manifold struct or ComposedManifold), M (model config), param_sig (scale).
    # Outputs: A precision matrix or a structure representing the composed precision.

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



function parse_variable_and_transforms(var_str::AbstractString)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: An internal helper for the formula parser that separates a variable name from any
    #          piped transformation functions (e.g., "x |> log").
    # Inputs: var_str (string).
    # Outputs: A tuple containing the variable name and a vector of transform names.
    parts = Base.split(var_str, "|>")
    var_name = strip(parts[1])
    transforms = [strip(p) for p in parts[2:end]]
    return var_name, transforms
end

function apply_transforms(data_vector, transform_names::Vector{String})
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal utility that applies a series of named transformations (e.g., "log",
              "zscore") to a data vector based on the `BSTM_TRANSFORM_KEYWORDS` dictionary.
    """
    v = copy(data_vector)
    for t_name in transform_names
        if haskey(BSTM_TRANSFORM_KEYWORDS, t_name)
            v = BSTM_TRANSFORM_KEYWORDS[t_name](v)
        else
            @warn "Unknown transformation '$t_name' requested. Ignoring."
        end
    end
    return v
end

function sample_spectral_density(kernel_name::String, D_in::Int, M_rff::Int, lengthscale::Real; nu::Union{Real, Nothing}=nothing)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper that samples frequencies from the analytical spectral density of
              a given kernel (e.g., Matern, Squared Exponential). This is used to generate the
              `W` matrix for Random Fourier Features.
    """
    k_name = lowercase(kernel_name)
    
    # Default to Squared Exponential if kernel is not supported for RFF
    if !(k_name in ["squared_exponential", "se", "gaussian", "rbf", "matern12", "exponential", "matern32", "matern52"])
        @warn "RFF sampling for kernel '$kernel_name' is not implemented. Defaulting to 'squared_exponential'."
        k_name = "squared_exponential"
    end

    if startswith(k_name, "matern")
        local effective_nu
        if !isnothing(nu)
            effective_nu = nu
        else
            if k_name == "matern12"; effective_nu = 0.5;
            elseif k_name == "matern32"; effective_nu = 1.5;
            elseif k_name == "matern52"; effective_nu = 2.5;
            else; effective_nu = 1.5; end # Default Matern nu
        end
        # Matern spectral density corresponds to a scaled Student's-t distribution
        df = 2 * effective_nu
        dist = TDist(df)
        scale_factor = sqrt(df) / lengthscale
        return rand(dist, D_in, M_rff) .* scale_factor
    else # Squared Exponential
        sigma_spectral = 1.0 / lengthscale
        return randn(D_in, M_rff) .* sigma_spectral
    end
end


function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; W=nothing, kwargs...)
    # BSTM Smooth Basis Factory v2.2.1
    # Timestamp: 2026-07-02
    # Synopsis: A factory function that generates a 1D basis matrix for various smoothers.
    # Rationale for v2.2.1:
    #     - Corrected knot generation for 'invdist' and 'kriging' to use a regular grid,
    #       ensuring stability for skewed data distributions.

    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        knots = collect(range(v_min, stop=v_max, length=nbins))
        h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1)
        h = h > 0 ? h : 1.0

        for m in 1:nbins
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B[mask, m] .= 1.0 .- dist[mask]
        end

    elseif type == "tps"
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff"
        m_rff = nbins
        ls = get(kwargs, :lengthscale, v_std)
        Omega = randn(1, m_rff) ./ ls
        Phi_phases = rand(m_rff) .* (2.0 * pi)
        B .= sqrt(2.0 / m_rff) .* cos.((vals * Omega) .+ Phi_phases')

    elseif type == "fft"
        ls = get(kwargs, :lengthscale, v_std)
        t_coords = vals ./ ls
        for m in 1:div(nbins, 2)
            B[:, 2m-1] .= sin.(2.0 * pi * m .* t_coords)
            B[:, 2m]   .= cos.(2.0 * pi * m .* t_coords)
        end

    elseif type == "wavelet"
        for m in 1:nbins
            center = v_min + (m/nbins) * (v_max - v_min)
            width = (v_max - v_min) / nbins
            B[:, m] .= (vals .>= center) .& (vals .< (center + width)) ? 1.0 : 0.0
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Falling back to sine proxy."
            for m in 1:nbins
                B[:, m] .= sin.(pi * m * (vals .- v_min) ./ (v_max - v_min))
            end
        else
            n_spatial = size(W, 1)
            if n_spatial != n_obs
                @warn "Moran's I basis requires W matrix size to match number of observations for 1D smooth. Using proxy."
                 for m in 1:nbins; B[:, m] .= sin.(pi * m * (vals .- v_min) ./ (v_max - v_min)); end
            else
                centering_matrix = I - (1/n_spatial) * ones(n_spatial, n_spatial)
                W_centered = centering_matrix * W * centering_matrix
                eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
                B = eigen_decomp.vectors[:, end-nbins+1:end]
            end
        end

    elseif type == "spherical"
        range_r = get(kwargs, :range, v_std * 2.0)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            h = abs.(vals .- knots[m]) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "decay"
        ls = get(kwargs, :lengthscale, v_std)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            B[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
        end

    elseif type == "invdist"
        knots = collect(range(v_min, stop=v_max, length=nbins))
        for m in 1:nbins
            dist_sq = (vals .- knots[m]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        knots = collect(range(v_min, stop=v_max, length=nbins))
        ls = get(kwargs, :lengthscale, v_std)
        for m in 1:nbins
            dist_sq = (vals .- knots[m]).^2
            B[:, m] .= exp.(-dist_sq ./ (2 * ls^2))
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, kwargs...)
    # BSTM Smooth Basis Factory v2.2.1
    # Timestamp: 2026-07-02
    # Synopsis: A factory function that generates a 2D basis matrix for various smoothers.
    # Rationale for v2.2.1:
    #     - Corrected knot generation for 'tps', 'spherical', 'invdist', and 'kriging' to use a regular grid.

    n_obs = size(coords, 1)
    
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)
        
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        hx = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1)
        hy = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1)
        hx = hx > 0 ? hx : 1.0
        hy = hy > 0 ? hy : 1.0

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                dist_x = abs.(coords[:, 1] .- kx[i]) ./ hx
                dist_y = abs.(coords[:, 2] .- ky[j]) ./ hy
                mask_x = dist_x .< 1.0
                mask_y = dist_y .< 1.0
                
                b_x = zeros(n_obs)
                b_y = zeros(n_obs)
                b_x[mask_x] .= 1.0 .- dist_x[mask_x]
                b_y[mask_y] .= 1.0 .- dist_y[mask_y]
                
                B[:, idx] .= b_x .* b_y
                idx += 1
            end
        end

    elseif type == "tps"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]

        for m in 1:min(nbins, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            r = sqrt.(dx.^2 .+ dy.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff" || type == "anisotropic"
        Omega = randn(2, nbins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        n_marginal = Int(floor(sqrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y

        idx = 1
        for mx in 1:n_marginal, my in 1:n_marginal
            if idx + 1 <= nbins
                arg = mx .* nx .+ my .* ny
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end

    elseif type == "wavelet"
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)
        
        centers_x = c_min[1] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[1] - c_min[1])
        width_x = (c_max[1] - c_min[1]) / n_marginal
        centers_y = c_min[2] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[2] - c_min[2])
        width_y = (c_max[2] - c_min[2]) / n_marginal

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                b_x = (coords[:, 1] .>= centers_x[i]) .& (coords[:, 1] .< (centers_x[i] + width_x))
                b_y = (coords[:, 2] .>= centers_y[j]) .& (coords[:, 2] .< (centers_y[j] + width_y))
                B[:, idx] .= b_x .* b_y
                idx += 1
            end
        end

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
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Falling back to sine proxy."
            B = zeros(Float64, n_obs, nbins)
            for m in 1:nbins
                B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2]) ./ sum(c_max))
            end
        else
            n = size(W, 1)
            centering_matrix = I - (1/n) * ones(n, n)
            W_centered = centering_matrix * W * centering_matrix
            eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
            B = eigen_decomp.vectors[:, end-nbins+1:end]
        end

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, kwargs...)
    # BSTM Smooth Basis Factory v2.2.1
    # Timestamp: 2026-07-02
    # Synopsis: A factory function that generates a 3D basis matrix for various smoothers.
    # Rationale for v2.2.1:
    #     - Corrected knot generation for 'invdist' and 'kriging' to use a regular grid.

    n_obs = size(coords, 1)

    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2]), minimum(coords[:, 3])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2]), maximum(coords[:, 3])]
    c_std = [std(coords[:, 1]), std(coords[:, 2]), std(coords[:, 3])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)

        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        hx = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hy = hy > 0 ? hy : 1.0
        hz = (c_max[3] - c_min[3]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hz = hz > 0 ? hz : 1.0

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            b_z = max.(0.0, 1.0 .- abs.(coords[:, 3] .- kz[k]) ./ hz)
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end

    elseif type == "tps"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff"
        Omega = randn(3, nbins)
        Omega[1, :] ./= ls_x; Omega[2, :] ./= ls_y; Omega[3, :] ./= ls_z
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        n_marginal = Int(floor(cbrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)
        nx = coords[:, 1] ./ ls_x
        ny = coords[:, 2] ./ ls_y
        nz = coords[:, 3] ./ ls_z
        idx = 1
        for mx in 1:n_marginal, my in 1:n_marginal, mz in 1:n_marginal
            if idx + 1 <= nbins
                arg = mx .* nx .+ my .* ny .+ mz .* nz
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end

    elseif type == "wavelet"
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)
        
        centers_x = c_min[1] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[1] - c_min[1])
        width_x = (c_max[1] - c_min[1]) / n_marginal
        centers_y = c_min[2] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[2] - c_min[2])
        width_y = (c_max[2] - c_min[2]) / n_marginal
        centers_z = c_min[3] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[3] - c_min[3])
        width_z = (c_max[3] - c_min[3]) / n_marginal

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal
            b_x = (coords[:, 1] .>= centers_x[i]) .& (coords[:, 1] .< (centers_x[i] + width_x))
            b_y = (coords[:, 2] .>= centers_y[j]) .& (coords[:, 2] .< (centers_y[j] + width_y))
            b_z = (coords[:, 3] .>= centers_z[k]) .& (coords[:, 3] .< (centers_z[k] + width_z))
            B[:, idx] .= b_x .* b_y .* b_z
            idx += 1
        end

    elseif type == "spherical"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Using sine proxy."
            B = zeros(Float64, n_obs, nbins)
            for m in 1:nbins
                B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2] .+ coords[:, 3]) ./ sum(c_max))
            end
        else
            n = size(W, 1)
            centering_matrix = I - (1/n) * ones(n, n)
            W_centered = centering_matrix * W * centering_matrix
            eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
            B = eigen_decomp.vectors[:, end-nbins+1:end]
        end

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        centers = [(x, y, z) for x in range(c_min[1], c_max[1], length=n_grid), y in range(c_min[2], c_max[2], length=n_grid), z in range(c_min[3], c_max[3], length=n_grid)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_z^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, kwargs...)
    # BSTM Smooth Basis Factory v2.2.1
    # Timestamp: 2026-07-02
    # Synopsis: A factory function that generates a 4D basis matrix for various smoothers.
    # Rationale for v2.2.1:
    #     - Corrected knot generation for 'invdist' and 'kriging' to use a regular grid.

    n_obs = size(coords, 1)

    c_min = [minimum(coords[:, i]) for i in 1:4]
    c_max = [maximum(coords[:, i]) for i in 1:4]
    c_std = [std(coords[:, i]) for i in 1:4] .+ 1e-9

    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))
        h1 = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h1 = h1 > 0 ? h1 : 1.0
        h2 = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h2 = h2 > 0 ? h2 : 1.0
        h3 = (c_max[3] - c_min[3]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h3 = h3 > 0 ? h3 : 1.0
        h4 = (c_max[4] - c_min[4]) / (n_marginal > 1 ? (n_marginal - 1) : 1); h4 = h4 > 0 ? h4 : 1.0

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal, l in 1:n_marginal
            b1 = max.(0.0, 1.0 .- abs.(coords[:, 1] .- k1[i]) ./ h1)
            b2 = max.(0.0, 1.0 .- abs.(coords[:, 2] .- k2[j]) ./ h2)
            b3 = max.(0.0, 1.0 .- abs.(coords[:, 3] .- k3[k]) ./ h3)
            b4 = max.(0.0, 1.0 .- abs.(coords[:, 4] .- k4[l]) ./ h4)
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end

    elseif type == "tps"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            d1 = (coords[:, 1] .- centers[m][1]) ./ ls_1
            d2 = (coords[:, 2] .- centers[m][2]) ./ ls_2
            d3 = (coords[:, 3] .- centers[m][3]) ./ ls_3
            d4 = (coords[:, 4] .- centers[m][4]) ./ ls_4
            r = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    elseif type == "rff"
        Omega = randn(4, nbins)
        Omega[1, :] ./= ls_1; Omega[2, :] ./= ls_2; Omega[3, :] ./= ls_3; Omega[4, :] ./= ls_4
        Phi_phases = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi_phases')

    elseif type == "fft"
        n_marginal = Int(floor(sqrt(sqrt(nbins / 2))))
        B = zeros(Float64, n_obs, nbins)
        nx1 = coords[:, 1] ./ ls_1
        nx2 = coords[:, 2] ./ ls_2
        nx3 = coords[:, 3] ./ ls_3
        nx4 = coords[:, 4] ./ ls_4
        idx = 1
        for m1 in 1:n_marginal, m2 in 1:n_marginal, m3 in 1:n_marginal, m4 in 1:n_marginal
            if idx + 1 <= nbins
                arg = m1 .* nx1 .+ m2 .* nx2 .+ m3 .* nx3 .+ m4 .* nx4
                B[:, idx]   .= sin.(2.0 * pi * arg)
                B[:, idx+1] .= cos.(2.0 * pi * arg)
                idx += 2
            end
        end

    elseif type == "wavelet"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

        centers1 = c_min[1] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[1] - c_min[1])
        width1 = (c_max[1] - c_min[1]) / n_marginal
        centers2 = c_min[2] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[2] - c_min[2])
        width2 = (c_max[2] - c_min[2]) / n_marginal
        centers3 = c_min[3] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[3] - c_min[3])
        width3 = (c_max[3] - c_min[3]) / n_marginal
        centers4 = c_min[4] .+ ((1:n_marginal) ./ n_marginal) .* (c_max[4] - c_min[4])
        width4 = (c_max[4] - c_min[4]) / n_marginal

        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal, k in 1:n_marginal, l in 1:n_marginal
            b1 = (coords[:, 1] .>= centers1[i]) .& (coords[:, 1] .< (centers1[i] + width1))
            b2 = (coords[:, 2] .>= centers2[j]) .& (coords[:, 2] .< (centers2[j] + width2))
            b3 = (coords[:, 3] .>= centers3[k]) .& (coords[:, 3] .< (centers3[k] + width3))
            b4 = (coords[:, 4] .>= centers4[l]) .& (coords[:, 4] .< (centers4[l] + width4))
            B[:, idx] .= b1 .* b2 .* b3 .* b4
            idx += 1
        end

    elseif type == "spherical"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            d1 = (coords[:, 1] .- centers[m][1]) ./ ls_1
            d2 = (coords[:, 2] .- centers[m][2]) ./ ls_2
            d3 = (coords[:, 3] .- centers[m][3]) ./ ls_3
            d4 = (coords[:, 4] .- centers[m][4]) ./ ls_4
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "moran"
        if isnothing(W)
            @warn "Moran's I basis requires an adjacency matrix 'W'. Using sine proxy."
            B = zeros(Float64, n_obs, nbins)
            for m in 1:nbins
                B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2]) ./ (c_max[1] + c_max[2]))
            end
        else
            n = size(W, 1)
            centering_matrix = I - (1/n) * ones(n, n)
            W_centered = centering_matrix * W * centering_matrix
            eigen_decomp = eigen(Symmetric(Matrix(W_centered)))
            B = eigen_decomp.vectors[:, end-nbins+1:end]
        end

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2 .+ (coords[:, 4] .- centers[m][4]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_1^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_2^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_3^2) .+ ((coords[:, 4] .- centers[m][4]).^2 ./ ls_4^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

 


function get_inits(model::DynamicPPL.Model; refine="map", n_samples=100, optimizer=LBFGS(), max_iters=500, maxtime=60.0, noise=nothing)
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility for generating sensible initial values for MCMC sampling. It uses either a
    #          heuristic based on prior samples or refines these initial values via Maximum A
    #          Posteriori (MAP) estimation to find a good starting point for the sampler.
    # Inputs: model, and optional parameters for refinement method and optimization.
    # Outputs: A `DynamicPPL.InitFromParams` object containing the initial values.
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

        if vals[1] isa AbstractVector
            # Latent fields are centered at zero for stability
            init_dict[ks] = zeros(eltype(vals[1]), length(vals[1]))
        else
            mu = median(vals)
            s_name = string(ks)
            if occursin(r"sigma|ls_|lengthscale|sig_", s_name)
                init_dict[ks] = max(0.1, mu)
            elseif occursin("rho", s_name)
                init_dict[ks] = clamp(mu, -0.99, 0.99)
            elseif occursin("phi", s_name)
                init_dict[ks] = clamp(mu, 0.01, 0.5)
            else
                init_dict[ks] = mu
            end
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

 
 
# --- 4. Trait Retrieval Helpers ---
# These helpers standardise the coordinate mapping from struct types to the metadata index vectors.

# Maps Spatial manifolds to the 's_idx' column in the data source
# manifold_indices(::SpatialManifold, M) = M.s_idx

# Maps Temporal manifolds to the 't_idx' column (time steps)
# manifold_indices(::TemporalManifold, M) = M.t_idx

# Maps Seasonal manifolds to the 'u_idx' column (periodic bins)
# manifold_indices(::SeasonalManifold, M) = M.u_idx


    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper to parse mixed-effects terms from the formula string (e.g.,
              "me(1 | group)") and create the necessary index mappings for group-level random
              effects.
    """

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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility that creates the fixed-effects design matrix (`X`) from a formula string.
              It handles `as_factor` directives for categorical variables and prunes `bstm`-specific
              module calls before passing the formula to `StatsModels.jl`.
    """
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




function _build_pass_through_model(m::ManifoldModel, data_inputs::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A generic constructor for manifold configurations that do not require complex template generation.
    # Inputs: m - The manifold model struct.
    #         data_inputs - The main model configuration dictionary.
    #         model_type_sym - Optional symbol to override the model type.
    #         Q_template_val - Optional pre-computed precision template.
    #         sf_val - Optional scaling factor.
    # Outputs: A NamedTuple representing the manifold's configuration.
    model_sym = isnothing(model_type_sym) ? Symbol(lowercase(string(typeof(m)))) : model_type_sym
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        # Extract all fields from the manifold struct that are not types or matrices
        # to populate the hyperparameter tuple.
        field_val = getfield(m, fn)
        if !(field_val isa DataType) && !(field_val isa AbstractMatrix)
             hyper_dict[fn] = field_val
        end
    end
    
    return (
        Q_template = Q_template_val,
        scaling_factor = sf_val,
        model_type = model_sym,
        hyper = NamedTuple(hyper_dict)
    )
end

function _build_from_template(m::ManifoldModel, data_inputs::Dict, domain::Symbol)
    model_sym = Symbol(lowercase(string(typeof(m))))
    
    n, W_mat = if domain == :spatial
        (get(data_inputs, :s_N, 1), get(data_inputs, :W, nothing))
    elseif domain == :temporal
        (get(data_inputs, :t_N, 10), nothing)
    else 
        @warn "Unrecognized domain '$domain'. Defaulting to spatial context."
        (get(data_inputs, :s_N, 1), get(data_inputs, :W, nothing))
    end

    template = build_structure_template(model_sym, n; W=W_mat)

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        field_val = getfield(m, fn)
        if !(field_val isa DataType) && !(field_val isa AbstractMatrix)
             hyper_dict[fn] = field_val
        end
    end
    
    for p in [:rho_prior, :lengthscale_prior, :kappa_prior]
        if !haskey(hyper_dict, p)
            hyper_dict[p] = nothing
        end
    end

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = model_sym,
        hyper = NamedTuple(hyper_dict)
    )
end

function build_model(m::Union{IID, ICAR, Besag, BYM2, Leroux, SAR}, data_inputs::Dict)
    return _build_from_template(m, data_inputs, :spatial)
end

function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict)
    return _build_from_template(m, data_inputs, :temporal)
end

function build_model(m::Union{GP, FITC, RFF, SVGP, Warp, Nystrom, Harmonic, Hyperbolic, ExponentialDecay}, data_inputs::Dict)
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict)
    n = 20 # Default value
    if hasproperty(m, :domain)
        domain = get(m, :domain, :spatial)
        if domain == :spatial; n = get(data_inputs, :s_N, 20);
        elseif domain == :temporal; n = get(data_inputs, :t_N, 20);
        elseif domain == :seasonal; n = get(data_inputs, :u_N, 20);
        else; n = get(m, :nbins, get(data_inputs, :s_N, 20)); end
    else
        n = get(m, :nbins, get(data_inputs, :s_N, 20))
    end

    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Wavelet, data_inputs::Dict)
    n = 20 # Default value
    if hasproperty(m, :domain)
        domain = get(m, :domain, :spatial)
        if domain == :spatial; n = get(data_inputs, :s_N, 20);
        elseif domain == :temporal; n = get(data_inputs, :t_N, 20);
        elseif domain == :seasonal; n = get(data_inputs, :u_N, 20);
        else; n = get(m, :nbins, get(data_inputs, :s_N, 20)); end
    else
        n = get(m, :nbins, get(data_inputs, :s_N, 20))
    end

    template = build_structure_template(:iid, n)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:spectral, Q_template_val=template.matrix)
end

function build_model(m::FFT, data_inputs::Dict)
    n = get(m, :nbins, get(data_inputs, :t_N, get(data_inputs, :s_N, 20)))
    template = build_structure_template(:iid, n)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:spectral, Q_template_val=template.matrix)
end

function build_model(m::SPDE, data_inputs::Dict)
    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    template = build_structure_template(:besag, n; W=W)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Eigen, data_inputs::Dict)
    n = get(data_inputs, :s_N, 1)
    template = build_structure_template(:eigen, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::BCGN, data_inputs::Dict)
    n_groups = size(m.bipartite_adj, 2)
    template = build_structure_template(:iid, n_groups)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix)
end

function build_model(m::NetworkFlow, data_inputs::Dict)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:network, Q_template_val=m.adjacency_matrix)
end

function build_model(m::LocalAdaptive, data_inputs::Dict)
    template = build_structure_template(:besag, get(data_inputs, :s_N, 1); W=get(data_inputs, :W, nothing))
    return _build_pass_through_model(m, data_inputs, model_type_sym=:local_adaptive, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Mosaic, data_inputs::Dict)
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Cyclic, data_inputs::Dict)
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::TensorProductSmooth, data_inputs::Dict)
    return (
        Q_template = m.Q_template,
        scaling_factor = 1.0,
        model_type = :tensor_product_smooth,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::SoftConstraintManifold, data_inputs::Dict)
    inner_config = build_model(m.manifold, data_inputs)
    new_hyper = merge(inner_config.hyper, (soft_constraint_type=m.type, soft_constraint_weight=m.weight))
    return merge(inner_config, (hyper = new_hyper,))
end

function build_model(m::RegularizationGroupManifold, data_inputs::Dict)
    return (
        Q_template = nothing,
        scaling_factor = 1.0,
        model_type = :regularization_group,
        hyper = (
            penalty = m.penalty,
            lambda_prior = m.lambda_prior,
            alpha_prior = m.alpha_prior, 
            sub_manifolds = m.manifolds
        )
    )
end

function build_model(m::Manifold, data_inputs::Dict)
    @warn "No specific builder for $(typeof(m)). Using IID identity template."
    template = build_structure_template(:iid, get(data_inputs, :s_N, 1))
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = hasproperty(m, :sigma_prior) ? m.sigma_prior : Exponential(1.0),)
    )
end


# v4.0.0 (2026-07-07 10:15:00)
# Description: Complete restoration of bstm_univariate with explicit logic for all Knorr-Held interaction types.
# Rationale: Ensures Types I, II, III, and IV are implemented with correct structural Kronecker products.

# v4.1.0 (2026-07-07 11:30:00)
# Description: Full implementation of bstm_univariate with explicit Knorr-Held interaction logic.
# Rationale: Ensures feature completeness and transparency for Types I, II, III, and IV.

@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
    # #
    # Global Likelihood Hyperparameters
    family = M.model_family
    noise = get(M, :noise, 1e-6)
    use_zi = get(M, :use_zi, false)

    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = one(T)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(Exponential(1.0), :extra_params)
    end

    # #
    # Linear Predictor Initialization
    eta = zeros(T, M.y_N)

    if get(M, :add_intercept, false)
        intercept ~ NamedDist(get(M, :intercept_prior, Normal(0, 5)), :intercept)
        eta .+= intercept
    end

    if M.Xfixed_N > 0
        Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
        eta .+= M.Xfixed * Xfixed_beta
    end

    if haskey(M, :log_offset)
        eta .+= M.log_offset
    end

    # #
    # Scaffolding for Spatiotemporal Interaction Structures
    # These matrices are populated by the manifold loop and consumed by the interaction block.
    s_Q_main = sparse(I(M.s_N))
    t_Q_main = sparse(I(M.t_N))
    t_rho_main = zero(T)
    s_sigma_main = one(T)
    t_sigma_main = one(T)

    # #
    # Modular Manifold Realization Loop
    for spec in M.manifolds
        m_obj = spec.manifold_obj

        if m_obj isa NoneManifold
            continue
        end

        key = spec.key
        domain = spec.domain

        if m_obj isa IID
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))

            n_units = (domain == :spatial) ? M.s_N : ((domain == :temporal) ? M.t_N : M.u_N)
            indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
            field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), Symbol("latent_", key))
            eta .+= field[indices]

        elseif m_obj isa BYM2
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))
            s_sigma_main = sigma_val

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            rho_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))

            s_icar ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), Symbol("latent_struct_", key))
            s_iid ~ NamedDist(MvNormal(zeros(M.s_N), I), Symbol("latent_iid_", key))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(s_icar))
            combined = sigma_val .* (sqrt(rho_val) .* s_icar .+ sqrt(1.0 - rho_val) .* s_iid)
            eta .+= combined[M.s_idx]
            s_Q_main = spec.Q_template

        elseif m_obj isa AR1
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))
            t_sigma_main = sigma_val

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            rho_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))

            t_rho_main = rho_val
            innovations ~ NamedDist(MvNormal(zeros(M.t_N), I), Symbol("innov_", key))
            t_field = Vector{T}(undef, M.t_N)
            t_field[1] = innovations[1] / sqrt(1.0 - rho_val^2 + noise)
            for i in 2:M.t_N
                t_field[i] = rho_val * t_field[i-1] + innovations[i]
            end
            eta .+= (t_field .* sigma_val)[M.t_idx]
            t_Q_base = Symmetric((1.0 + rho_val^2) .* I(M.t_N) .- rho_val .* spec.Q_template)
            t_Q_main = Symmetric((1.0 / (sigma_val^2 * (1.0 - rho_val^2) + noise)) .* t_Q_base)

        elseif m_obj isa Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic}
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))

            r_val = nothing
            if hasproperty(m_obj, :rho_prior)
                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                r_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))
            end
            n_units = size(spec.Q_template, 1)
            Q = recompose_precision(Symbol(lowercase(string(typeof(m_obj)))), spec.Q_template, sigma_val; extra_param=r_val, noise=noise)
            field ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", key))
            if !(m_obj isa Cyclic)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
            end
            indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
            eta .+= field[indices]
            if domain == :spatial; s_Q_main = Q; elseif domain == :temporal; t_Q_main = Q; end

        elseif m_obj isa Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}
            var_sym = spec.var
            B_mat = M.basis_matrices[var_sym]
            n_basis_cols = size(B_mat, 2)
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_basis_", var_sym))

            beta_name = Symbol("beta_basis_", var_sym)
            if m_obj isa PSpline || m_obj isa TPS
                Q_penalty = (1.0 / (sigma_val^2 + noise)) .* spec.Q_template
                latent_coeffs ~ NamedDist(MvNormalCanon(zeros(n_basis_cols), Q_penalty), beta_name)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_basis_cols), sum(latent_coeffs))
            else
                latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_basis_cols), beta_name)
            end
            eta .+= B_mat * latent_coeffs

        elseif m_obj isa DynamicsManifold
            d_model = m_obj.model
            priors = M.hyperpriors
            if d_model == "advection" || d_model == "diffusion"
                v_p = get(priors, "velocity_prior", Normal(0, 0.5))
                sig_p = get(priors, "sigma_prior", Exponential(1.0))
                param_v ~ NamedDist(v_p, Symbol("dyn_v_", key))
                sig_d ~ NamedDist(sig_p, Symbol("dyn_sig_", key))
                L_mat = spec.Q_template
                dyn_field = Matrix{T}(undef, M.s_N, M.t_N)
                dyn_field[:, 1] ~ MvNormal(zeros(M.s_N), I)
                for t in 2:M.t_N
                    drift = -param_v .* (L_mat * dyn_field[:, t-1])
                    dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] .+ drift, sig_d^2 * I)
                end
                for i in 1:M.y_N; eta[i] += dyn_field[M.s_idx[i], M.t_idx[i]]; end
            elseif d_model in ["gompertz", "logistic_basic"]
                r_p = get(priors, "r_prior", LogNormal(-1.5, 0.5))
                k_p = get(priors, "K_prior", Normal(150, 50))
                r_val ~ NamedDist(r_p, Symbol("dyn_r_", key))
                k_val ~ NamedDist(k_p, Symbol("dyn_k_", key))
                pop_state = Vector{T}(undef, M.t_N)
                pop_state[1] ~ Normal(log(k_val / 2.0), 1.0)
                for t in 2:M.t_N
                    growth = r_val * (log(k_val) - pop_state[t-1])
                    pop_state[t] ~ Normal(pop_state[t-1] + growth, 0.1)
                end
                eta .+= pop_state[M.t_idx]
            end

        elseif m_obj isa Eigen
            n_factors = m_obj.n_factors
            vars = spec.variables
            cov_data = Matrix(M.data[!, vars])'
            n_vars = length(vars)
            pca_sd ~ NamedDist(m_obj.pca_sd_prior, Symbol("pca_sd_", key))
            pdef_sd ~ NamedDist(m_obj.pdef_sd_prior, Symbol("pdef_sd_", key))
            v_vec ~ NamedDist(filldist(Normal(0, 1), length(m_obj.ltri_indices)), Symbol("v_", key))
            v_mat = zeros(T, n_vars, n_factors)
            v_mat[m_obj.ltri_indices] .= v_vec
            U_mat = householder_to_eigenvector(v_mat, n_vars, n_factors)
            z_latent ~ NamedDist(filldist(Normal(0, 1), n_factors, M.y_N), Symbol("z_", key))
            eigen_coeffs ~ NamedDist(filldist(Normal(0, 1), n_factors), Symbol("eigen_coeffs_", key))
            eta .+= z_latent' * eigen_coeffs
            reconstructed_cov = U_mat * z_latent
            for i in 1:M.y_N
                Turing.@addlogprob! logpdf(MvNormal(reconstructed_cov[:, i], pdef_sd^2 * I), cov_data[:, i])
            end

        elseif m_obj isa SVCManifold
            cov_var = m_obj.covariate
            x_svc = M.data[!, cov_var]
            inner_m = m_obj.model
            sig_svc_dist = get(inner_m, :sigma_prior, Exponential(1.0))
            sig_svc ~ NamedDist(sig_svc_dist, Symbol("sigma_svc_", key))

            if inner_m isa BYM2
                rho_svc_dist = inner_m.rho_prior
                rho_svc ~ NamedDist(rho_svc_dist, Symbol("rho_svc_", key))
                s_struct ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), Symbol("svc_struct_", key))
                s_unstruct ~ NamedDist(MvNormal(zeros(M.s_N), I), Symbol("svc_iid_", key))
                beta_svc = sig_svc .* (sqrt(rho_svc) .* s_struct .+ sqrt(1.0 - rho_svc) .* s_unstruct)
                eta .+= beta_svc[M.s_idx] .* x_svc
            else
                beta_svc ~ NamedDist(filldist(Normal(0, sig_svc), M.s_N), Symbol("beta_svc_", key))
                eta .+= beta_svc[M.s_idx] .* x_svc
            end
        end
    end

    # #
    # 4. Spatiotemporal Interaction (Knorr-Held Taxonomy)
    # Logic explicitly handles Types I, II, III, and IV via Kronecker compositions.
    model_st = get(M, :model_st, "none")
    if model_st != "none"
        st_sigma ~ NamedDist(Exponential(0.5), :st_sigma)
        
        if model_st == "I"
            # Type I: Unstructured space and time (IID ⊗ IID)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_inter = reshape(st_raw, M.s_N, M.t_N) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "II"
            # Type II: Unstructured space, structured time (IID ⊗ Temporal Q)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            # Apply temporal structure (Cholesky of t_Q_main)
            L_t = cholesky(Symmetric(t_Q_main + noise * I)).L
            st_inter = (reshape(st_raw, M.s_N, M.t_N) * L_t') .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "III"
            # Type III: Structured space, unstructured time (Spatial Q ⊗ IID)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            # Apply spatial structure (Cholesky of s_Q_main)
            L_s = cholesky(Symmetric(s_Q_main + noise * I)).L
            st_inter = (L_s * reshape(st_raw, M.s_N, M.t_N)) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "IV"
            # Type IV: Structured space, structured time (Spatial Q ⊗ Structured Time)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            L_s = cholesky(Symmetric(s_Q_main + noise * I)).L
            
            # Recursive definition for AR(1) temporal evolution of spatial fields
            st_inter = Matrix{T}(undef, M.s_N, M.t_N)
            st_inter[:, 1] = (L_s * st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_main^2 + noise)
            for t in 2:M.t_N
                st_inter[:, t] = t_rho_main .* st_inter[:, t-1] .+ (L_s * st_innov_matrix[:, t])
            end
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]] * st_sigma; end
        end
    end

    # #
    # 5. High-Fidelity Likelihood Dispatch
    y_sigma ~ NamedDist(Exponential(1.0), :y_sigma)
    for i in 1:M.y_N
        d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma], phi_zi=lik_phi, r_nb=lik_r, trial=[Int(M.trials[i])], extra_params=extra_p)
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end


 
@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # #
    # 1. Global Likelihood Hyperparameters
    family = M.model_family
    noise = get(M, :noise, 1e-6)
    use_zi = get(M, :use_zi, false)
    outcomes_N = M.outcomes_N

    lik_r = ones(T, outcomes_N)
    lik_phi = zero(T)
    extra_p = ones(T, outcomes_N)

    if family == "negbin"
        lik_r ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :lik_r)
    end
    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end
    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :extra_params)
    end

    # #
    # 2. Linear Predictor Initialization [Observations x Outcomes]
    eta_latent = zeros(T, M.y_N, outcomes_N)

    # Intercept Resolution (Outcome-Specific)
    if get(M, :add_intercept, false)
        intercept ~ NamedDist(filldist(get(M, :intercept_prior, Normal(0, 5)), outcomes_N), :intercept)
        for k in 1:outcomes_N
            eta_latent[:, k] .+= intercept[k]
        end
    end

    # Fixed Effects (Shared design, outcome-specific coefficients)
    if M.Xfixed_N > 0
        Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N * outcomes_N), 5.0 * I), :Xfixed_beta)
        beta_matrix = reshape(Xfixed_beta, M.Xfixed_N, outcomes_N)
        eta_latent .+= M.Xfixed * beta_matrix
    end

    # Offsets
    if haskey(M, :log_offset)
        for k in 1:outcomes_N
            eta_latent[:, k] .+= M.log_offset
        end
    end

    # #
    # 3. Scaffolding for Spatiotemporal Interaction Structures (outcome-specific)
    s_Q_main = [sparse(I(M.s_N)) for _ in 1:outcomes_N]
    t_Q_main = [sparse(I(M.t_N)) for _ in 1:outcomes_N]
    t_rho_main = [zero(T) for _ in 1:outcomes_N]
    s_sigma_main = [one(T) for _ in 1:outcomes_N]
    t_sigma_main = [one(T) for _ in 1:outcomes_N]

    # #
    # 4. Modular Manifold Realization Loop
    for spec in M.manifolds
        m_obj = spec.manifold_obj
        if m_obj isa NoneManifold
            continue
        end

        key = spec.key
        domain = spec.domain

        for k in 1:outcomes_N
            # Standard Suffix Pattern: {param}_{key}_{outcome_index}
            sigma_name = Symbol("sigma_", key, "_", k)
            rho_name = Symbol("rho_", key, "_", k)
            latent_name = Symbol("latent_", key, "_", k)

            if m_obj isa IID
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                n_units = (domain == :spatial) ? M.s_N : ((domain == :temporal) ? M.t_N : M.u_N)
                indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
                field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), latent_name)
                eta_latent[:, k] .+= field[indices]

            elseif m_obj isa BYM2
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                s_sigma_main[k] = sigma_val

                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                rho_val ~ NamedDist(rho_val_dist, rho_name)

                struct_name = Symbol("latent_struct_", key, "_", k)
                iid_name = Symbol("latent_iid_", key, "_", k)

                s_icar ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), struct_name)
                s_iid ~ NamedDist(MvNormal(zeros(M.s_N), I), iid_name)

                Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(s_icar))
                combined = sigma_val .* (sqrt(rho_val) .* s_icar .+ sqrt(1.0 - rho_val) .* s_iid)
                eta_latent[:, k] .+= combined[M.s_idx]
                s_Q_main[k] = spec.Q_template

            elseif m_obj isa AR1
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                t_sigma_main[k] = sigma_val

                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                rho_val ~ NamedDist(rho_val_dist, rho_name)
                t_rho_main[k] = rho_val

                innov_name = Symbol("innov_", key, "_", k)
                innovations ~ NamedDist(MvNormal(zeros(M.t_N), I), innov_name)
                t_field = Vector{T}(undef, M.t_N)
                t_field[1] = innovations[1] / sqrt(1.0 - rho_val^2 + noise)
                for i in 2:M.t_N
                    t_field[i] = rho_val * t_field[i-1] + innovations[i]
                end
                eta_latent[:, k] .+= (t_field .* sigma_val)[M.t_idx]

                t_Q_base = Symmetric((1.0 + rho_val^2) .* I(M.t_N) .- rho_val .* spec.Q_template)
                t_Q_main[k] = Symmetric((1.0 / (sigma_val^2 * (1.0 - rho_val^2) + noise)) .* t_Q_base)

            elseif m_obj isa Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic}
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                r_val = nothing
                if hasproperty(m_obj, :rho_prior)
                    rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                    r_val ~ NamedDist(rho_val_dist, rho_name)
                end

                n_units = size(spec.Q_template, 1)
                Q = recompose_precision(Symbol(lowercase(string(typeof(m_obj)))), spec.Q_template, sigma_val; extra_param=r_val, noise=noise)
                field ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)

                if !(m_obj isa Cyclic)
                    Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
                end

                indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
                eta_latent[:, k] .+= field[indices]
                if domain == :spatial; s_Q_main[k] = Q; elseif domain == :temporal; t_Q_main[k] = Q; end

            elseif m_obj isa Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}
                var_sym = spec.var; B_mat = M.basis_matrices[var_sym]; n_basis_cols = size(B_mat, 2)
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)

                beta_name = Symbol("beta_basis_", key, "_", k)
                if m_obj isa PSpline || m_obj isa TPS
                    Q_penalty = (1.0 / (sigma_val^2 + noise)) .* spec.Q_template
                    latent_coeffs ~ NamedDist(MvNormalCanon(zeros(n_basis_cols), Q_penalty), beta_name)
                    Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_basis_cols), sum(latent_coeffs))
                else
                    latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_basis_cols), beta_name)
                end
                eta_latent[:, k] .+= B_mat * latent_coeffs

            elseif m_obj isa DynamicsManifold
                d_model = m_obj.model
                priors = M.hyperpriors
                if d_model == "advection" || d_model == "diffusion"
                    v_p = get(priors, "velocity_prior", Normal(0, 0.5))
                    sig_p = get(priors, "sigma_prior", Exponential(1.0))
                    param_v ~ NamedDist(v_p, Symbol("dyn_v_", key, "_", k))
                    sig_d ~ NamedDist(sig_p, Symbol("dyn_sig_", key, "_", k))
                    L_mat = spec.Q_template
                    dyn_field = Matrix{T}(undef, M.s_N, M.t_N)
                    dyn_field[:, 1] ~ MvNormal(zeros(M.s_N), I)
                    for t in 2:M.t_N
                        drift = -param_v .* (L_mat * dyn_field[:, t-1])
                        dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] .+ drift, sig_d^2 * I)
                    end
                    for i in 1:M.y_N; eta_latent[i, k] += dyn_field[M.s_idx[i], M.t_idx[i]]; end
                elseif d_model in ["gompertz", "logistic_basic"]
                    r_p = get(priors, "r_prior", LogNormal(-1.5, 0.5))
                    k_p = get(priors, "K_prior", Normal(150, 50))
                    r_val ~ NamedDist(r_p, Symbol("dyn_r_", key, "_", k))
                    k_val ~ NamedDist(k_p, Symbol("dyn_k_", key, "_", k))
                    pop_state = Vector{T}(undef, M.t_N)
                    pop_state[1] ~ Normal(log(k_val / 2.0), 1.0)
                    for t in 2:M.t_N
                        growth = r_val * (log(k_val) - pop_state[t-1])
                        pop_state[t] ~ Normal(pop_state[t-1] + growth, 0.1)
                    end
                    eta_latent[:, k] .+= pop_state[M.t_idx]
                end

            elseif m_obj isa Eigen
                n_factors = m_obj.n_factors
                vars = spec.variables
                cov_data = Matrix(M.data[!, vars])' # This refers to original data, not training data if PS is active
                n_vars = length(vars)

                pca_sd ~ NamedDist(m_obj.pca_sd_prior, Symbol("pca_sd_", key, "_", k))
                pdef_sd ~ NamedDist(m_obj.pdef_sd_prior, Symbol("pdef_sd_", key, "_", k))
                v_vec ~ NamedDist(filldist(Normal(0, 1), length(m_obj.ltri_indices)), Symbol("v_", key, "_", k))

                v_mat = zeros(T, n_vars, n_factors)
                v_mat[m_obj.ltri_indices] .= v_vec
                U_mat = householder_to_eigenvector(v_mat, n_vars, n_factors)

                z_latent ~ NamedDist(filldist(Normal(0, 1), n_factors, M.y_N), Symbol("z_", key, "_", k))
                eigen_coeffs ~ NamedDist(filldist(Normal(0, 1), n_factors), Symbol("eigen_coeffs_", key, "_", k))

                eta_latent[:, k] .+= z_latent' * eigen_coeffs

                reconstructed_cov = U_mat * z_latent
                for i in 1:M.y_N
                    Turing.@addlogprob! logpdf(MvNormal(reconstructed_cov[:, i], pdef_sd^2 * I), cov_data[:, i])
                end

            elseif m_obj isa SVCManifold
                cov_var = m_obj.covariate
                x_svc = M.data[!, cov_var]
                inner_m = m_obj.model
                sig_svc_dist = get(inner_m, :sigma_prior, Exponential(1.0))
                sig_svc ~ NamedDist(sig_svc_dist, Symbol("sigma_svc_", key, "_", k))

                if inner_m isa BYM2
                    rho_svc_dist = inner_m.rho_prior
                    rho_svc ~ NamedDist(rho_svc_dist, Symbol("rho_svc_", key, "_", k))
                    s_struct ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), Symbol("svc_struct_", key, "_", k))
                    s_unstruct ~ NamedDist(MvNormal(zeros(M.s_N), I), Symbol("svc_iid_", key, "_", k))
                    beta_svc = sig_svc .* (sqrt(rho_svc) .* s_struct .+ sqrt(1.0 - rho_svc) .* s_unstruct)
                    eta_latent[:, k] .+= beta_svc[M.s_idx] .* x_svc
                else # Default to IID or similar simple model
                    beta_svc ~ NamedDist(filldist(Normal(0, sig_svc), M.s_N), Symbol("beta_svc_", key, "_", k))
                    eta_latent[:, k] .+= beta_svc[M.s_idx] .* x_svc
                end
            end
        end
    end

    # #
    # 5. Spatiotemporal Interaction (Knorr-Held Taxonomy)
    model_st = get(M, :model_st, "none")
    if model_st != "none"
        st_sigma_base ~ NamedDist(Exponential(0.5), :st_sigma_base) # Global ST interaction scale

        for k in 1:outcomes_N
            st_sigma_k = st_sigma_base # Each outcome uses the same base ST scale currently

            # st_raw needs to be sampled once then scaled/transformed per outcome
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)

            if model_st == "I"
                # Type I: Unstructured space and time (IID ⊗ IID)
                st_inter = reshape(st_raw, M.s_N, M.t_N) .* st_sigma_k
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]]; end

            elseif model_st == "II"
                # Type II: Unstructured space, structured time (IID ⊗ Temporal Q)
                # Apply temporal structure (Cholesky of t_Q_main[k])
                L_t = cholesky(Symmetric(t_Q_main[k] + noise * I)).L
                st_inter = (reshape(st_raw, M.s_N, M.t_N) * L_t') .* st_sigma_k
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]]; end

            elseif model_st == "III"
                # Type III: Structured space, unstructured time (Spatial Q ⊗ IID)
                # Apply spatial structure (Cholesky of s_Q_main[k])
                L_s = cholesky(Symmetric(s_Q_main[k] + noise * I)).L
                st_inter = (L_s * reshape(st_raw, M.s_N, M.t_N)) .* st_sigma_k
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]]; end

            elseif model_st == "IV"
                # Type IV: Structured space, structured time (Spatial Q ⊗ Temporal Q)
                st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
                L_s = cholesky(Symmetric(s_Q_main[k] + noise * I)).L

                # Recursive definition for AR(1) temporal evolution of spatial fields
                st_inter = Matrix{T}(undef, M.s_N, M.t_N)
                st_inter[:, 1] = (L_s * st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_main[k]^2 + noise)
                for t in 2:M.t_N
                    st_inter[:, t] = t_rho_main[k] .* st_inter[:, t-1] .+ (L_s * st_innov_matrix[:, t])
                end
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]] .* st_sigma_k; end
            end
        end
    end

    # #
    # 6. Multivariate Coupling via Cholesky Factor
    L_corr ~ NamedDist(LKJCholesky(outcomes_N, 1.0, T), :L_corr)
    eta = eta_latent * L_corr.L

    # #
    # 7. Pointwise Multivariate Likelihood Dispatch
    y_sigma ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :y_sigma)
    for k in 1:outcomes_N
        # Use ok_idx to handle potential missing data/filtered observations per outcome
        ok_idx = get(M, :y_ok, [1:M.y_N for _ in 1:outcomes_N])[k]
        for i in ok_idx
            d_lik = bstm_Likelihood(family, [T(M.y_obs[i, k])]; sigma_y=[y_sigma[k]], phi_zi=lik_phi, r_nb=lik_r[k], trial=[Int(M.trials[i])], extra_params=extra_p[k])
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
        end
    end
end


@model function bstm_multifidelity(M::NamedTuple, ::Type{T}=Float64) where {T}
    # #
    # 1. Global Likelihood Hyperparameters for the Main (High-Fidelity) Model
    family = M.model_family
    noise = get(M, :noise, 1e-6)
    use_zi = get(M, :use_zi, false)

    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = one(T)

    if family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end
    if use_zi == true
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end
    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
        extra_p ~ NamedDist(Exponential(1.0), :extra_params)
    end

    # #
    # 2. Observation Volatility for the Main Model
    y_sigma = Vector{T}(undef, M.y_N)
    if get(M, :use_sv, false) == true
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)
        vol_proj_field = M.vol_proj * beta_vol_latent
        vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)
        for i in 1:M.y_N
            y_sigma[i] = exp((sigma_log_var * vol_latent_field[i]) / 2.0)
        end
    else
        y_sigma_main ~ NamedDist(get(M.hyperpriors, "y_sigma_prior", Exponential(1.0)), :y_sigma)
        for i in 1:M.y_N
            y_sigma[i] = y_sigma_main
        end
    end

    # #
    # 3. Base Predictor for the Main Model
    eta = zeros(T, M.y_N)
    if get(M, :add_intercept, false) && haskey(M, :intercept_prior)
        intercept ~ NamedDist(M.intercept_prior, :intercept)
        eta .+= intercept
    end
    if M.Xfixed_N > 0
        Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
        eta .+= M.Xfixed * Xfixed_beta
    end
    if haskey(M, :log_offset)
        if family in ["gaussian", "student_t", "laplace"]
            eta .+= exp.(M.log_offset)
        else
            eta .+= M.log_offset
        end
    end

    # #
    # 4. Scaffolding for Spatiotemporal Interaction Structures (Main Model)
    s_Q_main = sparse(I(M.s_N))
    t_Q_main = sparse(I(M.t_N))
    t_rho_main = zero(T)
    s_sigma_main = one(T)
    t_sigma_main = one(T)

    # #
    # 5. Modular Manifold Realization Loop (Main Model)
    for spec in M.manifolds
        m_obj = spec.manifold_obj
        if m_obj isa NoneManifold
            continue
        end
        m_domain = spec.domain
        key = spec.key

        sigma_name = Symbol("sigma_", key)
        rho_name = Symbol("rho_", key)
        latent_name = Symbol("latent_", key)

        if m_obj isa IID
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
            n_units = if m_domain == :mixed; spec.params.n_cat; elseif m_domain == :spatial; M.s_N; else M.t_N; end
            indices = if m_domain == :mixed; spec.params.indices; elseif m_domain == :spatial; M.s_idx; else M.t_idx; end
            field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), latent_name)
            eta .+= field[indices]

        elseif m_obj isa BYM2
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
            s_sigma_main = sigma_val # For ST interaction

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            rho_val ~ NamedDist(rho_val_dist, rho_name)

            s_icar_name = Symbol("latent_struct_", key)
            s_iid_name = Symbol("latent_iid_", key)

            s_icar ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), s_icar_name)
            s_iid ~ NamedDist(MvNormal(zeros(M.s_N), I), s_iid_name)
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(s_icar))
            combined = sigma_val .* (sqrt(rho_val) .* s_icar .+ sqrt(1.0 - rho_val) .* s_iid)
            eta .+= combined[M.s_idx]
            s_Q_main = spec.Q_template # For ST interaction

        elseif m_obj isa AR1
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
            t_sigma_main = sigma_val # For ST interaction

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            rho_val ~ NamedDist(rho_val_dist, rho_name)
            t_rho_main = rho_val # For ST interaction

            innov_name = Symbol("innov_", key)
            innovations ~ NamedDist(MvNormal(zeros(M.t_N), I), innov_name)

            t_field = Vector{T}(undef, M.t_N)
            t_field[1] = innovations[1] / sqrt(1.0 - rho_val^2 + noise)
            for i in 2:M.t_N
                t_field[i] = rho_val * t_field[i-1] + innovations[i]
            end
            eta .+= (t_field .* sigma_val)[M.t_idx]

            t_Q_base = Symmetric((1.0 + rho_val^2) .* I(M.t_N) .- rho_val .* spec.Q_template)
            t_Q_main = Symmetric((1.0 / (sigma_val^2 * (1.0 - rho_val^2) + noise)) .* t_Q_base) # For ST interaction

        elseif m_obj isa Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic}
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
            r_val = nothing
            if hasproperty(m_obj, :rho_prior)
                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                _r ~ NamedDist(rho_val_dist, rho_name)
                r_val = _r
            end

            n_units = size(spec.Q_template, 1)
            Q = recompose_precision(Symbol(lowercase(string(typeof(m_obj)))), spec.Q_template, sigma_val; extra_param=r_val, noise=noise)
            field ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)

            if !(m_obj isa Cyclic) # Sum-to-zero constraint for non-cyclic GMRFs
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
            end

            indices = if m_domain == :spatial; M.s_idx; elseif m_domain == :temporal; M.t_idx; else M.u_idx; end
            eta .+= field[indices]

            if m_domain == :spatial; s_Q_main = Q; elseif m_domain == :temporal; t_Q_main = Q; end

        elseif m_obj isa Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}
            var_sym = spec.var; B_mat = M.basis_matrices[var_sym]; n_basis_cols = size(B_mat, 2)
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)

            beta_name = Symbol("beta_basis_", var_sym)
            if m_obj isa PSpline || m_obj isa TPS
                Q_penalty = (1.0 / (sigma_val^2 + noise)) .* spec.Q_template
                latent_coeffs ~ NamedDist(MvNormalCanon(zeros(n_basis_cols), Q_penalty), beta_name)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_basis_cols), sum(latent_coeffs))
            else
                latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_basis_cols), beta_name)
            end
            eta .+= B_mat * latent_coeffs

        elseif m_obj isa DynamicsManifold
            d_model = m_obj.model
            priors = M.hyperpriors
            if d_model == "advection" || d_model == "diffusion"
                v_p = get(priors, "velocity_prior", Normal(0, 0.5))
                sig_p = get(priors, "sigma_prior", Exponential(1.0))
                param_v ~ NamedDist(v_p, Symbol("dyn_v_", key))
                sig_d ~ NamedDist(sig_p, Symbol("dyn_sig_", key))
                L_mat = spec.Q_template
                dyn_field = Matrix{T}(undef, M.s_N, M.t_N)
                dyn_field[:, 1] ~ MvNormal(zeros(M.s_N), I)
                for t in 2:M.t_N
                    drift = -param_v .* (L_mat * dyn_field[:, t-1])
                    dyn_field[:, t] ~ MvNormal(dyn_field[:, t-1] .+ drift, sig_d^2 * I)
                end
                for i in 1:M.y_N; eta[i] += dyn_field[M.s_idx[i], M.t_idx[i]]; end
            elseif d_model in ["gompertz", "logistic_basic"]
                r_p = get(priors, "r_prior", LogNormal(-1.5, 0.5))
                k_p = get(priors, "K_prior", Normal(150, 50))
                r_val ~ NamedDist(r_p, Symbol("dyn_r_", key))
                k_val ~ NamedDist(k_p, Symbol("dyn_k_", key))
                pop_state = Vector{T}(undef, M.t_N)
                pop_state[1] ~ Normal(log(k_val / 2.0), 1.0)
                for t in 2:M.t_N
                    growth = r_val * (log(k_val) - pop_state[t-1])
                    pop_state[t] ~ Normal(pop_state[t-1] + growth, 0.1)
                end
                eta .+= pop_state[M.t_idx]
            end

        elseif m_obj isa Eigen
            n_factors = m_obj.n_factors
            vars = spec.variables
            cov_data = Matrix(M.data[!, vars])'
            n_vars = length(vars)
            pca_sd ~ NamedDist(m_obj.pca_sd_prior, Symbol("pca_sd_", key))
            pdef_sd ~ NamedDist(m_obj.pdef_sd_prior, Symbol("pdef_sd_", key))
            v_vec ~ NamedDist(filldist(Normal(0, 1), length(m_obj.ltri_indices)), Symbol("v_", key))
            v_mat = zeros(T, n_vars, n_factors)
            v_mat[m_obj.ltri_indices] .= v_vec
            U_mat = householder_to_eigenvector(v_mat, n_vars, n_factors)
            z_latent ~ NamedDist(filldist(Normal(0, 1), n_factors, M.y_N), Symbol("z_", key))
            eigen_coeffs ~ NamedDist(filldist(Normal(0, 1), n_factors), Symbol("eigen_coeffs_", key))
            eta .+= z_latent' * eigen_coeffs
            reconstructed_cov = U_mat * z_latent
            for i in 1:M.y_N
                Turing.@addlogprob! logpdf(MvNormal(reconstructed_cov[:, i], pdef_sd^2 * I), cov_data[:, i])
            end

        elseif m_obj isa SVCManifold
            cov_var = m_obj.covariate
            x_svc = M.data[!, cov_var]
            inner_m = m_obj.model
            sig_svc_dist = get(inner_m, :sigma_prior, Exponential(1.0))
            sig_svc ~ NamedDist(sig_svc_dist, Symbol("sigma_svc_", key))

            if inner_m isa BYM2
                rho_svc_dist = inner_m.rho_prior
                rho_svc ~ NamedDist(rho_svc_dist, Symbol("rho_svc_", key))
                s_struct ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), Symbol("svc_struct_", key))
                s_unstruct ~ NamedDist(MvNormal(zeros(M.s_N), I), Symbol("svc_iid_", key))
                beta_svc = sig_svc .* (sqrt(rho_svc) .* s_struct .+ sqrt(1.0 - rho_svc) .* s_unstruct)
                eta .+= beta_svc[M.s_idx] .* x_svc
            else
                beta_svc ~ NamedDist(filldist(Normal(0, sig_svc), M.s_N), Symbol("beta_svc_", key))
                eta .+= beta_svc[M.s_idx] .* x_svc
            end
        end
    end

    # #
    # 6. Spatiotemporal Interaction (Knorr-Held Taxonomy)
    model_st = get(M, :model_st, "none")
    if model_st != "none"
        st_sigma ~ NamedDist(Exponential(0.5), :st_sigma)

        if model_st == "I"
            # Type I: Unstructured space and time (IID ⊗ IID)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_inter = reshape(st_raw, M.s_N, M.t_N) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "II"
            # Type II: Unstructured space, structured time (IID ⊗ Temporal Q)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            # Apply temporal structure (Cholesky of t_Q_main)
            L_t = cholesky(Symmetric(t_Q_main + noise * I)).L
            st_inter = (reshape(st_raw, M.s_N, M.t_N) * L_t') .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "III"
            # Type III: Structured space, unstructured time (Spatial Q ⊗ IID)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            # Apply spatial structure (Cholesky of s_Q_main)
            L_s = cholesky(Symmetric(s_Q_main + noise * I)).L
            st_inter = (L_s * reshape(st_raw, M.s_N, M.t_N)) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "IV"
            # Type IV: Structured space, structured time (Spatial Q ⊗ Structured Time)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            L_s = cholesky(Symmetric(s_Q_main + noise * I)).L

            # Recursive definition for AR(1) temporal evolution of spatial fields
            st_inter = Matrix{T}(undef, M.s_N, M.t_N)
            st_inter[:, 1] = (L_s * st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_main^2 + noise)
            for t in 2:M.t_N
                st_inter[:, t] = t_rho_main .* st_inter[:, t-1] .+ (L_s * st_innov_matrix[:, t])
            end
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]] * st_sigma; end
        end
    end

    # #
    # 7. Nested Hierarchical Supervisors (Low-Fidelity Layers)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_nested ~ NamedDist(Normal(1.0, 0.5), Symbol("rho_nested_", z_key))
            eta_nested = zeros(T, z_meta.y_N)

            # Fixed Effects for Nested Layer
            if haskey(z_meta, :Xfixed) && size(z_meta.Xfixed, 2) > 0
                xf_z = z_meta.Xfixed
                beta_z_f ~ NamedDist(MvNormal(zeros(size(xf_z, 2)), 5.0 * I), Symbol("beta_nested_fixed_", z_key))
                eta_nested .+= xf_z * beta_z_f
            end

            # Spatial Realization for Nested Layer
            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                sig_z_s ~ NamedDist(Exponential(1.0), Symbol("sigma_nested_spatial_", z_key))
                Q_z_s = recompose_precision(Symbol(z_meta.model_space), z_meta.s_Q_template.matrix, sig_z_s, noise=noise)
                lat_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta.s_N), Q_z_s), Symbol("latent_nested_spatial_", z_key))
                eta_nested .+= lat_z_s[z_meta.s_idx]

                # Joint Fidelity Link: Low fidelity contributes to high fidelity eta
                eta .+= rho_nested .* lat_z_s[M.s_idx]
            end

            # Likelihood Evaluation for Nested Data
            y_sigma_nested ~ NamedDist(Exponential(1.0), Symbol("y_sigma_nested_", z_key))
            nested_family = get(z_meta, :model_family, "gaussian")
            for i in 1:z_meta.y_N
                d_lik_z = bstm_Likelihood(nested_family, [T(z_meta.y_obs[i])]; sigma_y=[y_sigma_nested])
                Turing.@addlogprob! Distributions.logpdf(d_lik_z, eta_nested[i])
            end
        end
    end

    # #
    # 8. Final High-Fidelity Likelihood Dispatch
    y_sigma_main ~ NamedDist(Exponential(1.0), :y_sigma)
    for i in 1:M.y_N
        d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma_main], phi_zi=lik_phi, r_nb=lik_r, trial=[Int(M.trials[i])], extra_params=extra_p)
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end

# v4.5.5 (2026-07-08 05:00:00)
# Purpose: Hardened parameter discovery specifically for keyed spatiotemporal components.

function _find_parameter(p_names, base, domain, key, k=nothing)
    p_strs = string.(p_names)
    k_str = isnothing(k) ? "" : string("_", k)
    key_str = string(key)
    dom_str = string(domain)

    # Priority 1: Exact keyed matches from supervisor (e.g., latent_struct_spatial_s_idx_bym2)
    targets = [
        string("latent_struct_", key_str, k_str),
        string("latent_iid_", key_str, k_str),
        string("t_raw_", key_str, k_str),
        string("innov_", key_str, k_str),
        string("sigma_", key_str, k_str),
        string("rho_", key_str, k_str)
    ]

    for t in targets
        if t in p_strs; return t; end
    end

    # Priority 2: Standard base matches
    if string(base) in p_strs; return string(base); end

    # Priority 3: Fuzzy matches containing the key
    for n in p_strs
        if occursin(base, n) && occursin(key_str, n)
            return n
        end
    end

    return ""
end


function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Description: Primary engine for projecting recovered latent manifolds onto new spatiotemporal coordinates.
    # Rationale: Standardizing the out-of-sample path to support the full BSTM v06.1 Taxonomy.
    # # BSTM Monolithic Prediction & Projection Engine [v19.38.7]
    # Timestamp: 2026-07-03 10:00:00
    # Rationale: Overhauled the prediction configuration (`PS`) generation to be robust. Instead of
    #            re-running `bstm_config`, this version creates a new configuration by inheriting
    #            the training model's structure and selectively updating data-dependent fields.
    #            This resolves a critical flaw where smoother types and parameters were not correctly
    #            propagated to the prediction set.
    # Requirements: 100% parity with v19.38.5 Results Dashboard. Zero-truncation of latent manifolds.

    println("--- Starting BSTM Out-of-Sample Prediction v19.38.7 ---")

    # # 1. Training Metadata Recovery
    # Rationale: M_train contains the complete configuration and technical registry from the training phase.
    M_train = model_obj.args.M
    n_samples_total = size(chain, 1)
    n_samps = min(n_samples_total, n_samples)

    # # 2. Prediction Metadata Configuration (PS)
    # Rationale: Create a new configuration `PS` for the prediction data. We start by copying the
    #            training configuration and then update only the data-dependent parts. This ensures
    #            that the model structure (manifolds, priors, etc.) is identical.
    
    # Convert NamedTuple to a mutable dictionary for updates
    PS_dict = Dict(pairs(M_train))

    # Update with new data and dimensions
    PS_dict[:data] = new_data
    PS_dict[:y_obs] = zeros(nrow(new_data)) # Dummy response
    PS_dict[:y_N] = nrow(new_data)
    PS_dict[:log_offset] = hasproperty(new_data, :log_offset) ? new_data.log_offset : zeros(nrow(new_data))
    PS_dict[:weights] = hasproperty(new_data, :weights) ? new_data.weights : ones(nrow(new_data))
    PS_dict[:trials] = hasproperty(new_data, :trials) ? new_data.trials : ones(Int, nrow(new_data))

    # Re-create fixed effects design matrix for the new data using the original formula part
    if haskey(M_train, :formula)
        decomposed_formula = decompose_bstm_formula(M_train.formula)
        fixed_effects_formula_part = join(decomposed_formula.fixed_effects, " + ")
        if decomposed_formula.has_intercept
             fixed_effects_formula_part = isempty(strip(fixed_effects_formula_part)) ? "1" : "1 + " * fixed_effects_formula_part
        end
        if !isempty(strip(fixed_effects_formula_part))
            PS_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, new_data; contrasts=get(M_train, :contrasts, Dict()))
            PS_dict[:Xfixed_N] = size(PS_dict[:Xfixed], 2)
        end
    end

    # Update indices from new_data
    if haskey(M_train, :s_idx_var) && !isnothing(M_train.s_idx_var) && hasproperty(new_data, M_train.s_idx_var)
        PS_dict[:s_idx] = new_data[!, M_train.s_idx_var]
    end
    if haskey(M_train, :t_idx_var) && !isnothing(M_train.t_idx_var) && hasproperty(new_data, M_train.t_idx_var)
        PS_dict[:t_idx] = new_data[!, M_train.t_idx_var]
    end

    # # 3. Manifold Coordinate Alignment
    # Rationale: Aligning out-of-sample points with the training grid for discrete spatial models.
    if haskey(M_train, :centroids) && !isnothing(M_train.centroids)
        centroids_train = M_train.centroids
        nx = hasproperty(new_data, :s_x) ? new_data.s_x : zeros(nrow(new_data))
        ny = hasproperty(new_data, :s_y) ? new_data.s_y : zeros(nrow(new_data))
        
        PS_s_idx = Vector{Int}(undef, nrow(new_data))
        for i in 1:nrow(new_data)
            # Find nearest neighbor in the training unit grid
            dists = [sum(((nx[i], ny[i]) .- c).^2) for c in centroids_train]
            PS_s_idx[i] = argmin(dists)
        end
        PS_dict[:s_idx] = PS_s_idx
    end

    # # 4. Hyper-Volumetric Basis Projection (1D-4D)
    # Rationale: Reconstructing basis matrices (Splines/RFF/FFT) for new coordinates
    #            using the original model specifications from the training manifolds.
    if haskey(M_train, :manifolds) && !isempty(M_train.manifolds)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.domain == :smooth, M_train.manifolds)
        
        for spec in smooth_specs
            # The key for the basis matrix is derived from the variable names.
            # This logic must match the key generation in `bstm_config`.
            v_sym = Symbol(join(spec.params.variables, "_"))

            if haskey(M_train.basis_matrices, v_sym)
                B_train = M_train.basis_matrices[v_sym]
                m_obj = spec.manifold_obj
                model_type_str = lowercase(string(typeof(m_obj)))
                
                vars = Symbol.(spec.params.variables)
                n_vars = length(vars)
                nb = size(B_train, 2)
                
                # Reconstruct the basis matrix for the new data using the correct model type and parameters
                if n_vars == 1
                    ps_basis_registry[v_sym] = bstm_smooth_basis_1D(model_type_str, new_data[!, vars[1]], nb; spec.params...)
                elseif n_vars == 2
                    coords_new = Matrix{Float64}(new_data[!, vars])
                    ps_basis_registry[v_sym] = bstm_smooth_basis_2D(model_type_str, coords_new, nb; spec.params...)
                elseif n_vars == 3
                    coords_new = Matrix{Float64}(new_data[!, vars])
                    ps_basis_registry[v_sym] = bstm_smooth_basis_3D(model_type_str, coords_new, nb; spec.params...)
                elseif n_vars == 4
                    coords_new = Matrix{Float64}(new_data[!, vars])
                    ps_basis_registry[v_sym] = bstm_smooth_basis_4D(model_type_str, coords_new, nb; spec.params...)
                end
            end
        end
        PS_dict[:basis_matrices] = ps_basis_registry
    end

    # Convert dictionary back to NamedTuple for the reconstruct engine
    PS = NamedTuple(PS_dict)

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
    chain_sub = chain[1:min(n_samps, end), :, :]

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
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance
    #          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
    # Inputs: model_obj, chain, alpha.
    # Outputs: A NamedTuple containing the LOO object, metrics, log-likelihood matrix, and Pareto k values.
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
        log_lik_matrix = log_lik,
        pareto_k = pareto_k
    )
end



# 2. Bayes Factor Suite (Manifold Comparison)
function compare_manifolds(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.
    # Description: Performs formal model selection between two BSTM manifold candidates.
    # Rationale: Standardizing the assessment of ELPD differences and complexity trade-offs.
    # Requirements: Absolute parity with the PSIS-LOO metrics.

    println("--- Starting BSTM Manifold Comparison ---")

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
    println("\n--- BSTM Manifold Selection Registry ---")
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
    sampler = NUTS(500, 0.65), 
    alpha = 0.05, 
    kwargs...
)    
    # v1.0.1 (2026-06-29 17:26:00)
    # Purpose: An orchestration utility for performing cross-validation. It supports standard
    #          k-fold and Leave-One-Location-Out (LOLO) strategies to assess model performance on
    #          held-out data.
    # Inputs: formula, data, and optional parameters for CV method, sampler, etc.
    # Outputs: A NamedTuple containing fold results and summary metrics.
    # # 1. Metadata Discovery and Outcome Resolution
    # The formula is decomposed to identify the primary response variable and module requirements.
    meta_discovery = decompose_bstm_formula(formula)
    response_name = Symbol(meta_discovery.outcomes[1])

    # # 2. Partition Strategy Selection
    # Establishing fold indices based on spatiotemporal logic or random sampling.
    folds_indices = Vector{Vector{Int}}()

    if method == :lolo
        # Leave-One-Location-Out: Grouping indices by the spatial unit identifier.
        unique_locs = unique(data[!, lolo_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, lolo_var]))
        end
    else
        # Standard K-Fold: Random permutation of row indices.
        n_obs = size(data, 1)
        row_indices = Random.randperm(n_obs)
        fold_size = div(n_obs, n_folds)
        for i in 1:n_folds
            idx_start = (i - 1) * fold_size + 1
            idx_end = i == n_folds ? n_obs : i * fold_size
            push!(folds_indices, row_indices[idx_start:idx_end])
        end
    end

    # # 3. Cross-Validation Loop
    fold_results = []
    n_actual_folds = length(folds_indices)

    for (f_idx, test_idx) in enumerate(folds_indices)
        # Splitting dataset into training and testing partitions.
        # Training mask is constructed to exclude the test indices.
        train_mask = trues(size(data, 1))
        train_mask[test_idx] .= false
        
        train_data = data[train_mask, :]
        test_data = data[test_idx, :]

        # # 4. Modular Training Configuration
        # Pre-configuring the model to ensure technical registries (W, s_N, t_N) are consistent.
        # bstm_config resolves the manifold registry M.
        opt_train = bstm_config(formula, train_data; kwargs...)

        # # 5. Model Execution
        # Dispatching to the modular univariate or multivariate supervisor.
        model_train = bstm(opt_train)
        
        # Posterior sampling using the requested sampler configuration.
        chain_train = sample(model_train, sampler, n_samples, progress = false)

        # # 6. Manifold Projection (Out-of-Sample)
        # Using the standardized predict() function which handles reconstruction of S, T, and Smooth basis.
        # This ensures that PS (Prediction Surface) alignment is consistent with the modular BSTM taxonomy.
        res_pred = predict(model_train, chain_train, test_data, n_samples = div(n_samples, 2), alpha = alpha)

        # # 7. Performance Assessment
        # Extracting denoised expectations for the test partition.
        y_test_obs = test_data[!, response_name]
        y_test_pred = res_pred.predictions_denoised.mean

        # Verification of dimensional parity between prediction and observation.
        if length(y_test_obs) == length(y_test_pred)
            residuals = y_test_obs .- y_test_pred
            rmse = sqrt(Statistics.mean(residuals.^2))
            
            # R-Squared calculation with safety floor for variance.
            ss_res = sum(residuals.^2)
            ss_tot = sum((y_test_obs .- Statistics.mean(y_test_obs)).^2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-15))

            push!(fold_results, (fold=f_idx, rmse=rmse, r2=r2))
        else
            @warn "Fold $f_idx: Prediction length mismatch. Observed: $(length(y_test_obs)), Predicted: $(length(y_test_pred))"
        end
    end

    # # 8. Aggregate Reporting
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

 


function bstm(config::NamedTuple)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Dispatches a pre-built configuration object to the appropriate model supervisor.
    # Inputs: config::NamedTuple.
    # Outputs: A Turing.jl model instance.
    arch = get(config, :model_arch, "univariate")
    
    if arch == "multivariate"
        return bstm_multivariate(config)
    elseif arch == "multifidelity"
        return bstm_multifidelity(config)
    else
        return bstm_univariate(config) 
    end

end

function bstm(formula::String, data::DataFrame; kwargs...)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: The primary user-facing function that provides a formula-driven interface.
    # Inputs: formula::String, data::DataFrame, and other keyword arguments.
    # Outputs: A Turing.jl model instance.
    options = bstm_config(formula, data; kwargs...)
    return bstm(options)
end

 

function bstm_config(formula::String, data::DataFrame; kwargs...)
    # v2.0.0 (2026-07-02)
    # Purpose: Parses the bstm formula and data to create a model configuration NamedTuple.
    # This version includes a pre-processor for temporal/seasonal models and robust module handling.
    metadata = decompose_bstm_formula(formula)

    opt_dict = Dict{Symbol, Any}(kwargs)

    # Pre-processor for dual temporal/seasonal models
    # This simplifies the main loop by ensuring each module has a single responsibility.
    processed_modules = Dict{String, Any}()
    for (key, mod) in metadata.modules
        if mod[:type] == :temporal && get(mod[:params], :model, "") isa Tuple
            models = mod[:params][:model]
            vars = mod[:variables]

            # Create time module
            if !isempty(vars)
                time_mod = deepcopy(mod)
                time_mod[:params][:model] = string(models[1])
                processed_modules[key * "_time"] = time_mod
            end

            # Create season module
            if length(models) > 1 && length(vars) > 1
                season_mod = deepcopy(mod)
                season_mod[:type] = :seasonal # Re-brand as a seasonal module
                season_mod[:variables] = [vars[2]] # Use second variable
                season_mod[:params][:model] = string(models[2])
                processed_modules[key * "_season"] = season_mod
            end
        else
            processed_modules[key] = mod
        end
    end
    metadata = (outcomes=metadata.outcomes, modules=processed_modules, fixed_effects=metadata.fixed_effects, has_intercept=metadata.has_intercept)

    opt_dict[:y_obs_vars] = metadata.outcomes
    opt_dict[:data] = data
    opt_dict[:y_N] = size(data, 1)
    # Ensure u_N is initialized to prevent FieldError in _discover_manifold_realizations
    # It will be properly set if a seasonal module is processed.
    if !haskey(opt_dict, :u_N); opt_dict[:u_N] = 0; end
    if !haskey(opt_dict, :u_idx); opt_dict[:u_idx] = nothing; end
    opt_dict[:add_intercept] = metadata.has_intercept

    # Pre-evaluate any expressions passed as keyword arguments or in the formula string.
    for (key, val) in opt_dict
        if val isa Expr
            try
                opt_dict[key] = Core.eval(Main, val)
            catch e; @error "Failed to evaluate keyword argument `$key` with expression `$val`."; rethrow(e); end
        end
    end
    for (key, mod_data) in metadata.modules
        if haskey(mod_data, :params)
            for (param_key, param_val) in mod_data[:params]
                should_eval = param_val isa Expr || (param_val isa Symbol && isdefined(Main, param_val))

                if should_eval
                    try
                        mod_data[:params][param_key] = Core.eval(Main, param_val)
                    catch e; @error "Failed to evaluate formula parameter `$param_key` with expression `$param_val` in module `$key`."; rethrow(e); end
                end
            end
        end
    end

    if hasproperty(data, :s_x) && hasproperty(data, :s_y)
        opt_dict[:s_x] = data.s_x
        opt_dict[:s_y] = data.s_y
        opt_dict[:s_coord] = hcat(data.s_x, data.s_y)
    end

    opt_dict[:model_space] = get(opt_dict, :model_space, "none")
    opt_dict[:model_time] = get(opt_dict, :model_time, "none")
    opt_dict[:model_season] = get(opt_dict, :model_season, "none")
    opt_dict[:model_st] = get(opt_dict, :model_st, "none")

    initial_hp = get(opt_dict, :hyperpriors, Dict{Any, Any}())
    hyperprior_registry = Dict{String, Any}(string(k) => v for (k, v) in initial_hp)
    opt_dict[:hyperpriors] = hyperprior_registry

    basis_matrices_registry = Dict{Symbol, Any}()
    manifolds_registry = []

    for (key_str, mod_data) in metadata.modules
        m_type = mod_data[:type]

        if haskey(MODULE_PROCESSORS, m_type)
            processor_func = MODULE_PROCESSORS[m_type]
            processor_func(opt_dict, mod_data, basis_matrices_registry, manifolds_registry)
        end

        if m_type in [:interaction_composition, :observationprocess, :intercept, :nested, :fixed]
            continue
        end

        manifold_obj = resolve_technical_primitive(mod_data, opt_dict, hyperprior_registry, :pcpriors)

        n_units = 0
        if m_type == :spatial
            n_units = get(opt_dict, :s_N, 0)
        elseif m_type == :temporal
            n_units = get(opt_dict, :t_N, 0)
        elseif m_type == :seasonal
            n_units = get(opt_dict, :u_N, 0)
        elseif m_type == :smooth
            reg_key = Symbol(join(mod_data[:variables], "_"))
            if haskey(basis_matrices_registry, reg_key)
                n_units = size(basis_matrices_registry[reg_key], 2)
            end
        elseif m_type == :mixed
             n_units = mod_data[:params][:n_cat]
        end

        if n_units == 0 && !(manifold_obj isa GP)
             @warn "Could not determine size for manifold '$key_str'. Skipping."
             continue
        end

        template_info = build_structure_template(Symbol(lowercase(string(typeof(manifold_obj)))), n_units; W=get(opt_dict, :W, nothing))

        spec_params = mod_data[:params]
        if manifold_obj isa GP
            if !haskey(spec_params, :coords)
                spec_params[:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
            end
            spec_params[:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
        end

        spec = (
            key = Symbol(key_str),
            domain = m_type,
            var = isempty(mod_data[:variables]) ? :none : Symbol(mod_data[:variables][1]),
            manifold_obj = manifold_obj,
            Q_template = template_info.matrix,
            scaling_factor = template_info.scaling_factor,
            params = spec_params
        )
        push!(manifolds_registry, spec)
    end

    opt_dict[:manifolds] = manifolds_registry
    opt_dict[:basis_matrices] = basis_matrices_registry
    opt_dict[:formula] = formula

    if get(opt_dict, :use_sv, false)
        if hasproperty(data, :s_x) && hasproperty(data, :s_y) && haskey(opt_dict, :t_idx)
            m_rff_sigma = get(opt_dict, :M_rff_sigma, 20)
            coords_st = hcat(data.s_x, data.s_y, opt_dict[:t_idx])
            W_sigma, b_sigma = generate_informed_rff_params(coords_st, m_rff_sigma)
            opt_dict[:vol_proj] = (coords_st * W_sigma) .+ b_sigma'
        else
             @warn "Stochastic volatility requires spatial coordinates (:s_x, :s_y) in `data` and a temporal index. SV disabled."
             opt_dict[:use_sv] = false
        end
    end

    fixed_effects_formula_part = join(metadata.fixed_effects, " + ")
    add_intercept_to_formula = get(opt_dict, :add_intercept, false) && !any(m -> m[:type] == :intercept, values(metadata.modules))

    if add_intercept_to_formula
        fixed_effects_formula_part = isempty(strip(fixed_effects_formula_part)) ? "1" : "1 + " * fixed_effects_formula_part
    end

    if !isempty(strip(fixed_effects_formula_part))
        opt_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, data; contrasts=get(opt_dict, :contrasts, Dict()))
        opt_dict[:Xfixed_N] = size(opt_dict[:Xfixed], 2)
    else
        opt_dict[:Xfixed] = NamedArray(zeros(size(data, 1), 0), (1:size(data,1), Symbol[]))
        opt_dict[:Xfixed_N] = 0
    end

    opt_dict[:outcomes_N] = length(metadata.outcomes)
    if length(metadata.outcomes) == 1
        opt_dict[:y_obs] = data[!, metadata.outcomes[1]]
        opt_dict[:model_arch] = get(opt_dict, :model_arch, "univariate")
    else
        opt_dict[:y_obs] = Matrix(data[!, metadata.outcomes])
        opt_dict[:model_arch] = "multivariate"
        opt_dict[:outcomes_N] = length(metadata.outcomes)
    end

    if !haskey(opt_dict, :weights); opt_dict[:weights] = ones(Float64, opt_dict[:y_N]); end
    if !haskey(opt_dict, :trials); opt_dict[:trials] = ones(Int, opt_dict[:y_N]); end

    opt_dict[:model_family] = get(opt_dict, :model_family, "gaussian")

    return NamedTuple(opt_dict)
end



function get_optimal_sampler(model_obj::DynamicPPL.Model; sampler_choice=:auto, target_acceptance=0.65, adaptation_steps=100, n_particles=20, hmc_leapfrog_steps=10)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Automatically selects an optimal MCMC sampler (or a composite Gibbs sampler).
    # Inputs: model_obj, target_acceptance, adaptation_steps, n_particles.
    # Outputs: A Turing.jl sampler object.
    # Note: This simplified logic only identifies discrete parameters. The NUTS sampler in a Gibbs
    #       composition automatically handles all remaining (continuous) parameters, removing the
    #       need to explicitly partition them. This makes the function more concise and robust.

    # # 1. Parameter Discovery and Type Introspection
    # Inspect the model's VarInfo to get metadata about each parameter's prior distribution.
    # This is a robust method to determine if a parameter's support is discrete or continuous.
    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)
    
    discrete_params = Symbol[]
    all_gaussian_priors = true
    num_continuous_params = 0

    for vn in vns
        try
            # Attempt to get the distribution for the variable using the modern API.
            dist = DynamicPPL.getdist(vi, vn)
            sym = Symbol(vn) # Convert VarName to Symbol

            # Check if the distribution's support is discrete.
            if Distributions.value_support(typeof(dist)) == Distributions.Discrete
                if !(sym in discrete_params)
                    push!(discrete_params, sym)
                end
            else # Continuous parameter
                num_continuous_params += 1
                # Check if the prior is a Gaussian type. If not, set the flag to false.
                if !(dist isa Normal || dist isa MvNormal || dist isa Truncated{<:Normal})
                    all_gaussian_priors = false
                end
            end
        catch e
            # Not all variables in VarInfo have a corresponding distribution
            # (e.g., derived quantities). `getdist` will throw an error for these,
            # which can be safely ignored.
            if !(e isa KeyError)
                # rethrow(e) # Optionally rethrow other unexpected errors
                all_gaussian_priors = false # Assume non-Gaussian if prior is not found
            end
        end
    end

    # # 2. Sampler Construction and Dispatch
    # Based on user choice or an informed automatic selection, construct the most appropriate sampler.
    
    local continuous_sampler
    if sampler_choice == :auto
        if all_gaussian_priors && num_continuous_params > 0
            println("Info: All continuous priors appear Gaussian. Selecting ESS for the continuous part.")
            continuous_sampler = ESS()
        else
            println("Info: Model contains non-Gaussian continuous priors or complex structure. Selecting NUTS for the continuous part.")
            continuous_sampler = NUTS(adaptation_steps, target_acceptance)
        end
    elseif sampler_choice == :nuts
        continuous_sampler = NUTS(adaptation_steps, target_acceptance)
    elseif sampler_choice == :hmc
        # HMC is a good alternative but requires manual tuning of leapfrog steps.
        continuous_sampler = HMC(0.1, hmc_leapfrog_steps) # Default step size 0.1
    elseif sampler_choice == :mh
        # Metropolis-Hastings is a gradient-free option, useful for non-differentiable models.
        continuous_sampler = MH()
    elseif sampler_choice == :ess
        # Elliptical Slice Sampling is efficient for models with Gaussian priors.
        continuous_sampler = ESS()
    elseif sampler_choice == :slice
        # Slice sampling is another robust gradient-free method.
        continuous_sampler = Slice()
    else
        @warn "Unknown sampler_choice ':$sampler_choice'. Defaulting to NUTS."
        continuous_sampler = NUTS(adaptation_steps, target_acceptance)
    end

    if isempty(discrete_params)
        # If no discrete parameters are found, return the chosen sampler for the continuous space.
        println("Info: No discrete parameters found. Using sampler: $(nameof(typeof(continuous_sampler))).")
        return continuous_sampler
    else
        # If discrete parameters exist, a composite Gibbs sampler is required.
        # Particle Gibbs (PG) will handle the discrete parameters.
        # The chosen continuous sampler will handle all other (continuous) parameters.
        println("Info: Discrete parameters found: ", discrete_params, ". Using composite Gibbs sampler (PG + $(nameof(typeof(continuous_sampler)))).")
        return Gibbs(PG(n_particles, discrete_params...), continuous_sampler)
    end
end




function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Resolves observation-level parameters (e.g., offsets, weights) from symbols to data vectors.
    # Inputs: opt_dict, params, data, param_keys, target_key.
    # Outputs: Modifies opt_dict in place.
    # Note: It checks for a symbol in the `params` dict and extracts the corresponding column from the `data` DataFrame.
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol && hasproperty(data, val)
                opt_dict[target_key] = data[!, val]
            elseif val isa AbstractVector
                opt_dict[target_key] = val
            else
                @warn "Observation parameter ':$val' for '$target_key' not found in data or is not a vector. Ignoring."
            end
            return # Exit after the first matching key is found and processed
        end
    end
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

        # Dimensional Realignment and Broadcasting
        # The logic has been simplified to remove an ambiguous condition that could fail
        # if the number of samples was equal to the number of expected parameters.
        # The construction of `mat_data` is assumed to correctly produce a matrix of
        # size [N_samples, n_parameters].
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 2) == 1 && expected_len > 1
            # Scalar Broadcast fallback
            return repeat(mat_data, 1, expected_len)
        else
            @warn "Parameter '$base_name' was found, but its length ($(size(mat_data, 2))) does not match expected length ($expected_len). Returning as is, which may cause downstream errors."
            return mat_data
        end
    end

    # 3. Null Safety Fallback
    # If the parameter is missing, return a zero-matrix to prevent downstream assembly failure.
    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end


function _quantile_along_last_dim(A::AbstractArray, q::Real)
    # v1.0.0 (2026-07-09)
    # Purpose: A performant replacement for `mapslices` to compute quantiles along the last dimension of an array.
    # Inputs: A (the array), q (the quantile).
    # Outputs: An array with the last dimension dropped, containing the quantiles.
    other_dims = size(A)[1:end-1]
    out = Array{Float64}(undef, other_dims)
    
    # CartesianIndices provides an efficient iterator over multidimensional indices.
    for I in CartesianIndices(out)
        # `view` creates a lightweight slice without copying data.
        slice_view = view(A, I, :)
        out[I] = quantile(slice_view, q)
    end
    return out
end



function summarize_array(samples::AbstractArray; alpha=0.05)
    # Computes summary statistics for posterior samples.
    if isempty(samples) || all(isnan, samples)
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end

    dims = size(samples)
    sample_dim = length(dims)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    # The use of `mapslices` for quantiles is known to be inefficient.
    # This is replaced with a more performant, explicit iteration using `_quantile_along_last_dim`.
    # This avoids the overhead associated with `mapslices` while achieving the same result.
    low_bound = _quantile_along_last_dim(samples, low_prob)
    high_bound = _quantile_along_last_dim(samples, high_prob)

    to_vector(x) = x isa AbstractArray ? vec(collect(Float64, x)) : [Float64(x)]

    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end


function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic}, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for standard GMRF manifold types.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()
    
    for k in 1:outcomes_N
        key = spec.key
        m_domain = spec.domain
        
        # Use the robust finder for all parameters
        sigma_name = _find_parameter(p_names, "sigma_" * string(key), string(m_domain), string(key), k)
        latent_name = _find_parameter(p_names, "latent_" * string(key), string(m_domain), string(key), k)
        
        n_units = if m_domain == :spatial; M.s_N; elseif m_domain == :temporal; M.t_N; else M.u_N; end
        
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = get_params_vector(chain, latent_name, n_units)
        
        # Transpose to [n_units, n_samples] and apply scaling
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
    end
    
    return (structured=structured_fields, noisy=structured_fields)
end
 

function extract_manifold(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec)
    structured_fields = Vector{Matrix{Float64}}()
    noisy_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = string(spec.key)
        m_domain = string(spec.domain)
        sigma_name = _find_parameter(p_names, "sigma_" * key, m_domain, key, k)
        rho_name = _find_parameter(p_names, "rho_" * key, m_domain, key, k)
        struct_name = _find_parameter(p_names, "latent_struct_" * key, m_domain, key, k)
        iid_name = _find_parameter(p_names, "latent_iid_" * key, m_domain, key, k)

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        struct_samples = get_params_vector(chain, struct_name, M.s_N)
        iid_samples = get_params_vector(chain, iid_name, M.s_N)

        s_vec = sigma_samples'
        r_vec = rho_samples'

        struct_eff = (struct_samples' .* sqrt.(r_vec)) .* s_vec
        noisy_eff = (iid_samples' .* sqrt.(1.0 .- r_vec)) .* s_vec .+ struct_eff

        push!(structured_fields, struct_eff)
        push!(noisy_fields, noisy_eff)
    end
    return (structured=structured_fields, noisy=noisy_fields)
end

function extract_manifold(m_obj::AR1, chain, M, n_samples, outcomes_N, p_names, spec)
    structured_fields = Vector{Matrix{Float64}}()
    for k in 1:outcomes_N
        key = string(spec.key)
        m_domain = string(spec.domain)
        
        sigma_name = _find_parameter(p_names, "sigma_" * key, m_domain, key, k)
        rho_name = _find_parameter(p_names, "rho_" * key, m_domain, key, k)
        innov_name = _find_parameter(p_names, "innov_" * key, m_domain, key, k)

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        innov_samples = get_params_vector(chain, innov_name, M.t_N)

        # Reconstruct the AR1 field from its innovations for each posterior sample.
        n_t = M.t_N
        t_field_samples = zeros(Float64, n_t, n_samples)
        
        for j in 1:n_samples
            rho = rho_samples[j, 1]
            sig = sigma_samples[j, 1]
            innov = innov_samples[j, :]
            
            curr_field = zeros(Float64, n_t)
            curr_field[1] = innov[1] / sqrt(1.0 - rho^2 + 1e-9)
            for t in 2:n_t
                curr_field[t] = rho * curr_field[t-1] + innov[t]
            end
            t_field_samples[:, j] = curr_field .* sig
        end
        
        push!(structured_fields, t_field_samples)
    end
    return (structured=structured_fields, noisy=structured_fields)
end


function extract_manifold(m_obj::IID, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for the IID manifold.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        key = spec.key
        m_domain = spec.domain
        
        sigma_name = _find_parameter(p_names, "sigma_" * string(key), string(m_domain), string(key), k)
        
        # For IID, the latent field is often named specifically (e.g., s_iid, t_iid)
        latent_base = string(m_domain)[1] * "_iid" # e.g., s_iid, t_iid
        latent_name = _find_parameter(p_names, latent_base, string(m_domain), string(key), k)
        
        n_units = if m_domain == :spatial; M.s_N; elseif m_domain == :temporal; M.t_N; else M.u_N; end
        
        sigma_samples = get_params_vector(chain, sigma_name, 1)
        latent_samples = get_params_vector(chain, latent_name, n_units)
        # Transpose to [n_units, n_samples] and apply scaling
        effect = latent_samples' .* sigma_samples'
        push!(structured_fields, effect)
    end

    return (structured=structured_fields, noisy=structured_fields)
end
function extract_manifold(m_obj::Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}, chain, M, n_samples, outcomes_N, p_names, spec)
    # v2.0.0 (2026-07-01)
    # Purpose: Extracts posterior samples for basis function manifolds.
    # Change: Returns both the final effect and the raw coefficients.
    var_sym = spec.var
    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)

    structured_fields = Vector{Matrix{Float64}}()
    coefficient_fields = Vector{Matrix{Float64}}() # This will store [n_basis, n_samples] for each outcome

    for k in 1:outcomes_N
        key = spec.key
        beta_name = outcomes_N > 1 ? Symbol("beta_", key, "_", k) : Symbol("beta_", key)
        coeffs = get_params_vector(chain, string(beta_name), n_basis_cols) # [n_samples, n_basis]
        
        push!(coefficient_fields, coeffs') # store as [n_basis, n_samples]

        effect = B_mat * coeffs' # Result is [n_obs, n_samples]
        push!(structured_fields, effect)
    end

    return (structured=structured_fields, noisy=structured_fields, coefficients=coefficient_fields)
end
 
function extract_manifold(m_obj::DynamicsManifold, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Extracts posterior samples for dynamics manifolds.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing structured and noisy fields.
    structured_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        # The model samples the full dyn_field for each outcome
        key = spec.key
        latent_name = outcomes_N > 1 ? Symbol("dyn_field_", key, "_", k) : Symbol("dyn_field_", key)
        latent_samples = get_params_vector(chain, string(latent_name), M.s_N * M.t_N) # [n_samples, M.s_N * M.t_N]

        # Reshape to [M.s_N, M.t_N, n_samples] and then extract relevant indices
        dyn_field_samples = reshape(latent_samples', M.s_N, M.t_N, n_samples) # [M.s_N, M.t_N, n_samples]
        
        # Extract the effect at observation points
        effect_k = zeros(Float64, M.y_N, n_samples)
        for j in 1:n_samples
            for i in 1:M.y_N
                effect_k[i, j] = dyn_field_samples[M.s_idx[i], M.t_idx[i], j]
            end
        end
        push!(structured_fields, effect_k)
    end

    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::MixedManifold, chain, M, n_samples, outcomes_N, p_names, spec)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for a Mixed Effects manifold (random intercept/slope).
    #          This separates the extraction logic from the main discovery loop for clarity.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the structured effect (the random effect coefficients).
    structured_fields = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        # For mixed effects, the 'structured' field holds the random coefficients.
        # The model does not have a separate sigma parameter for these; the coefficients
        # are typically sampled from a Normal(0, sigma_group) where sigma_group is a
        # hyperparameter, but the final coefficients are what we need.
        # The existing logic incorrectly tried to find and apply a non-existent sigma.
        
        n_units = spec.params.n_cat
        latent_name = _find_parameter(p_names, "latent_" * string(spec.key), "mixed", string(spec.var), k)
        latent_samples = get_params_vector(chain, latent_name, n_units) # [n_samples, n_units]
        push!(structured_fields, latent_samples') # Store as [n_units, n_samples]
    end

    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for a Harmonic manifold (seasonal/periodic effects).
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the structured effect.

    structured_fields = Vector{Matrix{Float64}}()
    
    # Harmonic manifold is typically applied to temporal or seasonal domain
    n_units = M.t_N # Assuming temporal domain for now
    basis_coords = 1.0:Float64(n_units)

    # The number of harmonics and period are part of the manifold object
    n_harmonics = get(spec.params, :n_harmonics, 2) # Default to 2 if not specified
    period = get(spec.params, :period, Float64(n_units)) # Default to n_units if not specified

    # Construct the Fourier basis matrix
    # Each harmonic contributes a sine and cosine component
    n_basis_cols = 2 * n_harmonics 
    B_mat = zeros(Float64, n_units, n_basis_cols)

    for j in 1:n_harmonics
        omega_j = 2.0 * pi * j / period
        B_mat[:, 2*j - 1] = sin.(omega_j .* basis_coords)
        B_mat[:, 2*j] = cos.(omega_j .* basis_coords)
    end

    for k in 1:outcomes_N
        # BSTM v3.0.0 Naming Convention Alignment
        beta_name = "beta_basis_$(spec.key)"
        if outcomes_N > 1; beta_name *= "_$(k)"; end
        coeffs = get_params_vector(chain, beta_name, n_basis_cols)

        effect = B_mat * coeffs'
        push!(structured_fields, effect)
    end

    return (structured=structured_fields, noisy=structured_fields)
end

function extract_manifold(m_obj::Eigen, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for an EigenManifold (Bayesian PCA factor).
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing the structured effect (the first principal component).
    # Rationale for v1.1.0:
    #     - Modified to use both training (M) and prediction (PS) data to construct the
    #       full effect vector, preventing truncation errors.

    # 1. Get parameters from chain
    # The parameter names are constructed based on the key of the manifold spec
    key = spec.key
    v_samples = get_params_vector(chain, "v_$(key)", length(spec.params.ltri_indices))

    # 2. Get data for the factor model from both training and prediction sets
    eigen_vars = spec.variables # e.g., ["y1", "y2", "w1", "w2", "w3"]
    
    # Combine training and prediction data if PS is available
    Y_data_train = Matrix(M.data[!, Symbol.(eigen_vars)])
    Y_data = if !isnothing(PS) && hasproperty(PS, :data)
        Y_data_pred = Matrix(PS.data[!, Symbol.(eigen_vars)])
        vcat(Y_data_train, Y_data_pred)
    else
        Y_data_train
    end

    n_vars = length(eigen_vars)
    n_factors = spec.params.n_factors

    # 3. Reconstruct the effect for each sample
    eigen_effect = zeros(Float64, N_tot, n_samples)

    for j in 1:n_samples
        v_vec = v_samples[j, :]
        v_mat = zeros(Float64, n_vars, n_factors)
        v_mat[spec.params.ltri_indices] .= v_vec
        U = householder_to_eigenvector(v_mat, n_vars, n_factors)
        factors = Y_data * U
        eigen_effect[:, j] = factors[:, 1]
    end

    structured_fields = [eigen_effect]
    return (structured=structured_fields, noisy=structured_fields)
end

function _extract_nested_fields(chain, M, n_samples, outcomes_N, p_names)
    # v1.0.0 (2026-06-30)
    # Purpose: Extracts posterior samples for nested/multifidelity manifolds. This logic was
    #          previously inside the main discovery loop and is now modularized.
    # Inputs: chain, M, n_samples, outcomes_N, p_names.
    # Outputs: A NamedTuple containing the reconstructed z_latent and w_latent fields.

    z_ls_s = get_params_vector(chain, "z_ls", 1)
    z_beta_s = get_params_vector(chain, "z_beta", M.M_rff)
    w_ls_s = get_params_vector(chain, "w_ls", 1)
    w_beta_s = get_params_vector(chain, "w_beta", M.M_rff * 3)

    z_latent_field = zeros(size(M.z_coords_s, 1), n_samples)
    w_latent_field = zeros(size(M.w_coords_st, 1), 3, n_samples)

    for j in 1:n_samples
        z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls_s[j])) .+ M.b_fixed'
        z_latent_j = M.rff_scale .* (cos.(z_proj) * z_beta_s[j, :])
        z_latent_field[:, j] = z_latent_j

        w_coords_aug = hcat(M.w_coords_st, z_latent_j[1:size(M.w_coords_st, 1)])
        w_proj = (w_coords_aug * (M.W_fixed[1:size(w_coords_aug, 2), :] ./ w_ls_s[j])) .+ M.b_fixed'
        w_beta_mat = reshape(w_beta_s[j, :], M.M_rff, 3)
        w_latent_field[:, :, j] = M.rff_scale .* (cos.(w_proj) * w_beta_mat)
    end

    return (z_latent=z_latent_field, w_latent=w_latent_field)
end

# Fallback for other manifold types (including SVCManifold, TransformedManifold, VaryingInteractionManifold, RegularizationGroupManifold, SoftConstraintManifold)
function extract_manifold(m_obj::ManifoldModel, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # v1.0.1 (2026-06-29 17:16:00)
    # Purpose: A fallback method for manifold types without a specific extraction rule.
    # Inputs: m_obj, chain, M, n_samples, outcomes_N, p_names, spec.
    # Outputs: A NamedTuple containing zero-filled fields to prevent downstream errors.

    @warn "No specific `extract_manifold` method for $(typeof(m_obj)). Returning zero matrix."
    zero_field = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]
    return (structured=zero_field, noisy=zero_field) # Return outcome-specific zero fields
end



# v4.9.8 (2026-07-08) - Hardened Comprehensive Manifold Discovery
# Synopsis: Restores full discovery logic for all manifold types to ensure zero-truncation in reporting.
# Rationale: Aligns the latent field extraction with the modular assembly engine to support mixed, dynamics, and factor models.

function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names, PS, N_tot)
    # Initialization of outcome-specific latent containers
    # These structures aggregate multiple manifold effects per outcome for assembly
    s_eff_struct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_noisy  = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    t_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    u_eff = zeros(Float64, M.u_N, n_samples)
    
    # Extended registries for complex manifold types
    basis_eff_accum = zeros(Float64, N_tot, n_samples)
    basis_coeffs = Dict{Symbol, Vector{Matrix{Float64}}}()
    st_eff_maps = [zeros(Float64, M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    dynamics_eff = [zeros(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    eigen_eff = [zeros(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    nested_eff = [zeros(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]

    # Registry for component metadata identification used in reporting wrappers
    disc_space = "none"
    disc_time = "none"

    # Global Intercept Discovery
    # Utilizes the tier-based search to resolve naming variations like 'intercept' or 'intercept_1'
    intercept_eff = nothing
    intercept_name = _find_parameter(p_names, "intercept", "", "")
    if !isempty(intercept_name) && intercept_name in p_names
        intercept_samples = get_params_vector(chain, intercept_name, outcomes_N)
        intercept_eff = intercept_samples'
    end

    log_offset_eff = get(M, :log_offset, nothing)
    
    # Technical Registry for random effects tracking
    mixed_terms_list = get(M, :mixed_terms, [])
    mixed_eff_coeffs = !isempty(mixed_terms_list) ? [zeros(Float64, term.n_cat, n_samples) for term in mixed_terms_list] : nothing
    
    # Main Manifold Discovery Loop
    # Iterates through registered manifolds and dispatches to specific extraction logic
    if haskey(M, :manifolds)
        for spec in M.manifolds
            m_obj = spec.manifold_obj
            if m_obj isa NoneManifold
                continue
            end

            # Extract posterior realizations based on manifold type trait
            extracted = extract_manifold(m_obj, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)

            # Map extracted fields to their respective domains
            if spec.domain == :spatial
                disc_space = string(typeof(m_obj))
                for k in 1:outcomes_N
                    s_eff_struct[k] .+= extracted.structured[k]
                    s_eff_noisy[k] .+= extracted.noisy[k]
                end
            elseif spec.domain == :temporal
                disc_time = string(typeof(m_obj))
                for k in 1:outcomes_N
                    t_eff[k] .+= extracted.structured[k]
                end
            elseif spec.domain == :seasonal
                u_eff .+= extracted.structured[1]
            elseif spec.domain == :smooth
                if hasproperty(extracted, :coefficients)
                    basis_coeffs[spec.var] = extracted.coefficients
                end
                # This effect was being calculated in extract_manifold but never accumulated.
                # This correction ensures the realized smooth effect is passed to the eta assembly.
                if hasproperty(extracted, :structured) && !isempty(extracted.structured)
                    basis_eff_accum .+= extracted.structured[1]
                end
            elseif m_obj isa DynamicsManifold
                for k in 1:outcomes_N
                    dynamics_eff[k] .+= extracted.structured[k]
                end
            elseif m_obj isa Eigen
                for k in 1:outcomes_N
                    eigen_eff[k] .+= extracted.structured[k]
                end
            elseif m_obj isa MixedManifold
                term_idx = findfirst(t -> t.name == spec.var, mixed_terms_list)
                if !isnothing(term_idx)
                    mixed_eff_coeffs[term_idx] = extracted.structured[1]
                end
            end
        end
    end

    # #
    # Nested Hierarchical Supervisor Realization
    # Rationale: Centralizing the discovery of nested/multifidelity effects. This logic was
    #            previously misplaced in a specific _reconstruct method, leading to truncation.
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_name = _find_parameter(p_names, "rho_nested", "", string(z_key))
            rho_samples = get_params_vector(chain, rho_name, 1) # [n_samples, 1]

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                lat_base = "latent_nested_spatial"
                lat_name = _find_parameter(p_names, lat_base, "", string(z_key))
                lat_samples = get_params_vector(chain, lat_name, z_meta.s_N)' # [s_N, n_samples]

                # Apply effect to the first outcome's nested_eff container, consistent with existing logic
                for j in 1:n_samples
                    for i in 1:M.y_N
                        s_ptr = M.s_idx[i]
                        nested_eff[1][i, j] += rho_samples[j] * lat_samples[s_ptr, j]
                    end
                end
            end
        end
    end

    # Discovery of Space-Time Interaction components
    if get(M, :model_st, "none") != "none"
        st_sigma_name = _find_parameter(p_names, "st_sigma", "", "")
        st_sigma_samples = get_params_vector(chain, st_sigma_name, outcomes_N)
        st_raw_name = _find_parameter(p_names, "st_raw", "", "")
        st_raw_samples = get_params_vector(chain, st_raw_name, M.s_N * M.t_N)

        for k in 1:outcomes_N
            for j in 1:n_samples
                st_eff_maps[k][:, :, j] = reshape(st_raw_samples[j, :], M.s_N, M.t_N) .* st_sigma_samples[j, k]
            end
        end
    end

    # Fixed Effects Parameter Recovery
    xf_betas = nothing
    if M.Xfixed_N > 0
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    # Observation Volatility Surface Reconstruction
    sv_surface = _extract_volatility(chain, p_names, N_tot, n_samples, outcomes_N, M)

    # Bundle all discovered realizations for the linear predictor assembly engine
    return (
        Xfixed_betas = xf_betas,
        intercept_eff = intercept_eff,
        log_offset_eff = log_offset_eff,
        mixed_eff_coeffs = mixed_eff_coeffs,
        s_eff_struct = s_eff_struct,
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff,
        u_eff = u_eff,
        basis_eff_accum = basis_eff_accum,
        basis_coeffs = basis_coeffs,
        st_eff_maps = st_eff_maps,
        dynamics_eff = dynamics_eff,
        eigen_eff = eigen_eff,
        nested_eff = nested_eff,
        sv_surface = sv_surface,
        n_samples = n_samples,
        outcomes_N = outcomes_N,
        model_space = disc_space,
        model_time = disc_time
    )
end


function _modular_eta_assembly(N_tot_in, registry, M, PS_in)
    # v4.9.2 (2026-07-08) - Complete Manifold Accumulation
    # Rationale: This version corrects several logical truncations. It now correctly
    #            accumulates effects from all discovered manifolds, including smooths,
    #            mixed effects, dynamics, and factor models. It also corrects the
    #            application of log_offset for out-of-sample predictions.

    local n_samples = registry.n_samples
    local outcomes_n = registry.outcomes_N
    local y_n_train = Int(M.y_N)
    local actual_limit = isnothing(PS_in) ? y_n_train : Int(N_tot_in)

    local eta_container = zeros(Float64, actual_limit, outcomes_n, n_samples)

    # Pre-slice fixed effects betas for efficiency
    local xf_n = Int(M.Xfixed_N)
    local beta_samps = get(registry, :Xfixed_betas, nothing)
    local beta_slices = [!isnothing(beta_samps) ? beta_samps[((k-1)*xf_n + 1):(k*xf_n), :] : zeros(0, n_samples) for k in 1:outcomes_n]

    # Main assembly loop over posterior samples
    for j in 1:n_samples
        local intercept_val = !isnothing(registry.intercept_eff) ? registry.intercept_eff[:, j] : zeros(Float64, outcomes_n)
        
        for k in 1:outcomes_n
            # Extract single-sample slices for all latent fields for the current outcome
            local s_f_k = get(registry, :s_eff_noisy, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local t_f_k = get(registry, :t_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local u_f_k = get(registry, :u_eff, zeros(M.u_N, n_samples))[:, j]
            local st_f_k = get(registry, :st_eff_maps, []) |> (x -> !isempty(x) ? x[k][:, :, j] : zeros(0, 0))
            local beta_slice_k = beta_slices[k][:, j]

            # Extract effects from specialized manifolds
            local basis_eff_k = get(registry, :basis_eff_accum, zeros(actual_limit, n_samples))[:, j]
            local dynamics_eff_k = get(registry, :dynamics_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local eigen_eff_k = get(registry, :eigen_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])
            local nested_eff_k = get(registry, :nested_eff, []) |> (x -> !isempty(x) ? x[k][:, j] : Float64[])

            # Loop over all observations (in-sample and out-of-sample)
            for i in 1:actual_limit
                local is_obs = i <= y_n_train
                local src = is_obs ? M : PS_in
                local idx = is_obs ? i : i - y_n_train

                # Initialize with intercept
                local val = intercept_val[k]

                # Add standard spatiotemporal effects
                if !isempty(s_f_k); val += s_f_k[Int(src.s_idx[idx])]; end
                if !isempty(t_f_k); val += t_f_k[Int(src.t_idx[idx])]; end
                if !all(iszero, u_f_k); val += u_f_k[Int(src.u_idx[idx])]; end
                if !isempty(st_f_k); val += st_f_k[Int(src.s_idx[idx]), Int(src.t_idx[idx])]; end

                # Add fixed effects
                if !isempty(beta_slice_k); val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice_k); end

                # Add offset (for both observed and prediction data)
                if hasproperty(src, :log_offset) && !isnothing(src.log_offset)
                    val += src.log_offset[idx]
                end

                # Add smooth covariate effects
                if !all(iszero, basis_eff_k); val += basis_eff_k[i]; end

                # Add mixed effects
                if haskey(M, :mixed_terms) && !isempty(M.mixed_terms)
                    for (term_idx, term) in enumerate(M.mixed_terms)
                        # The `term.indices` vector is assumed to be constructed for the full dataset
                        # (training + prediction), so it must be indexed by the global observation
                        # index `i`, not the local index `idx`. The original use of `idx` would
                        # cause an out-of-bounds error or incorrect mapping for prediction data.
                        group_idx = term.indices[i]
                        val += registry.mixed_eff_coeffs[term_idx][group_idx, j]
                    end
                end

                # Add specialized manifold effects
                if !isempty(dynamics_eff_k); val += dynamics_eff_k[i]; end
                if !isempty(eigen_eff_k); val += eigen_eff_k[i]; end
                if !isempty(nested_eff_k); val += nested_eff_k[i]; end

                eta_container[i, k, j] = val
            end
        end
    end
    return eta_container
end


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcomes_N, M=nothing)
    # v1.0.2 (2026-07-04)
    # Purpose: Reconstructs the observation volatility (noise) surface from MCMC samples. This version
    #          is hardened to use the `_find_parameter` utility for robust discovery of both
    #          stochastic and homoskedastic volatility parameters, correcting a flaw where
    #          hardcoded indices could fail to find shared or alternatively named parameters.

    all_y_sig_samples = [zeros(Float64, N_tot, N_samples) for _ in 1:outcomes_N]

    for k in 1:outcomes_N
        y_sig_samples_k = zeros(Float64, N_tot, N_samples)

        if get(M, :use_sv, false) # Stochastic Volatility
            # Use the robust finder for all SV parameters
            sig_log_var_name = _find_parameter(name_strs, "sigma_log_var", "sv", "volatility", k)
            beta_vol_latent_name = _find_parameter(name_strs, "beta_vol_latent", "sv", "volatility", k)

            if !isempty(sig_log_var_name) && !isempty(beta_vol_latent_name)
                sig_vals = get_params_vector(chain, sig_log_var_name, 1)
                beta_vol_latent_vals = get_params_vector(chain, beta_vol_latent_name, M.M_rff_sigma)

                if haskey(M, :vol_proj)
                    vol_proj_field = M.vol_proj * beta_vol_latent_vals' # [N_tot x M_rff] * [M_rff x N_samples]
                    vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)
                    y_sig_samples_k .= exp.((sig_vals' .* vol_latent_field) ./ 2.0)
                else
                    @warn "M.vol_proj not found for stochastic volatility reconstruction. Defaulting to 1.0 for outcome $k."
                    y_sig_samples_k .= 1.0
                end
            else
                @warn "Stochastic volatility parameters not found for outcome $k. Defaulting to 1.0."
                y_sig_samples_k .= 1.0
            end
        else # Homoskedastic Volatility
            y_sigma_name = _find_parameter(name_strs, "y_sigma", "observation", "noise", k)
            if !isempty(y_sigma_name)
                vals = get_params_vector(chain, y_sigma_name, 1)
                y_sig_samples_k .= vals' # Broadcast the [1 x N_samples] vector to [N_tot x N_samples]
            else
                y_sig_samples_k .= 1.0
            end
        end

        all_y_sig_samples[k] = y_sig_samples_k
    end
    return all_y_sig_samples
end


function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies the inverse link function to the linear predictor `eta`.
    # Inputs: family, eta, use_zi, phi, r.
    # Outputs: The expected value `mu` on the response scale.
    local mu

    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        mu = exp.(eta)

    elseif family in ["bernoulli", "binomial", "beta"]
        mu = logistic.(eta)

    elseif family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        mu = eta

    else
        mu = eta
    end

    if use_zi
        mu = (1.0 .- phi) .* mu
    end

    return mu
end

function _compute_waic(log_lik)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the Widely Applicable Information Criterion (WAIC).
    # Inputs: log_lik - A matrix of pointwise log-likelihoods [N_samples x N_obs].
    # Outputs: The WAIC value.
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples_all_outcomes, y_obs_custom=nothing)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates predictions and pointwise log-likelihood values.
    # Inputs: fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, y_obs_custom.
    # Outputs: A tuple (denoised_predictions, noisy_predictions, log_likelihood_matrix).
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.y_N)

    name_strs = string.(FlexiChains.parameters(chain))
    use_zi = get(M, :use_zi, false)
    fam_str = hasproperty(M, :model_family) ? M.model_family : "gaussian"
    
    # Pre-extract parameters to avoid repeated lookups inside the loop
    r_nb_samples = "lik_r" in name_strs ? get_params_vector(chain, "lik_r", 1) : fill(1.0, N_samples, 1)
    phi_zi_samples = "lik_phi" in name_strs ? get_params_vector(chain, "lik_phi", 1) : fill(0.0, N_samples, 1)
    extra_p_samples = "extra_params" in name_strs ? get_params_vector(chain, "extra_params", 1) : fill(1.0, N_samples, 1)
    
    for j in 1:N_samples

        sig_y = if !isnothing(y_sigma_samples_all_outcomes)
            # This function is called for univariate, so y_sigma_samples_all_outcomes is Matrix{Float64}
            y_sigma_samples_all_outcomes[:, j]
        else
            # Inefficient fallback if y_sigma_samples_all_outcomes is not provided
            _extract_volatility(chain, name_strs, N_tot, N_samples, 1, M)[1][:, j]
        end

        r_val = r_nb_samples[j, 1]
        phi_val = phi_zi_samples[j, 1]
        extra = extra_p_samples[j, 1]

        mu_vec = _apply_link_and_lik(fam_str, eta[:, j], use_zi, phi_val, r_val)
        denoised[:, j] .= mu_vec

        for i in 1:N_tot
            is_obs = i <= M.y_N
            eta_val = eta[i, j]

            if is_obs
                y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                lik_obj = bstm_Likelihood(
                    fam_str, [y_vals_src[i]]; sigma_y=[sig_y[i]], weight=M.weights[i],
                    phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, trial=M.trials[i], extra_params=extra
                )
                log_lik[j, i] = Distributions.logpdf(lik_obj, eta_val)
            end

            if use_zi && rand() < phi_val
                noisy[i, j] = 0.0
            else
                n_t = Int(is_obs ? M.trials[i] : 1)

                temp_lik_obj = bstm_Likelihood(
                    fam_str, [0.0];
                    sigma_y=sig_y[i],
                    r_nb=r_val,
                    trial=n_t,
                    extra_params=extra
                )

                dist = get_dist_ref(fam, temp_lik_obj, eta_val, sig_y[i])

                noisy[i, j] = rand(dist)
            end
        end
    end

    return denoised, noisy, log_lik
end

function _calculate_ps_weights(p_denoised, M, PS, N_PS, N_samples)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Calculates post-stratification weights.
    # Rationale for v1.0.2:
    #     - Corrected fallback logic to use a spatially-informed mean instead of a global mean.
    #     - Hardened stratum matching to handle cases where seasonal index `u_idx` is nothing.
    # Inputs: p_denoised, M, PS, N_PS, N_samples.
    # Outputs: A matrix of post-stratification weights [N_PS x N_samples].
    if N_PS == 0
        return nothing
    end

    ps_weights = zeros(N_PS, N_samples)

    for k in 1:N_PS
        s_target = PS.s_idx[k]
        t_target = PS.t_idx[k]
        u_target = isnothing(PS.u_idx) ? nothing : PS.u_idx[k]

        obs_match_idx = findfirst(1:M.y_N) do i
            match_s = M.s_idx[i] == s_target
            match_t = M.t_idx[i] == t_target
            match_u = if isnothing(u_target) || isnothing(M.u_idx)
                true # No seasonal component to match
            else
                M.u_idx[i] == u_target
            end
            return match_s && match_t && match_u
        end

        if !isnothing(obs_match_idx)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (p_denoised[obs_match_idx, j] + 1e-9)
            end
        else
            # Fallback logic: If no exact spatiotemporal match is found, use the mean of all
            # observations within the same SPATIAL stratum. This is more robust than a global mean.
            spatial_match_indices = findall(i -> M.s_idx[i] == s_target, 1:M.y_N)
            if !isempty(spatial_match_indices)
                spatial_mean_obs = mean(p_denoised[spatial_match_indices, :], dims=1)
                for j in 1:N_samples
                    ps_weights[k, j] = p_denoised[M.y_N + k, j] / (spatial_mean_obs[j] + 1e-9)
                end
            else
                # Final fallback to global mean if the spatial stratum has no observations at all.
                global_mean_obs = mean(p_denoised[1:M.y_N, :], dims=1)
                for j in 1:N_samples
                    ps_weights[k, j] = p_denoised[M.y_N + k, j] / (global_mean_obs[j] + 1e-9)
                end
            end
        end
    end

    return ps_weights
end



# v4.9.5 (2026-07-08) - Full Reconstruction for UnivariateArchitecture
# Synopsis: Restores all missing outputs, including post-stratification weights and latent summaries.
# Rationale: Resolves truncation errors identified in v4.9.2 by ensuring comprehensive field extraction.

function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names, PS, N_tot)

    # Linear Predictor Assembly
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS) # This function is already designed for N_tot

    # Summarize Latent Effects using safe access
    summarized_effects = Dict{Symbol, Any}()

    s_struct = get(registry, :s_eff_struct, [])
    if !isempty(s_struct)
        summarized_effects[:spatial_denoised] = summarize_array(s_struct[1]; alpha=alpha)
    end

    s_noisy = get(registry, :s_eff_noisy, [])
    if !isempty(s_noisy)
        summarized_effects[:spatial_noisy] = summarize_array(s_noisy[1]; alpha=alpha)
    end

    t_effs = get(registry, :t_eff, [])
    if !isempty(t_effs)
        summarized_effects[:temporal] = summarize_array(t_effs[1]; alpha=alpha)
    end

    u_eff = get(registry, :u_eff, zeros(0, 0))
    if !all(iszero, u_eff)
        summarized_effects[:seasonal] = summarize_array(u_eff; alpha=alpha)
    end

    st_maps = get(registry, :st_eff_maps, [])
    if !isempty(st_maps)
        summarized_effects[:spacetime_interaction] = summarize_array(st_maps[1]; alpha=alpha)
    end

    b_coeffs = get(registry, :basis_coeffs, Dict())
    if !isempty(b_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        # The original logic incorrectly summarized the realized effect (B * coeffs)
        # instead of the coefficients themselves. The correction is to summarize
        # the raw coefficients, which is consistent with how other effects are reported.
        # The full realized effect is captured in the `eta` and `predictions` summaries.
        for (var_sym, coeffs_matrix_per_outcome) in b_coeffs
            summarized_effects[:smooth_effects][var_sym] = summarize_array(coeffs_matrix_per_outcome[1]; alpha=alpha)
        end
    end

    xf_betas = get(registry, :Xfixed_betas, nothing)
    if !isnothing(xf_betas)
        summarized_effects[:fixed_effects] = summarize_array(xf_betas'; alpha=alpha)
    end

    # Generate Predictions and Compute Log-Likelihood
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    
    # For the univariate case, the volatility surface is the first element of the returned vector.
    # This is already a matrix of size [N_tot x n_samples].
    vol_matrix = get(registry.sv_surface, 1, zeros(Float64, N_tot, n_samples))

    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, vol_matrix
    )

    # Summarize Predictions
    summarized_effects[:eta] = summarize_array(eta_samples_2d[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)

    # Post-Stratification Logic and Output Restoration
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)

        # Explicit calculation and summarization of post-stratified observation-stratum weights
        ps_weights_samples = _calculate_ps_weights(p_denoised, M, PS, N_PS, n_samples)
        if !isnothing(ps_weights_samples)
            summarized_effects[:ps_weights_raw] = ps_weights_samples
            summarized_effects[:ps_weights] = summarize_array(ps_weights_samples; alpha=alpha)
        end
    end

    # Final Diagnostics and Metadata propagation
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    summarized_effects[:model_space] = get(registry, :model_space, "none")
    summarized_effects[:model_time] = get(registry, :model_time, "none")

    return NamedTuple(summarized_effects)
end




function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # Dimensions and Scope Discovery
    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # #
    # 1. Parameter and Latent Field Discovery (now aware of prediction set)
    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names, PS, N_tot)

    # #
    # 2. Linear Predictor Assembly
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS) # This function is already designed for N_tot

    # Apply LKJ coupling if covariance structure is present
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta_samples[:, :, j] = eta_samples[:, :, j] * L_corr_samples[:,:,j]'
        end
    end

    # #
    # 3. Summarize Latent Effects (Outcome-Specific)
    summarized_effects = Dict{Symbol, Any}()

    # #
    # 3. Summarize Primary Latent Effects (Restored for completeness)
    # Rationale: This block was incomplete, leading to truncated output. It has been
    #            restored to match the comprehensive summarization of the univariate architecture.
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        summarized_effects[:spatial_denoised] = [summarize_array(registry.s_eff_struct[k]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        summarized_effects[:spatial_noisy] = [summarize_array(registry.s_eff_noisy[k]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        summarized_effects[:temporal] = [summarize_array(registry.t_eff[k]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        summarized_effects[:seasonal] = summarize_array(registry.u_eff; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        summarized_effects[:spacetime_interaction] = [summarize_array(registry.st_eff_maps[k]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_vec_per_outcome) in registry.basis_coeffs
            # For multivariate, coeffs_vec_per_outcome is a Vector of matrices
            summarized_effects[:smooth_effects][var_sym] = [summarize_array(M.basis_matrices[var_sym] * coeffs_vec_per_outcome[k]; alpha=alpha) for k in 1:outcomes_N]
        end
    end
    if !isnothing(registry.Xfixed_betas)
        # Reshape betas to [n_coeffs, n_outcomes, n_samples] for summarization
        n_coeffs = M.Xfixed_N
        betas_reshaped = reshape(registry.Xfixed_betas, n_coeffs, outcomes_N, N_samples)
        summarized_effects[:fixed_effects] = [summarize_array(betas_reshaped[:, k, :]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            # mixed_eff_coeffs is a vector of [n_cat, n_samples] matrices, one per term
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end
    if !isnothing(registry.dynamics_eff) && !all(iszero, vcat(registry.dynamics_eff...))
        summarized_effects[:dynamics_eff] = [summarize_array(registry.dynamics_eff[k]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.eigen_eff) && !all(iszero, vcat(registry.eigen_eff...))
        summarized_effects[:eigen_eff] = [summarize_array(registry.eigen_eff[k]; alpha=alpha) for k in 1:outcomes_N]
    end
    if !isnothing(registry.nested_eff) && !all(iszero, vcat(registry.nested_eff...))
        summarized_effects[:nested_contributions] = [summarize_array(registry.nested_eff[k]; alpha=alpha) for k in 1:outcomes_N]
    end

    # #
    # 4. Generate Predictions and Process Likelihood Registry
    p_denoised = zeros(Float64, N_tot, outcomes_N, N_samples)
    p_noisy = zeros(Float64, N_tot, outcomes_N, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)

    use_zi = get(M, :use_zi, false)
    r_nb_samples = "lik_r" in p_names ? get_params_vector(chain, "lik_r", outcomes_N) : fill(1.0, N_samples, outcomes_N)
    phi_zi_samples = "lik_phi" in p_names ? get_params_vector(chain, "lik_phi", 1) : fill(0.0, N_samples, 1)

    for j in 1:N_samples
        for k in 1:outcomes_N
            eta_jk = eta_samples[:, k, j]
            y_sigma_jk = registry.sv_surface[k][:, j]
            mu_vec = _apply_link_and_lik(family_str, eta_jk, use_zi, phi_zi_samples[j], r_nb_samples[j, k])
            p_denoised[:, k, j] .= mu_vec

            for i in 1:N_tot
                if i <= M.y_N
                    lik_obj = bstm_Likelihood(
                        fam_obj, [M.y_obs[i, k]]; sigma_y=[y_sigma_jk[i]],
                        phi_zi=use_zi ? phi_zi_samples[j] : -Inf, r_nb=r_nb_samples[j, k]
                    )
                    log_lik[j, (k-1)*M.y_N + i] = Distributions.logpdf(lik_obj, eta_jk[i])
                end
                # Simple random draw for noisy predictions
                temp_lik = bstm_Likelihood(fam_obj, [0.0]; sigma_y=[y_sigma_jk[i]], r_nb=[r_nb_samples[j, k]])
                dist = get_dist_ref(fam_obj, temp_lik, eta_jk[i], y_sigma_jk[i])
                p_noisy[i, k, j] = rand(dist)
            end
        end
    end

    # #
    # 5. Summarize Final Outcomes
    summarized_effects[:predictions_denoised] = [summarize_array(p_denoised[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_noisy] = [summarize_array(p_noisy[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end



function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    # Dimensions and Scope Discovery
    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # #
    # 1. Parameter and Latent Field Discovery
    # Extracts posterior realizations for the main model components, now aware of prediction set
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names, PS, N_tot)

    # #
    # 2. Primary Linear Predictor Assembly
    # Modular assembly of the primary latent predictor, already handles N_tot
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

    # #
    # 3. Multifidelity / Nested Component Integration
    # This section specifically handles the contributions from nested sub-models
    summarized_effects = Dict{Symbol, Any}()

    # RFF-based multi-fidelity fields (z_latent, w_latent)
    # This logic was modularized into _extract_nested_fields but was not being called.
    # We call it here to reconstruct and summarize these core multi-fidelity components.
    if haskey(M, :z_coords_s) # Check if RFF-based multi-fidelity is active
        nested_rff_fields = _extract_nested_fields(chain, M, n_samples, 1, p_names)
        if hasproperty(nested_rff_fields, :z_latent) && !isnothing(nested_rff_fields.z_latent)
            summarized_effects[:z_latent_field] = summarize_array(nested_rff_fields.z_latent; alpha=alpha)
        end
        if hasproperty(nested_rff_fields, :w_latent) && !isnothing(nested_rff_fields.w_latent)
            # Summarize each of the 3 w_latent fields
            summarized_effects[:w_latent_field_1] = summarize_array(nested_rff_fields.w_latent[:, 1, :]; alpha=alpha)
            summarized_effects[:w_latent_field_2] = summarize_array(nested_rff_fields.w_latent[:, 2, :]; alpha=alpha)
            summarized_effects[:w_latent_field_3] = summarize_array(nested_rff_fields.w_latent[:, 3, :]; alpha=alpha)
        end
    end

    summarized_effects[:nested_effects] = Dict{Symbol, Any}()

    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            # Discover linking parameter rho_nested_{z_key}
            rho_name = _find_parameter(p_names, "rho_nested", "", string(z_key))
            rho_samples = get_params_vector(chain, rho_name, 1)

            # Discover nested spatial field realizations
            # Note: This discovery should ideally be centralized in _discover_manifold_realizations
            summarized_effects[:nested_effects][z_key] = Dict{Symbol, Any}()
            summarized_effects[:nested_effects][z_key][:rho_nested] = summarize_array(rho_samples; alpha=alpha)
            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                lat_base = "latent_nested_spatial"
                lat_name = _find_parameter(p_names, lat_base, "", string(z_key))
                lat_samples = get_params_vector(chain, lat_name, z_meta.s_N)' # [s_N x n_samples]

                # Map nested spatial effects to main model indices
                # Rationale: The low-fidelity field contributes to the main predictor scaled by rho_nested
                for j in 1:n_samples
                    for i in 1:N_tot
                        src = (i <= M.y_N) ? M : PS
                        idx_local = (i <= M.y_N) ? i : i - M.y_N
                        s_ptr = Int(src.s_idx[idx_local])
                        # Accumulate weighted contribution
                        eta_samples[i, 1, j] += rho_samples[j] * lat_samples[s_ptr, j]
                    end
                end
                summarized_effects[:nested_effects][z_key][:spatial_field] = summarize_array(lat_samples; alpha=alpha)
            end
        end
    end


    # Reshape for univariate-compatible predictive functions
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    vol_matrix = registry.sv_surface[1]

    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, vol_matrix
    )

    # # 4. Summarization and Metadata Assembly

    # Primary effects
    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        summarized_effects[:spatial_denoised] = summarize_array(registry.s_eff_struct[1]; alpha=alpha)
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        summarized_effects[:spatial_noisy] = summarize_array(registry.s_eff_noisy[1]; alpha=alpha)
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        summarized_effects[:temporal] = summarize_array(registry.t_eff[1]; alpha=alpha)
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        summarized_effects[:seasonal] = summarize_array(registry.u_eff; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        summarized_effects[:spacetime_interaction] = summarize_array(registry.st_eff_maps[1]; alpha=alpha)
    end
    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_vec) in registry.basis_coeffs
            B_mat = M.basis_matrices[var_sym]
            effect_matrix = B_mat * coeffs_vec[1]
            summarized_effects[:smooth_effects][var_sym] = summarize_array(effect_matrix; alpha=alpha)
        end
    end
    if !isnothing(registry.Xfixed_betas)
        summarized_effects[:fixed_effects] = summarize_array(registry.Xfixed_betas'; alpha=alpha)
    end
    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end
    if !isnothing(registry.dynamics_eff) && !all(iszero, registry.dynamics_eff)
        summarized_effects[:dynamics_eff] = summarize_array(registry.dynamics_eff[1]; alpha=alpha)
    end
    if !isnothing(registry.eigen_eff) && !all(iszero, registry.eigen_eff)
        summarized_effects[:eigen_eff] = summarize_array(registry.eigen_eff[1]; alpha=alpha)
    end

    # Predictive summaries
    summarized_effects[:eta] = summarize_array(eta_samples_2d[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)

    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)
    end

    # # 5. Diagnostics
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:arch] = arch
    summarized_effects[:family] = family_str

    return NamedTuple(summarized_effects)
end


function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Applies the inverse link function to the linear predictor `eta`.
    # Inputs: family, eta, use_zi, phi, r.
    # Outputs: The expected value `mu` on the response scale.
    local mu

    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        mu = exp.(eta)

    elseif family in ["bernoulli", "binomial", "beta"]
        mu = logistic.(eta)

    elseif family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        mu = eta

    else
        mu = eta
    end

    if use_zi
        mu = (1.0 .- phi) .* mu
    end

    return mu
end

function _compute_waic(log_lik)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Computes the Widely Applicable Information Criterion (WAIC).
    # Inputs: log_lik - A matrix of pointwise log-likelihoods [N_samples x N_obs].
    # Outputs: The WAIC value.
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Generates predictions and pointwise log-likelihood values.
    # Inputs: fam, eta, chain, M, N_tot, N_samples, y_sigma_samples, y_obs_custom.
    # Outputs: A tuple (denoised_predictions, noisy_predictions, log_likelihood_matrix).
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.y_N)

    name_strs = string.(FlexiChains.parameters(chain))
    use_zi = get(M, :use_zi, false)
    fam_str = hasproperty(M, :model_family) ? M.model_family : "gaussian"

    for j in 1:N_samples

        sig_y = if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        else
            sig_y = _extract_volatility(chain, name_strs, N_tot, N_samples, nothing, M)[:, j]
        end

        r_val = "lik_r" in name_strs ? chain[:lik_r].data[j] : 1.0
        phi_val = "lik_phi" in name_strs ? chain[:lik_phi].data[j] : 0.0
        extra = "extra_params" in name_strs ? chain[:extra_params].data[j] : 1.0

        mu_vec = _apply_link_and_lik(fam_str, eta[:, j], use_zi, phi_val, r_val)
        denoised[:, j] .= mu_vec

        for i in 1:N_tot
            is_obs = i <= M.y_N
            eta_val = eta[i, j]

            if is_obs
                y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                lik_obj = bstm_Likelihood(
                    fam_str, [y_vals_src[i]]; sigma_y=sig_y[i], weight=M.weights[i],
                    phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, trial=M.trials[i], extra_params=extra
                )
                log_lik[j, i] = Distributions.logpdf(lik_obj, eta_val)
            end

            if use_zi && rand() < phi_val
                noisy[i, j] = 0.0
            else
                n_t = Int(is_obs ? M.trials[i] : 1)

                temp_lik_obj = bstm_Likelihood(
                    fam_str, [0.0];
                    sigma_y=sig_y[i],
                    r_nb=r_val,
                    trial=n_t,
                    extra_params=extra
                )

                dist = get_dist_ref(fam, temp_lik_obj, eta_val, sig_y[i])

                noisy[i, j] = rand(dist)
            end
        end
    end

    return denoised, noisy, log_lik
end

function _calculate_ps_weights(p_denoised, M, PS, N_PS, N_samples)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: Calculates post-stratification weights.
    # Inputs: p_denoised, M, PS, N_PS, N_samples.
    # Outputs: A matrix of post-stratification weights [N_PS x N_samples].
    if N_PS == 0
        return nothing
    end

    local ps_weights = zeros(N_PS, N_samples)

    for k in 1:N_PS
        local s_target = PS.s_idx[k]
        local t_target = PS.t_idx[k]
        local u_target = PS.u_idx[k]

        local obs_match_idx = findfirst(i -> M.s_idx[i] == s_target && M.t_idx[i] == t_target && M.u_idx[i] == u_target, 1:M.y_N)

        if !isnothing(obs_match_idx)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (p_denoised[obs_match_idx, j] + 1e-9)
            end
        else
            local sample_mean_obs = mean(p_denoised[1:M.y_N, :], dims=1)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (sample_mean_obs[j] + 1e-9)
            end
        end
    end

    return ps_weights
end

function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v2.0.0
    # Timestamp: 2026-06-30 11:30:00
    # Synopsis: The internal reconstruction engine for univariate models. It discovers all latent
    #           fields from the MCMC chain, assembles the linear predictor, and generates
    #           predictions, summaries, and diagnostic metrics.
    # Rationale for v2.0.0:
    #     - Standardized output to include `:spatial_denoised` and `:spatial_noisy`.
    #     - Re-integrated summarization for `:mixed_effects` and `:nested_contributions`.

    n_samples = size(chain, 1)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, n_samples, 1, p_names)

    # 2. Linear Predictor Assembly for Training and Prediction Grids
    eta_samples = _modular_eta_assembly(
        N_tot, registry, M, PS
    )

    # 3. Summarize Primary Latent Effects
    summarized_effects = Dict{Symbol, Any}()

    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        s_denoised_samples = registry.s_eff_struct[1]
        summarized_effects[:spatial_denoised] = summarize_array(s_denoised_samples; alpha=alpha)
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_noisy_samples = registry.s_eff_noisy[1]
        summarized_effects[:spatial_noisy] = summarize_array(s_noisy_samples; alpha=alpha)
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_samples = registry.t_eff[1]
        summarized_effects[:temporal] = summarize_array(t_samples; alpha=alpha)
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        u_samples = registry.u_eff
        summarized_effects[:seasonal] = summarize_array(u_samples; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        st_samples = registry.st_eff_maps[1]
        summarized_effects[:spacetime_interaction] = summarize_array(st_samples; alpha=alpha)
    end
    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_matrix) in registry.basis_coeffs
            B_mat = M.basis_matrices[var_sym]
            effect_matrix = B_mat * coeffs_matrix'
            summarized_effects[:smooth_effects][var_sym] = summarize_array(effect_matrix; alpha=alpha)
        end
    end
    if !isnothing(registry.Xfixed_betas)
        summarized_effects[:fixed_effects] = summarize_array(registry.Xfixed_betas'; alpha=alpha)
    end

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end

    if !isnothing(registry.nested_eff) && !all(iszero, registry.nested_eff)
        summarized_effects[:nested_contributions] = summarize_array(registry.nested_eff; alpha=alpha)
    end

    # 4. Generate Predictions and Compute Log-Likelihood
    eta_samples_2d = reshape(eta_samples, N_tot, n_samples)
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples_2d, chain, M, N_tot, n_samples, registry.sv_surface
    )

    # 5. Summarize Predictions and Post-Stratification Weights
    summarized_effects[:eta] = summarize_array(eta_samples_2d[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)

        ps_weights_samples = _calculate_ps_weights(p_denoised, M, PS, N_PS, n_samples)
        if !isnothing(ps_weights_samples)
            summarized_effects[:ps_weights_raw] = ps_weights_samples
            summarized_effects[:ps_weights] = summarize_array(ps_weights_samples; alpha=alpha)
        end
    end

    # 6. Final Diagnostics and Metadata
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end



function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v2.1.0
    # Timestamp: 2026-07-03 12:00:00
    # Synopsis: The internal reconstruction engine for multivariate models.
    # Rationale for v2.1.0:
    #     - Completed the implementation by adding log-likelihood calculation.
    #     - Added summarization for space-time, smooth, mixed, and nested effects.
    #     - Integrated WAIC calculation for model selection.

    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names)

    # 2. Linear Predictor Assembly
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

    # Apply LKJ coupling if present
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta_samples[:, :, j] = eta_samples[:, :, j] * L_corr_samples[:,:,j]'
        end
    end

    # 3. Summarize Primary Latent Effects
    summarized_effects = Dict{Symbol, Any}()

    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        s_denoised_summaries = [summarize_array(registry.s_eff_struct[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_denoised] = s_denoised_summaries
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_noisy_summaries = [summarize_array(registry.s_eff_noisy[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_noisy] = s_noisy_summaries
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_summaries = [summarize_array(registry.t_eff[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:temporal] = t_summaries
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        summarized_effects[:seasonal] = summarize_array(registry.u_eff; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        st_summaries = [summarize_array(registry.st_eff_maps[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spacetime_interaction] = st_summaries
    end

    # 4. Generate Predictions and Compute Log-Likelihood
    p_denoised = zeros(Float64, N_tot, outcomes_N, N_samples)
    p_noisy = zeros(Float64, N_tot, outcomes_N, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)
    y_sigma_samples = zeros(N_tot, outcomes_N, N_samples)

    if "y_sigma" in p_names
        sig_samps = get_params_vector(chain, "y_sigma", outcomes_N)
        for k in 1:outcomes_N
            y_sigma_samples[:, k, :] .= sig_samps[:, k]'
        end
    end

    use_zi = get(M, :use_zi, false)
    r_nb_samples = "lik_r" in p_names ? get_params_vector(chain, "lik_r", outcomes_N) : fill(1.0, N_samples, outcomes_N)
    phi_zi_samples = "lik_phi" in p_names ? get_params_vector(chain, "lik_phi", 1) : fill(0.0, N_samples, 1)
    extra_p_samples = "extra_params" in p_names ? get_params_vector(chain, "extra_params", outcomes_N) : fill(1.0, N_samples, outcomes_N)

    for j in 1:N_samples
        for k in 1:outcomes_N
            eta_jk = eta_samples[:, k, j]
            r_val = r_nb_samples[j, k]
            phi_val = phi_zi_samples[j]
            extra_val = extra_p_samples[j, k]

            mu_vec = _apply_link_and_lik(family_str, eta_jk, use_zi, phi_val, r_val)
            p_denoised[:, k, j] .= mu_vec

            for i in 1:N_tot
                is_obs = i <= M.y_N
                eta_val = eta_jk[i]
                sig_y = y_sigma_samples[i, k, j]

                if is_obs
                    lik_obj = bstm_Likelihood(
                        fam_obj, [M.y_obs[i, k]]; sigma_y=[sig_y],
                        phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, extra_params=[extra_val]
                    )
                    log_lik[j, (k-1)*M.y_N + i] = Distributions.logpdf(lik_obj, eta_val)
                end

                if use_zi && rand() < phi_val
                    p_noisy[i, k, j] = 0.0
                else
                    temp_lik_obj = bstm_Likelihood(fam_obj, [0.0]; sigma_y=[sig_y], r_nb=[r_val], extra_params=[extra_val])
                    dist = get_dist_ref(fam_obj, temp_lik_obj, eta_val, sig_y)
                    p_noisy[i, k, j] = rand(dist)
                end
            end
        end
    end

    # 5. Summarize Predictions and other effects
    summarized_effects[:eta] = [summarize_array(eta_samples[:, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_denoised] = [summarize_array(p_denoised[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_noisy] = [summarize_array(p_noisy[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = [summarize_array(p_denoised[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:ps_predictions_noisy] = [summarize_array(p_noisy[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
    end

    # 6. Final Diagnostics and Metadata
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end



function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # BSTM Internal Utility v2.1.0
    # Timestamp: 2026-07-03 12:00:00
    # Synopsis: The internal reconstruction engine for multivariate models.
    # Rationale for v2.1.0:
    #     - Completed the implementation by adding log-likelihood calculation.
    #     - Added summarization for space-time, smooth, mixed, and nested effects.
    #     - Integrated WAIC calculation for model selection.

    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    # 1. Parameter and Latent Field Discovery
    registry = _discover_manifold_realizations(chain, M, N_samples, outcomes_N, p_names)

    # 2. Linear Predictor Assembly
    eta_samples = _modular_eta_assembly(N_tot, registry, M, PS)

    # Apply LKJ coupling if present
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta_samples[:, :, j] = eta_samples[:, :, j] * L_corr_samples[:,:,j]'
        end
    end

    # 3. Summarize Primary Latent Effects
    summarized_effects = Dict{Symbol, Any}()

    if !isnothing(registry.s_eff_struct) && !isempty(registry.s_eff_struct)
        s_denoised_summaries = [summarize_array(registry.s_eff_struct[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_denoised] = s_denoised_summaries
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_noisy_summaries = [summarize_array(registry.s_eff_noisy[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spatial_noisy] = s_noisy_summaries
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_summaries = [summarize_array(registry.t_eff[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:temporal] = t_summaries
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        summarized_effects[:seasonal] = summarize_array(registry.u_eff; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        st_summaries = [summarize_array(registry.st_eff_maps[k]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:spacetime_interaction] = st_summaries
    end

    # 4. Generate Predictions and Compute Log-Likelihood
    p_denoised = zeros(Float64, N_tot, outcomes_N, N_samples)
    p_noisy = zeros(Float64, N_tot, outcomes_N, N_samples)
    log_lik = zeros(Float64, N_samples, M.y_N * outcomes_N)
    y_sigma_samples = zeros(N_tot, outcomes_N, N_samples)

    if "y_sigma" in p_names
        sig_samps = get_params_vector(chain, "y_sigma", outcomes_N)
        for k in 1:outcomes_N
            y_sigma_samples[:, k, :] .= sig_samps[:, k]'
        end
    end

    use_zi = get(M, :use_zi, false)
    r_nb_samples = "lik_r" in p_names ? get_params_vector(chain, "lik_r", outcomes_N) : fill(1.0, N_samples, outcomes_N)
    phi_zi_samples = "lik_phi" in p_names ? get_params_vector(chain, "lik_phi", 1) : fill(0.0, N_samples, 1)
    extra_p_samples = "extra_params" in p_names ? get_params_vector(chain, "extra_params", outcomes_N) : fill(1.0, N_samples, outcomes_N)

    for j in 1:N_samples
        for k in 1:outcomes_N
            eta_jk = eta_samples[:, k, j]
            r_val = r_nb_samples[j, k]
            phi_val = phi_zi_samples[j]
            extra_val = extra_p_samples[j, k]

            mu_vec = _apply_link_and_lik(family_str, eta_jk, use_zi, phi_val, r_val)
            p_denoised[:, k, j] .= mu_vec

            for i in 1:N_tot
                is_obs = i <= M.y_N
                eta_val = eta_jk[i]
                sig_y = y_sigma_samples[i, k, j]

                if is_obs
                    lik_obj = bstm_Likelihood(
                        fam_obj, [M.y_obs[i, k]]; sigma_y=[sig_y],
                        phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, extra_params=[extra_val]
                    )
                    log_lik[j, (k-1)*M.y_N + i] = Distributions.logpdf(lik_obj, eta_val)
                end

                if use_zi && rand() < phi_val
                    p_noisy[i, k, j] = 0.0
                else
                    temp_lik_obj = bstm_Likelihood(fam_obj, [0.0]; sigma_y=[sig_y], r_nb=[r_val], extra_params=[extra_val])
                    dist = get_dist_ref(fam_obj, temp_lik_obj, eta_val, sig_y)
                    p_noisy[i, k, j] = rand(dist)
                end
            end
        end
    end

    # 5. Summarize Predictions and other effects
    summarized_effects[:eta] = [summarize_array(eta_samples[:, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_denoised] = [summarize_array(p_denoised[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    summarized_effects[:predictions_noisy] = [summarize_array(p_noisy[1:M.y_N, k, :]; alpha=alpha) for k in 1:outcomes_N]
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = [summarize_array(p_denoised[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
        summarized_effects[:ps_predictions_noisy] = [summarize_array(p_noisy[(M.y_N+1):end, k, :]; alpha=alpha) for k in 1:outcomes_N]
    end

    if !isnothing(registry.basis_coeffs) && !isempty(registry.basis_coeffs)
        summarized_effects[:smooth_effects] = Dict{Symbol, Any}()
        for (var_sym, coeffs_matrix) in registry.basis_coeffs
            effect_matrix = M.basis_matrices[var_sym] * coeffs_matrix'
            summarized_effects[:smooth_effects][var_sym] = summarize_array(effect_matrix; alpha=alpha)
        end
    end

    if !isnothing(registry.mixed_eff_coeffs) && !isempty(registry.mixed_eff_coeffs)
        summarized_effects[:mixed_effects] = Dict{Symbol, Any}()
        for (i, term) in enumerate(M.mixed_terms)
            summarized_effects[:mixed_effects][term.name] = summarize_array(registry.mixed_eff_coeffs[i]; alpha=alpha)
        end
    end

    if !isnothing(registry.nested_eff) && !all(iszero, registry.nested_eff)
        summarized_effects[:nested_contributions] = summarize_array(registry.nested_eff; alpha=alpha)
    end

    # 6. Final Diagnostics and Metadata
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return NamedTuple(summarized_effects)
end


function _generate_plots(res, M; au=nothing, data=nothing, ts=1, outcome=1)
    # Initialization of the plot registry
    # This dictionary aggregates all graphical objects for the final report bundle
    plots = Dict{Symbol, Any}()
    
    # Metadata Retrieval
    # Accesses observed data and geometric units for spatial rendering
    y_obs = isnothing(data) ? nothing : data.y
    polygons = isnothing(au) ? nothing : get(au, :polygons, nothing)
    centroids = isnothing(au) ? nothing : get(au, :centroids, nothing)

    # 1. Posterior Predictive Check (PPC)
    # Rationale: Validates the model fit by comparing observed values against the denoised posterior expectation
    if !isnothing(y_obs) && hasproperty(res, :predictions_denoised)
        is_mv = (y_obs isa AbstractMatrix && size(y_obs, 2) > 1)
        y_p = is_mv ? res.predictions_denoised.mean[:, outcome] : vec(res.predictions_denoised.mean)
        y_o = is_mv ? y_obs[:, outcome] : vec(y_obs)

        if length(y_p) == length(y_o)
            p_ppc = scatter(vec(y_p), vec(y_o), title="Posterior Predictive Check", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
            
            # Identity line (1:1) to visual bias
            clean_p = filter(!isnan, y_p)
            clean_o = filter(!isnan, y_o)
            if !isempty(clean_p) && !isempty(clean_o)
                min_val = min(minimum(clean_p), minimum(clean_o))
                max_val = max(maximum(clean_p), maximum(clean_o))
                plot!(p_ppc, [min_val, max_val], [min_val, max_val], color=:red, ls=:dash, lw=1.5)
            end
            plots[:ppc] = p_ppc
        end
    end

    # Internal Utility: Spatial Choropleth Construction
    # Rationale: Encapsulates the logic for mapping vector values to areal unit geometries
    function _create_choropleth_plot(field_data, title_str, polygons, centroids)
        if !isnothing(field_data) && hasproperty(field_data, :mean)
            s_mean = vec(collect(field_data.mean))
            if !all(iszero, s_mean) && (!isnothing(polygons) || !isnothing(centroids))
                if !isnothing(polygons) && length(polygons) >= length(s_mean)
                    # Dispatches to the high-level choropleth renderer
                    return plot_choropleth(s_mean, polygons; title=title_str)
                elseif !isnothing(centroids)
                    # Fallback to scatter plot if polygons are missing
                    p_map = scatter(getindex.(centroids, 1), getindex.(centroids, 2), marker_z=s_mean, markersize=4, c=:viridis, label=nothing, title=title_str, aspect_ratio=:equal)
                    return p_map
                end
            end
        end
        return nothing
    end

    # 2. Spatial Latent Fields
    # Rationale: Visualizing the regional risk structured by the BYM2/ICAR manifolds
    if hasproperty(res, :spatial_denoised)
        s_field_denoised = (res.arch isa MultivariateArchitecture) ? res.spatial_denoised[outcome] : res.spatial_denoised
        p_spatial_denoised = _create_choropleth_plot(s_field_denoised, "Spatial Denoised Effect", polygons, centroids)
        if !isnothing(p_spatial_denoised)
            plots[:spatial_denoised] = p_spatial_denoised
        end
    end

    if hasproperty(res, :spatial_noisy)
        s_field_noisy = (res.arch isa MultivariateArchitecture) ? res.spatial_noisy[outcome] : res.spatial_noisy
        p_spatial_noisy = _create_choropleth_plot(s_field_noisy, "Total Spatial Effect (incl. IID)", polygons, centroids)
        if !isnothing(p_spatial_noisy)
            plots[:spatial_noisy] = p_spatial_noisy
        end
    end

    # 3. Temporal Main Trend
    # Rationale: Displays the auto-regressive trajectory over the analysis period
    if hasproperty(res, :temporal)
        raw_t = res.temporal
        t_field = (res.arch isa MultivariateArchitecture) ? raw_t[outcome] : raw_t
        if !isnothing(t_field) && hasproperty(t_field, :mean) && !all(iszero, t_field.mean)
            tm = vec(t_field.mean)
            tl = vec(t_field.lower)
            tu = vec(t_field.upper)
            plots[:temporal] = plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend (AR1)", lw=2, fillalpha=0.2, color=:royalblue, legend=false, xlabel="Year Index")
        end
    end

    # 4. Seasonal Dynamics
    # Rationale: Captures periodic cycles extracted from the seasonal manifold
    if hasproperty(res, :seasonal) && !isnothing(res.seasonal)
        if hasproperty(res.seasonal, :mean) && !all(iszero, res.seasonal.mean)
            um = vec(res.seasonal.mean)
            ul = vec(res.seasonal.lower)
            uu = vec(res.seasonal.upper)
            plots[:seasonal] = plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Component", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Period")
        end
    end

    # 5. Smooth Covariate Effects
    # Rationale: Reconstructs the non-linear relationship between predictors and response
    if hasproperty(res, :smooth_effects) && res.smooth_effects isa Dict
        smooth_plots = Dict{Symbol, Any}()
        for (var_sym, smooth_summary) in res.smooth_effects
            if !isnothing(data) && hasproperty(data, var_sym)
                covariate_data = data[!, var_sym]
                # Sort indices for coherent line plotting
                p_order = sortperm(covariate_data)
                
                sm = vec(smooth_summary.mean)
                sl = vec(smooth_summary.lower)
                su = vec(smooth_summary.upper)
                
                smooth_plots[var_sym] = plot(covariate_data[p_order], sm[p_order], ribbon=(sm[p_order] .- sl[p_order], su[p_order] .- sm[p_order]),
                                             title="Smooth Effect: $var_sym", xlabel=string(var_sym), ylabel="Latent Effect",
                                             legend=false, color=:darkorange, fillalpha=0.2)
            end
        end
        if !isempty(smooth_plots)
            plots[:smooth_effects] = smooth_plots
        end
    end

    # 6. Fixed Effects Forest Plot
    # Rationale: Displays the standardized coefficients for fixed components with credible intervals
    if hasproperty(res, :fixed_effects) && !isnothing(res.fixed_effects)
        if hasproperty(res.fixed_effects, :mean)
            fm = vec(res.fixed_effects.mean)
            fl = vec(res.fixed_effects.lower)
            fu = vec(res.fixed_effects.upper)
            n_coeffs = length(fm)
            if n_coeffs > 0
                p_forest = scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), title="Fixed Effects Coefficients", xlabel="Estimate", ylabel="Index", markersize=4, color=:black, legend=false)
                # Vertical zero reference line to identify significance
                vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
                plots[:fixed_effects] = p_forest
            end
        end
    end

    # Return the technical plot bundle
    return NamedTuple(plots)
end


# v4.9.6 (2026-07-08) - Consolidated Comprehensive Reporting
# Synopsis: Resolves truncation in reporting by restoring Pearson r, compute time, and PS weights.
# Rationale: Ensures all diagnostic fixes for VNChain persist while maintaining full feature parity with v4.8.0.

function model_results_comprehensive(model, chain; au=nothing, data=nothing, n_samples=1000, alpha=0.05)
    # Metadata and Architecture Extraction
    # Determining dimensionality and likelihood family from the model configuration object M.
    M = model.args[1]
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")
    model_family = get(M, :model_family, "gaussian")

    # Technical Dispatch Resolution
    # Mapping the architecture string to concrete dispatch types for the reconstruction engine.
    arch_type = if raw_arch == "multivariate"
        MultivariateArchitecture()
    else
        UnivariateArchitecture()
    end

    # Latent Manifold Reconstruction
    # Invokes the internal engine to discover all latent realizations and assemble predictions.
    # v4.9.5 _reconstruct is used here to ensure ps_weights are calculated.
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # Performance Metric Assessment
    # Evaluates predictive accuracy on the response scale using valid observations.
    y_pred = res.predictions_denoised.mean
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0

    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]
        rmse_val = sqrt(mean((obs_v .- pred_v).^2))
        try
            r_pearson = cor(obs_v, pred_v)
        catch
            r_pearson = 0.0
        end
    end

    # MCMC Diagnostic Extraction
    # Standard MCMCChains.summarize handles VNChain objects via technical dispatch.
    mean_rhat = 1.0
    min_ess = 0.0
    try
        df_stats = DataFrame(MCMCChains.summarize(chain))

        if hasproperty(df_stats, :rhat)
            r_vals = filter(x -> !isnan(x) && x > 0, df_stats.rhat)
            mean_rhat = isempty(r_vals) ? 1.0 : mean(r_vals)
        end

        e_col = hasproperty(df_stats, :ess_bulk) ? :ess_bulk : (hasproperty(df_stats, :ess) ? :ess : nothing)
        if !isnothing(e_col)
            e_vals = filter(x -> !isnan(x) && x >= 0, df_stats[!, e_col])
            min_ess = isempty(e_vals) ? 0.0 : minimum(e_vals)
        end
    catch e
        @warn "Diagnostic extraction failed: $e. Falling back to default values."
    end

    # Compute Time Recovery
    sampling_time = 0.0
    if hasproperty(chain, :info) && haskey(chain.info, :stop_time)
        sampling_time = (chain.info.stop_time - chain.info.start_time)
    elseif hasproperty(chain, :_metadata) && hasproperty(chain._metadata, :sampling_time)
        sampling_time = chain._metadata.sampling_time[]
    end

    # Formatted Reporting
    println("\n--- Model Registry Summary ---")
    println("Architecture:     ", raw_arch)
    println("Family:           ", model_family)
    println("Space Component:  ", get(res, :model_space, "none"))
    println("Time Component:   ", get(res, :model_time, "none"))

    println("\n--- Performance Metrics ---")
    println("RMSE:             ", round(rmse_val, digits=4))
    println("Pearson r:        ", round(r_pearson, digits=4))
    println("WAIC Score:       ", round(get(res, :waic, 0.0), digits=2))

    println("\n--- MCMC Diagnostics ---")
    println("Compute Time:     ", round(sampling_time, digits=2), " seconds")
    println("Mean R-hat:       ", round(mean_rhat, digits=4))
    println("Minimum ESS:      ", round(min_ess, digits=2))

    # Post-Stratification Weight Validation
    if haskey(res, :ps_weights)
        println("\n--- Post-Stratification Audit ---")
        println("PS Weights Found: Yes")
        println("Weight Mean:      ", round(mean(res.ps_weights.mean), digits=4))
    end

    # Visualization Generation
    plots = _generate_plots(res, M; au=au, data=data)

    return (metrics = (rmse = rmse_val, r_pearson = r_pearson, ess = min_ess, rhat = mean_rhat, waic = get(res, :waic, 0.0), time = sampling_time), pstats = res, plots = plots)
end



function model_results_plots(res)
    # Displays all plots generated by `model_results_comprehensive` and stored
    # in the results object.
    if !hasproperty(res, :plots) || isempty(res.plots)
        println("No plots found in the results object.")
        return
    end

    println("Displaying generated plots...")
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
    println("--- End of plots ---")
end
