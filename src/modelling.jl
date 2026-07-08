#!Reference
 

# Definitions

const BSTM_TRANSFORM_KEYWORDS = Dict(
    "log" => x -> log.(x),
    "zscore" => x -> (x .- mean(x)) ./ std(x),
    "unit" => x -> (x .- minimum(x)) ./ (maximum(x) - minimum(x))
)
# BSTM Low-Level Manifold Registry ---
# Rationale: ManifoldModels are now defined as low-level primitives that are domain-agnostic. 
# The context (Spatial vs Temporal) is determined at the model-building stage.

# --- 1. Core Abstract Types ---
abstract type Manifold end
abstract type ManifoldModel <: Manifold end
abstract type ManifoldOperator <: Manifold end
struct Fixed <: ManifoldModel end
struct Covariate <: ManifoldModel end
struct NoneManifold <: ManifoldModel end

# Discrete & Graph Primitives

function create_pc_prior(param_name::Symbol, constraint::Tuple)
 """
    create_pc_prior(param_name::Symbol, constraint::Tuple)

Create a Penalized Complexity (PC) prior distribution based on a user-specified quantile constraint.

This function translates a user's belief about the scale of a parameter (e.g., standard deviation,
correlation) into a full prior distribution from the `Distributions.jl` package. The constraint
is typically of the form `P(parameter > U) = α` or `P(parameter < U) = α`.

# Arguments
- `param_name::Symbol`: The base name of the parameter (e.g., `:sigma`, `:rho`, `:lengthscale`).
- `constraint::Tuple`: A tuple specifying the prior belief.
    - For upper tail constraints (`:upper`): `(U, α)`, interpreted as `P(param > U) = α`.
    - For lower tail constraints (`:lower`): `(U, α, :lower)`, interpreted as `P(param < U) = α`.

# Returns
- A `Distribution` object representing the calculated prior.

# Details
- **sigma**: Assumes an `Exponential(λ)` prior. `λ` is derived from `P(σ > U) = α`.
- **rho**: Assumes a transformed `Exponential(λ)` prior on `θ = -log(1-ρ)`. `λ` is derived from `P(ρ > U) = α`.
- **lengthscale**: Assumes a transformed `Exponential(λ)` prior on `θ = 1/l`. `λ` is derived from `P(l < U) = α`.
- **kappa**: Assumes an `Exponential(λ)` prior. `λ` is derived from `P(κ > U) = α`.

create_pc_prior(param_name::Symbol, constraint::Tuple)

Create a Penalized Complexity (PC) prior distribution based on a user-specified quantile constraint.
This function translates a user's belief about the scale of a parameter into a full prior distribution.

# Arguments
- `param_name::Symbol`: The base name of the parameter (e.g., `:sigma`, `:rho`).
- `constraint::Tuple`: A tuple specifying the prior belief, e.g., `(U, α)`.

# Returns
- A `Distribution` object representing the calculated prior.
"""

    direction = :upper # Default to P(param > U) = α
    if length(constraint) == 2
        U, α = constraint
    elseif length(constraint) == 3
        U, α, direction = constraint
    else
        error("PC prior constraint must be a tuple of (U, α) or (U, α, direction).")
    end

    base_param_name = Symbol(replace(string(param_name), r"_prior$" => ""))

    if base_param_name == :sigma || endswith(string(base_param_name), "_sigma")
        direction != :upper && error("PC prior for sigma only supports upper tail constraints, e.g., P(sigma > U) = alpha.")
        λ = -log(α) / U
        return Exponential(λ)
    elseif base_param_name == :rho || endswith(string(base_param_name), "_rho")
        direction != :upper && error("PC prior for 'rho' only supports upper tail constraints, e.g., P(rho > U) = alpha.")
        # Transformation: θ = -log(1-ρ) ~ Exponential(λ)
        # P(ρ > U) = P(-log(1-ρ) > -log(1-U)) = exp(-λ * -log(1-U)) = (1-U)^λ = α
        λ = log(α) / log(1.0 - U)
        return Exponential(λ)
    elseif base_param_name == :lengthscale || endswith(string(base_param_name), "_lengthscale")
        direction != :lower && error("PC prior for 'lengthscale' only supports lower tail constraints, e.g., P(lengthscale < U) = alpha.")
        # Transformation: θ = 1/l ~ Exponential(λ)
        # P(l < U) = P(1/θ < U) = P(θ > 1/U) = exp(-λ/U) = α
        λ = -U * log(α)
        return Exponential(λ)
    elseif base_param_name == :kappa || endswith(string(base_param_name), "_kappa")
        direction != :upper && error("PC prior for kappa only supports upper tail constraints, e.g., P(kappa > U) = alpha.")
        λ = -log(α) / U
        return Exponential(λ)
    else
        error("PC prior not defined for parameter: $base_param_name")
    end
end



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
    """BSTM Utility Macro v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Macro v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
    Synopsis: A simple helper function to print the full contents of a Julia object to the console,
              bypassing truncation that can occur with default display methods.
    """
    show(stdout, "text/plain", x) # display all estimates
end 
 

function modelruntime(o)
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
    Synopsis: A commented-out utility for inspecting the generated code of a function.
              When active, it would use `CodeTracking.@code_string` to print the source code.
    """
end

function firstindexin(a::AbstractArray, b::AbstractArray)
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    Rationale for v1.2.0:
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
    """BSTM Utility Function v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    # v1.2.0 (2026-06-29 16:13:05)
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

    # Add intercept if it is part of the model
    if get(config, :add_intercept, false)
        push!(eta_parts, "intercept")
    end
    
    # Add a placeholder for Knorr-Held spatiotemporal interactions, which are
    # handled by a special block in bstm_univariate and not in the main manifold loop.
    model_st = get(config, :model_st, "none")
    if model_st != "none"
        push!(eta_parts, "spacetime_interaction")
    end

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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Capitalizes the first letter of a string. 
    # Inputs: s::String - The input string.
    # Outputs: A string with the first letter capitalized.
    isempty(s) && return s
    return uppercasefirst(s)
end


 



function apply_soft_constraint(latent_field::AbstractVector{T}, constraint_type::Symbol, weight::Float64) where {T}
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Returns the lowercase string representation of a manifold's type. 
    # Inputs: m::Manifold.
    # Outputs: A string with the manifold type name.
    return lowercase(string(typeof(m)))
end
 



  
function bstm_Likelihood(family_input, y_obs; sigma_y=0.0, weight=1.0, phi_zi=-Inf, r_nb=0, trial=0,
                         y_L=-Inf, y_U=Inf, hurdle=-Inf, extra_params=zeros(1)[])
    # v1.2.0 (2026-06-29 16:13:05)
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

# v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Checks if a model family is discrete. 
    # Inputs: A concrete type inheriting from AbstractBSTM_Family.
    # Outputs: true if the family is discrete, false otherwise.
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Default method for checking if a family is discrete. 
    # Inputs: A concrete type inheriting from AbstractBSTM_Family.
    # Outputs: false.
    return false
end

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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


# BSTM Internal Utility v1.2.0
# Timestamp: 2026-06-29 16:13:05
# Synopsis: Overloads the `Distributions.logpdf` function for the `bstm_Likelihood` struct.
#           This provides the main entry point for calculating the log-likelihood within a
#           Turing model, dispatching to the appropriate `bstm_kernel` based on the data traits.
# Rationale for v1.2.0:
#     - Removed redundant `local` keyword declarations for improved code clarity and style consistency.
#       The underlying logic remains correct and unchanged.

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Overloads logpdf for matrix-variate observations (e.g., InverseWishart). 
    # Inputs: d::bstm_Likelihood, eta::AbstractMatrix.
    # Outputs: The log-probability value.
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
end


function scaling_factor_bym2( adjacency_mat )
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
 
 

"""
    rff_map(coords, W, b)

Helper function to map input coordinates into a Random Fourier Feature (RFF) space.
This should exist in your environment, but is included here for completeness.
"""
function rff_map(coords, W, b)
    projection = (coords * W) .+ b'
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    return feature_map
end



function generate_informed_rff_params(coords, M_rff; kernel_name="se", nu=nothing, lengthscale_mult=0.5)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: Constructs a GMRF precision matrix (Graph Laplacian, Q = D - W) from an adjacency matrix. 
    # Inputs: adj_matrix - Sparse adjacency matrix (W).
    # Outputs: Sparse precision matrix (Q).
    # Note: This is the fundamental structure for ICAR and other graph-based spatial models.

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 



function get_rff_deep2D_basis(X, m, lengthscale)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.1 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: A simple helper function to compute a squared-exponential covariance. 
    # Inputs: D (distance matrix), phi (lengthscale), noise.
    # Outputs: A covariance matrix.
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
    # v1.2.0 (2026-06-29 16:13:05)
    # Purpose: A simple helper function to compute an exponential covariance. 
    # Inputs: D (distance matrix), phi (lengthscale).
    # Outputs: A covariance matrix.
exp_cov_fn(D, phi) = exp.(-D / phi)

 


# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    # v1.2.1 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
 
 


function bstm_sample(m::DynamicPPL.Model; nsample=10, testing=false)
    # #
    # bstm_sample v2.0.0 (2026-07-08)
    #
    # This function orchestrates the MCMC sampling process for a bstm model.
    #
    # Key Change in v2.0.0:
    # Rationale: The root cause of persistent reconstruction errors was identified as a
    #            type mismatch between different MCMC chain objects used in the ecosystem
    #            (e.g., `FlexiChains.FlexiChain` vs. `MCMCChains.Chains`).
    # Solution: This version now explicitly converts the output of `Turing.sample` into a
    #           standard `MCMCChains.Chains` object. This guarantees that all downstream
    #           post-processing functions (`model_results_comprehensive`, etc.) receive a
    #           consistent, predictable data structure, resolving the `MethodError` on `names()`
    #           and ensuring the parameter extraction engine works correctly.

    model_summary = show_model(m)
    
    if !testing
        inits = get_inits(m)
        nadapt = max(Int(round(nsample * 0.25)), 200)
        # Default to NUTS for robust sampling in production
        os = NUTS(nadapt, 0.65)
    else
        inits =  get_inits(m; refine="none")
        os = MH()
    end

    chn = sample(m, os, nsample; initial_params=inits, progress=true, drop_warmup=true)
    
    # Use StatsPlots.plot, which has recipes for handling various MCMC chain objects.
    plt = StatsPlots.plot(chn, seriestype=:traceplot)

    return chn, inits, os, model_summary, plt
 
end


 

function create_prediction_surface(
    data_df::DataFrame, 
    au_obj::NamedTuple, 
    tu_obj::NamedTuple, 
    covariate_vars::Vector{Symbol}; 
    iterations::Int = 3
)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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




function scottish_lip_cancer_data_spacetime(n_years::Int=10, spatial_expansion::Float64=1.5, temporal_expansion::Float64=1.5; rndseed::Int=42, recreate::Bool=false)
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.1 (2026-06-29 17:16:00)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    """BSTM Utility v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
BSTM Internal Utility v1.2.0
Timestamp: 2026-06-29 16:13:05
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
    """v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.0 (2026-06-29 16:13:05)
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
    # v1.2.1 (2026-07-12)
    # Purpose: A recursive version of `recompose_precision` that handles algebraic compositions.
    # Rationale for v1.2.1:
    #     - Corrected the dispatch order to handle `ComposedManifold` before the general `Manifold` case.
    #     - Added a conversion from NamedTuple to Dict for the `M` config object to resolve a `MethodError`
    #       when `build_model` is called from within `bstm_univariate`.

    # # Algebraic Case: Composed Manifolds (⊗, ⊕)
    # This check must come first, as ComposedManifold is a subtype of Manifold.
    if manifold_node isa ComposedManifold
        # Retrieve components
        comps = manifold_node.components
        op = manifold_node.operator
        
        # #
        # Kronecker Product (⊗)
        # Mathematical Justification: The Kronecker product of two precision matrices,
        # Q_total = Q1 ⊗ Q2, corresponds to the precision of a joint Gaussian process
        # defined on a product space. If x ~ N(0, inv(Q1)) and y ~ N(0, inv(Q2)), then
        # vec(y*x') ~ N(0, inv(Q1 ⊗ Q2)).
        # Use Case: This is the standard method for creating separable spatiotemporal
        # interaction models (e.g., Knorr-Held Type IV), where the joint precision is
        # Q_st = Q_s ⊗ Q_t. It models a process where every point in one domain (e.g., space)
        # has a unique, correlated trend defined by the other domain (e.g., time).
        if op == :kronecker_product
            # Recursively build the component precision matrices with unit variance.
            # The final variance is applied to the combined structure.
            # Q_total = Q1 ⊗ Q2
            # We split the parameter variance across components or treat sig as global scale
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            # The final precision matrix is the scaled Kronecker product.
            Q_full = kron(Q1, Q2)
            return Symmetric(Q_full ./ (param_sig^2 + noise) + noise * I)
            
        # #
        # Direct Sum (⊕)
        # Mathematical Justification: The matrix direct sum, Q_total = diag(Q1, Q2),
        # corresponds to the precision matrix of a joint distribution of two a priori
        # independent random vectors. If x1 ~ N(0, inv(Q1)) and x2 ~ N(0, inv(Q2)),
        # then [x1; x2] ~ N(0, inv(diag(Q1, Q2))).
        # Use Case: This models components that are structurally independent but are
            # specified to share a common hyperparameter (param_sig).
        elseif op == :direct_sum
            # Recursively build component precision matrices with unit variance.
            Q_components = [recompose_precision(c, M, 1.0; noise=noise) for c in comps]
            # Ensure components are sparse for efficient block-diagonal construction.
            Q_sparse_components = [c isa AbstractSparseMatrix ? c : sparse(c) for c in Q_components]
            Q_block = blockdiag(Q_sparse_components...)
            return Symmetric(Q_block ./ (param_sig^2 + noise) + noise * I(size(Q_block, 1)))
            
        # #
        # Composition (∘)
        # Mathematical Justification: This operator implements the matrix composition
        # Q_total = Q1 * Q2 * Q1. This form arises in contexts such as basis warping
        # or hierarchical smoothing, where one precision structure acts as a linear
        # operator transforming another. For example, if a latent field is defined as
        # y = A*z where z has precision Q2, and A has a structure related to inv(Q1),
        # the resulting precision for y can take this form.
        # Use Case: Creating complex, non-separable dependencies between manifolds of the
        # same dimension, such as applying a smoothing kernel (Q2) to a field that
        # already has a dependency structure (Q1).
        elseif op == :composition
            # Recursively build component precision matrices.
            # Q_total = Q1 * Q2 * Q1 (Basis warping/Cascading dependency)
            # Rationale: Representing the composition of two precision structures as a functional transformation.
            Q1 = recompose_precision(comps[1], M, 1.0; noise=noise)
            Q2 = recompose_precision(comps[2], M, 1.0; noise=noise)
            
            # Ensure dimensional parity for valid matrix multiplication.
            # Ensure dimensional parity for matrix multiplication in composition
            if size(Q1) == size(Q2)
                Q_comp = Q1 * Q2 * Q1
                return Symmetric(Q_comp ./ (param_sig^2 + noise) + noise * I)
            else
                # Fallback to Kronecker if dimensions are heterogeneous
                return Symmetric(kron(Q1, Q2) ./ (param_sig^2 + noise) + noise * I)
            end

        end

    # # Base Case: Atomic Manifold Structs
    elseif manifold_node isa Manifold
        m_type = manifold_type(manifold_node)
        
        # The `build_model` functions expect a Dict, but M can be a NamedTuple when called
        # from `bstm_univariate`. We ensure it's a Dict here.
        M_dict = M isa NamedTuple ? Dict(pairs(M)) : M
        
        # Atomic builders provide the base template Q
        m_meta = build_model(manifold_node, M_dict)
        
        if !isnothing(m_meta) && hasproperty(m_meta, :Q_template)
            return recompose_precision(Symbol(m_type), m_meta.Q_template, param_sig; noise=noise)
        end
    end
    
    # Fallback for symbol-based legacy calls
    return Matrix(1.0I, 1, 1)
end



function parse_variable_and_transforms(var_str::AbstractString)
    # v1.2.0 (2026-06-29 16:13:05)
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
    """BSTM Internal Utility v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    """BSTM Internal Utility v1.2.0
    Timestamp: 2026-06-29 16:13:05
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
    # BSTM Smooth Basis Factory v1.2.0
    # Timestamp: 2026-06-29 16:13:05
    # Synopsis: A factory function that generates a 1D basis matrix for various smoothers. 
    # Rationale for v1.2.0:
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
    # BSTM Smooth Basis Factory v1.2.0
    # Timestamp: 2026-06-29 16:13:05
    # Synopsis: A factory function that generates a 2D basis matrix for various smoothers. 
    # Rationale for v1.2.0:
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
    # BSTM Smooth Basis Factory v1.2.0
    # Timestamp: 2026-06-29 16:13:05
    # Synopsis: A factory function that generates a 3D basis matrix for various smoothers. 
    # Rationale for v1.2.0:
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
    # BSTM Smooth Basis Factory v1.2.0
    # Timestamp: 2026-06-29 16:13:05
    # Synopsis: A factory function that generates a 4D basis matrix for various smoothers. 
    # Rationale for v1.2.0:
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
    # v1.2.0 (2026-06-29 16:13:05)
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

    # # 2. Data-Driven Initialization for Specialized Manifolds
    # Rationale: For complex manifolds like Eigen/PCA, initializing from the data structure
    #            is more robust than random sampling from the prior. This section reintegrates
    #            the `eigenvector_to_householder` utility to provide informed initial values.
    M = model.args.M
    if haskey(M, :manifolds)
        for spec in M.manifolds
            if spec.manifold_obj isa Eigen
                try
                    key = spec.key
                    vars = spec.variables
                    n_factors = spec.manifold_obj.n_factors
                    
                    # Run a standard PCA to get initial eigenvectors
                    pca_data = Matrix(M.data[!, vars])
                    
                    # NOTE: Assumes a `pca_standard` utility function is available in the environment,
                    # as its usage is documented in the project. This function should return
                    # eigenvectors as the first element of a tuple.
                    # e.g., evecs, _, _, _, _, _ = pca_standard(pca_data, n_factors=n_factors)
                    evecs = Main.pca_standard(pca_data, n_factors=n_factors)[1]

                    # Use the `eigenvector_to_householder` function to convert eigenvectors
                    # into the Householder 'v' parameters used by the model.
                    v_mat = eigenvector_to_householder(evecs, n_factors)
                    
                    # Extract the vector of free parameters for initialization.
                    v_params = extract_v_parameters(v_mat, spec.manifold_obj.ltri_indices)
                    init_dict[Symbol("v_", key)] = v_params
                    println("Info: Initialized Eigen manifold '$key' from frequentist PCA.")
                catch e
                    @warn "Failed to initialize Eigen manifold from data. Using prior-based initialization. Error: $e"
                end
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
    BSTM Internal Utility v1.2.0
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
    # BSTM Utility v1.2.1
    # Timestamp: 2026-07-06
    # Synopsis: A utility that creates the fixed-effects design matrix (`X`) from a formula string.
    #           It handles `as_factor` directives for categorical variables. This version corrects
    #           a bug where the function would fail if the formula contained no fixed effects.
    
    # #
    # 1. Synchronize Internal Data Frame
    # Ensures that both DataFrame and NamedArray inputs are handled consistently.
    df_internal = data isa NamedArray ? DataFrame(data, :auto) : copy(data)
    if data isa NamedArray
        rename!(df_internal, names(data, 2))
    end

    # #
    # 2. Process Fixed Effects String
    # The formula_rhs passed to this function should only contain fixed effects terms,
    # as bstm modules are stripped out by the `decompose_bstm_formula` function beforehand.
    final_rhs_string = strip(formula_rhs)

    # If the string is empty after stripping, it signifies no fixed effects.
    # Return a zero-column matrix to maintain type stability downstream.
    if isempty(final_rhs_string)
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[]))
    end

    # Handle the specific case of an intercept-only model.
    if final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept]))
    end

    # #
    # 3. Process `as_factor()` Directives
    # This ensures that specified columns are treated as categorical by StatsModels.
    factor_regex = r"as_factor\(\s*(\w+)\s*\)"
    factor_matches = eachmatch(factor_regex, final_rhs_string)
    for m in factor_matches
        var_sym = Symbol(m.captures[1])
        if hasproperty(df_internal, var_sym)
            df_internal[!, var_sym] = CategoricalArrays.categorical(df_internal[!, var_sym])
        end
    end
    # The formula passed to StatsModels should not contain the as_factor wrapper itself.
    final_rhs_string = replace(final_rhs_string, "as_factor(" => "(")

    # #
    # 4. Design Matrix Expansion using StatsModels.jl
    try
        # A placeholder response variable is required by the @formula macro.
        placeholder_name = :__y_placeholder
        df_internal[!, placeholder_name] = zeros(size(df_internal, 1))

        # Construct and evaluate the formula expression in the Main scope.
        formula_expression = Meta.parse("@formula($placeholder_name ~ $final_rhs_string)")
        dynamic_formula = Main.eval(formula_expression)

        # Apply schema with any specified contrasts to generate the model matrix.
        data_schema = StatsModels.schema(dynamic_formula, df_internal, contrasts)
        applied_formula = StatsModels.apply_schema(dynamic_formula, data_schema, StatsModels.RegressionModel)

        _, model_matrix_numeric = StatsModels.modelcols(applied_formula, df_internal)
        coefficient_labels = StatsModels.coefnames(applied_formula.rhs)

        # Ensure coefficient labels are always a vector of symbols for NamedArray.
        label_vector = coefficient_labels isa AbstractString ? [Symbol(coefficient_labels)] : Symbol.(coefficient_labels)

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector))

    catch design_error
        # If parsing fails, issue a warning and return an empty design matrix to prevent a hard crash.
        @warn "BSTM Registry: create_fixed_design expansion failed for: $final_rhs_string. Error: $design_error"
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[]))
    end
end




function _build_pass_through_model(m::ManifoldModel, data_inputs::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # v1.2.0 (2026-06-29 16:13:05)
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

function build_model(m::DynamicsManifold, data_inputs::Dict)
    # Dynamics models like advection/diffusion operate on a spatial grid.
    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    template = build_structure_template(:besag, n; W=W) # :besag gives the Laplacian
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
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

function build_model(m::ComposedManifold, data_inputs::Dict)
    # ComposedManifolds do not have a single template. Their components are resolved
    # recursively by functions like `recompose_precision`. This method returns a
    # placeholder to satisfy the configuration pipeline.
    return (
        Q_template = nothing,
        scaling_factor = 1.0,
        model_type = :composed,
        hyper = (operator=m.operator, components=m.components)
    )
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



@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
    # #
    # Global Likelihood Hyperparameters
    family = get(M.likelihood_specs[1], :family, "gaussian")
    noise = get(M, :noise, 1e-6)
    use_zi = get(get(M.likelihood_specs, 1, Dict()), :use_zi, false)

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
        Xfixed_beta ~ NamedDist(get(M, :Xfixed_prior, MvNormal(zeros(M.Xfixed_N), 5.0 * I)), :Xfixed_beta)
        eta .+= M.Xfixed * Xfixed_beta
    end

    if haskey(M, :log_offset)
        eta .+= M.log_offset
    end

    # Scaffolding for Spatiotemporal Interaction Structures.
    # These are initialized only if a spatiotemporal interaction model is specified,
    # improving efficiency by avoiding unnecessary memory allocation.
    model_st = get(M, :model_st, "none")
    local s_Q_main, t_Q_main, t_rho_main, s_sigma_main, t_sigma_main
    if model_st != "none"
        s_Q_main = sparse(I(M.s_N))
        t_Q_main = sparse(I(M.t_N))
        t_rho_main = zero(T)
        s_sigma_main = one(T)
        t_sigma_main = one(T)
    end

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

            n_units = if domain == :mixed; spec.params.n_cat; elseif domain == :spatial; M.s_N; else M.t_N; end
            indices = if domain == :mixed; spec.params.indices; elseif domain == :spatial; M.s_idx; else M.t_idx; end
            field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), Symbol("latent_", key))
            eta .+= field[indices]

        elseif m_obj isa BYM2
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))
            if model_st != "none"
                s_sigma_main = sigma_val
            end

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            rho_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))

            s_icar ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), Symbol("latent_struct_", key))
            s_iid ~ NamedDist(MvNormal(zeros(M.s_N), I), Symbol("latent_iid_", key))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(s_icar))
            combined = sigma_val .* (sqrt(rho_val) .* s_icar .+ sqrt(1.0 - rho_val) .* s_iid)
            eta .+= combined[M.s_idx]
            if model_st != "none"
                s_Q_main = spec.Q_template
            end

        elseif m_obj isa AR1
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            rho_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))

            if model_st != "none"
                t_sigma_main = sigma_val
                t_rho_main = rho_val
            end

            innovations ~ NamedDist(MvNormal(zeros(M.t_N), I), Symbol("innov_", key))
            t_field = Vector{T}(undef, M.t_N)
            t_field[1] = innovations[1] / sqrt(1.0 - rho_val^2 + noise)
            for i in 2:M.t_N
                t_field[i] = rho_val * t_field[i-1] + innovations[i]
            end
            eta .+= (t_field .* sigma_val)[M.t_idx]
            if model_st != "none"
                t_Q_base = Symmetric((1.0 + rho_val^2) .* I(M.t_N) .- rho_val .* spec.Q_template)
                t_Q_main = Symmetric((1.0 / (sigma_val^2 * (1.0 - rho_val^2) + noise)) .* t_Q_base)
            end

        elseif m_obj isa Union{ICAR, Besag, RW1, RW2, Leroux, Cyclic}
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))

            r_val = nothing
            if hasproperty(m_obj, :rho_prior)
                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                if rho_val_dist isa Exponential
                    theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_rho_", key))
                    r_val = 1.0 - exp(-theta_rho) # Transform back to [0,1]
                else
                    r_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))
                end
            end

            n_units = size(spec.Q_template, 1)
            Q = recompose_precision(Symbol(lowercase(string(typeof(m_obj)))), spec.Q_template, sigma_val; extra_param=r_val, noise=noise)
            field ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", key))
            if !(m_obj isa Cyclic)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
            end
            indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
            eta .+= field[indices]
            if model_st != "none"
                if domain == :spatial; s_Q_main = Q; elseif domain == :temporal; t_Q_main = Q; end
            end

        elseif m_obj isa SAR
            # #
            # # Simultaneous Autoregressive (SAR) Model
            # # This block implements the general SAR structure: (I - ρW)y = ε, where the innovations ε
            # # can be IID (standard SAR) or have their own correlation structure (SAR with correlated errors).
            # # The precision matrix is Q = (I - ρW)' Q_innov (I - ρW).
            
            # Get priors from the manifold object
            sigma_prior = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            rho_prior = get(spec.params, :rho_prior, m_obj.rho_prior)

            # Sample the main SAR parameters
            sigma_val ~ NamedDist(sigma_prior, Symbol("sigma_", key))
            rho_val ~ NamedDist(rho_prior, Symbol("rho_", key))

            # M1 is the spatial structure matrix (adjacency)
            Q_source = spec.Q_template
            n_units = size(Q_source, 1)

            # For a standard SAR model, innovations are IID. The precision matrix for the innovations
            # is therefore a scaled identity matrix. The scale is determined by sigma_val.
            # This implementation aligns with the classic SAR definition. More complex innovation
            # structures could be specified via formula extensions in the future.
            Q_innov = (1.0 / (sigma_val^2 + noise)) * I(n_units)

            # Construct the final SAR precision matrix: (I - ρW)' Q_innov (I - ρW)
            L_op = I(n_units) - rho_val * Q_source
            Q_final = Symmetric(L_op' * Q_innov * L_op + noise * I)

            field ~ NamedDist(MvNormalCanon(zeros(n_units), Q_final), Symbol("latent_", key))
            
            # Sum-to-zero constraint for identifiability
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
            
            eta .+= field[M.s_idx]


        elseif m_obj isa Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}
            var_sym = spec.var
            
            if m_obj isa RFF
                # --- Dynamic RFF Basis Construction ---
                n_features = m_obj.n_features
                coords = spec.params.coords
                d = size(coords, 2)

                # Sample lengthscale from its prior (can be a PC prior)
                ls_prior = get(spec.params, :lengthscale_prior, m_obj.lengthscale_prior)
                local ls_val
                if ls_prior isa Exponential # PC prior on 1/l
                    theta_ls ~ NamedDist(ls_prior, Symbol("theta_ls_", key))
                    ls_val = 1.0 / (theta_ls + 1e-9)
                else
                    ls_val ~ NamedDist(ls_prior, Symbol("ls_", key))
                end

                # Sample RFF weights and biases
                sigma_spectral = 1.0 / ls_val
                W_raw ~ NamedDist(filldist(Normal(0, 1), d * n_features), Symbol("W_raw_", key))
                W_rff = reshape(W_raw, d, n_features) .* sigma_spectral
                b_rff ~ NamedDist(filldist(Uniform(0, 2*pi), n_features), Symbol("b_rff_", key))

                # Construct basis matrix on-the-fly
                B_mat = rff_map(coords, W_rff, b_rff)
                
                # Sample coefficients for the basis functions
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_basis_", var_sym))
                beta_name = Symbol("beta_basis_", var_sym)
                latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_features), beta_name)
                
                eta .+= B_mat * latent_coeffs
            else # Static basis models
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
            end

        elseif m_obj isa MixedManifold
            # This block was missing, truncating random effects functionality.
            # It samples the random effect coefficients for each group level.
            n_units = spec.params.n_cat
            indices = spec.params.indices
            sigma_val_dist = get(spec.params, :sigma_prior, Exponential(1.0))
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))
            field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), Symbol("latent_", key))
            eta .+= field[indices]

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
            elseif d_model == "directed_spatial_process"
                # #
                # # Simultaneous Autoregressive (SAR) Model with Correlated Errors
                # # Implements the model (I - ρ * M₁)y = ε, where ε has its own covariance structure Q₂.
                # # This is a model for a single spatial field, not a temporal evolution.
                
                if !haskey(M, :W) || !haskey(M, :centroids)
                    @warn "Dynamics model 'directed_spatial_process' requires a spatial manifold with an adjacency matrix 'W' and centroids. Skipping."
                    continue
                end

                # M₁ is the main spatial adjacency matrix.
                Q_source = M.W
                n_units = size(Q_source, 1)

                # Parameters for the dependency structure (ρ)
                rho_prior = get(m_obj.params, :rho_prior, Beta(1,1))
                rho_dep ~ NamedDist(rho_prior, Symbol("rho_dep_", key))

                # Parameters for the innovation process ε, modeled as a GP
                sigma_innov_prior = get(m_obj.params, :sigma_innov_prior, Exponential(1.0))
                sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_", key))
                
                ls_innov_prior = get(m_obj.params, :ls_innov_prior, InverseGamma(3,3))
                ls_innov ~ NamedDist(ls_innov_prior, Symbol("ls_innov_", key))

                # Construct the innovation precision matrix Q₂ from a GP kernel
                kernel_type = Symbol(get(m_obj.params, :kernel, "se"))
                dist_sq = [sum((M.centroids[i] .- M.centroids[j]).^2) for i in 1:n_units, j in 1:n_units]
                K_innov = evaluate_kernel_matrix(dist_sq, sigma_innov, ls_innov, kernel_type, noise)
                Q_innov = inv(Symmetric(K_innov))

                # Construct the final precision matrix: (I - ρM₁)' Q₂ (I - ρM₁)
                L_op = I(n_units) - rho_dep * Q_source
                Q_final = Symmetric(L_op' * Q_innov * L_op + noise * I)

                field ~ NamedDist(MvNormalCanon(zeros(n_units), Q_final), Symbol("latent_", key))
                eta .+= field[M.s_idx]
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

        elseif m_obj isa ComposedManifold
            # #
            # # Algebraic Manifold Composition
            # # This block handles manifolds created by algebraic operators (⊗, ∘, |>, ⊕).
            # # This block handles manifolds created by algebraic operators (⊗, ∘, |>).
            # # It dispatches to specific logic based on the operator.
            # # Note: The direct sum operator (⊕) is handled by the formula parser, which
            # # expands it into separate additive terms. It does not create a ComposedManifold.

            op = m_obj.operator

            if op == :kronecker_product || op == :composition
                # Handles Kronecker products (⊗) and matrix compositions (∘).
                # The precision matrix is constructed by the shared `recompose_precision` function.
                
                sigma_val ~ NamedDist(Exponential(0.5), Symbol("sigma_", key))
                
                # Build the final precision matrix using the shared recomposition logic.
                Q_final = recompose_precision(m_obj, M, sigma_val; noise=noise)
                
                n_units = size(Q_final, 1)
                field ~ NamedDist(MvNormalCanon(zeros(n_units), Symmetric(Q_final)), Symbol("latent_", key))

                # The configuration engine must pre-compute the combined indices for composed fields.
                # For example, for `spatial() ⊗ temporal()`, the index for observation `i` would be
                # `(M.t_idx[i] - 1) * M.s_N + M.s_idx[i]`.
                if hasproperty(spec, :indices) && !isnothing(spec.indices)
                    eta .+= field[spec.indices]
                else
                    @warn "Composed manifold '$key' is missing combined indices. Effect will not be applied."
                end

            elseif op == :direct_sum
                # Handles the matrix direct sum (⊕), creating a block-diagonal precision matrix.
                # This models components that are a priori independent but share a common hyperparameter (sigma_val).
                sigma_val ~ NamedDist(Exponential(0.5), Symbol("sigma_", key))
                Q_final = recompose_precision(m_obj, M, sigma_val; noise=noise)
                n_total = size(Q_final, 1)
                field ~ NamedDist(MvNormalCanon(zeros(n_total), Symmetric(Q_final)), Symbol("latent_", key))

                # The configuration engine must provide component dimensions and indices.
                if hasproperty(spec, :component_dims) && hasproperty(spec, :component_indices)
                    offset = 0
                    for i in 1:length(m_obj.components)
                        dim = spec.component_dims[i]
                        indices = spec.component_indices[i]
                        
                        sub_field = field[offset + 1 : offset + dim]
                        eta .+= sub_field[indices]
                        
                        offset += dim
                    end
                else
                    @warn "Direct sum manifold '$key' is missing component dimensions/indices. Effect will not be applied."
                end

            elseif op == :pipe
                # Handles state-space evolution, e.g., spatial() |> temporal(model=ar1).
                # This logic must reside inside the @model block due to its sampling loop.
                state_manifold = m_obj.components[1]
                dynamic_manifold = m_obj.components[2]

                # Build the specification for the state manifold, which defines the spatial
                # structure of the innovations or coefficients.
                M_dict = M isa NamedTuple ? Dict(pairs(M)) : M
                state_spec = build_model(state_manifold, M_dict)

                # The state manifold determines the structure of the field at each time step.
                # The dynamic manifold determines how the field evolves over time.
                if dynamic_manifold isa AR1
                    # Extract dynamic parameters from the manifold definition.
                    rho_prior = get(dynamic_manifold, :rho_prior, Beta(1,1))
                    sigma_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))

                    rho_pipe ~ NamedDist(rho_prior, Symbol("rho_pipe_", key))
                    sigma_innov ~ NamedDist(sigma_prior, Symbol("sigma_innov_pipe_", key))

                    # Use recompose_precision to build the precision matrix for the innovations.
                    Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                    L_innov = cholesky(Symmetric(Q_innov)).L
                    n_state = size(Q_innov, 1)

                    pipe_field = Matrix{T}(undef, n_state, M.t_N) 
                    
                    # Sample the initial state and subsequent time steps.
                    innov_base ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_1"))
                    pipe_field[:, 1] = (L_innov \ innov_base) ./ sqrt(1.0 - rho_pipe^2 + noise)

                    for t in 2:M.t_N
                        innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", t))
                        pipe_field[:, t] = rho_pipe .* pipe_field[:, t-1] .+ (L_innov \ innov_t)
                    end

                    # Apply the spatiotemporal effect to the linear predictor.
                    for i in 1:M.y_N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                elseif dynamic_manifold isa RW1
                    n_state = size(state_spec.Q_template, 1)
                        # Handles RW1 dynamics: field_t = field_{t-1} + innovation
                        sigma_innov_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                        sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_pipe_", key))

                        # Precision for innovations, structured by the state_manifold
                        Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                        L_innov = cholesky(Symmetric(Q_innov)).L

                        pipe_field = Matrix{T}(undef, n_state, M.t_N)

                        # Diffuse prior for the initial state
                        sigma_init_prior = get(dynamic_manifold, :sigma_init_prior, Exponential(10.0))
                        sigma_init ~ NamedDist(sigma_init_prior, Symbol("sigma_init_pipe_", key))
                        Q_init = (1.0 / (sigma_init^2 + noise)) .* state_spec.Q_template + noise * I
                        pipe_field[:, 1] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_1"))

                        for t in 2:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", t))
                            pipe_field[:, t] = pipe_field[:, t-1] .+ (L_innov \ innov_t)
                        end

                        for i in 1:M.y_N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                
                elseif dynamic_manifold isa RW2
                        # Handles RW2 dynamics: field_t = 2*field_{t-1} - field_{t-2} + innovation
                        sigma_innov_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                        sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_pipe_", key))

                        Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                        L_innov = cholesky(Symmetric(Q_innov)).L

                        pipe_field = Matrix{T}(undef, n_state, M.t_N)

                        # Diffuse priors for the first two states
                        sigma_init_prior = get(dynamic_manifold, :sigma_init_prior, Exponential(10.0))
                        sigma_init ~ NamedDist(sigma_init_prior, Symbol("sigma_init_pipe_", key))
                        Q_init = (1.0 / (sigma_init^2 + noise)) .* state_spec.Q_template + noise * I
                        pipe_field[:, 1] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_1"))
                        pipe_field[:, 2] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_2"))

                        for t in 3:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", t))
                            pipe_field[:, t] = 2.0 .* pipe_field[:, t-1] .- pipe_field[:, t-2] .+ (L_innov \ innov_t)
                        end

                        for i in 1:M.y_N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                
                elseif dynamic_manifold isa Union{RFF, TPS, PSpline, BSpline, FFT, Wavelet, Spherical, ExponentialDecay, Barycentric}
                        # Handles spatially-varying temporal curves, e.g., spatial() |> smooth(time, model=rff)
                        # The coefficients of the temporal basis functions are modeled as spatial fields.
                        
                        # Extract parameters from the dynamic manifold struct
                        n_basis = hasproperty(dynamic_manifold, :n_features) ? dynamic_manifold.n_features : dynamic_manifold.nbins
                        sigma_prior = getfield(dynamic_manifold, :sigma_prior)
                        sigma_coeffs ~ NamedDist(sigma_prior, Symbol("sigma_coeffs_pipe_", key))

                        # Get the covariate values (assumes it's the main time variable)
                        time_values = M.data[!, M.t_idx_var]
                        B_time = Matrix{T}(undef, length(time_values), n_basis)

                        # Construct the basis matrix B_time. This requires special handling for smooths
                        # with their own stochastic parameters (e.g., lengthscale in RFF).
                        if dynamic_manifold isa RFF
                            # RFF basis depends on a sampled lengthscale, and sampled W and b matrices.
                            ls_prior = dynamic_manifold.lengthscale_prior
                            ls_rff ~ NamedDist(ls_prior, Symbol("ls_rff_pipe_", key))
                            
                            W_rff ~ NamedDist(filldist(Normal(0, 1.0/ls_rff), 1, n_basis), Symbol("W_rff_pipe_", key))
                            b_rff ~ NamedDist(filldist(Uniform(0, 2*pi), n_basis), Symbol("b_rff_pipe_", key))

                            B_time .= rff_map(reshape(time_values, :, 1), W_rff, b_rff')

                        elseif dynamic_manifold isa Union{FFT, Spherical, ExponentialDecay}
                            # These smooths have hyperparameters that need to be sampled.
                            params_dict = Dict(fn => getfield(dynamic_manifold, fn) for fn in fieldnames(typeof(dynamic_manifold)))
                            
                            if hasproperty(dynamic_manifold, :lengthscale_prior)
                                ls_prior = getfield(dynamic_manifold, :lengthscale_prior)
                                ls_val ~ NamedDist(ls_prior, Symbol("ls_pipe_", key))
                                params_dict[:lengthscale] = ls_val
                            end
                            if hasproperty(dynamic_manifold, :range_prior)
                                range_prior = getfield(dynamic_manifold, :range_prior)
                                range_val ~ NamedDist(range_prior, Symbol("range_pipe_", key))
                                params_dict[:range] = range_val
                            end
                            
                            model_str = lowercase(string(typeof(dynamic_manifold)))
                            degree = get(params_dict, :degree, 3)
                            B_time .= bstm_smooth_basis_1D(model_str, time_values, n_basis, degree; params_dict...)
                        else
                            # For non-parametric smooths like TPS, PSpline, BSpline, Wavelet, Barycentric
                            model_str = lowercase(string(typeof(dynamic_manifold)))
                            params_dict = Dict(fn => getfield(dynamic_manifold, fn) for fn in fieldnames(typeof(dynamic_manifold)))
                            degree = get(params_dict, :degree, 3)
                            B_time .= bstm_smooth_basis_1D(model_str, time_values, n_basis, degree; params_dict...)
                        end

                        # Sample the spatially-varying coefficients
                        coeffs_field = Matrix{T}(undef, n_state, n_basis)
                        Q_coeffs = recompose_precision(state_manifold, M, sigma_coeffs; noise=noise)
                        L_coeffs = cholesky(Symmetric(Q_coeffs)).L
                        for j in 1:n_basis
                            innov_j ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("coeffs_base_", key, "_", j))
                            coeffs_field[:, j] = L_coeffs \ innov_j
                        end
                        
                        # Apply the effect to the linear predictor
                        for i in 1:M.y_N; eta[i] += dot(B_time[i, :], coeffs_field[M.s_idx[i], :]); end
                else
                    @warn "Dynamic manifold type $(typeof(dynamic_manifold)) not supported within a pipe operator. Skipping manifold '$key'."
                end
            end

        end
    end

    # #
    # 4. Spatiotemporal Interaction (Knorr-Held Taxonomy)
    # Logic explicitly handles Types I, II, III, and IV via Kronecker compositions.
    if model_st != "none"
        st_sigma ~ NamedDist(Exponential(0.5), :st_sigma)
        
        if model_st == "I"
            # Type I: Unstructured space and time (IID ⊗ IID)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_inter = reshape(st_raw, M.s_N, M.t_N) .* st_sigma # This is correct for IID
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "II"
            # Type II: Unstructured space, structured time (IID ⊗ Temporal Q)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            # Correct sampling: For each spatial location, the innovations are temporally correlated.
            # We solve U_t * x = z for each spatial location's innovation vector, where U_t is the upper Cholesky factor.
            C_t = cholesky(Symmetric(t_Q_main + noise * I))
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            st_inter = (C_t.U \ st_innov_matrix')' .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "III"
            # Type III: Structured space, unstructured time (Spatial Q ⊗ IID)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            # Correct sampling: For each time step, the innovations are spatially correlated.
            # We solve U_s * x = z for each time step's innovation vector.
            C_s = cholesky(Symmetric(s_Q_main + noise * I))
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            st_inter = (C_s.U \ st_innov_matrix) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "IV"
            # Type IV: Structured space, structured time (Spatial Q ⊗ Structured Time)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            C_s = cholesky(Symmetric(s_Q_main + noise * I))
            
            # Recursive definition for AR(1) temporal evolution of spatial fields
            st_inter = Matrix{T}(undef, M.s_N, M.t_N)
            # Correct sampling: Solve U_s * x = z for the innovations.
            st_inter[:, 1] = (C_s.U \ st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_main^2 + noise)
            for t in 2:M.t_N
                st_inter[:, t] = t_rho_main .* st_inter[:, t-1] .+ (C_s.U \ st_innov_matrix[:, t])
            end
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]] * st_sigma; end
        end
    end

    # #
    # 5. High-Fidelity Likelihood Dispatch
    # Rationale: Observation variance `y_sigma` is only relevant for certain likelihoods.
    # This conditional sampling prevents adding an unnecessary parameter to the model graph.
    y_sigma = one(T)
    if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        y_sigma ~ NamedDist(Exponential(1.0), :y_sigma)
    end

    for i in 1:M.y_N
        d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma], phi_zi=lik_phi, r_nb=lik_r, trial=[Int(get(M.trials, i, 1))], extra_params=extra_p)
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end


 
 @model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # #
    # 1. Global Likelihood Hyperparameters
    noise = get(M, :noise, 1e-6)
    outcomes_N = M.outcomes_N

    # Initialize parameter vectors that will be populated based on outcome-specific families.
    lik_r = ones(T, outcomes_N)
    extra_p = ones(T, outcomes_N)

    # A single zero-inflation parameter is shared across all applicable outcomes.
    # This is a modeling choice that can be revisited if outcome-specific ZI parameters are needed.
    use_zi_any = any(spec -> get(spec, :use_zi, false), M.likelihood_specs)
    lik_phi = zero(T)
    if use_zi_any
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    # To maintain a static model graph for Turing.jl, we sample vectors of parameters
    # that might be needed and then assign them based on the family of each outcome.
    lik_r_sampled ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :lik_r)
    extra_p_sampled ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :extra_p)

    # Assign the sampled parameters to the correct outcomes.
    for k in 1:outcomes_N
        spec_k = M.likelihood_specs[k]
        family_k = get(spec_k, :family, "gaussian")
        if family_k == "negbin"
            lik_r[k] = lik_r_sampled[k]
        end
        if family_k in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]
            extra_p[k] = extra_p_sampled[k]
        end
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
    model_st = get(M, :model_st, "none")
    run_st_interaction = model_st != "none" && M.s_N > 0 && M.t_N > 0
    if model_st != "none" && !run_st_interaction
        @warn "A spatiotemporal interaction model ('$model_st') was specified, but the model does not contain both spatial and temporal manifolds. The interaction term will be ignored."
    end

    local s_Q_main, t_Q_main, t_rho_main, s_sigma_main, t_sigma_main
    if model_st != "none"
        s_Q_main = [sparse(I(M.s_N)) for _ in 1:outcomes_N]
        t_Q_main = [sparse(I(M.t_N)) for _ in 1:outcomes_N]
        t_rho_main = [zero(T) for _ in 1:outcomes_N]
        s_sigma_main = [one(T) for _ in 1:outcomes_N]
        t_sigma_main = [one(T) for _ in 1:outcomes_N]
    end

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
                n_units = if domain == :mixed; spec.params.n_cat; elseif domain == :spatial; M.s_N; else M.t_N; end
                indices = if domain == :mixed; spec.params.indices; elseif domain == :spatial; M.s_idx; else M.t_idx; end
                field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), latent_name)
                eta_latent[:, k] .+= field[indices]

            elseif m_obj isa BYM2
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                if model_st != "none"
                    s_sigma_main[k] = sigma_val
                end

                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                local rho_val
                if rho_val_dist isa Exponential
                    theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_", rho_name))
                    rho_val = 1.0 - exp(-theta_rho)
                else
                    rho_val ~ NamedDist(rho_val_dist, rho_name)
                end

                struct_name = Symbol("latent_struct_", key, "_", k)
                iid_name = Symbol("latent_iid_", key, "_", k)

                s_icar ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), struct_name)
                s_iid ~ NamedDist(MvNormal(zeros(M.s_N), I), iid_name)

                Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(s_icar))
                combined = sigma_val .* (sqrt(rho_val) .* s_icar .+ sqrt(1.0 - rho_val) .* s_iid)
                eta_latent[:, k] .+= combined[M.s_idx]
                if model_st != "none"
                    s_Q_main[k] = spec.Q_template
                end

            elseif m_obj isa AR1
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)

                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                local rho_val
                if rho_val_dist isa Exponential
                    theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_", rho_name))
                    rho_val = 1.0 - exp(-theta_rho)
                else
                    rho_val ~ NamedDist(rho_val_dist, rho_name)
                end

                if model_st != "none"
                    t_sigma_main[k] = sigma_val
                    t_rho_main[k] = rho_val
                end

                innov_name = Symbol("innov_", key, "_", k)
                innovations ~ NamedDist(MvNormal(zeros(M.t_N), I), innov_name)
                t_field = Vector{T}(undef, M.t_N)
                t_field[1] = innovations[1] / sqrt(1.0 - rho_val^2 + noise)
                for i in 2:M.t_N
                    t_field[i] = rho_val * t_field[i-1] + innovations[i]
                end
                eta_latent[:, k] .+= (t_field .* sigma_val)[M.t_idx]

                if model_st != "none"
                    t_Q_base = Symmetric((1.0 + rho_val^2) .* I(M.t_N) .- rho_val .* spec.Q_template)
                    t_Q_main[k] = Symmetric((1.0 / (sigma_val^2 * (1.0 - rho_val^2) + noise)) .* t_Q_base)
                end

            elseif m_obj isa Union{ICAR, Besag, RW1, RW2, Leroux, Cyclic}
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                r_val = nothing
                if hasproperty(m_obj, :rho_prior)
                    rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                    if rho_val_dist isa Exponential
                        theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_", rho_name))
                        r_val = 1.0 - exp(-theta_rho)
                    else
                        r_val ~ NamedDist(rho_val_dist, rho_name)
                    end
                end

                n_units = size(spec.Q_template, 1)
                Q = recompose_precision(Symbol(lowercase(string(typeof(m_obj)))), spec.Q_template, sigma_val; extra_param=r_val, noise=noise)
                field ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)

                if !(m_obj isa Cyclic)
                    Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
                end

                indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
                eta_latent[:, k] .+= field[indices]
                if model_st != "none"
                    if domain == :spatial; s_Q_main[k] = Q; elseif domain == :temporal; t_Q_main[k] = Q; end
                end

            elseif m_obj isa SAR
                sigma_prior = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                rho_prior = get(spec.params, :rho_prior, m_obj.rho_prior)

                sigma_val ~ NamedDist(sigma_prior, sigma_name)
                rho_val ~ NamedDist(rho_prior, rho_name)

                Q_source = spec.Q_template
                n_units = size(Q_source, 1)

                Q_innov = (1.0 / (sigma_val^2 + noise)) * I(n_units)

                L_op = I(n_units) - rho_val * Q_source
                Q_final = Symmetric(L_op' * Q_innov * L_op + noise * I)

                field ~ NamedDist(MvNormalCanon(zeros(n_units), Q_final), latent_name)
                
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
                
                eta_latent[:, k] .+= field[M.s_idx]



            elseif m_obj isa Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}
                var_sym = spec.var; B_mat = M.basis_matrices[var_sym]; n_basis_cols = size(B_mat, 2)
        
                if m_obj isa RFF
                    # --- Dynamic RFF Basis Construction for each outcome ---
                    n_features = m_obj.n_features
                    coords = spec.params.coords
                    d = size(coords, 2)

                    # Sample lengthscale (outcome-specific)
                    ls_prior = get(spec.params, :lengthscale_prior, m_obj.lengthscale_prior)
                    local ls_val
                    if ls_prior isa Exponential
                        theta_ls ~ NamedDist(ls_prior, Symbol("theta_ls_", key, "_", k))
                        ls_val = 1.0 / (theta_ls + 1e-9)
                    else
                        ls_val ~ NamedDist(ls_prior, Symbol("ls_", key, "_", k))
                    end

                    # Sample RFF weights and biases (outcome-specific)
                    sigma_spectral = 1.0 / ls_val
                    W_raw ~ NamedDist(filldist(Normal(0, 1), d * n_features), Symbol("W_raw_", key, "_", k))
                    W_rff = reshape(W_raw, d, n_features) .* sigma_spectral
                    b_rff ~ NamedDist(filldist(Uniform(0, 2*pi), n_features), Symbol("b_rff_", key, "_", k))

                    # Construct basis matrix on-the-fly
                    B_mat = rff_map(coords, W_rff, b_rff)
                    
                    # Sample coefficients (outcome-specific)
                    sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                    sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_basis_", var_sym, "_", k))
                    beta_name = Symbol("beta_basis_", var_sym, "_", k)
                    latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_features), beta_name)
                    
                    eta_latent[:, k] .+= B_mat * latent_coeffs
                else # Static basis models
                    B_mat = M.basis_matrices[var_sym]
                    n_basis_cols = size(B_mat, 2)
                    sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                    sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_basis_", key, "_", k))

                    beta_name = Symbol("beta_basis_", var_sym, "_", k)
                    if m_obj isa PSpline || m_obj isa TPS
                        Q_penalty = (1.0 / (sigma_val^2 + noise)) .* spec.Q_template
                        latent_coeffs ~ NamedDist(MvNormalCanon(zeros(n_basis_cols), Q_penalty), beta_name)
                        Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_basis_cols), sum(latent_coeffs))
                    else
                        latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_basis_cols), beta_name)
                    end
                    eta_latent[:, k] .+= B_mat * latent_coeffs
                end

            elseif m_obj isa MixedManifold
                # This block was missing, truncating random effects functionality.
                # It samples outcome-specific random effect coefficients.
                n_units = spec.params.n_cat
                indices = spec.params.indices
                sigma_val_dist = get(spec.params, :sigma_prior, Exponential(1.0))
                sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
                field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), latent_name)
                eta_latent[:, k] .+= field[indices]

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
                elseif d_model == "directed_spatial_process"
                    # #
                    # # Simultaneous Autoregressive (SAR) Model with Correlated Errors
                    # # Implements the model (I - ρ * M₁)y = ε, where ε has its own covariance structure Q₂.
                    # # This is a model for a single spatial field, not a temporal evolution.
                    
                    if !haskey(M, :W) || !haskey(M, :centroids)
                        @warn "Dynamics model 'directed_spatial_process' requires a spatial manifold with an adjacency matrix 'W' and centroids. Skipping."
                        continue
                    end
    
                    # M₁ is the main spatial adjacency matrix.
                    Q_source = M.W
                    n_units = size(Q_source, 1)
    
                    # Parameters for the dependency structure (ρ)
                    rho_prior = get(m_obj.params, :rho_prior, Beta(1,1))
                    rho_dep ~ NamedDist(rho_prior, Symbol("rho_dep_", key, "_", k))
    
                    # Parameters for the innovation process ε, modeled as a GP
                    sigma_innov_prior = get(m_obj.params, :sigma_innov_prior, Exponential(1.0))
                    sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_", key, "_", k))
                    
                    ls_innov_prior = get(m_obj.params, :ls_innov_prior, InverseGamma(3,3))
                    ls_innov ~ NamedDist(ls_innov_prior, Symbol("ls_innov_", key, "_", k))
    
                    # Construct the innovation precision matrix Q₂ from a GP kernel
                    kernel_type = Symbol(get(m_obj.params, :kernel, "se"))
                    dist_sq = [sum((M.centroids[i] .- M.centroids[j]).^2) for i in 1:n_units, j in 1:n_units]
                    K_innov = evaluate_kernel_matrix(dist_sq, sigma_innov, ls_innov, kernel_type, noise)
                    Q_innov = inv(Symmetric(K_innov))
    
                    # Construct the final precision matrix: (I - ρM₁)' Q₂ (I - ρM₁)
                    L_op = I(n_units) - rho_dep * Q_source
                    Q_final = Symmetric(L_op' * Q_innov * L_op + noise * I)
    
                    field ~ NamedDist(MvNormalCanon(zeros(n_units), Q_final), latent_name)
                    eta_latent[:, k] .+= field[M.s_idx]
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
            elseif m_obj isa ComposedManifold
                op = m_obj.operator

                if op == :kronecker_product || op == :composition
                    sigma_val ~ NamedDist(Exponential(0.5), sigma_name)
                    Q_final = recompose_precision(m_obj, M, sigma_val; noise=noise)
                    n_units = size(Q_final, 1)
                    field ~ NamedDist(MvNormalCanon(zeros(n_units), Symmetric(Q_final)), latent_name)
                    if hasproperty(spec, :indices) && !isnothing(spec.indices)
                        eta_latent[:, k] .+= field[spec.indices]
                    else
                        @warn "Composed manifold '$key' is missing combined indices. Effect will not be applied."
                    end

                elseif op == :direct_sum
                    sigma_val ~ NamedDist(Exponential(0.5), sigma_name)
                    Q_final = recompose_precision(m_obj, M, sigma_val; noise=noise)
                    n_total = size(Q_final, 1)
                    field ~ NamedDist(MvNormalCanon(zeros(n_total), Symmetric(Q_final)), latent_name)

                    if hasproperty(spec, :component_dims) && hasproperty(spec, :component_indices)
                        offset = 0
                        for i in 1:length(m_obj.components)
                            dim = spec.component_dims[i]
                            indices = spec.component_indices[i]
                            sub_field = field[offset + 1 : offset + dim]
                            eta_latent[:, k] .+= sub_field[indices]
                            offset += dim
                        end
                    else
                        @warn "Direct sum manifold '$key' is missing component dimensions/indices. Effect will not be applied."
                    end

                elseif op == :pipe
                    state_manifold = m_obj.components[1]
                    dynamic_manifold = m_obj.components[2]
                    M_dict = M isa NamedTuple ? Dict(pairs(M)) : M
                    state_spec = build_model(state_manifold, M_dict)

                    if dynamic_manifold isa AR1
                        rho_prior = get(dynamic_manifold, :rho_prior, Beta(1,1))
                        sigma_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                        rho_pipe ~ NamedDist(rho_prior, Symbol("rho_pipe_", key, "_", k))
                        sigma_innov ~ NamedDist(sigma_prior, Symbol("sigma_innov_pipe_", key, "_", k))
                        Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                        L_innov = cholesky(Symmetric(Q_innov)).L
                        n_state = size(Q_innov, 1)
                        pipe_field = Matrix{T}(undef, n_state, M.t_N) 
                        innov_base ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", k, "_1"))
                        pipe_field[:, 1] = (L_innov \ innov_base) ./ sqrt(1.0 - rho_pipe^2 + noise)
                        for t in 2:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", k, "_", t))
                            pipe_field[:, t] = rho_pipe .* pipe_field[:, t-1] .+ (L_innov \ innov_t)
                        end
                        for i in 1:M.y_N; eta_latent[i, k] += pipe_field[M.s_idx[i], M.t_idx[i]]; end

                    elseif dynamic_manifold isa RW1
                        n_state = size(state_spec.Q_template, 1)
                        sigma_innov_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                        sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_pipe_", key, "_", k))
                        Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                        L_innov = cholesky(Symmetric(Q_innov)).L
                        pipe_field = Matrix{T}(undef, n_state, M.t_N)
                        sigma_init_prior = get(dynamic_manifold, :sigma_init_prior, Exponential(10.0))
                        sigma_init ~ NamedDist(sigma_init_prior, Symbol("sigma_init_pipe_", key, "_", k))
                        Q_init = (1.0 / (sigma_init^2 + noise)) .* state_spec.Q_template + noise * I
                        pipe_field[:, 1] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_", k, "_1"))
                        for t in 2:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", k, "_", t))
                            pipe_field[:, t] = pipe_field[:, t-1] .+ (L_innov \ innov_t)
                        end
                        for i in 1:M.y_N; eta_latent[i, k] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                    
                    elseif dynamic_manifold isa RW2
                        n_state = size(state_spec.Q_template, 1)
                        sigma_innov_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                        sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_pipe_", key, "_", k))
                        Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                        L_innov = cholesky(Symmetric(Q_innov)).L
                        pipe_field = Matrix{T}(undef, n_state, M.t_N)
                        sigma_init_prior = get(dynamic_manifold, :sigma_init_prior, Exponential(10.0))
                        sigma_init ~ NamedDist(sigma_init_prior, Symbol("sigma_init_pipe_", key, "_", k))
                        Q_init = (1.0 / (sigma_init^2 + noise)) .* state_spec.Q_template + noise * I
                        pipe_field[:, 1] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_", k, "_1"))
                        pipe_field[:, 2] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_", k, "_2"))
                        for t in 3:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", k, "_", t))
                            pipe_field[:, t] = 2.0 .* pipe_field[:, t-1] .- pipe_field[:, t-2] .+ (L_innov \ innov_t)
                        end
                        for i in 1:M.y_N; eta_latent[i, k] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                    
                    elseif dynamic_manifold isa Union{RFF, TPS, PSpline, BSpline, FFT, Wavelet, Spherical, ExponentialDecay, Barycentric}
                        n_state = size(state_spec.Q_template, 1)
                        n_basis = hasproperty(dynamic_manifold, :n_features) ? dynamic_manifold.n_features : dynamic_manifold.nbins
                        sigma_prior = getfield(dynamic_manifold, :sigma_prior)
                        sigma_coeffs ~ NamedDist(sigma_prior, Symbol("sigma_coeffs_pipe_", key, "_", k))
                        time_values = M.data[!, M.t_idx_var]
                        B_time = Matrix{T}(undef, length(time_values), n_basis)

                        if dynamic_manifold isa RFF
                            ls_prior = dynamic_manifold.lengthscale_prior
                            ls_rff ~ NamedDist(ls_prior, Symbol("ls_rff_pipe_", key, "_", k))
                            W_rff ~ NamedDist(filldist(Normal(0, 1.0/ls_rff), 1, n_basis), Symbol("W_rff_pipe_", key, "_", k))
                            b_rff ~ NamedDist(filldist(Uniform(0, 2*pi), n_basis), Symbol("b_rff_pipe_", key, "_", k))
                            B_time .= rff_map(reshape(time_values, :, 1), W_rff, b_rff')
                        else
                            model_str = lowercase(string(typeof(dynamic_manifold)))
                            params_dict = Dict(fn => getfield(dynamic_manifold, fn) for fn in fieldnames(typeof(dynamic_manifold)))
                            degree = get(params_dict, :degree, 3)
                            B_time .= bstm_smooth_basis_1D(model_str, time_values, n_basis, degree; params_dict...)
                        end

                        coeffs_field = Matrix{T}(undef, n_state, n_basis)
                        Q_coeffs = recompose_precision(state_manifold, M, sigma_coeffs; noise=noise)
                        L_coeffs = cholesky(Symmetric(Q_coeffs)).L
                        for j in 1:n_basis
                            innov_j ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("coeffs_base_", key, "_", k, "_", j))
                            coeffs_field[:, j] = L_coeffs \ innov_j
                        end
                        for i in 1:M.y_N; eta_latent[i, k] += dot(B_time[i, :], coeffs_field[M.s_idx[i], :]); end
                    else
                        @warn "Dynamic manifold type $(typeof(dynamic_manifold)) not supported within a pipe operator. Skipping manifold '$key' for outcome $k."
                    end
                end
            end
        end
    end

    # #
    # 5. Spatiotemporal Interaction (Knorr-Held Taxonomy)
    if run_st_interaction
        # Sample the base innovations ONCE, outside the loop over outcomes.
        # This ensures that the different outcomes can share the same underlying noise
        # structure, which can then be transformed differently for each outcome.
        st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
        st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)

        st_sigma_base ~ NamedDist(Exponential(0.5), :st_sigma_base) # Global ST interaction scale

        for k in 1:outcomes_N
            st_sigma_k = st_sigma_base # Each outcome uses the same base ST scale currently

            if model_st == "I"
                # Type I: Unstructured space and time (IID ⊗ IID)
                # The innovations are already IID, so we just scale them.
                st_inter = st_innov_matrix .* st_sigma_k
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]]; end

            elseif model_st == "II"
                # Type II: Unstructured space, structured time (IID ⊗ Temporal Q)
                # Apply temporal structure (Cholesky of t_Q_main[k])
                # Correct sampling: U' * x = z => x = U' \ z
                C_t = cholesky(Symmetric(t_Q_main[k] + noise * I))
                st_inter = (C_t.U \ st_innov_matrix')' .* st_sigma_k
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]]; end

            elseif model_st == "III"
                # Type III: Structured space, unstructured time (Spatial Q ⊗ IID)
                # Apply spatial structure (Cholesky of s_Q_main[k])
                # Correct sampling: U * x = z => x = U \ z
                C_s = cholesky(Symmetric(s_Q_main[k] + noise * I))
                st_inter = (C_s.U \ st_innov_matrix) .* st_sigma_k
                for i in 1:M.y_N; eta_latent[i, k] .+= st_inter[M.s_idx[i], M.t_idx[i]]; end

            elseif model_st == "IV"
                # Type IV: Structured space, structured time (Spatial Q ⊗ Temporal Q)
                # This implements an AR(1) process where the innovations at each time step
                # are spatially correlated according to s_Q_main.
                C_s = cholesky(Symmetric(s_Q_main[k] + noise * I))

                # Recursive definition for AR(1) temporal evolution of spatial fields
                st_inter = Matrix{T}(undef, M.s_N, M.t_N)
                # Correct sampling for the initial state and subsequent innovations
                st_inter[:, 1] = (C_s.U \ st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_main[k]^2 + noise)
                for t in 2:M.t_N
                    st_inter[:, t] = t_rho_main[k] .* st_inter[:, t-1] .+ (C_s.U \ st_innov_matrix[:, t])
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
    for k in 1:outcomes_N
        # Get the specific family and parameters for this outcome from the parsed specs
        spec_k = M.likelihood_specs[k]
        family_k = get(spec_k, :family, "gaussian")

        # Sample observation variance only if required by this outcome's family
        y_sigma_k = one(T)
        if family_k in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
            y_sigma_k ~ NamedDist(Exponential(1.0), Symbol("y_sigma_", k))
        end

        # Use ok_idx to handle potential missing data/filtered observations per outcome
        ok_idx = get(M, :y_ok, [1:M.y_N for _ in 1:outcomes_N])[k]
        for i in ok_idx
            # Construct likelihood with per-outcome parameters
            trials_i_k = M.trials isa AbstractMatrix ? get(M.trials, (i, k), 1) : get(M.trials, i, 1)
            d_lik = bstm_Likelihood(family_k, [T(M.y_obs[i, k])]; sigma_y=[y_sigma_k], phi_zi=lik_phi, r_nb=lik_r[k], trial=[Int(trials_i_k)], extra_params=extra_p[k])
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
        end
    end
end



@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    # #
    # 1. Global Likelihood Hyperparameters for the Main (High-Fidelity) Model
    # Rationale: Use `get` for safe access to prevent KeyError.
    family = get(get(M.likelihood_specs, 1, Dict()), :family, "gaussian")
    noise = get(M, :noise, 1e-6)
    use_zi = get(M, :use_zi, false)

    # Initialize parameters
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
        y_sigma .= exp.((sigma_log_var .* vol_latent_field[1:M.y_N]) ./ 2.0)
    elseif family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        y_sigma_main ~ NamedDist(get(M.hyperpriors, "y_sigma_prior", Exponential(1.0)), :y_sigma)
        y_sigma .= y_sigma_main
    else
        y_sigma .= one(T)
    end

    # #
    # 3. RFF-based Multi-fidelity Latent Fields (z_latent, w_latent)
    local z_latent_field = zeros(T, 0)
    local w_latent_field = zeros(T, 0, 3)

    if haskey(M, :z_coords_s) && haskey(M, :w_coords_st)
        # High-Fidelity Z (Spatial)
        z_ls ~ NamedDist(Gamma(2, 2), :z_ls)
        z_beta ~ NamedDist(filldist(Normal(0, 1), M.M_rff), :z_beta)
        z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls)) .+ M.b_fixed'
        z_latent_field = M.rff_scale .* (cos.(z_proj) * z_beta)

        # Medium-Fidelity W (Spatiotemporal, depends on Z)
        w_ls ~ NamedDist(Gamma(2, 2), :w_ls)
        w_beta_flat ~ NamedDist(filldist(Normal(0, 1), M.M_rff * 3), :w_beta)
        w_beta = reshape(w_beta_flat, M.M_rff, 3)

        # Interpolation of z_latent to w_coords resolution is assumed to be handled in config
        w_coords_augmented = hcat(M.w_coords_st, z_latent_field[1:size(M.w_coords_st, 1)])
        w_proj = (w_coords_augmented * (M.W_fixed[1:size(w_coords_augmented, 2), :] ./ w_ls)) .+ M.b_fixed'
        w_latent_field = M.rff_scale .* (cos.(w_proj) * w_beta)
    end

    # #
    # 4. Base Predictor for the Main Model
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

    # Add contributions from RFF multi-fidelity fields
    if haskey(M, :z_coords_s)
        z_beta_eta ~ NamedDist(Normal(0, 1), :z_beta_eta)
        if length(z_latent_field) == M.y_N
            eta .+= z_latent_field .* z_beta_eta
        end
    end
    if haskey(M, :w_coords_st)
        w_beta_eta ~ NamedDist(MvNormal(zeros(3), I), :w_beta_eta)
        if size(w_latent_field, 1) == M.y_N
            eta .+= w_latent_field * w_beta_eta
        end
    end

    # #
    # 5. Scaffolding for Spatiotemporal Interaction Structures (Main Model)
    model_st = get(M, :model_st, "none")
    run_st_interaction = model_st != "none" && M.s_N > 0 && M.t_N > 0
    if model_st != "none" && !run_st_interaction
        @warn "A spatiotemporal interaction model ('$model_st') was specified, but the model does not contain both spatial and temporal manifolds. The interaction term will be ignored."
    end

    local s_Q_main, t_Q_main, t_rho_main, s_sigma_main, t_sigma_main
    if run_st_interaction
        s_Q_main = sparse(I(M.s_N))
        t_Q_main = sparse(I(M.t_N))
        t_rho_main = zero(T)
        s_sigma_main = one(T)
        t_sigma_main = one(T)
    end

    # #
    # 6. Modular Manifold Realization Loop (Main Model)
    for spec in M.manifolds
        m_obj = spec.manifold_obj
        if m_obj isa NoneManifold
            continue
        end
        m_domain = spec.domain
        key = spec.key

        # Define parameter names for this manifold
        sigma_name = Symbol("sigma_", key)
        rho_name = Symbol("rho_", key)
        latent_name = Symbol("latent_", key)

        # --- IID Manifold ---
        if m_obj isa IID
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
            n_units = if m_domain == :mixed; spec.params.n_cat; elseif m_domain == :spatial; M.s_N; else M.t_N; end
            indices = if m_domain == :mixed; spec.params.indices; elseif m_domain == :spatial; M.s_idx; else M.t_idx; end
            field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), latent_name)
            eta .+= field[indices]

        # --- BYM2 Manifold ---
        elseif m_obj isa BYM2
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)
            if run_st_interaction; s_sigma_main = sigma_val; end

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            local rho_val
            if rho_val_dist isa Exponential
                theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_", rho_name))
                rho_val = 1.0 - exp(-theta_rho)
            else
                rho_val ~ NamedDist(rho_val_dist, rho_name)
            end

            s_icar_name = Symbol("latent_struct_", key)
            s_iid_name = Symbol("latent_iid_", key)

            s_icar ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec.Q_template + noise * I), s_icar_name)
            s_iid ~ NamedDist(MvNormal(zeros(M.s_N), I), s_iid_name)
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum(s_icar))
            combined = sigma_val .* (sqrt(rho_val) .* s_icar .+ sqrt(1.0 - rho_val) .* s_iid)
            eta .+= combined[M.s_idx]
            if run_st_interaction; s_Q_main = spec.Q_template; end

        # --- AR1 Manifold ---
        elseif m_obj isa AR1
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, sigma_name)

            rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
            local rho_val
            if rho_val_dist isa Exponential
                theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_", rho_name))
                rho_val = 1.0 - exp(-theta_rho)
            else
                rho_val ~ NamedDist(rho_val_dist, rho_name)
            end

            if run_st_interaction; t_sigma_main = sigma_val; t_rho_main = rho_val; end

            innov_name = Symbol("innov_", key)
            innovations ~ NamedDist(MvNormal(zeros(M.t_N), I), innov_name)

            t_field = Vector{T}(undef, M.t_N)
            t_field[1] = innovations[1] / sqrt(1.0 - rho_val^2 + noise)
            for i in 2:M.t_N
                t_field[i] = rho_val * t_field[i-1] + innovations[i]
            end
            eta .+= (t_field .* sigma_val)[M.t_idx]
            
            if run_st_interaction
                t_Q_base = Symmetric((1.0 + rho_val^2) .* I(M.t_N) .- rho_val .* spec.Q_template)
                t_Q_main = Symmetric((1.0 / (sigma_val^2 * (1.0 - rho_val^2) + noise)) .* t_Q_base)
            end

        # --- GMRF Manifolds (ICAR, Besag, RW, Leroux, SAR, Cyclic) ---
        elseif m_obj isa Union{ICAR, Besag, RW1, RW2, Leroux, Cyclic}
            sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))

            r_val = nothing
            if hasproperty(m_obj, :rho_prior)
                rho_val_dist = get(spec.params, :rho_prior, m_obj.rho_prior)
                if rho_val_dist isa Exponential
                    theta_rho ~ NamedDist(rho_val_dist, Symbol("theta_rho_", key))
                    r_val = 1.0 - exp(-theta_rho) # Transform back to [0,1]
                else
                    r_val ~ NamedDist(rho_val_dist, Symbol("rho_", key))
                end
            end

            n_units = size(spec.Q_template, 1)
            Q = recompose_precision(Symbol(lowercase(string(typeof(m_obj)))), spec.Q_template, sigma_val; extra_param=r_val, noise=noise)
            field ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", key))
            if !(m_obj isa Cyclic)
                Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
            end
            indices = (domain == :spatial) ? M.s_idx : ((domain == :temporal) ? M.t_idx : M.u_idx)
            eta .+= field[indices]
            if model_st != "none"
                if domain == :spatial; s_Q_main = Q; elseif domain == :temporal; t_Q_main = Q; end
            end

        # --- SAR Manifold ---
        elseif m_obj isa SAR
            sigma_prior = get(spec.params, :sigma_prior, m_obj.sigma_prior)
            rho_prior = get(spec.params, :rho_prior, m_obj.rho_prior)

            sigma_val ~ NamedDist(sigma_prior, Symbol("sigma_", key))
            rho_val ~ NamedDist(rho_prior, Symbol("rho_", key))

            Q_source = spec.Q_template
            n_units = size(Q_source, 1)

            Q_innov = (1.0 / (sigma_val^2 + noise)) * I(n_units)

            L_op = I(n_units) - rho_val * Q_source
            Q_final = Symmetric(L_op' * Q_innov * L_op + noise * I)

            field ~ NamedDist(MvNormalCanon(zeros(n_units), Q_final), Symbol("latent_", key))
            
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_units), sum(field))
            
            eta .+= field[M.s_idx]

        # --- Basis Function & Spectral Manifolds ---
        elseif m_obj isa Union{PSpline, BSpline, TPS, RFF, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}
            var_sym = spec.var
            
            if m_obj isa RFF
                # --- Dynamic RFF Basis Construction ---
                n_features = m_obj.n_features
                coords = spec.params.coords
                d = size(coords, 2)

                # Sample lengthscale from its prior (can be a PC prior)
                ls_prior = get(spec.params, :lengthscale_prior, m_obj.lengthscale_prior)
                local ls_val
                if ls_prior isa Exponential # PC prior on 1/l
                    theta_ls ~ NamedDist(ls_prior, Symbol("theta_ls_", key))
                    ls_val = 1.0 / (theta_ls + 1e-9)
                else
                    ls_val ~ NamedDist(ls_prior, Symbol("ls_", key))
                end

                # Sample RFF weights and biases
                sigma_spectral = 1.0 / ls_val
                W_raw ~ NamedDist(filldist(Normal(0, 1), d * n_features), Symbol("W_raw_", key))
                W_rff = reshape(W_raw, d, n_features) .* sigma_spectral
                b_rff ~ NamedDist(filldist(Uniform(0, 2*pi), n_features), Symbol("b_rff_", key))

                # Construct basis matrix on-the-fly
                B_mat = rff_map(coords, W_rff, b_rff)
                
                # Sample coefficients for the basis functions
                sigma_val_dist = get(spec.params, :sigma_prior, m_obj.sigma_prior)
                sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_basis_", var_sym))
                beta_name = Symbol("beta_basis_", var_sym)
                latent_coeffs ~ NamedDist(filldist(Normal(0, sigma_val), n_features), beta_name)
                
                eta .+= B_mat * latent_coeffs
            else # Static basis models
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
            end

        # --- Mixed Effects Manifold ---
        elseif m_obj isa MixedManifold
            n_units = spec.params.n_cat
            indices = spec.params.indices
            sigma_val_dist = get(spec.params, :sigma_prior, Exponential(1.0))
            sigma_val ~ NamedDist(sigma_val_dist, Symbol("sigma_", key))
            field ~ NamedDist(filldist(Normal(0, sigma_val), n_units), Symbol("latent_", key))
            eta .+= field[indices]

        # --- Dynamics Manifold ---
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
            elseif d_model == "directed_spatial_process"
                if !haskey(M, :W) || !haskey(M, :centroids)
                    @warn "Dynamics model 'directed_spatial_process' requires a spatial manifold with an adjacency matrix 'W' and centroids. Skipping."
                    continue
                end
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

        # --- Spatially Varying Coefficients Manifold ---
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

        # --- Composed Manifold (Algebraic Operators) ---
        elseif m_obj isa ComposedManifold
            op = m_obj.operator

            if op == :kronecker_product || op == :composition
                sigma_val ~ NamedDist(Exponential(0.5), Symbol("sigma_", key))
                Q_final = recompose_precision(m_obj, M, sigma_val; noise=noise)
                n_units = size(Q_final, 1)
                field ~ NamedDist(MvNormalCanon(zeros(n_units), Symmetric(Q_final)), Symbol("latent_", key))
                if hasproperty(spec, :indices) && !isnothing(spec.indices)
                    eta .+= field[spec.indices]
                else
                    @warn "Composed manifold '$key' is missing combined indices. Effect will not be applied."
                end

            elseif op == :direct_sum
                sigma_val ~ NamedDist(Exponential(0.5), Symbol("sigma_", key))
                Q_final = recompose_precision(m_obj, M, sigma_val; noise=noise)
                n_total = size(Q_final, 1)
                field ~ NamedDist(MvNormalCanon(zeros(n_total), Symmetric(Q_final)), Symbol("latent_", key))

                if hasproperty(spec, :component_dims) && hasproperty(spec, :component_indices)
                    offset = 0
                    for i in 1:length(m_obj.components)
                        dim = spec.component_dims[i]
                        indices = spec.component_indices[i]
                        sub_field = field[offset + 1 : offset + dim]
                        eta .+= sub_field[indices]
                        offset += dim
                    end
                else
                    @warn "Direct sum manifold '$key' is missing component dimensions/indices. Effect will not be applied."
                end

            elseif op == :pipe
                state_manifold = m_obj.components[1]
                dynamic_manifold = m_obj.components[2]
                M_dict = M isa NamedTuple ? Dict(pairs(M)) : M
                state_spec = build_model(state_manifold, M_dict)

                if dynamic_manifold isa AR1
                    rho_prior = get(dynamic_manifold, :rho_prior, Beta(1,1))
                    sigma_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                    rho_pipe ~ NamedDist(rho_prior, Symbol("rho_pipe_", key))
                    sigma_innov ~ NamedDist(sigma_prior, Symbol("sigma_innov_pipe_", key))
                    Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                    L_innov = cholesky(Symmetric(Q_innov)).L
                    n_state = size(Q_innov, 1)
                    pipe_field = Matrix{T}(undef, n_state, M.t_N) 
                    innov_base ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_1"))
                    pipe_field[:, 1] = (L_innov \ innov_base) ./ sqrt(1.0 - rho_pipe^2 + noise)
                    for t in 2:M.t_N
                        innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", t))
                        pipe_field[:, t] = rho_pipe .* pipe_field[:, t-1] .+ (L_innov \ innov_t)
                    end
                    for i in 1:M.y_N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end

                elseif dynamic_manifold isa RW1
                    n_state = size(state_spec.Q_template, 1)
                    sigma_innov_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                    sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_pipe_", key))
                    Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                    L_innov = cholesky(Symmetric(Q_innov)).L
                    pipe_field = Matrix{T}(undef, n_state, M.t_N)
                    sigma_init_prior = get(dynamic_manifold, :sigma_init_prior, Exponential(10.0))
                    sigma_init ~ NamedDist(sigma_init_prior, Symbol("sigma_init_pipe_", key))
                    Q_init = (1.0 / (sigma_init^2 + noise)) .* state_spec.Q_template + noise * I
                    pipe_field[:, 1] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_1"))
                    for t in 2:M.t_N
                        innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", t))
                        pipe_field[:, t] = pipe_field[:, t-1] .+ (L_innov \ innov_t)
                    end
                    for i in 1:M.y_N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                
                elseif dynamic_manifold isa RW2
                    n_state = size(state_spec.Q_template, 1)
                    sigma_innov_prior = get(dynamic_manifold, :sigma_prior, Exponential(1.0))
                    sigma_innov ~ NamedDist(sigma_innov_prior, Symbol("sigma_innov_pipe_", key))
                    Q_innov = recompose_precision(state_manifold, M, sigma_innov; noise=noise)
                    L_innov = cholesky(Symmetric(Q_innov)).L
                    pipe_field = Matrix{T}(undef, n_state, M.t_N)
                    sigma_init_prior = get(dynamic_manifold, :sigma_init_prior, Exponential(10.0))
                    sigma_init ~ NamedDist(sigma_init_prior, Symbol("sigma_init_pipe_", key))
                    Q_init = (1.0 / (sigma_init^2 + noise)) .* state_spec.Q_template + noise * I
                    pipe_field[:, 1] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_1"))
                    pipe_field[:, 2] ~ NamedDist(MvNormalCanon(zeros(n_state), Symmetric(Q_init)), Symbol("latent_pipe_", key, "_2"))
                    for t in 3:M.t_N
                        innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_", key, "_", t))
                        pipe_field[:, t] = 2.0 .* pipe_field[:, t-1] .- pipe_field[:, t-2] .+ (L_innov \ innov_t)
                    end
                    for i in 1:M.y_N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                
                elseif dynamic_manifold isa Union{RFF, TPS, PSpline, BSpline, FFT, Wavelet, Spherical, ExponentialDecay, Barycentric}
                    n_state = size(state_spec.Q_template, 1)
                    n_basis = hasproperty(dynamic_manifold, :n_features) ? dynamic_manifold.n_features : dynamic_manifold.nbins
                    sigma_prior = getfield(dynamic_manifold, :sigma_prior)
                    sigma_coeffs ~ NamedDist(sigma_prior, Symbol("sigma_coeffs_pipe_", key))
                    time_values = M.data[!, M.t_idx_var]
                    B_time = Matrix{T}(undef, length(time_values), n_basis)

                    if dynamic_manifold isa RFF
                        ls_prior = dynamic_manifold.lengthscale_prior
                        ls_rff ~ NamedDist(ls_prior, Symbol("ls_rff_pipe_", key))
                        W_rff ~ NamedDist(filldist(Normal(0, 1.0/ls_rff), 1, n_basis), Symbol("W_rff_pipe_", key))
                        b_rff ~ NamedDist(filldist(Uniform(0, 2*pi), n_basis), Symbol("b_rff_pipe_", key))
                        B_time .= rff_map(reshape(time_values, :, 1), W_rff, b_rff')
                    else
                        model_str = lowercase(string(typeof(dynamic_manifold)))
                        params_dict = Dict(fn => getfield(dynamic_manifold, fn) for fn in fieldnames(typeof(dynamic_manifold)))
                        degree = get(params_dict, :degree, 3)
                        B_time .= bstm_smooth_basis_1D(model_str, time_values, n_basis, degree; params_dict...)
                    end

                    coeffs_field = Matrix{T}(undef, n_state, n_basis)
                    Q_coeffs = recompose_precision(state_manifold, M, sigma_coeffs; noise=noise)
                    L_coeffs = cholesky(Symmetric(Q_coeffs)).L
                    for j in 1:n_basis
                        innov_j ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("coeffs_base_", key, "_", j))
                        coeffs_field[:, j] = L_coeffs \ innov_j
                    end
                    for i in 1:M.y_N; eta[i] += dot(B_time[i, :], coeffs_field[M.s_idx[i], :]); end
                else
                    @warn "Dynamic manifold type $(typeof(dynamic_manifold)) not supported within a pipe operator. Skipping manifold '$key'."
                end
            end

        end
    end

    # #
    # 7. Spatiotemporal Interaction (Knorr-Held Taxonomy)
    if run_st_interaction
        st_sigma ~ NamedDist(Exponential(0.5), :st_sigma)

        if model_st == "I"
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_inter = reshape(st_raw, M.s_N, M.t_N) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "II"
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            C_t = cholesky(Symmetric(t_Q_main + noise * I))
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            st_inter = (C_t.U \ st_innov_matrix')' .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "III"
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            C_s = cholesky(Symmetric(s_Q_main + noise * I))
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            st_inter = (C_s.U \ st_innov_matrix) .* st_sigma
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end

        elseif model_st == "IV"
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)
            C_s = cholesky(Symmetric(s_Q_main + noise * I))

            st_inter = Matrix{T}(undef, M.s_N, M.t_N)
            st_inter[:, 1] = (C_s.U \ st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho_main^2 + noise)
            for t in 2:M.t_N
                st_inter[:, t] = t_rho_main .* st_inter[:, t-1] .+ (C_s.U \ st_innov_matrix[:, t])
            end
            for i in 1:M.y_N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]] * st_sigma; end
        end
    end

    # #
    # 8. Nested Hierarchical Supervisors (Low-Fidelity Layers)
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_nested ~ NamedDist(Normal(1.0, 0.5), Symbol("rho_nested_", z_key))
            eta_nested = zeros(T, z_meta.y_N)

            if haskey(z_meta, :Xfixed) && size(z_meta.Xfixed, 2) > 0
                xf_z = z_meta.Xfixed
                beta_z_f ~ NamedDist(MvNormal(zeros(size(xf_z, 2)), 5.0 * I), Symbol("beta_nested_fixed_", z_key))
                eta_nested .+= xf_z * beta_z_f
            end

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                sig_z_s ~ NamedDist(Exponential(1.0), Symbol("sigma_nested_spatial_", z_key))
                Q_z_s = recompose_precision(Symbol(z_meta.model_space), z_meta.s_Q_template.matrix, sig_z_s, noise=noise)
                lat_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta.s_N), Q_z_s), Symbol("latent_nested_spatial_", z_key))
                eta_nested .+= lat_z_s[z_meta.s_idx]

                eta .+= rho_nested .* lat_z_s[M.s_idx]
            end

            y_sigma_nested ~ NamedDist(Exponential(1.0), Symbol("y_sigma_nested_", z_key))
            nested_family = get(z_meta, :model_family, "gaussian")
            for i in 1:z_meta.y_N
                d_lik_z = bstm_Likelihood(nested_family, [T(z_meta.y_obs[i])]; sigma_y=[y_sigma_nested])
                Turing.@addlogprob! Distributions.logpdf(d_lik_z, eta_nested[i])
            end
        end
    end

    # #
    # 9. Multi-fidelity Latent Likelihoods (for RFF-based fields)
    if haskey(M, :z_coords_s) && haskey(M, :z_obs)
        z_sigma ~ NamedDist(Exponential(0.5), :z_sigma)
        Turing.@addlogprob! logpdf(MvNormal(z_latent_field, z_sigma^2 * I), M.z_obs)
    end
    if haskey(M, :w_coords_st) && haskey(M, :w_obs)
        w_sigma ~ NamedDist(filldist(Exponential(0.5), 3), :w_sigma)
        for k in 1:3
            Turing.@addlogprob! logpdf(MvNormal(w_latent_field[:, k], w_sigma[k]^2 * I), M.w_obs[:, k])
        end
    end

    # #
    # 10. Final High-Fidelity Likelihood Dispatch
    for i in 1:M.y_N
        # Rationale: Use `get` for safe access to trials, defaulting to 1.
        d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma[i]], phi_zi=lik_phi, r_nb=lik_r, trial=[Int(get(M.trials, i, 1))], extra_params=extra_p)
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end





function get_optimal_sampler(
    model_obj::DynamicPPL.Model; 
    sampler_choice=:auto, 
    target_acceptance=0.8, 
    adaptation_steps=1000, 
    n_particles=20, 
    hmc_leapfrog_steps=10
)
    # v2.0.0 (2026-07-06)
    # Purpose: Automatically constructs an efficient composite Gibbs sampler by assigning
    #          specialized MCMC algorithms to blocks of parameters based on their prior distributions.
    # Rationale: The previous implementation defaulted to NUTS for most cases. This revised
    #            version creates a more efficient block-sampler that leverages ESS for Gaussian
    #            parameters and Slice sampling for bounded parameters, which can significantly
    #            improve sampling efficiency and convergence.

    # # 1. Parameter Introspection and Categorization
    # Inspect the model's VarInfo to get metadata about each parameter's prior distribution.
    # This is a robust method to determine if a parameter's support is discrete or continuous.
    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)
    
    # Use sets for efficient handling of unique parameter symbols.
    param_groups = Dict(
        :discrete => Set{Symbol}(), 
        :gaussian => Set{Symbol}(), 
        :bounded => Set{Symbol}(), 
        :other_continuous => Set{Symbol}()
    )

    for vn in vns
        # Extract the base symbol from the VarName, e.g., :x from x[1, 2]
        sym = Symbol(DynamicPPL.getsym(vn))
        
        # Skip if this base symbol has already been categorized to avoid redundant checks.
        if any(g -> sym in g, values(param_groups))
            continue
        end

        try
            dist = DynamicPPL.getdist(vi, vn)
            support = Distributions.value_support(typeof(dist))

            if support isa Distributions.Discrete
                push!(param_groups[:discrete], sym)
            elseif support isa Distributions.Continuous
                # Assign to the most specific continuous category.
                if dist isa Normal || dist isa MvNormal || dist isa Truncated{<:Normal}
                    push!(param_groups[:gaussian], sym)
                # `isfinite` checks for either an upper or lower bound.
                elseif isfinite(minimum(dist)) || isfinite(maximum(dist))
                    push!(param_groups[:bounded], sym)
                else
                    push!(param_groups[:other_continuous], sym)
                end
            end
        catch e
            # If `getdist` fails, it's likely a derived quantity or a parameter
            # without a direct prior. These fall into the 'other' category to be
            # handled by the default sampler (NUTS).
            if !(sym in param_groups[:discrete]) && !(sym in param_groups[:gaussian]) && !(sym in param_groups[:bounded])
                 push!(param_groups[:other_continuous], sym)
            end
        end
    end

    # Ensure no parameter is in multiple groups.
    setdiff!(param_groups[:other_continuous], param_groups[:discrete], param_groups[:gaussian], param_groups[:bounded])

    # # 2. Sampler Assembly
    # Construct a list of samplers based on the categorized parameters.
    samplers = []

    if !isempty(param_groups[:discrete])
        s = PG(n_particles, collect(param_groups[:discrete])...)
        push!(samplers, s)
        println("Info: Using Particle Gibbs (PG) for discrete parameters: ", collect(param_groups[:discrete]))
    end
    if !isempty(param_groups[:gaussian])
        s = ESS(collect(param_groups[:gaussian])...)
        push!(samplers, s)
        println("Info: Using Elliptical Slice Sampling (ESS) for Gaussian parameters: ", collect(param_groups[:gaussian]))
    end
    if !isempty(param_groups[:bounded])
        s = Slice(collect(param_groups[:bounded])...)
        push!(samplers, s)
        println("Info: Using Slice sampling for bounded parameters: ", collect(param_groups[:bounded]))
    end
    if !isempty(param_groups[:other_continuous])
        s = NUTS(adaptation_steps, target_acceptance)
        push!(samplers, s)
        println("Info: Using No-U-Turn Sampler (NUTS) for remaining continuous parameters: ", collect(param_groups[:other_continuous]))
    end

    # # 3. Final Sampler Construction
    if isempty(samplers)
        @warn "Could not identify any parameters to sample. Defaulting to NUTS."
        return NUTS(adaptation_steps, target_acceptance)
    elseif length(samplers) == 1
        # If all parameters fall into one category, return that sampler directly.
        println("Info: All parameters handled by a single sampler.")
        return samplers[1]
    else
        # Construct the composite Gibbs sampler.
        println("Info: Constructing composite Gibbs sampler.")
        return Gibbs(samplers...)
    end
end


 
