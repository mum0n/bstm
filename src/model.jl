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


abstract type Manifold end
abstract type ManifoldModel <: Manifold end
abstract type ManifoldOperator <: Manifold end

struct NoneManifold <: ManifoldModel end

struct IID <: ManifoldModel; sigma::UnivariateDistribution; end
struct ICAR <: ManifoldModel; sigma::UnivariateDistribution; end
struct Besag <: ManifoldModel; sigma::UnivariateDistribution; end

"""
    BYM2 <: ManifoldModel

The Besag-York-Mollié 2 (BYM2) model, which provides an intuitive and well-identified
parameterization for spatial effects by separating them into a structured (ICAR)
and an unstructured (IID) component.

# Fields
- `rho`: The prior for the mixing parameter, controlling the proportion of
  variance attributed to the structured spatial effect.
- `sigma`: The prior for the overall marginal standard deviation of the
  total spatial effect.
"""
struct BYM2 <: ManifoldModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end

struct Leroux <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end
struct SAR <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end
struct DAG <: ManifoldModel; rho::UnivariateDistribution; sigma::UnivariateDistribution; end

struct GP <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; kernel::String; end
struct FITC <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct RFF <: ManifoldModel; lengthscale::UnivariateDistribution; sigma::UnivariateDistribution; n_features::Int; kernel::String; end
struct FFT <: ManifoldModel; sigma::UnivariateDistribution; nbins::Int; kernel::String; lengthscale::UnivariateDistribution; end
struct SPDE <: ManifoldModel; sigma::UnivariateDistribution; kappa::UnivariateDistribution; end
struct SVGP <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Warp <: ManifoldModel; lengthscale::UnivariateDistribution; sigma::UnivariateDistribution; n_features::Int; kernel::String; end
struct Nystrom <: ManifoldModel; lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}; sigma::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Hyperbolic <: ManifoldModel; curvature::Real; sigma::UnivariateDistribution; end
struct ExponentialDecay <: ManifoldModel; sigma::UnivariateDistribution; lengthscale::UnivariateDistribution; end

struct BCGN <: ManifoldModel; sigma::UnivariateDistribution; bipartite_adj::AbstractMatrix; end
struct NetworkFlow <: ManifoldModel; sigma::UnivariateDistribution; adjacency_matrix::AbstractMatrix; flow_direction::Symbol; end

"""
    LocalAdaptive <: ManifoldModel

A manifold for localized spatial effects. It combines a Leroux-style precision matrix
with cluster-specific means, allowing the spatial field to have different average
levels in different regions of the domain.
"""
struct LocalAdaptive <: ManifoldModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end



struct Wavelet <: ManifoldModel
    family::Symbol
    nbins::Int
    sigma::UnivariateDistribution
    lengthscale::UnivariateDistribution
end

struct Eigen <: ManifoldModel
    n_vars::Int
    n_factors::Int
    pca_sd::UnivariateDistribution
    pdef_sd::UnivariateDistribution
    ltri_indices::Vector{Int}
end
struct Moran <: ManifoldModel; sigma::UnivariateDistribution; end
struct Spherical <: ManifoldModel; sigma::UnivariateDistribution; range::UnivariateDistribution; end
struct Barycentric <: ManifoldModel; sigma::UnivariateDistribution; end
struct Mosaic <: ManifoldModel; sigma::UnivariateDistribution; n_regions::Int; end
struct TensorProductSmooth <: ManifoldModel; sigma::UnivariateDistribution; Q_template::AbstractMatrix; end

struct TPS <: ManifoldModel; nbins::Int; sigma::UnivariateDistribution; end
struct BSpline <: ManifoldModel; nbins::Int; degree::Int; sigma::UnivariateDistribution; end

struct PSpline <: ManifoldModel
    nbins::Int
    degree::Int
    diff_order::Int
    sigma::UnivariateDistribution
end

struct AR1 <: ManifoldModel
    rho::UnivariateDistribution
    sigma::UnivariateDistribution
end

struct AR2 <: ManifoldModel
    rho1::UnivariateDistribution
    rho2::UnivariateDistribution
    sigma::UnivariateDistribution
end

struct RW1 <: ManifoldModel; sigma::UnivariateDistribution; end
struct RW2 <: ManifoldModel; sigma::UnivariateDistribution; end

"""
    Harmonic <: ManifoldModel

A manifold for modeling periodic effects using harmonic (sine and cosine) functions.
This struct is included for context but is not modified. The new implementation
of the code generator uses a more robust two-coefficient parameterization internally.

# Fields
- `amplitude`: Prior for the amplitude of the wave. (Note: Ignored by the new implementation).
- `phase`: Prior for the phase shift of the wave. (Note: Ignored by the new implementation).
- `period`: The period of the wave, which can be a fixed number or a random variable.
"""
struct Harmonic <: ManifoldModel
    amplitude::UnivariateDistribution
    phase::UnivariateDistribution
    period::Union{Real, UnivariateDistribution}
end
 

struct Cyclic <: ManifoldModel; period::Int; sigma::UnivariateDistribution; end

struct ST_I <: ManifoldModel; sigma::UnivariateDistribution; end
struct ST_II <: ManifoldModel; sigma::UnivariateDistribution; end
struct ST_III <: ManifoldModel; sigma::UnivariateDistribution; end
struct ST_IV <: ManifoldModel; sigma::UnivariateDistribution; end

struct CustomManifold <: ManifoldModel
    code_fragment::String
    params::Dict{Symbol, Any}
end

struct DynamicsManifold <: ManifoldModel; model::String; params::Dict{Symbol, Any}; end


struct ComposedManifold <: ManifoldOperator; components::Vector{Manifold}; operator::Symbol; end
struct SVCManifold <: ManifoldOperator
    covariate::Symbol
    model::ManifoldModel
end

"""
    MixedManifold <: ManifoldOperator

Represents a random effect (intercept or slope) for a specified grouping variable.
The `lhs` field is now a `Vector{String}` to correctly handle multiple correlated
effects (e.g., `(1 + cov1 | group)`), fixing a bug where it was treated as a single string.
"""
struct MixedManifold <: ManifoldOperator
    group_var::Symbol
    lhs::Vector{String}
    model::ManifoldModel
end

abstract type ManifoldSupervisor <: Manifold end
struct NestedManifold <: ManifoldSupervisor
    var::Symbol
    formula::String
    data_source::Symbol
end


# SVAR allows the temporal correlation to vary across space
struct SVAR <: ManifoldModel
    rho_spatial::ManifoldModel # The model for the spatial distribution of rho
    sigma::UnivariateDistribution
end

  
# AdaptiveSmooth learns a coordinate transformation via a simple MLP before smoothing
struct AdaptiveSmooth <: ManifoldModel
    hidden_dim::Int
    nbins::Int
    sigma::UnivariateDistribution
end


# Threshold Autoregressive (TAR): Implements regime-switching temporal dynamics where model parameters depend on a covariate threshold logic.
struct TAR <: ManifoldModel
    threshold_var::Symbol
    rho_regimes::Vector{UnivariateDistribution}
    sigma_regimes::Vector{UnivariateDistribution}
end

# Define LGCP Struct
# Rationale: LGCP models point patterns by assuming the intensity function lambda(s) 
# is a realization of a Log-Gaussian process: log(lambda(s)) = Z(s).
struct LGCP <: ManifoldModel
    model::ManifoldModel
    sigma::UnivariateDistribution
end

struct Kriging <: ManifoldModel
    lengthscale::UnivariateDistribution
    sigma::UnivariateDistribution
    kernel::String
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
    :intercept, :spatial, :temporal, :smooth, :fixed, :nested, :eigen, :svar,
    :mixed, :dynamics, :spacetime, :interact, :custom, :lgcp
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
    "dirichlet_multinomial" => DirichletMultinomialFamily()
)

const STATSMODELS_CONTRASTS = Dict(
    :dummy => StatsModels.DummyCoding(),
    :effects => StatsModels.EffectsCoding(),
    :helmert => StatsModels.HelmertCoding(),
    :treatment => StatsModels.DummyCoding()
)

const MANIFOLD_CONSTRUCTORS = Dict{Symbol, Function}(
    :none => (p, params) -> NoneManifold(),
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
    :harmonic => (p, params) -> Harmonic(p.amplitude, p.phase, get(params, :period, 12.0)),
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
    :kriging => (p, params) -> GP(p.lengthscale, p.sigma, string(get(params, :kernel, "se"))),
    :localadaptive => (p, params) -> LocalAdaptive(p.rho, p.sigma),
    :mosaic => (p, params) -> Mosaic(p.sigma, get(params, :n_regions, 4)),
    :tar => (p, params) -> TAR(
        get(params, :threshold_var, error("TAR model requires a `threshold_var` parameter.")),
        get(params, :rho_regimes, [Beta(1,1), Beta(1,1)]),
        get(params, :sigma_regimes, [Exponential(1.0), Exponential(1.0)])
    ),
    :tensorproductsmooth => (p, params) -> TensorProductSmooth(p.sigma, get(params, :Q_template, sparse(zeros(1,1)))),
    :dynamics => (p, params) -> DynamicsManifold(string(get(params, :model, "none")), params),
    :custom => (p, params) -> CustomManifold(get(params, :code_fragment, ""), get(params, :params, Dict{Symbol, Any}()))
)

function Base.:|>(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :pipe)
end

∘(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :composition)
composition(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :composition)

otimes(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)

⊗(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)

# ==============================================================================
# SECTION 3: FORMULA PARSING ENGINE
# ==============================================================================

function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    # Purpose: Splits a string by a separator, but only when the separator is not inside parentheses or brackets.
    # Rationale: This is crucial for correctly parsing complex formula terms like `smooth(x, model=ar1) + spatial(y)`
    #            without splitting the arguments inside the parentheses.
    # Assumptions: Assumes balanced parentheses and brackets.
    # Inputs:
    #   - input: The string to split.
    #   - sep: The separator string.
    # Outputs: A vector of strings.
    terms = String[]
    current_term = ""
    depth = 0
    i = 1
    sep_len = length(sep)

    while i <= length(input)
        char = input[i]
        if char == '(' || char == '['
            depth += 1
        elseif char == ')' || char == ']'
            depth -= 1
        end

        if depth == 0 && i <= length(input) - sep_len + 1 && SubString(input, i, i + sep_len - 1) == sep
            push!(terms, Base.strip(current_term))
            current_term = ""
            i += sep_len
            continue
        end
        
        current_term *= char
        i += 1
    end
    if !isempty(Base.strip(current_term))
        push!(terms, Base.strip(current_term))
    end
    
    return terms
end

function _parse_value(val_str::String)
    # Purpose: Parses a string value from a formula argument into a Julia type.
    # Rationale: Handles numbers, booleans, symbols, strings, and unevaluated expressions.
    # Assumptions: Input is a single, well-formed value string.
    # Inputs:
    #   - val_str: The string representation of the value.
    # Outputs: A Julia object (e.g., Float64, Int, Bool, Symbol, String, Expr).
    val_str = Base.strip(val_str)
    if startswith(val_str, "(") && endswith(val_str, ")") # Tuple
        inner_val_str = val_str[2:end-1]
        tuple_parts = split_terms_at_depth(inner_val_str, ",")
        parsed_tuple = []
        for tp in tuple_parts
            if !isempty(tp)
                push!(parsed_tuple, _parse_value(tp))
            end
        end
        return Tuple(parsed_tuple)
    elseif (startswith(val_str, "'") && endswith(val_str, "'")) || (startswith(val_str, "\"") && endswith(val_str, "\""))
        return val_str[2:end-1]
    elseif val_str == "true"
        return true
    elseif val_str == "false"
        return false
    elseif occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", val_str) # Treat as a symbol if it's a valid identifier
        return Symbol(val_str)
    else
        try
            return Meta.parse(val_str) # For numbers, expressions, etc.
        catch
            return val_str # Fallback to string if parsing fails
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

function _parse_single_manifold_term(term_str::AbstractString)
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
        error("Internal Parser Error: _parse_single_manifold_term was called with a non-module term '$term_str'. This indicates a logic error in the parent parser.")
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

    possible_priors = [:sigma, :rho, :rho1, :rho2, :lengthscale, :kappa, :amplitude, :phase, :pca_sd, :pdef_sd, :range]
    
    resolved = Dict{Symbol, Any}()

    for p_sym in possible_priors
        p_base_name = p_sym

        if haskey(local_params, p_sym)
            prior_val = local_params[p_sym]
            if prior_val isa Tuple
                resolved[p_sym] = create_pc_prior(p_base_name, prior_val)
            elseif prior_val isa Expr
                try
                    resolved[p_sym] = Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for manifold '$model_name'. Error: $e")
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
        end

        if haskey(prior_defaults, string(p_base_name))
            resolved[p_sym] = prior_defaults[string(p_base_name)]
        end
    end

    return NamedTuple(resolved)
end




function _parse_rhs_expression(term_str::AbstractString)
    # Purpose: Recursively parses the right-hand side of a formula, respecting operator precedence.
    # Rationale: Builds an Abstract Syntax Tree (AST) representing the model structure.
    # Assumptions: Operators are space-padded (e.g., " ⊗ ").
    # v1.2.1 (2026-07-16)
    # Inputs:
    #   - term_str: The RHS string or a substring of it.
    # Outputs: A nested NamedTuple representing the parsed structure.
    term_str_stripped = Base.strip(term_str)
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

    if occursin(r"\(.*\)", term_str_stripped)
        return _parse_single_manifold_term(term_str_stripped)
    else
        return (module_type = :fixed, args = Dict(:positional_args => [term_str_stripped]))
    end
end

function _categorize_rhs_nodes!(nodes, modules, fixed_effects)
    # Purpose: Traverses the parsed AST and categorizes nodes into a flat dictionary of modules.
    # Rationale: Simplifies processing by converting the tree structure into a key-value store.
    # Assumptions: `nodes` is a vector of parsed AST nodes.
    # v1.2.1 (2026-07-16)
    # Inputs:
    #   - nodes: The vector of AST nodes from `_parse_rhs_expression`.
    #   - modules: The dictionary to populate.
    #   - fixed_effects: The list to populate with fixed effect variable names.
    # Outputs: None (mutates `modules`).
    for node in nodes
        if hasproperty(node, :type) && node.type == :operator
            key = "composed_$(length(modules)+1)"
            modules[key] = (module_type = :interact, args = Dict(:operator => node.op, :components => node.children))
            # --- v1.9.0 (2026-07-21) ---
            # Rationale: Generate a more descriptive key for composed manifolds to improve
            #            the readability of generated variable names. Instead of a generic
            #            "composed_1", the key will reflect the components and operator,
            #            e.g., "spatial_s_idx_pipe_smooth_time".
            
            function _get_composed_node_key_str(n)
                if hasproperty(n, :type) && n.type == :operator
                    op_str = string(n.op)
                    child_keys = [_get_composed_node_key_str(child) for child in n.children]
                    return join(child_keys, "_$(op_str)_")
                elseif hasproperty(n, :module_type)
                    key_parts = [string(n.module_type)]
                    pos_args = get(n.args, :positional_args, [])
                    
                    local primary_vars_str = ""
                    if n.module_type == :mixed && !isempty(pos_args)
                        mixed_expr = pos_args[1]
                        if mixed_expr isa Expr && mixed_expr.head == :call && mixed_expr.args[1] == :|
                            primary_vars_str = string(mixed_expr.args[3]) # The group variable
                        elseif length(pos_args) >= 2
                            primary_vars_str = string(pos_args[2]) # The group variable
                        else
                            primary_vars_str = join([string(a) for a in pos_args], "_") # Fallback
                        end
                    elseif !isempty(pos_args)
                        primary_vars_str = join([string(a) for a in pos_args], "_")
                    end
                    push!(key_parts, primary_vars_str)
                    return join(filter(!isempty, key_parts), "_")
                else
                    return "unknown"
                end
            end

            raw_key = _get_composed_node_key_str(node)
            sanitized_base_key = _sanitize_variablename(raw_key)
            module_key = isempty(sanitized_base_key) ? "composed_$(length(modules)+1)" : sanitized_base_key
            modules[module_key] = (module_type = :interact, args = Dict(:operator => node.op, :components => node.children))
        elseif hasproperty(node, :module_type) && node.module_type in BSTM_MODULE_KEYWORDS
            if node.module_type == :fixed
                if haskey(node.args, :positional_args)
                    append!(fixed_effects, string.(node.args[:positional_args]))
                end
            end
            key_parts = [string(node.module_type)]
            
            # Generate a more concise key based on the primary variable(s).
            local primary_vars_str = ""
            pos_args = get(node.args, :positional_args, [])
            if node.module_type == :mixed && !isempty(pos_args)
                # For mixed(effect | group), the key should be based on the group.
                mixed_expr = pos_args[1]
                if mixed_expr isa Expr && mixed_expr.head == :call && mixed_expr.args[1] == :|
                    primary_vars_str = string(mixed_expr.args[3]) # The group variable
                elseif length(pos_args) >= 2
                    primary_vars_str = string(pos_args[2]) # The group variable
                else
                    primary_vars_str = join([string(a) for a in pos_args], "_") # Fallback
                end
            elseif !isempty(pos_args)
                primary_vars_str = join([string(a) for a in pos_args], "_")
            end
            push!(key_parts, primary_vars_str)

            raw_key = join(filter(!isempty, key_parts), "_")
            sanitized_base_key = _sanitize_variablename(raw_key)
            module_key = sanitized_base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = sanitized_base_key * "_$counter"
            end
            modules[module_key] = node
        else
            @warn "Unrecognized node type '$(node.module_type)' during categorization. Skipping."
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

function decompose_bstm_formula(formula_str::String)
    # Purpose: The main entry point for formula parsing.
    # Rationale: Decomposes the entire formula string into its constituent parts: outcomes, modules, and fixed effects.
    # Assumptions: Formula is in the format `lhs ~ rhs`.
    # Inputs:
    #   - formula_str: The complete model formula.
    # Outputs: A NamedTuple with `:outcomes`, `:modules`, `:fixed_effects`, `:has_intercept`, and `:intercept_prior`.
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
            continue # Intercept handled by has_intercept flag
        elseif startswith(term_stripped, "intercept(")
            if intercept_module_found
                @warn "Multiple intercept() modules found. Only the first will be evaluated for intercept control."
            else
                intercept_module_found = true
                intercept_data = _parse_single_manifold_term(term_stripped)
                # The intercept() module with `false` is a way to explicitly remove the intercept.
                # e.g., `intercept(false)` or `intercept(prior=..., false)`
                has_intercept = get(intercept_data.args, :positional_args, [true])[1] != false
                if haskey(intercept_data.args, :prior)
                    intercept_prior = intercept_data.args[:prior]
                end
            end
        else
            # Pass all other terms to the robust parser.
            push!(module_terms, term_stripped)
        end
    end

    top_level_nodes = [_parse_rhs_expression(term) for term in module_terms]
    modules = Dict{String, Any}()
    fixed_effects = String[] # This will be populated by _categorize_rhs_nodes!
    _categorize_rhs_nodes!(top_level_nodes, modules, fixed_effects)
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
    # v1.3.8 (2026-07-18) - Aligned with new parser logic for scalar-per-outcome parameters.
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
    # Purpose: The main configuration function that translates a formula and data into a detailed model specification.
    # Rationale: This version is updated to correctly evaluate the `prior` argument for the `intercept()` module.
    # v1.2.8 (2026-07-17)
    # Assumptions: `formula` and `data` are provided correctly.
    # Inputs:
    #   - formula, data, calling_module, kwargs.
    # Outputs: A NamedTuple `M` containing the complete model configuration.
    decomposed_formula = decompose_bstm_formula(formula)
    all_vars = Set{Symbol}()
    for out_spec in decomposed_formula.outcomes; push!(all_vars, Symbol(out_spec[:var])); for key in [:log_offsets, :weights, :trials, :censor_lower, :censor_upper]; if haskey(out_spec[:params], key); val = out_spec[:params][key]; if val isa Symbol; push!(all_vars, val); end; end; end; end
    for fe in decomposed_formula.fixed_effects; push!(all_vars, Symbol(fe)); end
    for (_, mod_data) in pairs(decomposed_formula.modules); if haskey(mod_data.args, :positional_args); for arg in mod_data.args[:positional_args]; if arg isa Symbol; push!(all_vars, arg); elseif arg isa Expr; _extract_symbols_from_expr!(all_vars, arg); end; end; end; end
    df_processed = deepcopy(data)
    vars_to_categorize = Set{Symbol}()
    for (_, mod_data_nt) in decomposed_formula.modules; if mod_data_nt.module_type == :fixed; if haskey(mod_data_nt.args, :positional_args); for var_sym in mod_data_nt.args[:positional_args]; if var_sym isa Symbol; if haskey(mod_data_nt.args, :contrast) || get(mod_data_nt.args, :model, nothing) == :categorical; push!(vars_to_categorize, var_sym); end; end; end; end; end; end
    for var_sym in vars_to_categorize; if hasproperty(df_processed, var_sym) && !(eltype(df_processed[!, var_sym]) <: CategoricalValue); df_processed[!, var_sym] = categorical(df_processed[!, var_sym]); end; end
    valid_vars_in_data = filter(v -> hasproperty(df_processed, v), all_vars)
    if !isempty(valid_vars_in_data)
        filtered_data = DataFrame(df_processed[completecases(df_processed, collect(valid_vars_in_data)), :])
        if nrow(filtered_data) < nrow(data); @warn "Missing values detected. $(nrow(data) - nrow(filtered_data)) rows were removed."; end
    else
        filtered_data = df_processed
        # Only warn if variables were specified but not found.
        if !isempty(all_vars); @warn "None of the variables specified in the formula were found in the data: $(all_vars). Proceeding without removing rows for missing data."; end
    end
    M = _initialize_config(filtered_data, merge(Dict(kwargs), Dict(:calling_module => calling_module)))
    M[:formula] = formula
    _process_lhs!(M, decomposed_formula.outcomes)

    _precompute_likelihood_params!(M)

    M[:N_cov] = 0
    M[:add_intercept] = decomposed_formula.has_intercept
    if !isnothing(decomposed_formula.intercept_prior)
        prior_val = decomposed_formula.intercept_prior
        if prior_val isa Expr
            try
                M[:intercept_prior] = Core.eval(calling_module, prior_val)
            catch e
                error("Could not evaluate `prior` argument `$(prior_val)` in intercept() module. Error: $e")
            end
        else; M[:intercept_prior] = prior_val; end
    end
    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        mod_data_dict = Dict(:type => mod_data_nt.module_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)
        
        create_manifold_for_module = true
        if !isnothing(processor!); create_manifold_for_module = processor!(M, mod_data_dict, M, M[:hyperpriors]); end
        
        if mod_type == :fixed || !create_manifold_for_module; continue; end
        
        manifold_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
        manifold_spec_built = build_model(manifold_obj, M, mod_data_dict)
        spec = (key=Symbol(key), domain=mod_data_dict[:type], var=join(mod_data_dict[:variables], "_"), manifold_obj=manifold_obj, params=mod_data_dict[:params], Q_template=manifold_spec_built.Q_template, scaling_factor=manifold_spec_built.scaling_factor)
        push!(M[:manifolds], spec)
    end
    _process_fixed_effects!(M, unique(decomposed_formula.fixed_effects))
    _process_fixed_effects_priors!(M)
    
    _precompute_static_manifolds!(M)

    _finalize_config!(M)
    return NamedTuple(M)
end

 
function generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: A centralized function to generate consistent and informative variable names for all manifold components.
    # Rationale: This function replaces ad-hoc string concatenation throughout the code generators,
    #            ensuring that variable names are unique, descriptive, and consistently formatted.
    #            It handles multivariate and nested model contexts automatically.
    # v1.6.0 (2026-07-20)
    # Inputs:
    #   - spec: The technical specification for the manifold.
    #   - arch: The model architecture ("univariate" or "multivariate").
    #   - outcome_idx: The index of the outcome in a multivariate model.
    #   - prefix: An optional prefix for nested models.
    # Outputs: A NamedTuple containing all necessary variable names as Symbols.

    base_key = string(spec.key)
    full_prefix = isempty(prefix) ? base_key : "$(prefix)_$(base_key)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Suffix for hyperparameters in multivariate, non-shared models.
    hyperparam_suffix = (is_multivariate && !is_shared) ? "_$(outcome_idx)" : ""
    
    # Suffix for latent fields, which are always per-outcome in multivariate models.
    latent_field_suffix = is_multivariate ? "_$(outcome_idx)" : ""

    names = Dict{Symbol, Symbol}()
    
    # --- Standard Hyperparameters ---
    names[:sigma] = Symbol("$(full_prefix)_sigma$(hyperparam_suffix)")
    names[:rho] = Symbol("$(full_prefix)_rho$(hyperparam_suffix)")
    names[:rho1] = Symbol("$(full_prefix)_rho1$(hyperparam_suffix)")
    names[:rho2] = Symbol("$(full_prefix)_rho2$(hyperparam_suffix)")
    names[:rho_field] = Symbol("$(full_prefix)_rho_field$(latent_field_suffix)")
    names[:kappa] = Symbol("$(full_prefix)_kappa$(hyperparam_suffix)")
    names[:ls] = Symbol("$(full_prefix)_ls$(hyperparam_suffix)")
    names[:range] = Symbol("$(full_prefix)_range$(hyperparam_suffix)")
    names[:period] = Symbol("$(full_prefix)_period$(hyperparam_suffix)")
    
    # --- Harmonic Model ---
    names[:beta_cos] = Symbol("$(full_prefix)_beta_cos$(hyperparam_suffix)")
    names[:beta_sin] = Symbol("$(full_prefix)_beta_sin$(hyperparam_suffix)")
    names[:amplitude] = Symbol("$(full_prefix)_amplitude$(hyperparam_suffix)")
    names[:phase] = Symbol("$(full_prefix)_phase$(hyperparam_suffix)")

    # --- Latent Fields and Innovations ---
    names[:raw] = Symbol("$(full_prefix)_raw$(latent_field_suffix)")
    names[:innov] = Symbol("$(full_prefix)_innov$(latent_field_suffix)")
    names[:latent] = Symbol("$(full_prefix)_latent$(latent_field_suffix)")
    
    # --- BYM2 Components ---
    names[:struct] = Symbol("$(full_prefix)_struct$(latent_field_suffix)")
    names[:iid] = Symbol("$(full_prefix)_iid$(latent_field_suffix)")

    # --- Dynamics ---
    names[:velocity] = Symbol("$(full_prefix)_velocity$(hyperparam_suffix)")
    names[:diffusion] = Symbol("$(full_prefix)_diffusion$(hyperparam_suffix)")

    # --- Correlated Mixed Effects ---
    names[:L_corr] = Symbol("L_corr_$(full_prefix)$(hyperparam_suffix)")
    names[:sigma_effects] = Symbol("sigma_effects_$(full_prefix)$(hyperparam_suffix)")

    return NamedTuple(names)
end




function _generate_manifold_code_fragments(m::MixedManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # # Retrieve centralized variable names following the established standard
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    inner_model = m.model
    group_var = m.group_var
    lhs_effects = m.lhs
    k_effects = length(lhs_effects)
    n_groups = get(spec.params, :n_cat, 0)

    # # Resolve linear predictor target based on architecture
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "mixed_idx_$(group_var)"

    if k_effects == 1
        # # Process uncorrelated random effects (Intercept-only or single Slope)
        inner_frags = _generate_manifold_code_fragments(inner_model, spec, arch, outcome_idx, prefix=prefix)
        
        # # Strip generic addition to apply custom grouping indexing via views
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
        # # Mixed Effect Logic (Single): $(lhs_str) | $(group_var)
        $(update_inner_cleaned)
        $(application_code)
    end
    """
        return (priors=inner_frags.priors, update=update_str)

    else
        # # Process correlated multivariate random effects (e.g., 1 + cov | group)
        priors_acc = String[]
        push!(priors_acc, "$(v.L_corr) ~ NamedDist(LKJCholesky($(k_effects), 1.0), :$(v.L_corr))")
        push!(priors_acc, "$(v.sigma_effects) ~ NamedDist(filldist(Exponential(1.0), $(k_effects)), :$(v.sigma_effects))")
        push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_groups * k_effects)), I), :$(v.raw))")

        # # Resolve grouping precision structure (IID or GMRF)
        group_chol_logic = if inner_model isa IID || inner_model isa NoneManifold
            "local L_groups_inv_t_$(spec.key) = sparse(I, $(n_groups), $(n_groups))"
        else
            """
        local Q_groups_$(spec.key) = spec_registry[\"$(spec.key)\"].Q_template
        local F_groups_$(spec.key) = cholesky(Symmetric(Q_groups_$(spec.key) + noise * I))
        local L_groups_inv_t_$(spec.key) = F_groups_$(spec.key).U \\ I
        """
        end

        # # Iterate through terms and generate application strings
        # # This prevents UndefVarErrors by correctly mapping formula terms to DataFrame columns individually
        application_loop_acc = String[]
        for i in 1:k_effects
            term = lhs_effects[i]
            is_int = (term == "1" || term == "intercept()" || term == "(Intercept)")

            term_application = if is_int
                "        $(eta_target) .+= view(effects_matrix_$(spec.key), M.$(index_var), $(i))"
            else
                "        $(eta_target) .+= M.data[!, :$(Symbol(term))] .* view(effects_matrix_$(spec.key), M.$(index_var), $(i))"
            end
            push!(application_loop_acc, "# Effect Component: $(term)")
            push!(application_loop_acc, term_application)
        end

        update_str = """
    begin
        # # Correlated Mixed Effects Construction for $(group_var)
        local L_effects_t_$(spec.key) = (Diagonal($(v.sigma_effects)) * $(v.L_corr).L)'
        $(group_chol_logic)
        
        local innovations_matrix_$(spec.key) = reshape($(v.raw), $(n_groups), $(k_effects))
        local effects_matrix_$(spec.key) = L_groups_inv_t_$(spec.key) * innovations_matrix_$(spec.key) * L_effects_t_$(spec.key)

        # # Sequential contribution of decomposed terms to the linear predictor
$(join(application_loop_acc, "\n"))
    end
    """
        return (priors=join(priors_acc, "\n    "), update=update_str)
    end
end




function _generate_manifold_code_fragments(m::SVCManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing code for Spatially Varying Coefficients (SVC).
    # Rationale: Ensures covariates are correctly indexed from the DataFrame while 
    #            preventing invalid indexing if an intercept term is passed as a covariate.
    # v1.4.1 (2026-07-20) - Corrected intercept handling.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    cov_var = m.covariate
    
    # Generate base spatial field logic
    inner_frags = _generate_manifold_code_fragments(inner_model, spec, arch, outcome_idx, prefix=prefix)
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



### Version 1.9.1 - 2026-07-21 23:45:00
### Technical Descriptor: Reinforced Manifold Code Generator for NCP GMRF Architectures

function _generate_manifold_code_fragments(m::ManifoldModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates Turing-compatible priors and update logic for GMRF manifolds.
    # Rationale: Implements Non-Centered Parameterization (NCP) to decouple hyperparameters from the latent field structure.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    # Prior definition block
    # Hyperparameters (sigma, rho) are defined once if shared across outcomes, or independently if not.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma)
            push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        end
        if hasproperty(m, :rho)
            push!(priors_acc, "$(v.rho) ~ NamedDist($(_distribution_to_string(m.rho)), :$(v.rho))")
        end
    end

    # Latent field innovations block
    # Independent standard normal innovations are sampled for transformation into the structured field.
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    # Domain and Index resolution
    # Determines the mapping from latent units to observations based on the manifold domain.
    local index_var
    if spec.domain == :spatial
        index_var = "s_idx"
    elseif spec.domain == :temporal
        # Cyclic and Harmonic models use the seasonal unit index u_idx.
        index_var = (typeof(m) <: Union{Cyclic, Harmonic}) ? "u_idx" : "t_idx"
    elseif spec.domain == :mixed
        index_var = "mixed_idx_$(spec.var)"
    else
        index_var = string(spec.domain) * "_idx"
    end

    # Target resolution
    # Directs updates to the global linear predictor or a specific multivariate outcome column.
    local eta_target
    if is_multivariate
        eta_target = "eta_latent[:, $(outcome_idx)]"
    else
        eta_target = "eta"
    end

    # Effect application block
    # Differentiates between basis projections (smooth) and unit-level mapping (spatial/temporal).
    local effect_app_str
    if spec.domain == :smooth
        effect_app_str = "$(eta_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
    else
        effect_app_str = "$(eta_target) .+= view($(v.latent), M.$(index_var))"
    end

    # Update logic assembly
    # Implements the NCP transformation: latent = sigma * (U \ innovations)
    local update_str
    if get(spec, :is_static, false)
        update_str = """
    begin
        # Static Manifold Solve: $(spec.key)
        # Uses pre-computed Cholesky factor from the registry.
        F_$(spec.key) = spec_registry["$(spec.key)"].cholesky_factor
        $(v.latent) = $(v.sigma) .* (F_$(spec.key).U \\ $(v.raw))
        $(effect_app_str)
    end
    """
    else
        # Dynamic Manifold Solve with Precision Recomposition
        local flow_direction_kwarg = (m isa NetworkFlow) ? ", flow_direction=:$(m.flow_direction)" : ""

        update_str = """
    begin
        # Dynamic Manifold Solve: $(spec.key)
        Q_temp_$(spec.key) = spec_registry["$(spec.key)"].Q_template
        m_type_$(spec.key) = spec_registry["$(spec.key)"].manifold_obj |> typeof |> Symbol

        rho_val_$(spec.key) = $(hasproperty(m, :rho) ? v.rho : "nothing")

        Q_final_$(spec.key) = recompose_precision(m_type_$(spec.key), Q_temp_$(spec.key), 1.0; extra_param=rho_val_$(spec.key)$(flow_direction_kwarg))
        F_$(spec.key) = cholesky(Symmetric(Q_final_$(spec.key) + noise * I))

        # Project innovations through the upper Cholesky factor
        $(v.latent) = $(v.sigma) .* (F_$(spec.key).U \\ $(v.raw))
        $(effect_app_str)
    end
    """
    end

    return (priors=priors_str, update=update_str)
end


"""
    _generate_manifold_code_fragments(m::DynamicsManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for process-based `dynamics` models.

# Rationale for New Implementation
The previous implementation incorrectly used a generic GMRF generator, failing to model
the temporal evolution specified by the dynamics model. This new, specialized function
correctly implements a state-space model where the latent field at time `t` is a
function of the field at `t-1`.

It supports different evolution models:
- **`advection`**: Models transport using a first-order directed operator (`A_template`).
- **`diffusion`**: Models spreading using the graph Laplacian (`L_template`).
- **`advection_diffusion`**: Combines both operators.

The function defines priors for the physical parameters (`velocity`, `diffusion`, `sigma`)
and constructs the evolution loop within the Turing model.
"""
function _generate_manifold_code_fragments(m::DynamicsManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = m.params
    model_type = m.model
    is_multivariate = arch == "multivariate"
    is_shared = get(params, :shared, false)

    # Determine parameter names
    local sigma_name, velocity_name, diffusion_name
    if is_multivariate && !is_shared
        sigma_name = Symbol("$(prefixed_key)_sigma_$(outcome_idx)")
        velocity_name = Symbol("$(prefixed_key)_velocity_$(outcome_idx)")
        diffusion_name = Symbol("$(prefixed_key)_diffusion_$(outcome_idx)")
    else
        sigma_name = Symbol("$(prefixed_key)_sigma")
        velocity_name = Symbol("$(prefixed_key)_velocity")
        diffusion_name = Symbol("$(prefixed_key)_diffusion")
    end

    priors_acc = String[]

    # Define priors for the dynamics parameters
    if model_type in ["advection", "advection_diffusion"]
        vel_prior = get(params, :velocity_prior, Normal(0, 0.5))
        push!(priors_acc, "$(velocity_name) ~ NamedDist($(_distribution_to_string(vel_prior)), :$(velocity_name))")
    end
    if model_type in ["diffusion", "advection_diffusion"]
        diff_prior = get(params, :diffusion_prior, LogNormal(-1, 1))
        push!(priors_acc, "$(diffusion_name) ~ NamedDist($(_distribution_to_string(diff_prior)), :$(diffusion_name))")
    end
    
    sigma_prior = get(params, :sigma_prior, Exponential(1.0))
    push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(sigma_prior)), :$(sigma_name))")

    # Prior for the innovations at each time step
    innov_name = Symbol("innov_dyn_$(prefixed_key)")
    push!(priors_acc, "$(innov_name) ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :$(innov_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    # Get the operators from the hyper registry
    L_op = "spec_registry[\"$(key_str)\"].hyper.L_template"
    A_op = "spec_registry[\"$(key_str)\"].hyper.A_template"

    # Build the evolution logic based on model type
    local propagator_logic, evolution_step
    if model_type == "advection"
        propagator_logic = "propagator = lu(I(M.s_N) - $(velocity_name) * $(A_op) + noise * I)"
        evolution_step = "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]"
    elseif model_type == "diffusion"
        propagator_logic = "propagator = cholesky(Symmetric(I(M.s_N) - $(diffusion_name) * $(L_op) + noise * I))"
        evolution_step = "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]"
    elseif model_type == "advection_diffusion"
        propagator_logic = "propagator = lu(I(M.s_N) - $(velocity_name) * $(A_op) - $(diffusion_name) * $(L_op) + noise * I)"
        evolution_step = "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]"
    else
        # Default to a simple random walk if model is unknown
        propagator_logic = "propagator = I(M.s_N)"
        evolution_step = "dyn_field[:, t] = dyn_field[:, t-1] + innov_matrix[:, t]"
    end

    update_str = """
    begin
        # Dynamics model: $(model_type) for $(key_str)
        # This uses an implicit Euler scheme for numerical stability.
        # This uses an implicit Euler scheme for numerical stability and aligns with the reconstruction logic.
        
        # 1. Construct the time-step propagator matrix
        $(propagator_logic)

        dyn_field = zeros(T, M.s_N, M.t_N)
        innov_matrix = reshape($(innov_name), M.s_N, M.t_N)

        # Initialize the first time step from a standard normal innovation
        dyn_field[:, 1] = innov_matrix[:, 1] .* $(sigma_name)
        dyn_field[:, 1] = innov_matrix[:, 1]

        # Evolve through time using the state-space equation
        # u_t = P⁻¹ * (u_{t-1} + innov_t)
        # u_t = P⁻¹ * u_{t-1} + innov_t
        for t in 2:M.t_N
            $(evolution_step)
        end
        
        # Scale the entire field by sigma
        dyn_field .*= $(sigma_name)

        # Apply the 2D spatiotemporal field to the linear predictor
        # using observation-specific spatial and temporal indices.
        for i in 1:N
            $(eta_update_target)[i] += dyn_field[M.s_idx[i], M.t_idx[i]]
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end



function _generate_manifold_code_fragments(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # This is a specialized implementation for AR(1) processes.
    # It uses a state-space formulation for numerical stability and efficiency,
    # avoiding the construction and Cholesky decomposition of a dense precision matrix.
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)

    key_str = string(spec.key)
    if isempty(prefix)
        prefixed_key = key_str
    else
        prefixed_key = "$(prefix)_$(key_str)"
    end

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        # The prior for rho is truncated to ensure stationarity and numerical stability.
        push!(priors_acc, "$(v.rho) ~ NamedDist(truncated($(_distribution_to_string(m.rho)), -0.9999, 0.9999), :$(v.rho))")
    end

    push!(priors_acc, "$(v.innov) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.innov))")
    priors_str = join(priors_acc, "\n")

    # Determine the target for updating the linear predictor
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx" # AR1 is always temporal

    # The update block performs the sequential calculation
    update_str = """
    begin
        # AR1 state-space evolution for $(key_str)
        $(v.latent) = Vector{T}(undef, $(n_latent))
        
        rho_val = $(v.rho)

        # Initialize the first state and evolve
        $(v.latent)[1] = $(v.innov)[1] / sqrt(1.0 - rho_val^2 + noise)
        for t in 2:$(n_latent)
            $(v.latent)[t] = rho_val * $(v.latent)[t-1] + $(v.innov)[t]
        end
        
        # Scale by sigma and apply to eta
        $(v.latent) .*= $(v.sigma)
        $(eta_update_target) .+= view($(v.latent), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_manifold_code_fragments(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the Bayesian PCA (`eigen`) manifold.

# Rationale for Update
This is a complete rewrite of the original implementation to correctly model the
variables specified in the `eigen()` call and use the resulting latent factor as a
predictor, aligning the implementation with the documentation.

The corrected implementation performs the following steps within the Turing model:
1.  **Factor Model Priors**: Defines priors for the factor standard deviations (`pca_sds`),
    uniquenesses (`pdef_sds`), Householder reflector vectors (`v_raw`), and the latent
    factor scores (`factors_flat`).
2.  **Loadings Matrix Construction**: Constructs an orthonormal loadings matrix `U` from the
    Householder reflectors and scales it by the factor standard deviations to get the
    full loadings matrix `L`.
3.  **Factor Model Likelihood**: Defines the likelihood for the input data `Y_eigen_data`
    as a multivariate normal distribution with a mean reconstructed from the factors and
    loadings (`F * L'`) and a diagonal covariance matrix of uniquenesses (`Psi`). This
    log-probability is added to the model using `Turing.@addlogprob!`.
4.  **Predictor Extraction**: The first latent factor column (`F[:, 1]`) is extracted and
    added to the main model's linear predictor `eta`, fulfilling the documented purpose
    of using the dominant shared signal as a predictor.
"""
function _generate_manifold_code_fragments(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_vars = m.n_vars
    n_factors = m.n_factors
    n_obs = size(spec.hyper.eigen_data, 1)

    # Priors for the factor model parameters
    priors_str = """
    # Priors for eigen manifold: $(key_str)
    v_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, $(length(m.ltri_indices))), 1.0), :v_raw_$(prefixed_key))
    
    # Priors for factor standard deviations (related to eigenvalues)
    pca_sds_$(prefixed_key) ~ NamedDist(filldist($(m.pca_sd), $(n_factors)), :pca_sds_$(prefixed_key))
    
    # Priors for uniquenesses (residual standard deviations)
    pdef_sds_$(prefixed_key) ~ NamedDist(filldist($(m.pdef_sd), $(n_vars)), :pdef_sds_$(prefixed_key))
    
    # Latent factors (scores) are sampled from a standard normal
    factors_flat_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, $(n_obs * n_factors)), I), :factors_flat_$(prefixed_key))
    """

    # Main logic for the factor model and its contribution to eta
    update_str = """
    begin
        # --- Factor Model for Eigen Manifold: $(key_str) ---
        
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

        # 7. Add the first latent factor to the main model's linear predictor.
        #    This uses the dominant shared signal as a predictor for the main outcome.
        local first_factor = view(F_$(prefixed_key), :, 1)
        eta .+= first_factor
    end
    """
    return (priors=priors_str, update=update_str)
end


function _generate_manifold_code_fragments(m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # v1.4.2 (2026-07-20) - Added state-space implementation for RW1.
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


function _generate_manifold_code_fragments(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # v1.4.2 (2026-07-20) - Added state-space implementation for RW2.
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


"""
    _generate_manifold_code_fragments(m::Union{Besag, ICAR, BCGN}, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for intrinsic spatial models like Besag, ICAR, and BCGN.

# Rationale
This is a specialized method for intrinsic Gaussian Markov Random Fields (GMRFs). These models are defined by a singular precision matrix (typically a graph Laplacian), which makes their mean level non-identifiable from the global intercept.

To resolve this, this function implements two key features:
1.  **Non-Centered Parameterization**: It samples standard normal noise (`latent_raw_name`) and transforms it using the Cholesky factor of the precision matrix (`F.U \\ ...`). This improves sampler efficiency.
2.  **Sum-to-Zero Constraint**: After generating the latent field, it explicitly subtracts the mean (`latent_field_raw .- mean(latent_field_raw)`). This constraint "pins" the mean of the spatial effect to zero, ensuring that the global intercept is uniquely identifiable and stabilizing the MCMC sampler.

This method is dispatched for `Besag`, `ICAR`, and `BCGN` because all three result in an intrinsic GMRF structure that requires this constraint for robust inference.

# Arguments
- `m::Union{Besag, ICAR, BCGN}`: The manifold object.
- `spec::NamedTuple`: The technical specification for this manifold instance.
- `arch::String`: The model architecture ("univariate" or "multivariate").
- `outcome_idx::Union{Int, Nothing}`: The index of the outcome in a multivariate model.
- `prefix::String`: An optional prefix for parameter names, used in composed models.

# Returns
- A `NamedTuple` containing the `priors` and `update` code strings for the Turing model.
"""
function _generate_manifold_code_fragments(m::Union{Besag, ICAR, BCGN}, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # # Process: Generates Turing-compatible logic for intrinsic GMRF models.
    # # Rationale: These models are defined by singular precision matrices (rank deficiency 1).
    # # Implementation ensures identifiability via Non-Centered Parameterization and soft sum-to-zero constraints.

    # # Technical Audit: Retrieve centralized variable names
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    # # Prior definition block
    # # Hyperparameters defined based on shared/independent outcome logic
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma)
            push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
        end
    end

    # # Standard Normal innovations for NCP
    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n    ")

    # # Index and target resolution
    index_var = spec.domain == :spatial ? "s_idx" : string(spec.domain) * "_idx"
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"

    # # Effect application logic using the resolved standard latent name
    # # Note: v.latent is used here to match the reconstructed field in the update block
    local effect_application_str
    if spec.domain == :smooth
        effect_application_str = "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)"
    else
        effect_application_str = "$(eta_update_target) .+= view($(v.latent), M.$(index_var))"
    end

    # # Update logic assembly
    # # 1. Solve the intrinsic system via Cholesky
    # # 2. Apply soft sum-to-zero constraint (identifiability)
    # # 3. Scale and apply effect
    update_str = """
    begin
        # --- Intrinsic Manifold Solve: $(key_str) ---
        local Q_template_$(key_str) = spec_registry["$(key_str)"].Q_template
        local F_$(key_str) = cholesky(Symmetric(Q_template_$(key_str) + noise * I))
        
        # # Reconstruct field from innovations
        local latent_field_raw_$(key_str) = F_$(key_str).U \\ $(v.raw)

        # # Soft sum-to-zero for identifiability against the global intercept
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(latent_field_raw_$(key_str)))
        
        # # Scale and apply transformation
        $(v.latent) = latent_field_raw_$(key_str) .* $(v.sigma)
        $(effect_application_str)
    end
    """

    return (priors=priors_str, update=update_str)
end

function _generate_manifold_code_fragments(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
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
    _generate_manifold_code_fragments(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `Moran` spatial manifold.

# Rationale for Update
This is a new function that implements the spectral decomposition for the Moran manifold.
Instead of using a precision matrix (GMRF), it models the latent spatial effect as a
linear combination of the pre-computed Moran eigenvectors. The coefficients of this
combination are sampled from a Normal distribution, with their scale controlled by the
manifold's `sigma` hyperparameter. This correctly implements the `Moran's I Basis Manifold`
as a spectral model.
"""
function _generate_manifold_code_fragments(m::Moran, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
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
        local $(latent_name_str) = moran_eigenvectors * ($(coeffs_name_str) .* $(sigma_name_str))
        
        $(eta_update_target) .+= view($(latent_name_str), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_manifold_code_fragments(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `Spherical` Gaussian Process model.

# Rationale for Update
This new function implements the code generation for a full GP with a spherical
covariance function. It computes the pairwise distance matrix, evaluates the
spherical kernel based on the sampled `range` and `sigma` parameters, and then
samples the latent field from the resulting `MvNormal` distribution. This correctly
implements the `Spherical` manifold as a continuous GP model.
"""
function _generate_manifold_code_fragments(m::Spherical, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Determine parameter names
    local sigma_name_str, range_name_str, latent_raw_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        range_name_str = "$(prefixed_key)_range_$(outcome_idx)"
        latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        range_name_str = "$(prefixed_key)_range"
        if is_multivariate
            latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_raw_name_str = "$(prefixed_key)_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        push!(priors_acc, "$(Symbol(range_name_str)) ~ NamedDist($(_distribution_to_string(m.range)), :$(Symbol(range_name_str)))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(Symbol(latent_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(latent_raw_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Spherical GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # Compute pairwise Euclidean distances
        dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute spherical kernel matrix
        h = dist_matrix ./ $(range_name_str)
        K = zeros(T, size(h))
        mask = h .< 1.0
        K[mask] = ($(sigma_name_str)^2) .* (1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3)
        K += (noise * I)
        
        F = cholesky(Symmetric(K))
        $(latent_name_str) = F.L * $(latent_raw_name_str)
        
        $(eta_update_target) .+= $(latent_name_str)
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_manifold_code_fragments(m::LocalAdaptive, ...)

A specialized code generator for the `LocalAdaptive` manifold.

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
function _generate_manifold_code_fragments(m::LocalAdaptive, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Determine parameter names based on architecture and sharing.
    # Note: mu_clusters_name_str is removed as it's no longer needed with soft centering.
    local sigma_name_str, rho_name_str, latent_raw_name_str, latent_name_str, mu_clusters_raw_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        rho_name_str = "$(prefixed_key)_rho_$(outcome_idx)"
        latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        mu_clusters_raw_name_str = "$(prefixed_key)_mu_clusters_raw_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        rho_name_str = "$(prefixed_key)_rho"

        mu_clusters_raw_name_str = "$(prefixed_key)_mu_clusters_raw"
        mu_clusters_name_str = "$(prefixed_key)_mu_clusters"
        if is_multivariate
            latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_raw_name_str = "$(prefixed_key)_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]

    # Generate priors only once for shared parameters in multivariate models.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        push!(priors_acc, "$(Symbol(rho_name_str)) ~ NamedDist($(_distribution_to_string(m.rho)), :$(Symbol(rho_name_str)))")
        
        # Prior for the raw cluster means. They will be centered later for identifiability.
        n_clusters = spec.hyper.n_clusters
        push!(priors_acc, "$(Symbol(mu_clusters_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_clusters)), I), :$(Symbol(mu_clusters_raw_name_str)))")
    end

    # Prior for the main latent field (non-centered innovations)
    push!(priors_acc, "$(Symbol(latent_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(latent_raw_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "s_idx" # LocalAdaptive is always spatial

    update_str = """
    begin
        # LocalAdaptive model for $(key_str)
        
        # 1. Apply soft sum-to-zero constraint on cluster means for identifiability.
        local n_clusters = spec_registry["$(key_str)"].hyper.n_clusters
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_clusters), sum($(mu_clusters_raw_name_str)))
        
        # 2. Construct the mean vector for the GMRF by mapping cluster means to spatial units.
        local mean_vector = $(mu_clusters_raw_name_str)[M.cluster_assignments]

        # 3. Recompose the Leroux precision matrix.
        local Q_template = spec_registry["$(key_str)"].Q_template
        local m_type = spec_registry["$(key_str)"].manifold_obj |> typeof |> Symbol
        local rho_val = $(rho_name_str)
        local Q_final = recompose_precision(m_type, Q_template, 1.0; extra_param=rho_val)
        
        # 4. Sample from the non-zero mean GMRF using a non-centered parameterization.
        #    For x ~ N(μ, Q⁻¹), we can sample z ~ N(0,I) and compute x = μ + L'⁻¹z,
        #    where Q = LL'. Here, Q = F.U' * F.U, so L = F.U'. Then L'⁻¹ = (F.U)⁻¹ = F.U \\ I.
        #    This gives x = μ + (F.U \\ z).
        local F = cholesky(Symmetric(Q_final + noise * I))
        local latent_field_centered_part = F.U \\ $(latent_raw_name_str)
        local $(latent_name_str) = mean_vector .+ latent_field_centered_part

        # 5. Scale by sigma and apply to eta.
        $(latent_name_str) .*= $(sigma_name_str)
        $(eta_update_target) .+= view($(latent_name_str), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end




function _generate_manifold_code_fragments(m::BYM2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
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
    
    index_var = spec.domain == :spatial ? "s_idx" : string(spec.domain) * "_idx"
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_application_str = spec.domain == :smooth ? "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(v.latent)" : "$(eta_update_target) .+= view($(v.latent), M.$(index_var))"

    # Update block to combine the components
    update_str = """
    begin
        # BYM2 model for $(key_str)
        # 1. Reconstruct the structured (ICAR) component from its raw innovations.
        local Q_template = spec_registry["$(spec.key)"].Q_template
        if !isnothing(Q_template) && $(n_latent) > 0
            local F = cholesky(Symmetric(Matrix(Q_template) + noise * I))
            local struct_latent = F.U \\ $(v.struct)
            # 2. Apply soft sum-to-zero constraint for identifiability
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(struct_latent))
            # 3. Combine structured and unstructured components using Riebler parameterization.
            local bym2_effect = $(v.sigma) .* (sqrt($(v.rho)) .* struct_latent .+ sqrt(1.0 - $(v.rho)) .* $(v.iid))
            # 5. Add the final effect to the linear predictor.
            $(eta_update_target) .+= view(bym2_effect, M.$(index_var))
        end
    end
    """    
    
    return (priors=priors_str, update=update_str)
end



function _generate_manifold_code_fragments(m::Warp, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
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

    # Define parameter names for warping and main GP layers
    local sigma_name_str, ls_name_str, beta_main_name_str, W_main_name_str, b_main_name_str
    local beta_warp_name_str, W_warp_name_str, b_warp_name_str

    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        ls_name_str = "$(prefixed_key)_ls_$(outcome_idx)"
        beta_main_name_str = "$(prefixed_key)_beta_main_$(outcome_idx)"
        W_main_name_str = "$(prefixed_key)_W_main_$(outcome_idx)"
        b_main_name_str = "$(prefixed_key)_b_main_$(outcome_idx)"
        beta_warp_name_str = "$(prefixed_key)_beta_warp_$(outcome_idx)"
        W_warp_name_str = "$(prefixed_key)_W_warp_$(outcome_idx)"
        b_warp_name_str = "$(prefixed_key)_b_warp_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        ls_name_str = "$(prefixed_key)_ls"
        if is_multivariate
            beta_main_name_str = "$(prefixed_key)_beta_main_$(outcome_idx)"
            W_main_name_str = "$(prefixed_key)_W_main_$(outcome_idx)"
            b_main_name_str = "$(prefixed_key)_b_main_$(outcome_idx)"
            beta_warp_name_str = "$(prefixed_key)_beta_warp_$(outcome_idx)"
            W_warp_name_str = "$(prefixed_key)_W_warp_$(outcome_idx)"
            b_warp_name_str = "$(prefixed_key)_b_warp_$(outcome_idx)"
        else
            beta_main_name_str = "$(prefixed_key)_beta_main"
            W_main_name_str = "$(prefixed_key)_W_main"
            b_main_name_str = "$(prefixed_key)_b_main"
            beta_warp_name_str = "$(prefixed_key)_beta_warp"
            W_warp_name_str = "$(prefixed_key)_W_warp"
            b_warp_name_str = "$(prefixed_key)_b_warp"
        end
    end

    priors_acc = String[]

    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        push!(priors_acc, "$(Symbol(ls_name_str)) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(Symbol(ls_name_str)))")
    end

    n_features = m.n_features
    in_dims = size(spec.hyper.coords, 2)

    # Priors for the warping function's RFF parameters
    push!(priors_acc, "$(Symbol(W_warp_name_str)) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(Symbol(W_warp_name_str)))")
    push!(priors_acc, "$(Symbol(b_warp_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(Symbol(b_warp_name_str)))")
    push!(priors_acc, "$(Symbol(beta_warp_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(Symbol(beta_warp_name_str)))")

    # Priors for the main GP's RFF parameters
    push!(priors_acc, "$(Symbol(W_main_name_str)) ~ NamedDist(MvNormal(zeros(T, $(in_dims * n_features)), I), :$(Symbol(W_main_name_str)))")
    push!(priors_acc, "$(Symbol(b_main_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_features)), I), :$(Symbol(b_main_name_str)))")
    push!(priors_acc, "$(Symbol(beta_main_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_features)), $(Symbol(sigma_name_str))^2 * I), :$(Symbol(beta_main_name_str)))")
    
    priors_str = join(priors_acc, "\n")
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # 1. Construct and apply the warping function
        W_warp_matrix = reshape($(W_warp_name_str), $(in_dims), $(n_features))
        Phi_warp = sqrt(2.0 / $(n_features)) .* cos.((coords * W_warp_matrix) .+ $(b_warp_name_str)')
        warping_effect = Phi_warp * $(beta_warp_name_str)
        coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        W_main_matrix = reshape($(W_main_name_str), $(in_dims), $(n_features)) ./ $(ls_name_str)
        Phi_main = sqrt(2.0 / $(n_features)) .* cos.((coords_warped * W_main_matrix) .+ $(b_main_name_str)')
        main_effect = Phi_main * $(beta_main_name_str)

        $(eta_update_target) .+= main_effect
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 

 

function _generate_manifold_code_fragments(m::Nystrom, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Nystrom sparse Gaussian Process model.
    # This method uses a low-rank approximation based on inducing points.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Determine parameter names
    local sigma_name_str, ls_name_str, v_latent_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        ls_name_str = "$(prefixed_key)_ls_$(outcome_idx)"
        v_latent_name_str = "$(prefixed_key)_v_latent_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        ls_name_str = "$(prefixed_key)_ls"
        if is_multivariate
            v_latent_name_str = "$(prefixed_key)_v_latent_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            v_latent_name_str = "$(prefixed_key)_v_latent"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        if m.lengthscale isa Vector; ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", "); push!(priors_acc, "$(Symbol(ls_name_str)) ~ NamedDist(Product([$(ls_priors_str)]), :$(Symbol(ls_name_str)))");
        else; push!(priors_acc, "$(Symbol(ls_name_str)) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(Symbol(ls_name_str)))"); end
    end

    n_inducing = m.n_inducing
    push!(priors_acc, "$(Symbol(v_latent_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(Symbol(v_latent_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Nystrom sparse GP model for $(key_str)
        X_coords = spec_registry["$(key_str)"].hyper.coords
        Z_coords = spec_registry["$(key_str)"].hyper.Z_inducing
        
        K_UU = evaluate_kernel_matrix(Z_coords, $(sigma_name_str), $(ls_name_str), Symbol("$(m.kernel)"))
        K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(sigma_name_str), $(ls_name_str), Symbol("$(m.kernel)"))
        
        L_UU = cholesky(Symmetric(K_UU + noise * I)).L
        
        # Project standard normal noise through the Nystrom approximation
        # f(X) ≈ K_XU * inv(K_UU) * u, where u ~ N(0, K_UU)
        # Using non-centered parameterization: u = L_UU * v, where v ~ N(0, I)
        # f(X) ≈ K_XU * inv(K_UU) * L_UU * v = K_XU * inv(L_UU' * L_UU) * L_UU * v = K_XU * inv(L_UU) * inv(L_UU') * L_UU * v = K_XU * (L_UU' \\ v)
        $(latent_name_str) = K_XU * (L_UU' \\ $(v_latent_name_str))
        $(eta_update_target) .+= $(latent_name_str)
    end
    """
    
    return (priors=priors_str, update=update_str)
end


"""
    _generate_manifold_code_fragments(m::FITC, ...)

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
function _generate_manifold_code_fragments(m::FITC, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Determine parameter names
    local sigma_name_str, ls_name_str, u_raw_name_str, f_raw_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        ls_name_str = "$(prefixed_key)_ls_$(outcome_idx)"
        u_raw_name_str = "$(prefixed_key)_u_raw_$(outcome_idx)"
        f_raw_name_str = "$(prefixed_key)_f_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        ls_name_str = "$(prefixed_key)_ls"
        if is_multivariate
            u_raw_name_str = "$(prefixed_key)_u_raw_$(outcome_idx)"
            f_raw_name_str = "$(prefixed_key)_f_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            u_raw_name_str = "$(prefixed_key)_u_raw"
            f_raw_name_str = "$(prefixed_key)_f_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        if m.lengthscale isa Vector
            ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
            push!(priors_acc, "$(Symbol(ls_name_str)) ~ NamedDist(Product([$(ls_priors_str)]), :$(Symbol(ls_name_str)))")
        else
            push!(priors_acc, "$(Symbol(ls_name_str)) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(Symbol(ls_name_str)))")
        end
    end

    n_inducing = m.n_inducing
    n_latent = size(spec.Q_template, 1) # Number of data points
    
    # Priors for the latent values at inducing points and the final field innovations
    push!(priors_acc, "$(Symbol(u_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_inducing)), I), :$(Symbol(u_raw_name_str)))")
    push!(priors_acc, "$(Symbol(f_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(f_raw_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # FITC sparse GP model for $(key_str)
        X_coords = spec_registry["$(key_str)"].Q_template
        Z_coords = spec_registry["$(key_str)"].hyper.Z_inducing
        
        # 1. Compute kernel matrices
        K_UU = evaluate_kernel_matrix(Z_coords, $(sigma_name_str), $(ls_name_str), Symbol("$(m.kernel)"), noise)
        K_XU = evaluate_cross_kernel_matrix(X_coords, Z_coords, $(sigma_name_str), $(ls_name_str), Symbol("$(m.kernel)"))
        
        # 2. Sample latent values at inducing points (non-centered)
        L_UU = cholesky(Symmetric(K_UU)).L
        u_latent = L_UU * $(u_raw_name_str)
        
        # 3. Compute conditional mean and variance for FITC
        #    μ_f = K_XU * inv(K_UU) * u_latent
        #    diag_cov_f = diag(K_XX - K_XU * inv(K_UU) * K_XU')
        
        K_UU_inv_u = K_UU \\ u_latent
        mean_f = K_XU * K_UU_inv_u
        
        # Compute diagonal of K_XX - Q_ff efficiently
        # diag(K_XX) is sigma^2 for stationary kernels.
        diag_K_XX = fill($(sigma_name_str)^2, $(n_latent))
        
        # diag(K_XU * inv(K_UU) * K_XU') = sum((K_XU / L_UU.U).^2, dims=2)
        tmp = (L_UU' \\ K_XU')'
        diag_Q_ff = sum(tmp.^2, dims=2)
        
        lambda_diag = diag_K_XX - vec(diag_Q_ff)
        
        # 4. Sample final latent field (non-centered)
        $(latent_name_str) = mean_f + sqrt.(max.(lambda_diag, 0.0) .+ noise) .* $(f_raw_name_str)
        
        $(eta_update_target) .+= $(latent_name_str)
    end
    """
    
    return (priors=priors_str, update=update_str)
end

 
function _generate_manifold_code_fragments(m::Hyperbolic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Hyperbolic Gaussian Process model.
    # This model computes distances in a hyperbolic space (Poincaré disk) before applying a kernel.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Determine parameter names
    local sigma_name_str, curvature_name_str, latent_raw_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        curvature_name_str = "$(prefixed_key)_curvature_$(outcome_idx)"
        latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        curvature_name_str = "$(prefixed_key)_curvature"
        if is_multivariate
            latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_raw_name_str = "$(prefixed_key)_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        # Curvature is fixed for now, but could be given a prior.
        # push!(priors_acc, "$(Symbol(curvature_name_str)) ~ NamedDist(Normal(-1.0, 0.5), :$(Symbol(curvature_name_str)))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(Symbol(latent_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(latent_raw_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Hyperbolic GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        curvature = $(m.curvature) # Fixed curvature
        
        K = evaluate_hyperbolic_kernel_matrix(coords, $(sigma_name_str), curvature, noise)
        F = cholesky(Symmetric(K))
        $(latent_name_str) = F.L * $(latent_raw_name_str)
        
        $(eta_update_target) .+= $(latent_name_str)
    end
    """
    
    return (priors=priors_str, update=update_str)
end
 

function _generate_manifold_code_fragments(m::ExponentialDecay, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the Exponential Decay GP model.
    # This model uses an exponential kernel based on Euclidean distances between coordinates.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Determine parameter names
    local sigma_name_str, ls_name_str, latent_raw_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        ls_name_str = "$(prefixed_key)_ls_$(outcome_idx)"
        latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        ls_name_str = "$(prefixed_key)_ls"
        if is_multivariate
            latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_raw_name_str = "$(prefixed_key)_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        push!(priors_acc, "$(Symbol(ls_name_str)) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(Symbol(ls_name_str)))")
    end

    n_latent = size(spec.hyper.coords, 1)
    push!(priors_acc, "$(Symbol(latent_raw_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(Symbol(latent_raw_name_str)))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Exponential Decay GP model for $(key_str)
        coords = spec_registry["$(key_str)"].hyper.coords
        
        # Compute pairwise Euclidean distances
        dist_matrix = pairwise(Euclidean(), coords, dims=1)
        
        # Compute exponential decay kernel matrix
        K = ($(sigma_name_str)^2) .* exp.(-dist_matrix ./ $(ls_name_str)) .+ (noise * I)
        
        F = cholesky(Symmetric(K))
        $(latent_name_str) = F.L * $(latent_raw_name_str)
        
        $(eta_update_target) .+= $(latent_name_str)
    end
    """
    
    return (priors=priors_str, update=update_str)
end


function _generate_manifold_code_fragments(m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for AR(2) processes using a state-space formulation.
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
        $(v.latent) = Vector{T}(undef, $(n_latent))
        
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


function _generate_manifold_code_fragments(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for the SPDE model.
    # This ensures the `kappa` parameter is correctly sampled and passed to `recompose_precision`.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    # Determine parameter names based on architecture and sharing
    local sigma_name_str, kappa_name_str, latent_raw_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        kappa_name_str = "$(prefixed_key)_kappa_$(outcome_idx)"
        latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        kappa_name_str = "$(prefixed_key)_kappa"
        if is_multivariate
            latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_raw_name_str = "$(prefixed_key)_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    sigma_name = Symbol(sigma_name_str)
    kappa_name = Symbol(kappa_name_str)
    latent_raw_name = Symbol(latent_raw_name_str)
    latent_name = Symbol(latent_name_str)

    priors_acc = String[]

    # Generate priors only once for shared parameters in multivariate models.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(sigma_name))")
        push!(priors_acc, "$(kappa_name) ~ NamedDist($(_distribution_to_string(m.kappa)), :$(kappa_name))")
    end

    push!(priors_acc, "$(latent_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(latent_raw_name))")
    priors_str = join(priors_acc, "\n")
    
    index_var = spec.domain == :spatial ? "s_idx" : string(spec.domain) * "_idx"
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_application_str = spec.domain == :smooth ? "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(latent_name)" : "$(eta_update_target) .+= view($(latent_name), M.$(index_var))"

    update_str = """
    begin
        # SPDE model for $(key_str)
        Q_template = spec_registry["$(key_str)"].Q_template
        m_type = spec_registry["$(key_str)"].manifold_obj |> typeof |> Symbol
        kappa_val = $(kappa_name_str)
        
        Q_final = recompose_precision(m_type, Q_template, 1.0; extra_param=kappa_val)
        F = cholesky(Symmetric(Q_final + noise * I))
        $(latent_name) = $(sigma_name_str) .* (F.U \\ $(latent_raw_name))
        $(effect_application_str)
    end
    """
    
    return (priors=priors_str, update=update_str)
end



"""
    _generate_manifold_code_fragments(m::ComposedManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for composed manifolds, such as state-space models (`|>`)
and non-stationary variance models (`∘`).

# Rationale for Update
This function has been updated to correctly handle the `:composition` operator. It now
retrieves the custom `priors` and `update` code strings that were injected into the
module's parameters by `process_interact_module!`. This completes the implementation
for non-stationary variance models where a smoother modulates a spatial field. The
existing logic for the `:pipe` operator (spatially-varying curves) is preserved.
"""
function _generate_manifold_code_fragments(m::ComposedManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    if m.operator == :pipe
        state_manifold = m.components[1]
        dynamic_manifold = get(spec.params, :dynamic_manifold_obj, nothing)
        key_str = string(spec.key); domain_str = string(spec.domain)
        prefixed_key = isempty(prefix) ? "$(domain_str)_$(key_str)" : "$(prefix)_$(domain_str)_$(key_str)"
        is_multivariate = arch == "multivariate"; is_first_outcome = outcome_idx == 1; is_shared = get(spec.params, :shared, false)
        eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
        local sigma_name_str, rho_name_str, coeffs_raw_name_str
        if is_multivariate && !is_shared
            sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"; rho_name_str = "$(prefixed_key)_rho_$(outcome_idx)"; coeffs_raw_name_str = "$(prefixed_key)_coeffs_raw_$(outcome_idx)"
        else
            sigma_name_str = "$(prefixed_key)_sigma"; rho_name_str = "$(prefixed_key)_rho"; coeffs_raw_name_str = "$(prefixed_key)_coeffs_raw"
        end
        sigma_name = Symbol(sigma_name_str); rho_name = Symbol(rho_name_str); coeffs_raw_name = Symbol(coeffs_raw_name_str)
        state_spec = spec.hyper.state_spec; n_spatial = size(state_spec.Q_template, 1); n_basis = dynamic_manifold.nbins
        basis_key = get(spec.params, :dynamic_basis_key, nothing); if isnothing(basis_key); error("Could not find basis matrix key for piped manifold $(key_str)."); end
        priors_acc = String[]
        if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
            if hasproperty(state_manifold, :sigma); push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(state_manifold.sigma)), :$(sigma_name))"); end
            if hasproperty(state_manifold, :rho); push!(priors_acc, "$(rho_name) ~ NamedDist($(_distribution_to_string(state_manifold.rho)), :$(rho_name))"); end
        end
        push!(priors_acc, "$(coeffs_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_spatial * n_basis)), I), :$(coeffs_raw_name))")
        priors_str = join(priors_acc, "\n        ")
        is_state_static = get(state_spec, :is_static, false)
        local cholesky_block
        if is_state_static
            cholesky_block = "F_spatial = spec_registry[\"$(key_str)\"].hyper.state_spec.cholesky_factor"
        else
            cholesky_block = "Q_spatial_template = spec_registry[\"$(key_str)\"].hyper.state_spec.Q_template\n state_m_type = spec_registry[\"$(key_str)\"].hyper.state_spec.model_type\n rho_val = $(hasproperty(state_manifold, :rho) ? rho_name_str : "nothing")\n Q_spatial = recompose_precision(state_m_type, Q_spatial_template, 1.0; extra_param=rho_val)\n F_spatial = cholesky(Symmetric(Q_spatial + noise * I))"
        end
        update_str = "begin\n $(cholesky_block)\n coeffs_raw_matrix = reshape($(coeffs_raw_name), $(n_spatial), $(n_basis))\n spatial_coeffs = $(sigma_name_str) .* (F_spatial.U \\ coeffs_raw_matrix)\n B_smooth = M.basis_matrices[:$(basis_key)]\n $(eta_update_target) .+= sum(B_smooth .* spatial_coeffs[M.s_idx, :], dims=2)\nend"
        return (priors=priors_str, update=update_str)

    elseif m.operator == :composition
        # This handles the non-stationary variance model where code fragments were injected.
        priors_str = get(spec.params, :priors, "")
        update_str = get(spec.params, :update, "")
        if isempty(priors_str) && isempty(update_str)
            @warn "Composition manifold '$(spec.key)' was processed, but no custom code fragments were found. This interaction will have no effect."
        end
        return (priors=priors_str, update=update_str)
    end
    
    return (priors="", update="")
end




function _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String})
    # Purpose: Creates the design matrix for all fixed effects, excluding the intercept.
    # Rationale: Consolidates fixed effects from bare terms and `fixed()` modules into a single design matrix.
    #            The intercept is handled separately by the `intercept()` module and its assembler block.
    # v1.2.1 (2026-07-16)
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



function _precompute_static_manifolds!(M::Dict)
    # Purpose: Pre-computes the Cholesky factorization for static manifolds.
    # Rationale: Moves constant computations out of the MCMC loop to improve sampling speed.
    #            A manifold is "static" if its precision matrix structure does not depend on a
    # v1.2.1 (2026-07-16)
    #            hyperparameter that is sampled within the model (e.g., a `rho` parameter).
    # Inputs:
    #   - M: The model configuration dictionary, which is mutated.
    # Outputs: None.
    noise = M[:noise]
    new_manifolds = []
    # Define manifold types that do not have dynamic structure parameters like `rho`.
    static_manifold_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Spherical, Barycentric, TensorProductSmooth]

    for spec_in in M[:manifolds]
        current_spec = spec_in
        m_obj = current_spec.manifold_obj

        # --- v1.4.4 (2026-07-20) ---
        # Rationale: This block is added to correctly handle wrapper manifolds like
        #            MixedManifold and SVCManifold. It checks if their *inner* model
        #            is static. If so, it pre-computes the Cholesky factor and attaches
        #            it to the wrapper's spec. This resolves a FieldError where the
        #            code generator for the inner model would look for a `cholesky_factor`
        #            on the wrapper's spec, which didn't exist.
        if m_obj isa MixedManifold || m_obj isa SVCManifold
            inner_model = m_obj.model
            is_inner_static = any(T -> inner_model isa T, static_manifold_types)
            if is_inner_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
                try
                    Q_concrete = sparse(current_spec.Q_template)
                    F = cholesky(Symmetric(Q_concrete + noise * I))
                    final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                    push!(new_manifolds, final_spec)
                    continue # Proceed to the next manifold in the loop
                catch e
                    @warn "Cholesky factorization failed for static inner model in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                end
            end
        end

        if m_obj isa ComposedManifold && m_obj.operator == :pipe
            state_spec = get(current_spec.hyper, :state_spec, nothing)
            if !isnothing(state_spec)
                state_m_obj = state_spec.manifold_obj
                is_state_static = any(T -> state_m_obj isa T, static_manifold_types)

                if is_state_static && !isnothing(state_spec.Q_template) && size(state_spec.Q_template, 1) > 0
                    try
                        Q_concrete = sparse(state_spec.Q_template)
                        F = cholesky(Symmetric(Q_concrete + noise * I))
                        new_state_spec = merge(state_spec, (is_static=true, cholesky_factor=F))
                        new_hyper = merge(current_spec.hyper, (state_spec=new_state_spec,))
                        current_spec = merge(current_spec, (hyper=new_hyper,))
                    catch e
                        @warn "Cholesky factorization failed for static state manifold in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                    end
                end
            end
        end

        # Now check the main manifold object of the (potentially updated) current_spec
        is_main_static = !(current_spec.manifold_obj isa ComposedManifold) && any(T -> current_spec.manifold_obj isa T, static_manifold_types)

        if is_main_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
            try
                # Ensure Q_template is a concrete sparse matrix for Cholesky
                Q_concrete = sparse(current_spec.Q_template)
                F = cholesky(Symmetric(Q_concrete + noise * I))
                final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                push!(new_manifolds, final_spec)
            catch e
                @warn "Cholesky factorization failed for static manifold $(current_spec.key). Reverting to dynamic computation. Error: $e"
                final_spec = merge(current_spec, (is_static=false,))
                push!(new_manifolds, final_spec)
            end
        else
            # For composed manifolds or dynamic manifolds, just add them.
            # The is_static flag for the composed manifold itself remains false.
            final_spec = merge(current_spec, (is_static=get(current_spec, :is_static, false),))
            push!(new_manifolds, final_spec)
        end
    end
    M[:manifolds] = new_manifolds
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
    M[:manifolds] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end



function _process_lhs!(M::Dict, outcome_specs::Vector{Dict{Symbol, Any}})
    # Purpose: Processes the left-hand side of the formula to set up outcome variables and likelihood parameters.
    # Rationale: This version correctly handles multivariate likelihood specifications by iterating
    #            over each outcome. It uses specialized helpers to enforce that parameters like
    #            `censor_lower` are scalars per outcome, while others like `log_offsets` can be
    #            vectors per observation.
    # v1.3.8 (2026-07-18)
    # Assumptions: `outcome_specs` is a vector of dictionaries from `decompose_bstm_formula`.
    # Inputs:
    #   - M: The model configuration dictionary.
    #   - outcome_specs: Parsed information about the outcome(s).
    # Outputs: None (mutates `M`).
    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    M[:outcomes] = outcomes

    # Ensure a default likelihood family is set if not specified.
    for spec in likelihood_specs
        if !haskey(spec, :family)
            spec[:family] = "gaussian" # Default to Gaussian
            @warn "Likelihood `family` not specified. Defaulting to `family=gaussian`."
        end
    end

    # Check if all outcome variables exist in the data frame before proceeding.
    for out_sym in outcomes
        if !hasproperty(M[:data], out_sym)
            error("Outcome variable ':$out_sym' specified in the formula was not found as a column in the provided data frame. Please check for typos or ensure the column exists.")
        end
    end

    M[:outcomes_N] = length(outcomes)
    M[:likelihood_specs] = likelihood_specs

    if M[:outcomes_N] > 1
        M[:model_arch] = "multivariate"
        M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
    else
        M[:model_arch] = get(M, :model_arch, "univariate")
        M[:y_obs] = M[:data][!, M[:outcomes][1]]
    end

    calling_mod = get(M, :calling_module, Main)

    # --- Process scalar-per-outcome parameters ---
    scalar_param_keys = [:censor_lower, :censor_upper, :hurdle]
    for key in scalar_param_keys
        values_per_outcome = []
        any_provided = false
        for spec_params in likelihood_specs
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

    # --- Process per-observation parameters ---
    # For these, we assume they are shared across outcomes if specified only once.
    representative_params = !isempty(likelihood_specs) ? likelihood_specs[1] : Dict()
    _resolve_obs_param!(M, representative_params, M[:data], [:log_offsets], :log_offsets)
    _resolve_obs_param!(M, representative_params, M[:data], [:weights], :weights)
    _resolve_obs_param!(M, representative_params, M[:data], [:trials], :trials)

    # --- Process boolean flags ---
    _resolve_boolean_obs_param!(M, representative_params, :zero_inflated, :use_zi)
if get(M, :user_provided_hurdle, false) && get(M, :use_zi, false)
    @warn "Both `hurdle` and `zero_inflated` were specified. The hurdle model will be used and zero-inflation will be ignored."
    M[:use_zi] = false
end
    _resolve_boolean_obs_param!(M, representative_params, :volatility, :volatility)
end



function _resolve_outcome_scalar_param!(params::Dict, key::Symbol, calling_mod::Module)
    # Purpose: Resolves a likelihood parameter that must be a scalar value for a given outcome.
    # Rationale: This enforces that parameters like `censor_lower` cannot be specified as
    #            per-observation vectors from the data, only as single scalar values.
    # v1.3.8 (2026-07-18)
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
    # Rationale: This version is updated to correctly handle scalar numeric inputs for parameters like censor_lower, censor_upper,
    #            and hurdle, which was a source of user-reported errors. It also improves the warning message.
    # v1.2.4 (2026-07-17)
    # Assumptions: `data` is a DataFrame.
    # Inputs:
    #   - opt_dict: The configuration dictionary to update.
    #   - params: The likelihood parameters from the formula.
    #   - data: The input DataFrame.
    #   - param_keys: A list of possible keys for the parameter (e.g., [:log_offsets]).
    #   - target_key: The key to use in `opt_dict`.
    # Outputs: None (mutates `opt_dict`).
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol
                if hasproperty(data, val)
                    # The symbol refers to a column in the data frame.
                    opt_dict[target_key] = data[!, val]
                    opt_dict[Symbol("user_provided_", target_key)] = true
                else
                    # The symbol might refer to a variable in the calling scope.
                    calling_mod = get(opt_dict, :calling_module, Main)
                    try
                        evaluated_val = Core.eval(calling_mod, val)
                        if evaluated_val isa Number || evaluated_val isa AbstractVector
                            opt_dict[target_key] = evaluated_val
                            opt_dict[Symbol("user_provided_", target_key)] = true
                        else
                            @warn "Parameter '$val' for '$target_key' evaluated to an unsupported type '$(typeof(evaluated_val))'. Ignoring."
                        end
                    catch
                        @warn "Parameter '$val' for '$target_key' is not a valid column name and could not be evaluated as a variable in the calling module. Ignoring."
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
    # v1.3.9 (2026-07-19)
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
    # v1.2.8 (2026-07-17)
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
    # v1.2.1 (2026-07-16)
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

function process_temporal_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    # Purpose: Processes the `temporal()` or `seasonal()` module call.
    # Rationale: Sets up temporal indices (`t_idx`) or seasonal indices (`u_idx`) based on the model type.
    # v1.2.1 (2026-07-16)
    # Assumptions: `data` is present in `opt_dict`.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `opt_dict`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]
    model_type = get(params, :model, nothing)
    is_seasonal = (model_type in [:cyclic, :harmonic]) || haskey(params, :period)
    basis_models = [:pspline, :bspline, :tps, :rff, :fft, :moran, :spherical, :barycentric, :decay, :wavelet, :linear, :invdist, :kriging, :gp, :fitc, :svgp, :nystrom, :warp, :spde]
    is_basis_model = model_type in basis_models
    if !isempty(variables)
        t_var_sym = Symbol(variables[1])
        if hasproperty(data, t_var_sym)
            if is_seasonal
                opt_dict[:u_idx] = data[!, t_var_sym]
                opt_dict[:u_N] = length(unique(opt_dict[:u_idx]))
                opt_dict[:u_idx_var] = t_var_sym
            elseif is_basis_model
                process_smooth_module!(opt_dict, mod_data, opt_dict[:basis_matrices], opt_dict[:manifolds])
            else
                time_opts = Dict(:time_method => get(params, :time_method, "regular"))
                filter!(p -> !isnothing(p.second), time_opts)
                tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
                opt_dict[:t_idx] = tu_meta.t_idx
                opt_dict[:t_N] = tu_meta.tn
                opt_dict[:t_idx_var] = t_var_sym
            end
        else
            @warn "Temporal/Seasonal index variable ':$t_var_sym' not found in data. Effects may not be correctly applied."
        end
    else
        @warn "The `temporal()` or `seasonal()` module was called without specifying a time variable. Effects will be ignored."
    end
    return true
end

function process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, manifolds_registry)
    # Purpose: Processes the `smooth()` module call.
    # Rationale: Generates basis matrices for spline-based or spectral smoothers, or sets up coordinates for GP-based smoothers.
    # v1.2.1 (2026-07-16)
    # Assumptions: `data` is present in `opt_dict`.
    # Inputs:
    #   - opt_dict, mod_data, basis_matrices_registry, manifolds_registry.
    # Outputs: None (mutates `opt_dict` and registries).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    
    local registry_to_use
    if basis_matrices_registry isa Dict && haskey(basis_matrices_registry, :basis_matrices)
        # This indicates the main config dict `M` was passed as the third argument.
        registry_to_use = basis_matrices_registry[:basis_matrices]
    else
        # This indicates the basis_matrices dict itself was passed.
        registry_to_use = basis_matrices_registry
    end

    data = opt_dict[:data]
    params = mod_data[:params]
    # The default model for a smooth term is a P-spline.
    model_param = get(params, :model, "pspline")
    basis_models = ["pspline", "bspline", "tps", "fft", "moran", "spherical", "barycentric", "decay", "wavelet", "linear", "invdist"]
    continuous_kernel_models = ["gp", "fitc", "svgp", "nystrom", "warp", "spde", "exponentialdecay", "rff", "kriging"]
    gmrfs_on_bins_models = ["rw1", "rw2", "ar1", "icar", "besag", "cyclic"]
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
                if !isnothing(ls_i); kwargs_i[:lengthscale] = ls_i; end
                vals_i = data[!, var_sym]
                push!(B_list, bstm_smooth_basis_1D(model_i, vals_i, nbins_i, degree_i; kwargs_i...))
                template_type = if model_i in ["pspline", "rw2"]; :rw2; elseif model_i == "rw1"; :rw1; else :iid; end
                push!(Q_list, build_structure_template(template_type, nbins_i).matrix)
                push!(nbins_list, nbins_i)
            end
            B_final = B_list[end]
            for i in (n_vars-1):-1:1
                B_final = kron(B_final, B_list[i])
            end
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
            return
        end
    end
    model_str = string(model_param)
    if model_str in basis_models
        if !isempty(mod_data[:variables])
            nb = get(mod_data[:params], :nbins, 20)
            reg_key = Symbol(join(mod_data[:variables], "_"))
            if all(hasproperty(data, Symbol(v)) for v in mod_data[:variables])
                n_vars = length(mod_data[:variables])
                if n_vars == 1
                    v_vec = data[!, Symbol(mod_data[:variables][1])]
                    registry_to_use[reg_key] = bstm_smooth_basis_1D(model_str, v_vec, nb, get(mod_data[:params], :degree, 3); mod_data[:params]...)
                else
                    c_mat = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
                    if n_vars == 2; registry_to_use[reg_key] = bstm_smooth_basis_2D(model_str, c_mat, nb; mod_data[:params]...);
                    elseif n_vars == 3; registry_to_use[reg_key] = bstm_smooth_basis_3D(model_str, c_mat, nb; mod_data[:params]...);
                    elseif n_vars == 4; registry_to_use[reg_key] = bstm_smooth_basis_4D(model_str, c_mat, nb; mod_data[:params]...);
                    end
                end
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
            @warn "Continuous kernel smooth specified, but coordinate variables not found in data. Manifold may be misspecified."
        end
    elseif model_str in gmrfs_on_bins_models
        vars = mod_data[:variables]
        if length(vars) != 1; @warn "GMRF smooth on $(join(vars, ",")) requires exactly 1 variable. Skipping."; return; end
        
        # When a GMRF is applied to a binned covariate, it's treated as a mixed effect.
        # We need to store the generated indices in the main config object `M` so the
        # code generator can find them.
        index_key = Symbol("mixed_idx_$(string(vars[1]))")
        opt_dict[index_key] = indices
        var_sym = Symbol(vars[1])
        nbins = get(mod_data[:params], :nbins, 20)
        _, indices = apply_discretization_logic(data[!, var_sym], nbins)
        mod_data[:params][:indices] = indices
        mod_data[:params][:n_cat] = length(unique(indices))
        mod_data[:type] = :mixed
    end    
    # Ensure the resolved model type is stored back for the manifold constructor.
    mod_data[:params][:model] = model_param

    return true
end


function process_svar_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `svar()` module.
    # Rationale: This processor parses the inner spatial manifold that defines the structure
    #            of the spatially varying autoregressive coefficient, rho.
    
    components = get(mod_data[:params], :positional_args, [])
    if isempty(components) || !(components[1] isa NamedTuple) || get(components[1], :module_type, :none) != :spatial
        error("The `svar()` module requires a spatial manifold as its first argument, e.g., `svar(spatial(s_idx, model=icar))")
    end

    # The inner spatial manifold is the first positional argument.
    spatial_node = components[1]
    
    # Store the parsed spatial node in the parameters. The `resolve_technical_primitive`
    # function for `svar` will use this to recursively resolve the inner manifold.
    mod_data[:params][:rho_spatial_node] = spatial_node
    
    # The SVAR manifold itself will be created.
    return true
end




function process_dynamics_module!(opt_dict::Dict, mod_data::Dict, registries::Dict, hyperpriors::Dict)
    # 1. Initialize Spatial Context
    # Dynamics models inherently operate on a spatial graph. We must ensure the 
    # adjacency structure and spatial indices are resolved first.
    process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)

    params = mod_data[:params]
    data = opt_dict[:data]
    
    # 2. Model Type Verification
    # Supported models: "advection", "diffusion", "advection_diffusion"
    model_type = string(get(params, :model, "none"))
    if model_type == "none"
        error("Dynamics module requires a 'model' parameter (e.g., model='advection').")
    end

    # 3. Covariate and Parameter Validation
    # For mechanistic models, we check for required physical parameters or threshold variables.
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

    # 4. Spatiotemporal Indexing
    # Dynamics models evolve over time; we ensure temporal indices are present.
    if !haskey(opt_dict, :t_idx)
        @warn "Dynamics module detected but no temporal indices found. Attempting default temporal resolution."
        if hasproperty(data, :year)
            opt_dict[:t_idx] = data[!, :year] .- minimum(data[!, :year]) .+ 1
            opt_dict[:t_N] = length(unique(opt_dict[:t_idx]))
        else
            error("Dynamics models require temporal indices. Provide a time variable via temporal() or ensure 'year' is in data.")
        end
    end

    # 5. Mapping Spatiotemporal State
    # We pre-calculate the spatiotemporal flat index (st_idx) to allow the 
    # code generator to map the [s_N, t_N] state matrix to the observation vector N.
    s_idx = opt_dict[:s_idx]
    t_idx = opt_dict[:t_idx]
    s_N = opt_dict[:s_N]
    
    # st_idx = (t-1)*s_N + s
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
    
    return true # Proceed with manifold creation.
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
- `true` to indicate that a `MixedManifold` object should be created.
"""

function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # # Process: Orchestrates the setup for random effects (MixedManifold).
    # # Rationale: Resolves formula parsing issues by ensuring a valid response variable is present 
    # # during schema application.

    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    # # Retrieve the first outcome variable to serve as a valid response placeholder for StatsModels
    response_var = Symbol(opt_dict[:outcomes][1])

    local effect_expr, group_var_str

    # # Logical Dispatch: Determine formula syntax style
    if !isempty(vars) && vars[1] isa Expr && vars[1].head == :call && vars[1].args[1] == :|
        # # Case: mixed(effect | group)
        effect_expr = vars[1].args[2]
        group_expr = vars[1].args[3]
        group_var_str = string(group_expr)
    elseif length(vars) >= 2
        # # Case: mixed(effect, group)
        effect_expr = vars[1]
        group_var_str = string(vars[2])
    else
        @warn "The mixed() module requires syntax `mixed(effect | group)` or `mixed(effect, group)`. Skipping."
        return false
    end

    # # Technical Audit: Normalize 'intercept()' to '1' for StatsModels compatibility
    effect_expr_mod = _replace_bstm_modules_in_expr(effect_expr)

    # # Formula Parsing Engine
    schema = StatsModels.schema(data)
    local terms

    if effect_expr_mod isa Number
        # # Direct conversion for pure intercept/zero-intercept models
        terms = StatsModels.term(effect_expr_mod)
    else
        # # Robust Parsing: Use the actual response variable to avoid 'nothing' KeyError
        calling_mod = get(opt_dict, :calling_module, Main)
        
        # # Create a valid formula for schema validation
        form = Core.eval(calling_mod, :(@formula($response_var ~ $(effect_expr_mod))))
        
        # # Apply schema to resolve categorical contrasts and variable types
        applied_form = StatsModels.apply_schema(form, schema)
        terms = applied_form.rhs
    end

    # # Term Normalization: Consolidate effects into a string vector
    term_vec = terms isa Tuple ? collect(terms) : [terms]
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

    # # Index Resolution: Construct group assignments
    group_var_sym = Symbol(group_var_str)
    if !hasproperty(data, group_var_sym)
        error("Grouping variable ':$group_var_sym' for mixed() module not found in dataset.")
    end
    
    group_data = data[!, group_var_sym]
    unique_levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(unique_levels))
    indices = [group_map[v] for v in group_data]

    # # Registry Update: Store metadata for code generator
    index_key = Symbol("mixed_idx_$(group_var_str)")
    opt_dict[index_key] = indices
    
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = length(unique_levels)
    mod_data[:params][:lhs] = effect_names

    # # Domain mapping for downstream dispatch
    mod_data[:variables] = [group_var_str]
    
    return true
end



"""
    _generate_manifold_code_fragments(m::CustomManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for a user-defined `CustomManifold`.

# Rationale for New Implementation
The previous implementation incorrectly dispatched `CustomManifold` to a generic GMRF
generator, ignoring the user-provided code. This new, specialized function correctly
implements the intended behavior by directly injecting the user's code into the model.

The `code_fragment` provided by the user in the `custom()` module is expected to be a
complete and valid block of Turing model code. This block is inserted directly into the
model's main assembly block. The user is responsible for defining any necessary priors
and update logic within this fragment. The function returns an empty `priors` string
and places the entire user code into the `update` string, as Turing does not
distinguish between these contexts within the `@model` macro.
"""
function _generate_manifold_code_fragments(m::CustomManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # The user's code fragment is expected to be a self-contained block
    # that includes both prior definitions and update logic for the linear predictor.
    
    user_code = m.code_fragment
    
    if isempty(strip(user_code))
        @warn "Custom manifold '$(spec.key)' was specified but the `code_fragment` is empty. This component will have no effect."
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



"""
    process_interact_module!(opt_dict, mod_data, registries, hyperpriors)

Processes interaction modules created by operators like `⊗`, `∘`, and `|>`.

# Rationale for Update
This function has been comprehensively reviewed and updated to ensure all documented
interaction types are fully functional.
1.  **`smooth() ⊗ smooth()`**: The logic to handle interactions between two `smooth`
    terms has been implemented. It correctly constructs the tensor product basis
    matrix and the corresponding tensor product penalty matrix, enabling the modeling
    of non-linear interactions between two continuous covariates.
2.  **`spatial() ∘ smooth()`**: The logic for this non-stationary variance model, where
    a smoother modulates the variance of a spatial field, was already present. It
    correctly injects custom code fragments into the module's parameters. The fix is
    completed in the corresponding code generator.
3.  **`spatial() |> smooth()`**: The logic for spatially-varying curves was already
    correctly implemented and is preserved.
4.  **Other Interactions**: The handling of `fixed() ⊗ fixed()`, `fixed() ⊗ smooth()`,
    and `spatial() ⊗ temporal()` interactions has been verified as correct.
"""
function process_interact_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes interaction modules created by operators like `⊗`, `∘`, and `|>`.
    # Rationale: This version is updated to recognize the `fixed |> spatial` pattern, which
    #            defines a Spatially Varying Coefficient (SVC) model. It re-tags the module
    #            as `:svc` and stores the covariate and inner spatial model information,
    #            allowing the rest of the configuration pipeline to correctly build the SVC.
    # v1.4.1 (2026-07-20)
    op = get(mod_data, :operator, get(mod_data[:params], :operator, nothing))
    components = get(mod_data, :components, get(mod_data[:params], :components, []))
    if isnothing(op) || isempty(components)
        @warn "Interaction module found with no operator or components. Skipping."
        return false
    end

    if op == :pipe && length(components) == 2
        node1, node2 = components[1], components[2]
        
        is_spatially_varying_curve = node1.module_type == :spatial && node2.module_type == :smooth
        is_svc = node1.module_type == :fixed && node2.module_type == :spatial

        if is_spatially_varying_curve
            # This handles `spatial(...) |> smooth(...)`, creating spatially varying curves.
            dynamic_vars = get(node2.args, :positional_args, [])
            if isempty(dynamic_vars); @warn "The dynamic component (smooth) of a pipe operator is missing variables. Skipping."; return false; end
            
            smooth_mod_data = Dict(:type => :smooth, :variables => dynamic_vars, :params => node2.args)
            process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:manifolds])
            
            basis_key = Symbol(join(dynamic_vars, "_"))
            mod_data[:params][:dynamic_basis_key] = basis_key
            
            scheme = get(opt_dict, :prior_scheme, :pcpriors)
            dynamic_manifold_obj = resolve_technical_primitive(smooth_mod_data, opt_dict, hyperpriors, scheme)
            mod_data[:params][:dynamic_manifold_obj] = dynamic_manifold_obj
            
            return true # Proceed to create the ComposedManifold object.

        elseif is_svc
            # This is a Spatially Varying Coefficient model: `fixed() |> spatial()`.
            covariate_node = node1
            spatial_node = node2
            
            cov_args = get(covariate_node.args, :positional_args, [])
            if isempty(cov_args); @warn "SVC model is missing a covariate. Skipping."; return false; end
            
            covariate_name = cov_args[1]
            
            # Re-tag the module as an SVC module.
            mod_data[:type] = :svc
            mod_data[:variables] = [covariate_name, get(spatial_node.args, :positional_args, [])...]
            
            # Store the necessary information for the builder and generator.
            mod_data[:params][:covariate] = covariate_name
            mod_data[:params][:spatial_model_spec] = spatial_node # Pass the whole parsed node
            
            return true # Proceed to create the SVCManifold object.
        end
    end

    if op == :kronecker_product && length(components) == 2
        c1, c2 = components[1], components[2]
        c1_type, c2_type = get(c1, :module_type, :unknown), get(c2, :module_type, :unknown)

        fixed_node = c1_type == :fixed ? c1 : (c2_type == :fixed ? c2 : nothing)
        smooth_node = c1_type == :smooth ? c1 : (c2_type == :smooth ? c2 : nothing)
        is_fixed_fixed_interaction = c1_type == :fixed && c2_type == :fixed

        if is_fixed_fixed_interaction
            var1_sym = get(c1.args, :positional_args, [])[1]
            var2_sym = get(c2.args, :positional_args, [])[1]
            interaction_term = "$(string(var1_sym))&$(string(var2_sym))"
            if !haskey(opt_dict, :fixed_effects); opt_dict[:fixed_effects] = String[]; end
            push!(opt_dict[:fixed_effects], interaction_term)
            return false

        elseif !isnothing(fixed_node) && !isnothing(smooth_node)
            fixed_var_sym = Symbol(get(fixed_node.args, :positional_args, [])[1])
            smooth_vars = get(smooth_node.args, :positional_args, [])
            data = opt_dict[:data]
            if !hasproperty(data, fixed_var_sym); @warn "Grouping variable ':$fixed_var_sym' for interaction not found. Skipping."; return true; end
            if !(eltype(data[!, fixed_var_sym]) <: CategoricalValue); data[!, fixed_var_sym] = categorical(data[!, fixed_var_sym]); end
            group_levels = levels(data[!, fixed_var_sym])
            n_levels = length(group_levels)
            group_indices = levelcode.(data[!, fixed_var_sym])
            smooth_params = smooth_node.args
            smooth_model_str = string(get(smooth_params, :model, "pspline"))
            nbins = get(smooth_params, :nbins, 20)
            degree = get(smooth_params, :degree, 3)
            if length(smooth_vars) != 1; @warn "Interaction with multi-dimensional smooths not supported. Skipping."; return true; end
            smooth_var_sym = Symbol(smooth_vars[1])
            smooth_vals = data[!, smooth_var_sym]
            B_smooth = bstm_smooth_basis_1D(smooth_model_str, smooth_vals, nbins, degree; smooth_params...)
            k_bins = size(B_smooth, 2)
            B_interaction = spzeros(size(data, 1), n_levels * k_bins)
            for i in 1:size(data, 1); B_interaction[i, ((group_indices[i]-1)*k_bins+1):(group_indices[i]*k_bins)] = B_smooth[i, :]; end
            diff_order = get(smooth_params, :diff_order, 2)
            smooth_penalty_type = if smooth_model_str in ["pspline", "rw2"]; :rw2; elseif smooth_model_str == "rw1"; :rw1; else :iid; end
            Q_smooth_template = build_structure_template(smooth_penalty_type, k_bins).matrix
            Q_interaction = kron(sparse(I, n_levels, n_levels), Q_smooth_template)
            mod_data[:type] = :smooth
            mod_data[:variables] = [fixed_var_sym, smooth_var_sym]
            mod_data[:params][:model] = :tensorproductsmooth
            mod_data[:params][:Q_template] = Q_interaction
            mod_data[:params][:nbins] = n_levels * k_bins
            interaction_key = Symbol(join([fixed_var_sym, smooth_vars...], "_"))
            opt_dict[:basis_matrices][interaction_key] = B_interaction
            return true

        elseif c1_type == :smooth && c2_type == :smooth
            smooth_node1, smooth_node2 = c1, c2
            data = opt_dict[:data]
            vars1 = get(smooth_node1.args, :positional_args, []); if length(vars1) != 1; @warn "First smoother in interaction must be 1D. Skipping."; return true; end
            var1_sym = Symbol(vars1[1]); params1 = smooth_node1.args; model1_str = string(get(params1, :model, "pspline")); nbins1 = get(params1, :nbins, 20); degree1 = get(params1, :degree, 3)
            vals1 = data[!, var1_sym]; B1 = bstm_smooth_basis_1D(model1_str, vals1, nbins1, degree1; params1...); penalty_type1 = if model1_str in ["pspline", "rw2"]; :rw2; elseif model1_str == "rw1"; :rw1; else :iid; end
            Q1 = build_structure_template(penalty_type1, nbins1).matrix
            vars2 = get(smooth_node2.args, :positional_args, []); if length(vars2) != 1; @warn "Second smoother in interaction must be 1D. Skipping."; return true; end
            var2_sym = Symbol(vars2[1]); params2 = smooth_node2.args; model2_str = string(get(params2, :model, "pspline")); nbins2 = get(params2, :nbins, 20); degree2 = get(params2, :degree, 3)
            vals2 = data[!, var2_sym]; B2 = bstm_smooth_basis_1D(model2_str, vals2, nbins2, degree2; params2...); penalty_type2 = if model2_str in ["pspline", "rw2"]; :rw2; elseif model2_str == "rw1"; :rw1; else :iid; end
            Q2 = build_structure_template(penalty_type2, nbins2).matrix
            B_interaction = spzeros(size(data, 1), nbins1 * nbins2)
            for i in 1:size(data, 1); B_interaction[i, :] = kron(B2[i, :], B1[i, :]); end
            Q_interaction = kron(sparse(I, nbins2, nbins2), Q1) + kron(Q2, sparse(I, nbins1, nbins1))
            mod_data[:type] = :smooth; mod_data[:variables] = [var1_sym, var2_sym]; mod_data[:params][:model] = :tensorproductsmooth
            mod_data[:params][:Q_template] = Q_interaction; mod_data[:params][:nbins] = nbins1 * nbins2
            interaction_key = Symbol(join([var1_sym, var2_sym], "_")); opt_dict[:basis_matrices][interaction_key] = B_interaction
            return true
        end
    
    elseif op == :composition && length(components) == 2
        base_node, modifier_node = components[1], components[2]
        is_nonstationary_variance = base_node.module_type == :spatial && modifier_node.module_type == :smooth
        if is_nonstationary_variance
            modifier_vars = get(modifier_node.args, :positional_args, [])
            if isempty(modifier_vars); @warn "The modifier component (smooth) of a composition operator is missing variables. Skipping."; return false; end
            smooth_mod_data = Dict(:type => :smooth, :variables => modifier_vars, :params => modifier_node.args)
            process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:manifolds])
            basis_key = Symbol(join(modifier_vars, "_"))
            mod_data[:params][:modifier_basis_key] = basis_key
            scheme = get(opt_dict, :prior_scheme, :pcpriors)
            base_manifold_obj = resolve_technical_primitive(Dict(:type => base_node.module_type, :params => base_node.args, :variables => get(base_node.args, :positional_args, [])), opt_dict, hyperpriors, scheme)
            modifier_manifold_obj = resolve_technical_primitive(smooth_mod_data, opt_dict, hyperpriors, scheme)
            key_str = string(mod_data[:key]); nbins = modifier_manifold_obj.nbins
            mod_data[:params][:priors] = "sv_sigma_smoother_$(key_str) ~ NamedDist(Exponential(1.0), :sv_sigma_smoother_$(key_str))\n" * "sv_coeffs_smoother_$(key_str) ~ NamedDist(MvNormal(zeros(T, $(nbins)), I), :sv_coeffs_smoother_$(key_str))\n" * "sv_icar_raw_$(key_str) ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :sv_icar_raw_$(key_str))"
            mod_data[:params][:update] = "begin\n Q_smoother_template = build_structure_template(:rw2, $(nbins)).matrix; F_smoother = cholesky(Symmetric(Q_smoother_template + noise * I)); smoother_latent = sv_sigma_smoother_$(key_str) .* (F_smoother.U \\ sv_coeffs_smoother_$(key_str)); log_sigma_field = M.basis_matrices[:$(basis_key)] * smoother_latent; spatially_varying_sigma = exp.(log_sigma_field); Q_icar_template = spec_registry[\"$(key_str)\"].hyper.base_spec.Q_template; F_icar = cholesky(Symmetric(Q_icar_template + noise * I)); icar_latent = F_icar.U \\ sv_icar_raw_$(key_str); final_effect = view(icar_latent, M.s_idx) .* spatially_varying_sigma; eta .+= final_effect;\nend"
            mod_data[:params][:base_manifold_obj] = base_manifold_obj; mod_data[:params][:modifier_manifold_obj] = modifier_manifold_obj
            return true
        end
 
    elseif op == :kronecker_product
        # Capture custom prior for the interaction sigma if provided via `⊗`
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

    end

    if op == :kronecker_product
        if length(components) == 2
            s_idx = get(opt_dict, :s_idx, nothing); t_idx = get(opt_dict, :t_idx, nothing); s_N = get(opt_dict, :s_N, nothing)
            if !isnothing(s_idx) && !isnothing(t_idx) && !isnothing(s_N); mod_data[:params][:indices] = [(t - 1) * s_N + s for (s, t) in zip(s_idx, t_idx)];
            else @warn "Could not compute Kronecker product indices for '$(mod_data[:variables])'. Ensure spatial and temporal manifolds are defined."; end
            c1_type = get(components[1], :module_type, :unknown); c2_type = get(components[2], :module_type, :unknown)
            spatial_node = c1_type == :spatial ? components[1] : (c2_type == :spatial ? components[2] : nothing)
            temporal_node = c1_type == :temporal ? components[1] : (c2_type == :temporal ? components[2] : nothing)
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



function process_spacetime_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `spacetime()` convenience module.
    # Rationale: Determines the Knorr-Held interaction type based on the specified spatial and temporal models.
    # v1.2.1 (2026-07-16)
    # Assumptions: `model` parameter is a tuple of (spatial_model, temporal_model).
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `opt_dict`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    #          `false` for spacetime as it's handled by a global assembler block.
    models = get(mod_data[:params], :model, (:iid, :iid))
    spatial_model = string(models[1])
    temporal_model = string(models[2])
    has_structured_space = spatial_model != "iid"
    has_structured_time = temporal_model != "iid"
    if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
    elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
    elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
    else opt_dict[:model_st] = "I";
    end

    # Capture and resolve custom prior for the interaction sigma
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
    return false # Do not create a separate manifold object
end

function process_fixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `fixed()` module.
    # Rationale: Gathers information about fixed effects, including custom contrasts and priors.
    # v1.2.1 (2026-07-16)
    # Assumptions: `mod_data` contains variables and optional parameters.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `opt_dict`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
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
    return false # fixed() modules do not create manifolds in the main loop.
end

function process_custom_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `custom()` module.
    # Rationale: Placeholder for user-defined custom model components.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None.
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
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
    # v1.2.1 (2026-07-16)
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Boolean indicating if a manifold should be created.
    params = mod_data[:params]
    if !haskey(params, :W) || isempty(params[:W]) || all(iszero, params[:W])
        error("The `bcgn()` module requires a non-empty `:W` sparse matrix parameter.")
    end

    res = adjacency_to_bipartite(params[:W])
    params[:bipartite_adj] = res.bipartite_adj

    # Further validation could be added here (e.g., check if it's actually bipartite)
    return true # Proceed with manifold creation
end

function process_networkflow_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `networkflow()` module.
    # Rationale: Validates the provided adjacency matrix for the network flow model.
    # v1.2.1 (2026-07-16)
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Boolean indicating if a manifold should be created.
    params = mod_data[:params]
    if !haskey(params, :adjacency_matrix) || isempty(params[:adjacency_matrix]) || all(iszero, params[:adjacency_matrix])
        error("The `networkflow()` module requires a non-empty `:adjacency_matrix` sparse matrix parameter.")
    end
    return true # Proceed with manifold creation
end


function process_localadaptive_module!(opt_dict, mod_data, registries, hyperpriors)
    process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)

    if !haskey(opt_dict, :centroids)
        error("The `localadaptive()` model requires centroids for clustering, but they were not found. Ensure spatial coordinates are provided to infer areal units and their centroids.")
    end
    
    centroids = opt_dict[:centroids]
    params = mod_data[:params]
    
    n_clusters = get(params, :n_clusters, 5)
    
    if size(centroids, 1) < n_clusters
        @warn "Number of spatial units ($(size(centroids, 1))) is less than the requested number of clusters ($n_clusters). Adjusting n_clusters to $(size(centroids, 1))."
        n_clusters = size(centroids, 1)
    end
    
    centroids_matrix = hcat(collect.(centroids)...)
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
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
    if !haskey(opt_dict, :nested_manifolds); opt_dict[:nested_manifolds] = Dict{Symbol, Any}(); end
    
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
    
    opt_dict[:nested_manifolds][var] = sub_config
    
    return false
end


const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :spatial => process_spatial_module!,
    :temporal => process_temporal_module!,
    :smooth => process_smooth_module!,
    :dynamics => process_dynamics_module!,
    :eigen => process_eigen_module!,
    :mixed => process_mixed_module!,
    :nested => process_nested_module!,
    :interact => process_interact_module!,
    :spacetime => process_spacetime_module!,
    :fixed => process_fixed_module!,
    :svar => process_svar_module!,
    :localadaptive => process_localadaptive_module!,
    :mosaic => process_mosaic_module!,
    :custom => process_custom_module!,
    :bcgn => process_bcgn_module!,
    :networkflow => process_networkflow_module!
)



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
        println("  - Review the configuration of the manifold that caused the error. The number of latent units (e.g., `n_cat` for a mixed effect, or `s_N` for a spatial effect) might be miscalculated.")
        println("  - Check for off-by-one errors in manual indexing if using a `custom()` manifold.")
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


macro bstm(exprs...)
    # Purpose: The main user-facing macro for defining a bstm model. It supports two syntaxes:
    #   1. `m = @bstm(formula, data, ...)`: Returns the model object.
    #   2. `@bstm m = formula, data, ...`: Assigns the model to `m` and returns `nothing`.
    # Rationale: This simplified version offloads all complex logic to the `bstm` function,
    #            using the macro only to provide the unquoted formula syntax and capture the
    #            caller's module context. This avoids complex macro hygiene and scoping issues.
    # v1.3.6 (2026-07-18) - Corrected macro logic to remove legacy code and fix variable scoping.

    # --- Parse Macro Arguments ---
    local formula, data, kwargs
    local var_name = nothing
    local is_assignment = false

    if !isempty(exprs) && exprs[1] isa Expr && exprs[1].head == :(=)
        # Handles `@bstm m = formula, data, ...`
        is_assignment = true
        var_name = exprs[1].args[1]
        formula = exprs[1].args[2]
        data = length(exprs) > 1 ? exprs[2] : nothing
        kwargs = length(exprs) > 2 ? exprs[3:end] : ()
    else
        # Handles `m = @bstm(formula, data, ...)`
        formula = exprs[1]
        data = length(exprs) > 1 ? exprs[2] : nothing
        kwargs = length(exprs) > 2 ? exprs[3:end] : ()
    end

    # The macro's main job is to convert the formula to a string and escape arguments.
    formula_str = string(formula)
    data_esc = esc(data)
    kwargs_esc = [esc(kw) for kw in kwargs]

    # --- Core Logic: A simple call to the `bstm` function ---
    # We pass `__module__`, the special macro variable for the caller's module,
    # as a positional argument to the `bstm` function. the ":" is the quote operator
    core_logic = :(bstm($formula_str, $data_esc, $(__module__); $(kwargs_esc...)))

    # --- Final Macro Expansion ---
    if is_assignment
        # For `@bstm m = ...`, assign the result of core_logic to `m` and return `nothing`.
        return quote
            $(esc(var_name)) = $(core_logic)
            nothing
        end
    else
        # For `m = @bstm(...)`, return the core_logic block, which evaluates to the model instance.
        return core_logic
    end
end


function bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # # Process: Translates a high-level formula and dataset into a Turing model instance.
    # # Rationale: Separates the technical configuration from the dynamic code generation
    # # and subsequent model instantiation.

    # # Generate model configuration dictionary based on formula syntax and data schema
    options = bstm_config(formula, data, calling_module = calling_module, kwargs...)

    # # Invoke the codegen engine to produce the model source string and expression
    model_func_name, expr, new_config, registry = bstm_codegen(options)

    if get(new_config, :verbose, true)
        println("\n--- Dynamically Generated Model Code ---")
        println(new_config.generated_model_code)
        println("----------------------------------------\n")
    end

    # # Evaluate the generated @model macro expression in the target module scope
    calling_module.eval(expr)

    # # Access the function binding from the module's global scope
    # # getfield retrieves the handle to the function defined above
    model_func = getfield(calling_module, model_func_name)

    # # Instantiation of the Turing Model Object
    # # Julia 1.12+ requires invokelatest for accessing bindings defined in the same turn.
    # # This prevents WorldAge errors and warnings.
    model_instance = Base.invokelatest(model_func, new_config, registry)

    if get(new_config, :verbose, true)
        println("\n--- Running prior predictive check ---")
    end

    prior_sample = nothing
    try
        # # Prior Predictive Validation
        # # We also use invokelatest for the rand call to ensure the dynamic method 
        # # dispatch for the new model type is resolved correctly.
        prior_sample = Base.invokelatest(rand, model_instance)
        # # Redirect stderr to devnull to suppress world-age warnings during this check.
        redirect_stderr(devnull) do
            prior_sample = Base.invokelatest(rand, model_instance)
        end
    
        if get(new_config, :verbose, true) && !isnothing(prior_sample)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
    catch e 
        # # Error handling for structural or parameterization issues
        _bstm_error_handler(e, model_instance)
    end

    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end

    # # Return the fully configured and validated model object
    return model_instance
end

function bstm(formula::String, data::DataFrame; kwargs...)
    # # Convenience overload defaulting to the Main execution scope
    return bstm(formula, data, Main, kwargs...)
end




function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]

    if m_type == :svc
        covariate_sym = Symbol(get(m_params, :covariate, :unknown))
        spatial_model_spec_node = get(m_params, :spatial_model_spec, nothing)
        spatial_mod_data = Dict(
            :type => spatial_model_spec_node.module_type, 
            :params => spatial_model_spec_node.args, 
            :variables => get(spatial_model_spec_node.args, :positional_args, [])
        )
        inner_manifold_obj = resolve_technical_primitive(spatial_mod_data, M, priors_dict, scheme)
        return SVCManifold(covariate_sym, inner_manifold_obj)

    elseif m_type == :mixed
        group_var_sym = Symbol(module_metadata[:variables][1])
        lhs_str = module_metadata[:params][:lhs]
        model_name = string(get(m_params, :model, "iid"))
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        inner_model_obj = MANIFOLD_CONSTRUCTORS[Symbol(model_name)](resolved_priors, m_params)
        return MixedManifold(group_var_sym, lhs_str, inner_model_obj)

    elseif m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(
            :type => c_node.module_type, 
            :params => c_node.args, 
            :variables => get(c_node.args, :positional_args, [])), 
            components_data
        )
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
        return ComposedManifold(resolved_components, op)

    elseif m_type == :adaptivesmooth
        resolved_priors = resolve_hyperpriors("adaptivesmooth", priors_dict, m_params, scheme, M[:calling_module])
        h_dim = get(m_params, :hidden_dim, 10)
        n_bins = get(m_params, :nbins, 20)
        return AdaptiveSmooth(h_dim, n_bins, resolved_priors.sigma)

    else
        default_model = if m_type == :spatial; haskey(M, :W) ? "bym2" : "iid"; elseif m_type == :temporal; "rw2"; else "none"; end
        model_name = string(get(m_params, :model, default_model))
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        return MANIFOLD_CONSTRUCTORS[Symbol(model_name)](resolved_priors, m_params)
    end
end




function _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
    # Purpose: Computes a robust scaling factor for a precision matrix from its eigenvalues.
    # Rationale: The scaling factor is the geometric mean of the non-zero eigenvalues.
    #            This method avoids using a fixed tolerance to identify zero eigenvalues,
    #            which can be sensitive to floating-point noise. Instead, it uses the known
    #            rank deficiency of the GMRF model to correctly identify the structural zero
    #            eigenvalues.
    # v1.4.5 (2026-07-21)
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


"""
    build_structure_template(type::Symbol, n::Int; scale=true, W=nothing)

A factory for creating precision matrix templates for various GMRF models.

# Rationale for BYM2
For the `bym2` type, this function constructs the scaled precision matrix for the
intrinsic (ICAR) component. It computes the graph Laplacian `D-W` and then scales
it by the geometric mean of its non-zero eigenvalues. This scaling is crucial for
the interpretability of the `sigma` hyperparameter in the BYM2 model.
"""
function build_structure_template(type::Symbol, n::Int; scale=true, W=nothing)
    Q = nothing
    sf = 1.0

    if type == :iid || type == :none || type == :identity || type == :harmonic || type == :rff
        return (matrix = sparse(I(n)), scaling_factor = 1.0)
    elseif type in [:icar, :besag, :bym2, :leroux, :localadaptive, :spde]
        if isnothing(W); error("Adjacency matrix W required for manifold :$type"); end
        D_sp = spdiagm(0 => vec(sum(W, dims=2)))
        Q_raw = D_sp - W
        if scale
            evals = eigvals(Matrix(Q_raw))
            # Intrinsic CAR models on a connected graph have a rank deficiency of 1.
            sf = _compute_scaling_factor(evals, 1)
            Q = Q_raw ./ sf
        else
            Q = Q_raw
        end
    elseif type == :networkflow
        if isnothing(W); error("Adjacency matrix W required for manifold :networkflow"); end
        Q = W
        sf = 1.0
    elseif type == :sar
        if isnothing(W); error("Adjacency matrix W required for manifold :sar"); end
        row_sums = sum(W, dims=2)
        D_inv = spdiagm(0 => 1.0 ./ (vec(row_sums) .+ 1e-9))
        Q = D_inv * W
        sf = 1.0
    elseif type == :dag
        if isnothing(W); error("Adjacency matrix W required for manifold :dag"); end
        Q = tril(W, -1)
        sf = 1.0
    elseif type == :rw1
        # RW1 with Neumann boundary conditions is a proper (non-singular) GMRF.
        # It has rank deficiency 0.
        Q_raw = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        Q_raw[1,1] = Q_raw[n,n] = 1.0
        if scale
            evals = eigvals(Matrix(Q_raw))
            sf = _compute_scaling_factor(evals, 0)
            Q = Q_raw ./ sf
        else; Q = Q_raw; end
    elseif type == :ar1
        Q = spdiagm(-1 => fill(1.0, n-1), 1 => fill(1.0, n-1))
        sf = 1.0
    elseif type == :rw2
        # Intrinsic RW2 has a rank deficiency of 2.
        D_rw2 = spdiagm(-2 => ones(n-2), -1 => -2*ones(n-1), 0 => ones(n), 1 => -2*ones(n-1), 2 => ones(n-2))
        Q_raw = D_rw2' * D_rw2
        if scale
            evals = eigvals(Matrix(Q_raw))
            sf = _compute_scaling_factor(evals, 2)
            Q = Q_raw ./ sf
        else; Q = Q_raw; end
    elseif type == :cyclic

        # This constructs a circulant precision matrix for a cyclic random walk of order 1.
        # The `n-1` and `-(n-1)` diagonals handle the wrap-around connection between
        # the first and last elements.
        Q_raw = spdiagm(
            0 => fill(2.0, n), 
            -1 => fill(-1.0, n-1), 
            1 => fill(-1.0, n-1), 
            n-1 => [-1.0], 
            -(n-1) => [-1.0]
        )
        
        if scale
            evals = eigvals(Matrix(Q_raw))
            # A cyclic random walk is an intrinsic GMRF with rank deficiency 1.
            sf = _compute_scaling_factor(evals, 1)
            Q = Q_raw ./ sf
        else
            Q = Q_raw
        end
    
    else
        @warn "BSTM Registry Fallback: Manifold :$type not recognized. Initializing Identity."
        Q = sparse(I(n))
        sf = 1.0
    end
    return (matrix = Q, scaling_factor = sf)
end



# Generic builder for standard Manifold types
function build_model(m::Manifold, data_inputs::Dict, module_metadata::Dict)
    domain = get(module_metadata, :type, :spatial)
    return _build_from_template(m, data_inputs, domain, module_metadata)
end

function build_model(m::CustomManifold, data_inputs::Dict, module_metadata::Dict)
"""
    build_model(m::CustomManifold, data_inputs::Dict, module_metadata::Dict)

A model builder for the `CustomManifold`.

# Rationale
This is a new function that ensures `CustomManifold` is handled correctly by the
configuration engine. Since a custom manifold is defined entirely by user-provided
code, it does not have a predefined structure matrix (`Q_template`). This builder
uses the `_build_pass_through_model` helper to signal that no precision matrix
template is needed, preventing the framework from incorrectly trying to build one.
"""
    # This manifold is defined entirely by user code, so it doesn't have a Q_template.
    # We use the pass-through builder to indicate this.
    return _build_pass_through_model(m, data_inputs, module_metadata)
end


# Specialized builder for IID to ensure domain-specific template resolution
function build_model(m::IID, data_inputs::Dict, module_metadata::Dict)
    domain = get(module_metadata, :type, :spatial)
    return _build_from_template(m, data_inputs, domain, module_metadata)
end

# Builder for temporal Gaussian Markov Random Fields
function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :temporal, module_metadata)
end


function build_model(m::MixedManifold, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `MixedManifold`.
    # Rationale: This function correctly constructs the technical specification for a mixed effect model.
    #            It recursively calls `build_model` on the inner manifold (e.g., IID, RW2)
    #            to obtain its precision matrix template (`Q_template`), which defines the
    #            correlation structure of the random effects.
    # v1.4.2 (2026-07-20)
    
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



function build_model(m::MixedManifold, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `MixedManifold`.
    # Rationale: This function correctly constructs the technical specification for a mixed effect model.
    #            It recursively calls `build_model` on the inner manifold (e.g., IID, RW2)
    #            to obtain its precision matrix template (`Q_template`), which defines the
    #            correlation structure of the random effects.
    # v1.4.2 (2026-07-20)
    
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

A model builder for the `SPDE` manifold.
"""
function build_model(m::SPDE, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata)
end


"""
    build_model(m::LocalAdaptive, data_inputs::Dict, module_metadata::Dict)

A specialized model builder for the `LocalAdaptive` manifold.

# Rationale
This function ensures that the `n_clusters` and `cluster_assignments`, which are
pre-computed by `process_localadaptive_module!`, are correctly stored in the
manifold's `hyper` registry. This information is essential for the specialized
code generator to construct the model with cluster-specific means.
"""
function build_model(m::LocalAdaptive, data_inputs::Dict, module_metadata::Dict)
    # First, call the generic template builder to get the Q_template.
    base_spec = _build_from_template(m, data_inputs, :spatial, module_metadata)
    
    # Now, augment the hyper parameters with the clustering info.
    hyper_dict = Dict(pairs(base_spec.hyper))
    
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

function build_model(m::SVCManifold, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `SVCManifold`.
    # Rationale: This function correctly constructs the technical specification for an SVC model.
    #            It recursively calls `build_model` on the inner spatial manifold (e.g., BYM2, ICAR)
    #            to obtain its precision matrix template (`Q_template`). This template is then
    #            passed up to the main configuration, ensuring that the code generator for the
    #            SVC has the correct structural information to model the spatially varying coefficient.
    # v1.4.1 (2026-07-20)
    
    # The inner model (e.g., BYM2) determines the structure.
    # We call its builder to get the Q_template.
    
    spatial_model_spec_node = get(module_metadata[:params], :spatial_model_spec, nothing)
    if isnothing(spatial_model_spec_node); error("SVC builder is missing the inner spatial model specification."); end
    
    spatial_mod_data = Dict(:type => spatial_model_spec_node.module_type, :params => spatial_model_spec_node.args, :variables => get(spatial_model_spec_node.args, :positional_args, []))
    
    # Call the builder for the inner spatial model
    inner_spec = build_model(m.model, data_inputs, spatial_mod_data)
    
    # The SVC manifold itself doesn't have hyperparameters, but we pass them
    # from the inner model for the code generator to use.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Pass the inner spec's hyper-parameters up.
    hyper_dict[:inner_hyper] = inner_spec.hyper
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:svc, hyper=NamedTuple(hyper_dict))
end




"""
    build_model(m::Spherical, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Spherical` manifold when used as a full Gaussian Process.

# Rationale for Update
This is a new function that enables the `Spherical` manifold to be treated as a
continuous-space Gaussian Process, consistent with its definition which includes
priors for `sigma` and `range`. It ensures that the coordinate data from a `smooth()`
call is correctly captured and passed to the code generator.
"""
function build_model(m::Spherical, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords); error("Spherical manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`."); end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    
    return (Q_template=nothing, scaling_factor=1.0, model_type=:spherical, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::RFF, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `RFF` (Random Fourier Features) manifold.

# Rationale
This function configures the `RFF` model by:
1.  Ensuring that coordinate data is available, as RFF is a continuous-space model.
2.  Calculating a heuristic initial `lengthscale` from the provided prior distribution.
3.  Generating a set of fixed random projection weights (`W_fixed`) and biases (`b_fixed`)
    based on the initial lengthscale. These fixed features serve as the means for the
    priors on the adaptive `W` and `b` parameters in the model's code generator,
    providing a stable starting point for the MCMC sampler.
4.  Storing all necessary parameters and pre-computed features in the manifold's
    `hyper` registry for later use by the code generator.

# Arguments
- `m::RFF`: The RFF manifold object.
- `data_inputs::Dict`: The main model configuration dictionary.
- `module_metadata::Dict`: The parsed dictionary for the `smooth` or `spatial` module.

# Returns
- A `NamedTuple` containing the manifold's technical specification. `Q_template` is set to `nothing`
  as RFF models do not use a GMRF precision matrix template.
"""
function build_model(m::RFF, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("RFF manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or a spatial module with available coordinates.")
    end
    
    ls_prior = m.lengthscale
    ls_initial = if ls_prior isa Truncated
        mean(untruncated(ls_prior))
    elseif ls_prior isa Vector
        mean([mean(p) for p in ls_prior])
    else
        mean(ls_prior)
    end
    
    in_dims = size(coords, 2)
    W_fixed, b_fixed = generate_rff_params(in_dims, m.n_features, ls_initial, m.kernel)

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:coords] = coords
    hyper_dict[:W_fixed] = W_fixed
    hyper_dict[:b_fixed] = b_fixed
    return (Q_template=nothing, scaling_factor=1.0, model_type=:rff, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::FITC, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `FITC` (Fully Independent Training Conditional) manifold.

# Rationale for Update
This version cleans up the internal implementation by removing redundant code. Its primary
role remains to configure the `FITC` sparse Gaussian Process model by:
1.  Storing the observation coordinates (`coords`) in the `Q_template` field.
2.  Storing the pre-computed inducing point locations (`Z_inducing`) in the manifold's
    `hyper` registry for use by the code generator.
"""
function build_model(m::FITC, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("FITC manifold requires coordinates, but none were not found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("FITC manifold requires inducing points, but they were not found. This is an internal error in the `smooth` or `spatial` processor.")
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

A model builder for the `Moran` spatial manifold.

# Rationale for Update
This is a new function to correctly implement the Moran eigenvector spectral model.
It computes the Moran operator `M = (I - 11'/n)W(I - 11'/n)`, calculates its
eigenvectors, and stores them in the manifold's hyperparameter registry. These
eigenvectors serve as the basis functions for the spatial effect, aligning the
implementation with the documented behavior of `Moran's I Basis Manifold`.
"""
function build_model(m::Moran, data_inputs::Dict, module_metadata::Dict)
    W = get(data_inputs, :W, nothing)
    if isnothing(W)
        error("The `moran` manifold requires an adjacency matrix `W`, but it was not found in the model configuration.")
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

A model builder specifically for the `Mosaic` manifold.

# Rationale
This new builder method ensures that the pre-computed mosaic centers from the
`process_mosaic_module!` are correctly passed into the manifold's hyperparameter
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


"""
    _generate_manifold_code_fragments(m::Mosaic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")

Generates Turing code fragments for the `Mosaic` spatial model.

# Rationale for Implementation
This function implements the "mixture of experts" or "soft clustering" approach
for the mosaic model, aligning it with the documentation. The key steps are:
1.  **Priors for Local Experts**: It defines priors for the means of the local
    spatial effects (`mu_local`), one for each of the `n_regions`. These are the "experts".
2.  **Softmax Weighting**: For each observation, it calculates the distance to every
    mosaic center. These distances are then transformed via a softmax function to
    produce a vector of weights. These weights represent the "responsibility" or
    influence of each expert on that specific location, implementing the "soft
    boundary stitching".
3.  **Weighted Combination**: The final latent effect for each observation is computed
    as the weighted sum of the local expert means, where the weights are the softmax
    responsibilities.
4.  **Scaling**: The combined effect is scaled by the overall `sigma` hyperparameter.
"""
function _generate_manifold_code_fragments(m::Mosaic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_regions = m.n_regions
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Determine parameter names
    local sigma_name_str, mu_local_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        mu_local_name_str = "$(prefixed_key)_mu_local_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        mu_local_name_str = "$(prefixed_key)_mu_local"
        if is_multivariate
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(Symbol(sigma_name_str)) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(Symbol(sigma_name_str)))")
        # Priors for the means of the local experts.
        push!(priors_acc, "$(Symbol(mu_local_name_str)) ~ NamedDist(MvNormal(zeros(T, $(n_regions)), I), :$(Symbol(mu_local_name_str)))")
    end
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    update_str = """
    begin
        # Mosaic mixture-of-experts model for $(key_str)
        local centers = spec_registry["$(key_str)"].hyper.mosaic_centers
        local coords = hcat(M.s_x, M.s_y)
        
        # Calculate squared distances from each observation to each mosaic center
        local dist_sq = pairwise(SqEuclidean(), coords, centers', dims=1)
        
        # Use softmax on negative distances to get weights (responsibilities)
        # Closer centers get higher weights.
        local weights = softmax(-dist_sq, dims=2)
        
        # The latent effect is the weighted sum of the local expert means.
        local $(latent_name_str) = (weights * $(mu_local_name_str)) .* $(sigma_name_str)
        
        $(eta_update_target) .+= $(latent_name_str)
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
        error("Warp manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or a spatial module with available coordinates.")
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
        error("SVGP manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("SVGP manifold requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}(); for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:Z_inducing] = Z_inducing
    return (Q_template=coords, scaling_factor=1.0, model_type=:svgp, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Nystrom, data_inputs::Dict, module_metadata::Dict)
    # For Nystrom, we need both the observation coordinates and the inducing point coordinates.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Nystrom manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom manifold requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
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
        error("Nystrom manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom manifold requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
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
        error("GP manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
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
        error("Hyperbolic manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
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
        error("ExponentialDecay manifold requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or `spatial(lon, lat, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    return (Q_template=nothing, scaling_factor=1.0, model_type=:exponentialdecay, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Harmonic, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for continuous, spectral, and other advanced manifolds.
    # Rationale: These models do not rely on pre-computed templates in the same way as GMRFs, so they use a pass-through builder.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_pass_through_model(m, data_inputs, module_metadata)
end


"""
    build_model(m::NetworkFlow, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `NetworkFlow` manifold.

# Rationale
This function dispatches the `NetworkFlow` manifold to the template-based builder
with a `:spatial` context. This ensures that the adjacency matrix `W` from the main
model configuration is correctly identified and passed as the structural template
for the network model.

# Arguments
- `m::NetworkFlow`: The NetworkFlow manifold object.
- `data_inputs::Dict`: The main model configuration dictionary.
- `module_metadata::Dict`: The parsed dictionary for the module.

# Returns
- A `NamedTuple` containing the manifold's technical specification.
"""
function build_model(m::NetworkFlow, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata)
end


function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for spline-based smoothers.
    # Rationale: Determines the appropriate underlying GMRF template (RW1 or RW2) based on the spline type and penalty order.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    n = m.nbins
    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, module_metadata; Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Cyclic, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the `Cyclic` manifold.
    # Rationale: Creates a circulant precision matrix for smooth periodic effects.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, module_metadata; model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::BCGN, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the BCGN (Bipartite Graph Convolutional Network) manifold.
    # Rationale: Constructs the precision matrix template from the provided bipartite adjacency matrix.
    #            The precision is based on the graph Laplacian of the one-mode projection of the
    #            bipartite graph. This induces a GMRF structure on one set of nodes, where two
    #            nodes are considered "neighbors" if they share a common neighbor in the other partition.
    # v1.3.14 (2026-07-19)

    B = m.bipartite_adj
    if isempty(B) || all(iszero, B)
        error("BCGN manifold requires a non-empty `bipartite_adj` matrix, but it was not provided or is all zeros.")
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

A model builder specifically for the `Eigen` manifold.

# Rationale for Update
This new builder method ensures that the pre-processed data matrix required for the
Bayesian PCA is correctly passed from the module processor into the manifold's
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
    
    # Q_template is not used for the Eigen manifold, but a placeholder is returned for API consistency.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:eigen, hyper=NamedTuple(hyper_dict))
end



function build_model(m::DynamicsManifold, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the `DynamicsManifold`.
    # Rationale: This version correctly distinguishes between advection and diffusion operators.
    #            It generates a second-order Laplacian operator (L) for diffusion and a first-order
    #            directed operator (A) for advection. Both are stored in the manifold's
    #            hyperparameter registry, ensuring consistency with the reconstruction engine.
    # v1.3.0 (2026-07-17) - Corrected from v1.2.1
    # Assumptions: `W` (adjacency matrix) is available in `data_inputs`.
    # Inputs:
    #   - m: The DynamicsManifold object.
    #   - data_inputs: The model configuration dictionary.
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the manifold's technical specification, including L and A operators.

    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    if isnothing(W)
        error("DynamicsManifold requires an adjacency matrix W, but it was not found in the model configuration.")
    end

    # 1. Build the second-order diffusion operator (Graph Laplacian).
    # This is used for 'diffusion' and 'advection_diffusion' models.
    L_template = build_structure_template(:besag, n; W=W).matrix

    # 2. Build a first-order advection operator.
    # This creates a simple directed graph by orienting edges from higher to lower indices,
    # then row-normalizing to create a shift/transition operator. This is a basic approximation
    # of advection on a graph without an explicit velocity field.
    W_dir = tril(W, -1)
    out_degree = sum(W_dir, dims=2)[:]
    # Add a small epsilon to avoid division by zero for nodes with no incoming edges.
    D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
    A_template = D_inv * W_dir

    # 3. Store both operators in the manifold's hyperparameter registry.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:L_template] = L_template
    hyper_dict[:A_template] = A_template

    # The Q_template field is still required by the generic manifold pathway. We pass the
    # Laplacian as the default, but the specific dynamics logic will use L_template and A_template.
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


function build_model(m::TPS, data_inputs::Dict, module_metadata::Dict)
    n = m.nbins
    # The penalty for a thin plate spline is typically a second-order difference penalty,
    # which is equivalent to a Random Walk of order 2 (RW2).
    template = build_structure_template(:rw2, n)
    return _build_pass_through_model(m, data_inputs, module_metadata; Q_template_val=template.matrix, sf_val=template.scaling_factor)
end




"""
    build_model(m::ComposedManifold, data_inputs::Dict, module_metadata::Dict)

A model builder for composed manifolds, created by operators like `|>` and `∘`.

# Rationale
This function correctly handles the setup for composed manifolds.
- For a `:pipe` operation (e.g., `spatial |> smooth`), it recursively builds the
  specification for the `state` manifold (the spatial part) and attaches it to the
  composed manifold's `hyper` registry. This is crucial for the code generator, which
  needs the state manifold's precision matrix to model the spatially varying coefficients.
- For a `:composition` operation (e.g., `spatial ∘ smooth`), it similarly builds the
  specification for the `base` manifold.
- For other operators, it uses a pass-through builder, as the logic is handled
  elsewhere (e.g., by global interaction blocks or other manifold types).
"""
function build_model(m::ComposedManifold, data_inputs::Dict, module_metadata::Dict)
    if m.operator == :pipe
        if length(m.components) != 2
            error("Pipe operator requires exactly two components: state |> dynamic.")
        end
        state_manifold = m.components[1]
        
        state_spec = build_model(state_manifold, data_inputs, module_metadata)
        
        hyper_dict = Dict{Symbol, Any}()
        for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
        hyper_dict[:state_spec] = state_spec
        
        return (Q_template=nothing, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    elseif m.operator == :composition
        base_manifold = get(m.components, 1, nothing)
        if isnothing(base_manifold); error("Composition manifold is missing its base component."); end 
        
        base_spec = build_model(base_manifold, data_inputs, module_metadata)
        hyper_dict = Dict(:base_spec => base_spec)
        return (Q_template=base_spec.Q_template, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    else
        # For other operators like ⊗, no special template is needed at this stage,
        # as they are either handled by other processors or are not supported generically.
        return _build_pass_through_model(m, data_inputs, module_metadata)
    end
end



function _build_from_template(m::ManifoldModel, data_inputs::Dict, domain::Symbol, module_metadata::Dict)
    # Purpose: A generic builder for manifolds that use a pre-defined template.
    # Rationale: Reduces code duplication for common GMRF models.
    # v1.2.1 (2026-07-16)
    # Assumptions: The manifold type has a corresponding entry in `build_structure_template`.
    # Inputs:
    #   - m: The ManifoldModel object.
    #   - data_inputs: The model configuration dictionary.
    #   - domain: The domain of the manifold (:spatial, :temporal, :mixed).
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the manifold's technical specification.
    model_sym = Symbol(lowercase(string(typeof(m))))
    n, W_mat = if domain == :spatial
        (get(data_inputs, :s_N, 1), get(data_inputs, :W, nothing))
    elseif domain == :temporal
        (get(data_inputs, :t_N, 10), nothing)
    elseif domain == :mixed
        n_levels = get(get(module_metadata, :params, Dict()), :n_cat, 0)
        if n_levels == 0; error("Could not determine number of levels for mixed effect. `n_cat` not found in module parameters."); end
        (n_levels, nothing)
    else
        @warn "Unrecognized domain '$domain'. Defaulting to spatial context."
        (get(data_inputs, :s_N, 1), get(data_inputs, :W, nothing))
    end
    template = build_structure_template(model_sym, n; W=W_mat)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        if !(fn in [:Q_template]) ; hyper_dict[fn] = getfield(m, fn); end
    end
    return (Q_template = template.matrix, scaling_factor = template.scaling_factor, model_type = model_sym, hyper = NamedTuple(hyper_dict))
end

function _build_pass_through_model(m::ManifoldModel, data_inputs::Dict, module_metadata::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # Purpose: A generic builder for manifolds that do not require complex template generation.
    # Rationale: Used for models where the structure is defined by parameters (e.g., splines) or handled dynamically.
    # v1.2.1 (2026-07-16)
    #            This version ensures a default identity Q_template is created for basis-like models.
    # Assumptions: None.
    # Inputs:
    #   - m, data_inputs, and optional overrides.
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the manifold's technical specification.
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
    # v1.2.1 (2026-07-16)
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
# SECTION 7.5: MANIFOLD CODE GENERATORS
# ==============================================================================

function _generate_manifold_code_fragments(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing})
    # Purpose: Dispatches code generation to a specific method based on the manifold object type.
    # Rationale: This is the entry point for converting a high-level manifold specification
    # v1.2.1 (2026-07-16)
    #            into low-level Turing model code strings.
    return _generate_manifold_code_fragments(spec.manifold_obj, spec, arch, outcome_idx)
end
 



"""
    _generate_manifold_code_fragments(m::Harmonic, ...)

Generates Turing code fragments for the `Harmonic` manifold.

# Rationale for v1.8.0 Update
This function has been rewritten to correctly parameterize the harmonic model based on
`amplitude` and `phase`, as specified in the struct definition. The previous implementation
used `sigma` (marginal standard deviation) as the primary parameter, which was inconsistent.

The new implementation samples `amplitude` and `phase` directly from their user-specified
priors. The internal `beta_cos` and `beta_sin` coefficients are then computed from these
sampled values, ensuring the model's generative process matches its definition. The `sigma`
parameter is no longer part of the `Harmonic` struct or its code generation.
"""
function _generate_manifold_code_fragments(m::Harmonic, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(v.amplitude) ~ NamedDist($(_distribution_to_string(m.amplitude)), :$(v.amplitude))")
        push!(priors_acc, "$(v.phase) ~ NamedDist($(_distribution_to_string(m.phase)), :$(v.phase))")
        if m.period isa UnivariateDistribution
            push!(priors_acc, "$(v.period) ~ NamedDist($(_distribution_to_string(m.period)), :$(v.period))")
        end
    end
    
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    # Harmonic models use the seasonal index `u_idx` by convention.
    index_var = "u_idx" 
    
    update_str = """
    begin
        # Harmonic model for $(spec.key), parameterized by amplitude and phase.
        # The effect is A*cos(ωt - φ), where φ is the phase shift.
        local amplitude = $(v.amplitude)
        local phase_rad = 2.0 * pi * $(v.phase)

        local beta_cos = amplitude * cos(phase_rad)
        local beta_sin = amplitude * sin(phase_rad)
        
        time_points = M.$(index_var)
        angle = (2.0 * pi / $(v.period)) .* time_points
        
        cos_term = cos.(angle)
        sin_term = sin.(angle)
        
        local harmonic_effect = beta_cos .* cos_term .+ beta_sin .* sin_term # Note: No extra scaling by sigma
        
        $(eta_update_target) .+= harmonic_effect
    end
    """
    
    return (priors=priors_str, update=update_str)
end


function _generate_manifold_code_fragments(m::AR2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for AR(2) processes using a state-space formulation.
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
        $(v.latent) = Vector{T}(undef, $(n_latent))
        
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


function _generate_manifold_code_fragments(m::Eigen, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates code for the Bayesian PCA (`eigen`) manifold.
    # Rationale: Implements the Householder reflection parameterization for the loading matrix.
    # v1.2.1 (2026-07-16)

    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    n_vars = m.n_vars
    n_factors = m.n_factors

    if n_vars == 0
        error("Eigen manifold '$(prefixed_key)' has n_vars=0, which is invalid. This typically occurs if the manifold was not processed correctly from the formula (e.g., no variables specified for eigen()).")
    end

    priors_str = """
        pca_sd_priors = $(m.pca_sd)
        pdef_sd_priors = $(m.pdef_sd)

        v_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, $(length(m.ltri_indices))), 1.0), :v_raw_$(prefixed_key))
        d_raw_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, $(n_factors)), 1.0), :d_raw_$(prefixed_key))
        
        pca_sds_$(prefixed_key) ~ NamedDist(filldist(pca_sd_priors, $(n_factors)), :pca_sds_$(prefixed_key))
        pdef_sds_$(prefixed_key) ~ NamedDist(filldist(pdef_sd_priors, $(n_factors)), :pdef_sds_$(prefixed_key))

        factors_$(prefixed_key) ~ NamedDist(MvNormal(zeros(T, M.y_N * $(n_factors)), I), :factors_$(prefixed_key))
    """

    update_str = """
        begin
            v_mat_$(prefixed_key) = zeros(T, $(n_vars), $(n_factors))
            v_mat_$(prefixed_key)[$(m.ltri_indices)] .= v_raw_$(prefixed_key)

            U_$(prefixed_key) = householder_to_eigenvector(v_mat_$(prefixed_key), $(n_vars), $(n_factors))
            
            d_trans_$(prefixed_key) = exp.(d_raw_$(prefixed_key) .* pdef_sds_$(prefixed_key))
            D_mat_$(prefixed_key) = Diagonal(d_trans_$(prefixed_key))

            L_$(prefixed_key) = U_$(prefixed_key) * D_mat_$(prefixed_key)

            F_matrix_$(prefixed_key) = reshape(factors_$(prefixed_key), M.y_N, $(n_factors))
            F_scaled_$(prefixed_key) = F_matrix_$(prefixed_key) .* pca_sds_$(prefixed_key)'
            eigen_effects = F_scaled_$(prefixed_key) * L_$(prefixed_key)'

            eta .+= sum(eigen_effects, dims=2)
        end
    """
    return (priors=priors_str, update=update_str)
end 

 

function _generate_householder_reflection_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # Purpose: Generates code for the Householder reflection (spectral orientation) feature.
    # Rationale: This allows for rotating the latent space in multivariate models to better
    # v1.2.1 (2026-07-16)
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
    # Purpose: Dynamically generates the Turing `@model` function as a string.
    # Rationale: This refactored version uses helper functions to generate code blocks,
    # v1.2.1 (2026-07-16)
    #            improving readability and maintainability.
    # Assumptions: `M` is a complete and valid model configuration from `bstm_config`.
    # Inputs:
    #   - M: The model configuration NamedTuple.
    #   - model_func_name: A unique symbol for the generated model function.
    # Outputs: A tuple containing the model string, the parsed model expression, and the specification registry.
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
    
    for spec in M.manifolds
        spec_registry[string(spec.key)] = spec
        for k in 1:outcomes_N
            outcome_idx = is_multivariate ? k : nothing            
            frag = _generate_manifold_code_fragments(spec.manifold_obj, spec, arch, outcome_idx)
            if !isempty(Base.strip(frag.priors)); push!(priors_acc, frag.priors); end
            if !isempty(Base.strip(frag.update)); push!(updates_acc, frag.update); end
        end

        if spec.domain == :spatial && isnothing(main_spatial_spec)
            main_spatial_spec = spec
        end
        if spec.domain == :temporal && isnothing(main_temporal_spec)
            main_temporal_spec = spec
        end
    end

    # Helper to indent a block of code for clean code generation
    function _indent_block(text::String, level=1)
        if isempty(Base.strip(text)) return "" end
        indent_str = "    " ^ level
        return indent_str * replace(Base.strip(text), "\n" => "\n" * indent_str)
    end

    # Generate all code blocks
    likelihood_section = _generate_likelihood_section(M, is_multivariate)
    intercept_priors, intercept_update = _generate_intercept_block(M, is_multivariate, eta_name)
    offset_block = _generate_offset_block(M, is_multivariate, eta_name)
    fixed_effects_priors, fixed_effects_update = _generate_fixed_effects_block(M, is_multivariate, eta_name)
    st_interaction_block = _generate_st_interaction_block(M, main_spatial_spec, main_temporal_spec, is_multivariate, eta_name)
    householder_priors, householder_update = _generate_householder_reflection_block(M, is_multivariate, eta_name)
    final_likelihood = _generate_final_likelihood_block(M, is_multivariate)
    nested_priors, nested_updates, nested_likelihoods = _generate_nested_model_block(M, is_multivariate, eta_name)

    # Add the separated prior blocks to the main priors accumulator
    if !isempty(Base.strip(intercept_priors)); push!(priors_acc, intercept_priors); end
    if !isempty(Base.strip(fixed_effects_priors)); push!(priors_acc, fixed_effects_priors); end

    # Indent all code blocks before interpolation
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
    # v1.3.4 (2026-07-18)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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

# # Likelihood Kernel for Dirichlet-Multinomial
# # Rationale: Maps the linear predictor (eta) via softmax to the probability simplex.
"""
    get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)

Implements a true Dirichlet-Multinomial model. The linear predictor `eta`
determines the mean probabilities of the categories via softmax. The `sig`
parameter (aliased from `y_sigma`) controls the overall concentration `α₀`,
which governs the overdispersion. A large `α₀` approaches a standard
Multinomial distribution.
"""
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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

 
# Specialized bstm_Likelihood method for vector outcomes (Multinomial)
function bstm_kernel(fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec)
    dist = get_dist_ref(fam, d, eta_vec, sig)
    return logpdf(dist, y_vec)
end


# # Specialized bstm_Likelihood method for vector outcomes (Multinomial)
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
    # v1.2.1 (2026-07-16)
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
    #            is no longer part of the API for manifold structs.
    # v1.4.0 (2026-07-20)
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
    group_manifolds::Bool=true,
    n_particles=20,
    hmc_leapfrog_steps=10
)
    # Purpose: Automatically constructs an efficient composite Gibbs sampler for a `bstm` model.
    # Rationale: This utility inspects the model's parameters and their prior distributions to build a
    # v1.2.1 (2026-07-16)
    #            composite `Gibbs` sampler. It assigns specialized samplers (`ESS`, `Slice`, `PG`) to
    #            different parameter blocks to improve MCMC efficiency. It also provides an option to
    #            group manifold-specific parameters for joint sampling with `NUTS` to better handle
    #            posterior correlations.
    # Assumptions: The model has been instantiated.
    # Inputs:
    #    - model_obj: The instantiated Turing.jl model object.
    #    - sampler_choice: If a specific sampler algorithm is provided, it is used directly, bypassing auto-detection.
    #    - sampler_map: A dictionary to manually assign specific samplers to parameter symbols.
    #    - target_acceptance: The target acceptance rate for `NUTS`.
    #    - adaptation_steps: The number of adaptation steps for `NUTS`.
    #    - group_manifolds: If `true`, groups all parameters of a single manifold (e.g., its hyperparameters
    #                       and latent field) into a single `NUTS` block to handle posterior correlations.
    #    - n_particles: The number of particles for the `PG` sampler (for discrete parameters).
    #    - hmc_leapfrog_steps: Unused, kept for API consistency.
    # Outputs: A Turing.jl sampler object (e.g., `Gibbs`, `NUTS`).

    if sampler_choice isa AbstractMCMC.AbstractSampler
        @info "Using user-specified sampler: $(typeof(sampler_choice))"
        return sampler_choice
    end

    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)

    samplers = AbstractMCMC.AbstractSampler[]
    all_processed_symbols = Set{Symbol}()

    # 1. Handle user-provided sampler map first (highest precedence).
    for (param_sym, sampler) in sampler_map
        sym_vns = filter(vn -> DynamicPPL.getsym(vn) == param_sym, vns)
        if !isempty(sym_vns)
            push!(samplers, sampler)
            push!(all_processed_symbols, param_sym)
            @info "Applying user-defined sampler $(typeof(sampler)) for parameter: $(param_sym)"
        else
            @warn "Parameter :$(param_sym) in sampler_map not found in model."
        end
    end

    # 2. Handle manifold grouping if enabled.
    if group_manifolds
        @info "Manifold grouping enabled. Grouping hyperparameters and latent fields for joint sampling."
        manifold_groups = Dict{String, Set{Symbol}}()

        for vn in vns
            sym = DynamicPPL.getsym(vn)
            if sym in all_processed_symbols; continue; end

            # This regex identifies parameters belonging to specific manifolds by matching the pattern:
            # `manifoldtype_variablename_parametername`. It is designed to be comprehensive,
            # covering all manifold types and their associated parameter suffixes used in the bstm framework.
            m = match(r"^(spatial|temporal|smooth|mixed|dynamics|eigen|interact|st|nested)_(.+?)_(sigma|rho|ls|kappa|coeffs|latent|innov|struct|iid|v|d|r|K|factors|beta|coeffs_correlated|raw|pca_sd|pdef_sd)", string(sym))

            if !isnothing(m)
                manifold_key = m.captures[1] * "_" * m.captures[2]
                if !haskey(manifold_groups, manifold_key); manifold_groups[manifold_key] = Set{Symbol}(); end
                push!(manifold_groups[manifold_key], sym)
            end
        end

        for (key, params) in manifold_groups
            if !isempty(params)
                push!(samplers, NUTS(adaptation_steps, target_acceptance, collect(params)...))
                union!(all_processed_symbols, params)
                @info "Created NUTS block for manifold '$(key)' with parameters: $(params)"
            end
        end
    end

    # 3. Default grouping for all remaining parameters.
    remaining_vns = filter(vn -> !(DynamicPPL.getsym(vn) in all_processed_symbols), vns)
    if !isempty(remaining_vns)
        param_groups = Dict(:discrete => Set{Symbol}(), :gaussian => Set{Symbol}(), :bounded => Set{Symbol}(), :other_continuous => Set{Symbol}())

        for vn in remaining_vns
            sym = DynamicPPL.getsym(vn)
            if sym in all_processed_symbols; continue; end

            try
                dist = DynamicPPL.getdist(vi, vn)
                support = Distributions.value_support(typeof(dist))
                if support isa Distributions.Discrete; push!(param_groups[:discrete], sym);
                elseif support isa Distributions.Continuous
                    if dist isa Union{Normal, MvNormal, Truncated{<:Normal}}; push!(param_groups[:gaussian], sym);
                    elseif isfinite(minimum(dist)) || isfinite(maximum(dist)); push!(param_groups[:bounded], sym);
                    else; push!(param_groups[:other_continuous], sym); end
                end
            catch e; push!(param_groups[:other_continuous], sym); end
        end

        if !isempty(param_groups[:discrete]); params = collect(param_groups[:discrete]); push!(samplers, PG(n_particles, params...)); @info "Using Particle Gibbs (PG) for remaining discrete parameters: $(params)"; end
        if !isempty(param_groups[:gaussian]); params = collect(param_groups[:gaussian]); push!(samplers, ESS(params...)); @info "Using Elliptical Slice Sampling (ESS) for remaining Gaussian parameters: $(params)"; end
        if !isempty(param_groups[:bounded]); params = collect(param_groups[:bounded]); push!(samplers, Slice(params...)); @info "Using Slice sampling for remaining bounded parameters: $(params)"; end
        if !isempty(param_groups[:other_continuous]); params = collect(param_groups[:other_continuous]); push!(samplers, NUTS(adaptation_steps, target_acceptance, params...)); @info "Using NUTS for remaining continuous parameters: $(params)"; end
    end

    # 4. Construct the final sampler.
    if isempty(samplers)
        @warn "Could not identify any parameters to sample. Defaulting to NUTS for all."
        return NUTS(adaptation_steps, target_acceptance)
    elseif length(samplers) == 1
        return samplers[1]
    else
        return Gibbs(samplers...)
    end
end


function create_fixed_design(formula_rhs::AbstractString, data::DataFrame; contrasts=Dict{Symbol, Any}())
    # Purpose: Creates the fixed-effects design matrix (`X`) from a formula string.
    # Rationale: A wrapper around `StatsModels.jl` to handle formula parsing and contrast coding.
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.2.1 (2026-07-16)
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
    # v1.4.0 (2026-07-20)
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

    if haskey(config, :manifolds) && !isempty(config.manifolds)
        for spec in config.manifolds
            m_obj = spec.manifold_obj
            m_type_str = string(typeof(m_obj))
            key = spec.key
            
            push!(lines, "\n    # Priors for manifold: $(key) ($(m_type_str))")
            
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


# ==============================================================================
# SECTION 10: ADVANCED MODELING UTILITIES (Restored)
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
- A matrix of size `(length(x), n_basis)` where each column is a B-spline basis function evaluated at the points in `x`.
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
    B[x .== t[end], num_total_basis] .= 1.0

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

    # Return the final basis matrix, ensuring it has the requested number of columns.
    return B[:, 1:n_basis]
end




"""
    bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; knot_method=:quantile)

Generates a tensor product B-spline basis matrix for multidimensional data.

# Arguments
- `coords`: An `N x D` matrix of data points, where `N` is the number of observations and `D` is the number of dimensions.
- `nbins_per_dim`: A vector of length `D` specifying the number of basis functions for each dimension.
- `degrees_per_dim`: A vector of length `D` specifying the B-spline degree for each dimension.
- `knot_method`: The knot placement strategy (`:quantile` or `:range`) for the 1D B-spline bases.

# Returns
- A tensor product basis matrix of size `(N, prod(nbins_per_dim))`.
"""
function bstm_tensor_product_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, degrees_per_dim::Vector{Int}; knot_method::Symbol=:quantile)
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims || length(degrees_per_dim) != n_dims
        error("Number of dimensions in coords must match length of nbins_per_dim and degrees_per_dim.")
    end

    # Generate 1D basis matrices for each dimension.
    basis_matrices_1D = [bstm_bspline_basis(coords[:, i], nbins_per_dim[i], degrees_per_dim[i]; knot_method=knot_method) for i in 1:n_dims]

    # Compute the tensor product basis matrix using row-wise Kronecker products.
    n_obs = size(coords, 1)
    B_final = ones(n_obs, 1)

    for B_i in basis_matrices_1D
        n_cols_final = size(B_final, 2)
        n_cols_i = size(B_i, 2)
        new_B = zeros(n_obs, n_cols_final * n_cols_i)
        for row in 1:n_obs
            new_B[row, :] = kron(B_final[row, :], B_i[row, :])
        end
        B_final = new_B
    end
    return B_final
end


"""
    bstm_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, lengthscale::Float64)

Generates a 1D wavelet basis matrix using scaled and translated mother wavelets.

# Rationale
The original implementation used a simple Haar-like (boxcar) basis, which did not
respect the specified wavelet `family`. This implementation uses the `Wavelets.jl`
package to generate basis functions from the actual mother wavelet shape (e.g., Daubechies),
providing a more correct and flexible wavelet basis expansion.

The basis functions are created by scaling and translating the mother wavelet function
across the domain of the input data.

# Arguments
- `vals`: The vector of data points.
- `nbins`: The number of basis functions (wavelets) to generate.
- `family`: The wavelet family (e.g., `:db4`, `:sym6`).
- `lengthscale`: A parameter controlling the base width of the wavelets.

# Returns
- A matrix of size `(length(vals), nbins)` where each column is a basis function.
"""
function bstm_wavelet_basis_1D(vals::AbstractVector, nbins::Int, family::Symbol, lengthscale::Float64)
    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    v_min, v_max = minimum(vals), maximum(vals)
    v_range = v_max - v_min
    if v_range < 1e-9; v_range = 1.0; end
    
    # Get the wavelet function shape from Wavelets.jl
    wt = wavelet(family)
    # '8' gives a resolution of 2^8=256 points for the function shape.
    x_grid, (_, psi) = wavefun(wt, 8) 

    # Create a linear interpolation object for the mother wavelet function
    # The function is periodic on the grid, so we handle boundaries.
    itp = LinearInterpolation(x_grid, psi, extrapolation_bc=Flat())

    # Determine scales and translations for the basis functions
    # A simple scheme: use a few scales and distribute bins among them.
    n_scales = max(1, floor(Int, log2(nbins/4))) # e.g., 3 scales for 32 bins
    bins_per_scale = div(nbins, n_scales)
    
    current_bin = 1
    for j in 1:n_scales
        # Scale determines the "stretch" of the wavelet. Higher j = more stretched.
        # We tie it to lengthscale to make it user-controllable.
        scale_factor = lengthscale * (2.0^(j-1))
        
        # Determine the number of translations for this scale
        if j == n_scales
            n_translations = nbins - current_bin + 1
        else
            n_translations = bins_per_scale
        end
        if n_translations <= 0; continue; end

        # Place translation centers at quantiles of the data for good coverage
        probs = n_translations == 1 ? [0.5] : range(0, 1, length=n_translations)
        centers = quantile(vals, probs)
        
        for k in 1:n_translations
            if current_bin > nbins; break; end
            
            # Transform data points to the wavelet's local coordinate system
            # (vals - center) / width
            transformed_vals = (vals .- centers[k]) ./ (scale_factor * v_range)
            
            # Evaluate the interpolated wavelet function at the transformed points
            B[:, current_bin] = itp.(transformed_vals)
            
            current_bin += 1
        end
    end
    return B
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
    bstm_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, family::Symbol, lengthscale::Float64)

Generates a tensor product wavelet basis matrix for multidimensional data.
"""
function bstm_tensor_product_wavelet_basis(coords::AbstractMatrix, nbins_per_dim::Vector{Int}, family::Symbol, lengthscale::Float64)
    n_dims = size(coords, 2)
    if length(nbins_per_dim) != n_dims
        error("Number of dimensions in coords must match length of nbins_per_dim.")
    end

    # Generate 1D wavelet basis matrices for each dimension.
    basis_matrices_1D = [bstm_wavelet_basis_1D(coords[:, i], nbins_per_dim[i], family, lengthscale) for i in 1:n_dims]

    # Compute the tensor product basis matrix using row-wise Kronecker products.
    n_obs = size(coords, 1)
    B_final = ones(n_obs, 1)

    for B_i in basis_matrices_1D
        n_cols_final = size(B_final, 2)
        n_cols_i = size(B_i, 2)
        new_B = zeros(n_obs, n_cols_final * n_cols_i)
        for row in 1:n_obs
            new_B[row, :] = kron(B_final[row, :], B_i[row, :])
        end
        B_final = new_B
    end
    return B_final
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
    B = zeros(Float64, n_obs, nbins)
    
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

    if type in ["pspline", "bspline"]
        # Correctly generate a B-spline basis of the specified degree.
        return bstm_bspline_basis(vals, nbins, degree; knot_method=knot_method, custom_knots=custom_knots)

    elseif type in ["smooth", "barycentric", "linear"]
        # Retain the original linear tent function implementation for these aliases.
        h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1)
        h = h > 0 ? h : 1.0
        for m in 1:nbins
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B[mask, m] .= 1.0 .- dist[mask]
        end


    elseif type == "tps"
        # The radial basis function for 1D TPS (m=2, d=1) is r^3.
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B[:, m] .= r.^3
        end
        return B


        
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
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        return bstm_wavelet_basis_1D(vals, nbins, family, lengthscale)
 
  
    elseif type == "spherical"
        range_r = get(kwargs, :range, v_std * 2.0)
        for m in 1:nbins
            h = abs.(vals .- knots[m]) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    elseif type == "decay"
        ls = get(kwargs, :lengthscale, v_std)
        for m in 1:nbins
            B[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
        end
 
    elseif type == "invdist"
        for m in 1:nbins
            dist_sq = (vals .- knots[m]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
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
    bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; ...)

Generates a 2D basis matrix. This is an updated version.

# Rationale for Update
The implementation for `pspline` and `bspline` now calls `bstm_tensor_product_basis`
to generate a proper tensor product B-spline basis, respecting the `degree` parameter.
The original linear implementation is retained for other aliases.
"""
function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    n_obs = size(coords, 1)
    
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    if type in ["pspline", "bspline"]
        n_marginal = Int(floor(sqrt(nbins)))
        degree_val = get(kwargs, :degree, 3)
        nbins_per_dim = [n_marginal, n_marginal]
        degrees_per_dim = [degree_val, degree_val]
        return bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)
    end

    # Fallback to original logic for other types
    local kx, ky
    n_marginal = Int(floor(sqrt(nbins)))
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        kx, ky = custom_knots
    elseif knot_method == :quantile && !use_regular_grid
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
    end
    
    B = zeros(Float64, n_obs, n_marginal^2)

    if type == "barycentric"
        # Combine knot vectors into a vector of Point2D
        knot_points = [Point2D(x, y) for x in kx for y in ky]
        return bstm_barycentric_basis_2D(coords, knot_points)
    end

    if type in ["smooth", "linear"]
        hx = (c_max[1] - c_min[1]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hx = hx > 0 ? hx : 1.0
        hy = (c_max[2] - c_min[2]) / (n_marginal > 1 ? (n_marginal - 1) : 1); hy = hy > 0 ? hy : 1.0
        idx = 1
        for i in 1:n_marginal, j in 1:n_marginal
            b_x = max.(0.0, 1.0 .- abs.(coords[:, 1] .- kx[i]) ./ hx)
            b_y = max.(0.0, 1.0 .- abs.(coords[:, 2] .- ky[j]) ./ hy)
            B[:, idx] .= b_x .* b_y
            idx += 1
        end
    elseif type == "wavelet"
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal, n_marginal]
        return bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
    end


    if type == "tps"
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, min(nbins, m_total))
        centers = [(x, y) for x in kx, y in ky][:]
        
        for m in 1:min(nbins, m_total)
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            r = sqrt.(dx.^2 .+ dy.^2)
            # Use a small epsilon to avoid log(0)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-9)
        end
        return B
    end


    if type == "rff" || type == "anisotropic"
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
    end

    if type == "wavelet"
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
    end

    if type == "spherical"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        centers = [(x, y) for x in kx, y in ky][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:min(nbins, length(centers))
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            h = sqrt.(dx.^2 .+ dy.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    end
     

    if  type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end
    end

    if  type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    end


    return B
end

function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    # BSTM Smooth Basis Factory
    # v1.2.1 (2026-07-16)
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

    local kx, ky, kz
    n_marginal = Int(floor(cbrt(nbins)))

    # Certain basis types require a regular grid for knot placement for stability.
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 3 || length(custom_knots[1]) != n_marginal || length(custom_knots[2]) != n_marginal || length(custom_knots[3]) != n_marginal
            @warn "Custom knots for 3D smoother must be a Tuple{AbstractVector, AbstractVector, AbstractVector} with each vector of length n_marginal ($n_marginal). Falling back to :quantile method."
            kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
            ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
            kz = quantile(coords[:, 3], range(0, 1, length=n_marginal))
        else
            kx = custom_knots[1]
            ky = custom_knots[2]
            kz = custom_knots[3]
        end
    elseif knot_method == :quantile
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
        kz = quantile(coords[:, 3], range(0, 1, length=n_marginal))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
    end

    if type in ["pspline", "bspline"]
        n_marginal = Int(floor(cbrt(nbins)))
        degree_val = get(kwargs, :degree, 3)
        nbins_per_dim = [n_marginal, n_marginal, n_marginal]
        degrees_per_dim = [degree_val, degree_val, degree_val]
        return bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)

    elseif type == "wavelet"
        n_marginal = Int(floor(cbrt(nbins)))
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal, n_marginal, n_marginal]
        return bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)

    elseif type in ["smooth", "barycentric", "linear"]
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)

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

        m_total = n_marginal^3
        B = zeros(Float64, n_obs, min(nbins, m_total))
        centers = [(x, y, z) for x in kx, y in ky, z in kz][:]
        
        for m in 1:min(nbins, m_total)
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            dz = coords[:, 3] .- centers[m][3]
            # The radial basis function for 3D TPS is r.
            r = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            B[:, m] .= r
        end
        return B
 
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
  
    elseif type == "spherical"
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, min(nbins, m_total))
        centers = [(x, y, z) for x in kx, y in ky, z in kz][:]
        range_r = get(kwargs, :range, mean(c_std))
        
        for m in 1:min(nbins, m_total)
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            dz = coords[:, 3] .- centers[m][3]
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
        return B
 
    # The 'moran' basis type was removed from smoothers. It is a spatial basis.
    elseif type == "moran"
        error("The 'moran' basis is not a valid smoother for covariates. Use a spatial manifold like `spatial(..., model=eigen)` instead.")

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        for m in 1:min(m_total, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        m_total = n_marginal^3
        centers = [(x,y,z) for x in kx, y in ky, z in kz][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_x^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_y^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_z^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    # BSTM Smooth Basis Factory
    # v1.2.1 (2026-07-16)
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

    local k1, k2, k3, k4
    n_marginal = Int(floor(sqrt(sqrt(nbins))))

    # Certain basis types require a regular grid for knot placement for stability.
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 4 || length(custom_knots[1]) != n_marginal || length(custom_knots[2]) != n_marginal || length(custom_knots[3]) != n_marginal || length(custom_knots[4]) != n_marginal
            @warn "Custom knots for 4D smoother must be a Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector} with each vector of length n_marginal ($n_marginal). Falling back to :quantile method."
            k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal))
            k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal))
            k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal))
            k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal))
        else
            k1 = custom_knots[1]
            k2 = custom_knots[2]
            k3 = custom_knots[3]
            k4 = custom_knots[4]
        end
    elseif knot_method == :quantile
        k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal)); k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal)); k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal)); k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal))
    else # :range or if regular grid is required
        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal)); k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal)); k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal)); k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))
    end

    if type in ["pspline", "bspline"]
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        degree_val = get(kwargs, :degree, 3)
        nbins_per_dim = [n_marginal, n_marginal, n_marginal, n_marginal]
        degrees_per_dim = fill(degree_val, 4)
        return bstm_tensor_product_basis(coords, nbins_per_dim, degrees_per_dim; knot_method=knot_method)
    elseif type == "wavelet"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        family = get(kwargs, :family, :db4)
        lengthscale = get(kwargs, :lengthscale, 0.1)
        nbins_per_dim = [n_marginal, n_marginal, n_marginal, n_marginal]
        return bstm_tensor_product_wavelet_basis(coords, nbins_per_dim, family, lengthscale)
    end

    if type == "barycentric"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        knot_points = [Point4D(i, j, k, l) for l in k4 for k in k3 for j in k2 for i in k1]
        return bstm_barycentric_basis_4D(coords, knot_points, n_marginal)
    end

    if type in [ "smooth", "linear"]
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

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
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, min(nbins, m_total))
        centers = [(k1_i, k2_i, k3_i, k4_i) for k1_i in k1, k2_i in k2, k3_i in k3, k4_i in k4][:]

        for m in 1:min(nbins, m_total)
            d1 = coords[:, 1] .- centers[m][1]
            d2 = coords[:, 2] .- centers[m][2]
            d3 = coords[:, 3] .- centers[m][3]
            d4 = coords[:, 4] .- centers[m][4]
            # The radial basis function for 4D TPS (m=2, d=4) is log(r).
            r = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            B[:, m] .= log.(r .+ 1e-9)
        end
        return B

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
 
    elseif type == "spherical"
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, min(nbins, m_total))
        centers = [(k1_i, k2_i, k3_i, k4_i) for k1_i in k1, k2_i in k2, k3_i in k3, k4_i in k4][:]
        range_r = get(kwargs, :range, mean(c_std))

        for m in 1:min(nbins, m_total)
            d1 = coords[:, 1] .- centers[m][1]
            d2 = coords[:, 2] .- centers[m][2]
            d3 = coords[:, 3] .- centers[m][3]
            d4 = coords[:, 4] .- centers[m][4]
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end
        return B
 

    # The 'moran' basis type was removed from smoothers. It is a spatial basis.
    elseif type == "moran"
        error("The 'moran' basis is not a valid smoother for covariates. Use a spatial manifold like `spatial(..., model=eigen)` instead.")

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        m_total = n_marginal^4
        centers = [(w,x,y,z) for w in range(c_min[1],c_max[1],length=n_marginal), x in range(c_min[2],c_max[2],length=n_marginal), y in range(c_min[3],c_max[3],length=n_marginal), z in range(c_min[4],c_max[4],length=n_marginal)][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2 .+ (coords[:, 3] .- centers[m][3]).^2 .+ (coords[:, 4] .- centers[m][4]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
        m_total = n_marginal^4
        centers = [(w,x,y,z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:min(m_total, length(centers))
            dist_sq = ((coords[:, 1] .- centers[m][1]).^2 ./ ls_1^2) .+ ((coords[:, 2] .- centers[m][2]).^2 ./ ls_2^2) .+ ((coords[:, 3] .- centers[m][3]).^2 ./ ls_3^2) .+ ((coords[:, 4] .- centers[m][4]).^2 ./ ls_4^2)
            B[:, m] .= exp.(-dist_sq ./ 2.0)
        end

    else
        B = ones(n_obs, 1)
    end

    return B
end

function evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    # v1.3.1 (2026-07-18) - Added support for ARD kernels by accepting a vector of lengthscales.
    
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
        return (param_val^2) .* exp.(-0.5 .* dist_sq) .+ (noise * I)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq) # distance is now scaled
        return (param_val^2) .* exp.(-d) .+ (noise * I)
    
    # Matern 3/2
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(3.0) .* d
        return (param_val^2) .* (1.0 .+ val) .* exp.(-val) .+ (noise * I)
    
    # Matern 5/2
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(5.0) .* d
        return (param_val^2) .* (1.0 .+ val .+ (val.^2 ./ 3.0)) .* exp.(-val) .+ (noise * I)

    # Constant Kernel (Identity innovation)
    elseif kernel_type == :constant
        return fill(param_val^2, size(dist_sq))

    # Linear Kernel
    elseif kernel_type == :linear
        return (param_val^2) .* dist_sq

    # Wavelet Multiscale Kernel
    # Rationale: Approximates a wavelet covariance by superposing energy scales. 
    # In real-space, this behaves as a sum of kernels with varying lengthscales corresponding to levels.
    elseif kernel_type == :wavelet
        K_accum = zeros(eltype(dist_sq), size(dist_sq))
        for scale in 1:wavelet_levels
            ls_scale_sq = (ls isa Real ? ls^2 : 1.0) / (4^(scale-1))
            weight_scale = (param_val^2) * exp(-scale / ls)
            
            K_accum .+= weight_scale .* exp.(-0.5 .* dist_sq ./ ls_scale_sq)
        end
        return K_accum .+ (noise * I)

    # Fallback Dispatch
    else
        return (param_val^2) .* exp.(-0.5 .* dist_sq) .+ (noise * I)
    end
end
     



"""
    recompose_precision(...)

An internal factory function that constructs a final precision matrix from a template.

# Rationale for Update
This version includes a more robust and explicit implementation for the `:NetworkFlow`
manifold. For the `:bidirectional` case, it now creates a symmetric, unweighted
adjacency matrix by checking for connectivity in either direction (`(W_net + W_net') .> 0`),
which is a more standard and clearer way to represent an undirected graph for a SAR model.

# Returns
- A symmetric precision matrix.
"""
function recompose_precision(
    m_type::Symbol, 
    template_s::AbstractMatrix, 
    param_val::Real; 
    template_t=nothing, 
    extra_param=nothing, 
    noise=1e-4, 
    directed_adj=nothing, 
    flow_direction=:bidirectional,
    kwargs...
)
    n_s = size(template_s, 1)

    if m_type == :NoneManifold || m_type == :FIXED
        return Symmetric(sparse(I(n_s)))
    end

    if m_type == :Besag || m_type == :ICAR || m_type == :Cyclic
        return Symmetric(template_s)
    end

    if m_type == :AR1 || m_type == :RW1 || m_type == :RW2
        error("recompose_precision should not be called for $(m_type) models. Use the state-space implementation.")
    end

    if m_type == :Leroux || m_type == :LocalAdaptive
        lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(lambda_val .* template_s + (1.0 - lambda_val) .* sparse(I(n_s)))
    end

    if m_type == :ST_I || m_type == :ST_II || m_type == :ST_III || m_type == :ST_IV
        Q_full = isnothing(template_t) ? template_s : kron(template_t, template_s)
        return Symmetric(Q_full)
    end

    if m_type == :NetworkFlow
        rho_net = isnothing(extra_param) ? 0.8 : extra_param
        W_net = template_s
        
        L_op = if flow_direction == :upstream
            I(n_s) - rho_net .* W_net'
        elseif flow_direction == :downstream
            I(n_s) - rho_net .* W_net
        else # :bidirectional or default
            # Create a symmetric, unweighted adjacency matrix for the undirected case.
            W_symm = sparse((W_net + W_net') .> 0)
            # The operator for an undirected SAR model.
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

    if m_type == :SPDE
        kappa = isnothing(extra_param) ? 1.0 : extra_param
        L_spde = (kappa^2 .* I(n_s) + template_s)
        return Symmetric(L_spde' * L_spde)
    end

    if m_type == :RFF || m_type == :FFT || m_type == :BSpline || m_type == :PSpline || m_type == :TPS
        return Symmetric(template_s)
    end

    return Symmetric(template_s)
end
     

function _distribution_to_string(d::Distribution)
    # Purpose: Converts a Distribution object into a string that represents a valid constructor call.
    # Rationale: The default `string(d)` often produces a string with keyword arguments (e.g., "Normal{Float64}(μ=0.0, σ=1.0)"),
    # v1.2.1 (2026-07-16)
    #            which is not valid syntax for constructing the object. This function ensures a valid
    #            positional-argument constructor string is generated.
    T = eltype(d)
    dist_name = string(typeof(d).name.name)
    
    # For common distributions, use positional arguments which are always safe.
    if d isa Exponential
        # Constructor is Exponential(rate), object stores scale theta = 1/rate
        return "$(dist_name){$T}($(1.0 / d.θ))"
    elseif d isa Normal
        return "$(dist_name){$T}($(d.μ), $(d.σ))"
    elseif d isa Beta
        return "$(dist_name){$T}($(d.α), $(d.β))"
    elseif d isa InverseGamma
        return "$(dist_name){$T}($(d.α), $(d.θ))"
    elseif d isa Gamma
        return "$(dist_name){$T}($(d.α), $(d.θ))"
    elseif d isa Uniform
        return "$(dist_name){$T}($(d.a), $(d.b))"
    else
        # Fallback for less common distributions. This might fail if their
        # string representation uses keywords, but covers many cases.
        return string(d)
    end
end 

  

function _generate_likelihood_section(M::NamedTuple, is_multivariate::Bool)
    # Purpose: Generates code fragments for the likelihood-specific parameters (sigma, ZI, correlation).
    # Rationale: This version only defines `y_sigma` and `lik_phi_zi` if they are explicitly
    #            required by the model's likelihood family or options. This avoids generating
    #            unnecessary variables for simpler models like Poisson.
    # v1.2.3 (2026-07-17)
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
        y_sigma_prior_str = _distribution_to_string(Exponential(1.0))
        if is_multivariate
            sigma_block = "y_sigma ~ NamedDist(filldist($(y_sigma_prior_str), K), :y_sigma)"
        else
            sigma_block = "y_sigma ~ NamedDist($(y_sigma_prior_str), :y_sigma)"
        end
    end
    
    extra_params_block = ""
    # These families require an extra parameter (shape, precision, etc.)
    if any(f -> string(f) in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)
        extra_params_block = "lik_extra_params ~ NamedDist(Exponential(1.0), :lik_extra_params)"
    end

    corr_block = is_multivariate ? "L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)" : ""

    return """
    $(corr_block)
    $(sigma_block)
    $(hurdle_prior_block)
    $(zi_prior_block)
    $(nu_student_t_block)
    $(extra_params_block)
    """
end

function _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)
    # v1.2.3 (2026-07-17)
    # Rationale: This version makes the inclusion of `sigma_y` in the `bstm_Likelihood`
    #            call conditional. `sigma_y` is only included for likelihood families that require it
    #            (e.g., Gaussian), addressing user feedback about passing unnecessary parameters.
    
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    any_needs_sigma = any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)
    any_needs_nu = any(f -> f == "student_t", families)
    any_needs_extra = any(f -> f in ["gamma", "beta", "inverse_gaussian", "pareto", "half_student_t"], families)

    is_multinomial = any(f -> string(f) == "dirichlet_multinomial", families)

    if is_multivariate && is_multinomial
        # This assumes all outcomes are part of a single multinomial model.
        # A mix of multinomial and other outcomes is not supported by this logic.
        return """
        eta = eta_latent * L_corr.L # Apply correlation structure
        family = M.likelihood_specs[1][:family] # Assume first spec is representative
        for i in 1:N
            # For each observation, y_obs[i, :] is the vector of counts across categories.
            # eta[i, :] is the vector of linear predictors.
            # The total number of trials for the Multinomial is the sum of counts for that observation.
            d_lik = bstm_Likelihood(family, M.y_obs[i, :]; trial=sum(M.y_obs[i, :]))
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, :])
        end
        """
    end


    if is_multivariate
        kwargs_parts = String[]
        if any_needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma[k]"); end
        if get(M, :user_provided_trials, false)
            push!(kwargs_parts, "trial=Int(M.trials[i, k])")
        end
        if get(M, :user_provided_weights, false)
            push!(kwargs_parts, "weight=M.weights[i, k]")
        end
        if get(M, :user_provided_censor_lower, false)
            push!(kwargs_parts, "censor_lower=M.censor_lower[i, k]")
        end
        if get(M, :user_provided_censor_upper, false)
            push!(kwargs_parts, "censor_upper=M.censor_upper[i, k]")
        end
        if get(M, :user_provided_hurdle, false)
            push!(kwargs_parts, "hurdle=M.hurdle[i, k]")
        end
        if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle");
        elseif get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end

        # Logic to select the correct extra parameter
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
    else # Univariate
        kwargs_parts = String[]
        if any_needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma"); end
        if get(M, :user_provided_trials, false)
            push!(kwargs_parts, "trial=Int(M.trials[i, 1])")
        end
        if get(M, :user_provided_weights, false)
            push!(kwargs_parts, "weight=M.weights[i, 1]")
        end
        if get(M, :user_provided_censor_lower, false)
            push!(kwargs_parts, "censor_lower=M.censor_lower[i, 1]")
        end
        if get(M, :user_provided_censor_upper, false)
            push!(kwargs_parts, "censor_upper=M.censor_upper[i, 1]")
        end
        if get(M, :user_provided_hurdle, false)
            push!(kwargs_parts, "hurdle=M.hurdle[i, 1]")
        end
        if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "phi_hurdle=lik_phi_hurdle");
        elseif get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end

        # Logic to select the correct extra parameter
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
        family = M.likelihood_specs[1][:family] # All outcomes share the same family in univariate case
        $(extra_param_logic)
        for i in 1:N
            d_lik = bstm_Likelihood(family, T(M.y_obs[i]); $(kwargs_str))
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
        end
        """
    end
end




 

function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.3.5 (2026-07-18)
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
    # v1.2.1 (2026-07-16)
    if !haskey(M, :log_offset) return "" end
    if is_multivariate
        return "$(eta_name) .+= M.log_offset"
    else
        return "$(eta_name) .+= M.log_offsets"
    end
end



function _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.2.8 (2026-07-17)
    # Rationale: This version generalizes the check for identical priors beyond just the Normal
    #            distribution. It now uses `filldist` for any set of identical priors, falling
    #            back to `Product` only when priors truly differ. This improves efficiency for
    #            cases where multiple fixed effects share the same non-Normal prior. It is also
    #            refactored to return separate prior and update code blocks, fixing a MethodError.
    if get(M, :Xfixed_N, 0) == 0; return "", ""; end

    priors_vec = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
    
    all_same = !isempty(priors_vec) && all(p -> p == priors_vec[1], priors_vec)

    local prior_code, update_code

    if isempty(priors_vec)
        return "", "$(eta_name) .+= M.Xfixed * Xfixed_beta"
    end

    if is_multivariate
        K = M.outcomes_N
        beta_name = "Xfixed_beta_flat"
        update_code = "$(eta_name) .+= M.Xfixed * reshape($(beta_name), M.Xfixed_N, K)"

        if all_same && !isempty(priors_vec)
            prior_str = _distribution_to_string(priors_vec[1])
            prior_code = "$(beta_name) ~ NamedDist(filldist($(prior_str), M.Xfixed_N * K), :Xfixed_beta)"
        else
            full_priors_list = vcat([priors_vec for _ in 1:K]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            priors_product_str = "Product([$(join(priors_str_list, ", "))])"
            prior_code = "$(beta_name) ~ NamedDist($(priors_product_str), :Xfixed_beta)"
        end
    else # Univariate case
        beta_name = "Xfixed_beta"
        update_code = "$(eta_name) .+= M.Xfixed * $(beta_name)"

        if all_same && !isempty(priors_vec)
            prior_str = _distribution_to_string(priors_vec[1])
            prior_code = "$(beta_name) ~ NamedDist(filldist($(prior_str), M.Xfixed_N), :Xfixed_beta)"
        else
            priors_str_list = [_distribution_to_string(p) for p in priors_vec]
            priors_product_str = "Product([$(join(priors_str_list, ", "))])"
            prior_code = "$(beta_name) ~ NamedDist($(priors_product_str), :Xfixed_beta)"
        end
    end

    return prior_code, update_code
end
 
### Version 2.1.0 - 2025-05-22 20:30:00
### Technical Descriptor: Reinforced Spatiotemporal Interaction Assembler with Multivariate and Multi-fidelity Logic Support

# # Spatiotemporal Interaction Assembler
# # Rationale: Implements Knorr-Held Type IV interaction logic by projecting independent 
# # standard normal innovations through the Cholesky factors of the marginal spatial 
# # and temporal precision matrices. Supports univariate and multivariate linear predictors.

function _generate_st_interaction_block(M::NamedTuple, s_spec, t_spec, is_multivariate::Bool, eta_name::String)
    # # Technical Audit: Verification of interaction existence
    if get(M, :model_st, "none") == "none" 
        return ""
    end

    # # Technical Audit: Resource Availability
    if isnothing(s_spec) || isnothing(t_spec)
        @warn "Spatiotemporal interaction requested but marginal specifications are missing."
        return ""
    end

    # # Reference keys for registry lookups
    s_key = string(s_spec.key)
    t_key = string(t_spec.key)
    
    # # Precision Factor Retrieval Logic
    # # If static, use the pre-computed Cholesky factor from the registry.
    # # If dynamic, perform the decomposition within the model scope using the recomposed precision.
    
    s_chol_access = get(s_spec, :is_static, false) ? "spec_registry[\"$s_key\"].cholesky_factor" : "cholesky(Symmetric(spec_registry[\"$s_key\"].Q_template + noise * I))"
    t_chol_access = get(t_spec, :is_static, false) ? "spec_registry[\"$t_key\"].cholesky_factor" : "cholesky(Symmetric(spec_registry[\"$t_key\"].Q_template + noise * I))"

    # # Multivariate outcome dimensions
    K = get(M, :outcomes_N, 1)

    # # Solver Implementation
    # # Mathematical identity for Type IV: X = L_s^{-T} * Z * L_t^{-1}
    # # Implemented via backslash solve: C.U \\ Z provides (L')^{-1} * Z

    if is_multivariate
        interaction_code = """
    # --- Spatiotemporal Interaction Priors ---
    local st_sigma_prior_dist_str = haskey(M, :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) : "Exponential(1.0)"
    st_interaction_sigma ~ NamedDist(filldist($(st_sigma_prior_dist_str), $K), :st_interaction_sigma)
    
    # --- Spatiotemporal Interaction Innovations ---
    st_interaction_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $K), I), :st_interaction_raw)

    let
        # # Marginal Cholesky factors
        C_s = $s_chol_access
        C_t = $t_chol_access
        
        # # Reshape flat innovations into (Spatial x Temporal x Outcome) tensor
        Z_tensor = reshape(st_interaction_raw, M.s_N, M.t_N, $K)
        
        for k in 1:$K
            # # Extract outcome-specific innovation matrix
            Z_k = view(Z_tensor, :, :, k)
            
            # # Vectorized solve: X_k = L_s^{-T} * Z_k * L_t^{-1}
            tmp_spatial = C_s.U \\ Z_k
            st_field_k_unscaled = (transpose(C_t.U \\ transpose(tmp_spatial)))
            
            # Apply soft sum-to-zero constraint for identifiability
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_k_unscaled))
            
            st_field_k = st_field_k_unscaled .* st_interaction_sigma[k]

            # # Apply to multivariate linear predictor
            for i in 1:N
                $(eta_name)[i, k] += st_field_k[M.s_idx[i], M.t_idx[i]]
            end
        end
    end
    """
    else
        interaction_code = """
    # --- Spatiotemporal Interaction Priors ---
    local st_sigma_prior_dist_str = haskey(M, :st_interaction_sigma_prior) ? _distribution_to_string(M.st_interaction_sigma_prior) : "Exponential(1.0)"
    st_interaction_sigma ~ NamedDist($(st_sigma_prior_dist_str), :st_interaction_sigma)

    st_interaction_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_interaction_raw)

    let
        C_s = $s_chol_access
        C_t = $t_chol_access
        
        Z_matrix = reshape(st_interaction_raw, M.s_N, M.t_N)
        
        # # Vectorized solve: X = L_s^{-T} * Z * L_t^{-1}
        tmp_spatial = C_s.U \\ Z_matrix
        st_field_unscaled = (transpose(C_t.U \\ transpose(tmp_spatial)))
        
        # Apply soft sum-to-zero constraint for identifiability
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
        
        st_field = st_field_unscaled .* st_interaction_sigma

        # # Map field to observation indices
        for i in 1:N
            $(eta_name)[i] += st_field[M.s_idx[i], M.t_idx[i]]
        end
    end
    """
    end
    
    return interaction_code
end

 



"""
    _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)

Generates the complete Turing code block for all nested sub-models.

# Rationale
This function was audited and found to be correct. It iterates through the pre-configured
sub-models stored in `M[:nested_manifolds]` and generates the necessary Turing code for each.
Its key responsibilities include:
1.  **Generating Priors**: It creates the prior definitions for all components of the sub-model,
    including its own manifolds, fixed effects, and intercept, ensuring all parameter names
    are prefixed to avoid collisions with the main model.
2.  **Assembling the Sub-Model Predictor**: It generates the code to construct the linear
    predictor (`eta_sub`) for the sub-model.
3.  **Generating the Linking Term**: It defines a `rho_nested` parameter that scales the
    sub-model's effect and adds it to the main model's linear predictor (`eta`). This
    parameter allows the main model to learn the strength and direction of the relationship
    between the low-fidelity proxy and the high-fidelity outcome.
4.  **Generating the Sub-Model Likelihood**: It generates the code that adds the log-probability
    of the sub-model's observations to the total model log-probability, ensuring the
    sub-model is jointly estimated.
"""
function _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)
    if !haskey(M, :nested_manifolds) || isempty(M.nested_manifolds)
        return "", "", ""
    end

    all_nested_priors = String[]
    all_nested_updates = String[]
    all_nested_likelihoods = String[]

    for (var_key, sub_config) in pairs(M.nested_manifolds)
        prefix = string(var_key)
        sub_eta_name = "eta_$(prefix)"

        # --- Generate Priors for Sub-Model ---
        # Manifold priors
        for spec in sub_config.manifolds
            frag = _generate_manifold_code_fragments(spec.manifold_obj, spec, "univariate", nothing; prefix=prefix)
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
        if get(sub_config, :add_intercept, false); update_block *= "\n    $(sub_eta_name) .+= $(prefix)_intercept"; end
        if haskey(sub_config, :log_offset); update_block *= "\n    $(sub_eta_name) .+= M.nested_manifolds[:$(var_key)].log_offset"; end
        if get(sub_config, :Xfixed_N, 0) > 0; update_block *= "\n    $(sub_eta_name) .+= M.nested_manifolds[:$(var_key)].Xfixed * $(prefix)_Xfixed_beta"; end
        
        # Manifold updates
        for spec in sub_config.manifolds
            frag = _generate_manifold_code_fragments(spec.manifold_obj, spec, "univariate", nothing; prefix=prefix)
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
                param_val = get(sub_config, key, nothing)
                val_code = param_val isa AbstractVector ? "sub_M.$(key)[i]" : "sub_M.$(key)"
                if key == :trials; val_code = "Int(" * val_code * ")"; end
                push!(kwargs_parts, "$(key)=$(val_code)")
            end
        end
        kwargs_str = join(kwargs_parts, ", ")

        lik_loop = """
        # Likelihood for nested model: $(prefix)
        let sub_M = M.nested_manifolds[:$(var_key)]
            sub_family = sub_M.likelihood_specs[1][:family]
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
    # v1.2.1 (2026-07-16)
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
    # SVAR requires both spatial and temporal metadata
    s_N = get(data_inputs, :s_N, 1)
    t_N = get(data_inputs, :t_N, 1)
    
    # The inner spatial manifold object is already resolved and stored in m.rho_spatial
    # by `resolve_technical_primitive`. We just need to build its technical spec.
    
    # The inner model is always spatial.
    inner_mod_data = Dict(:type => :spatial, :params => Dict())
    rho_spatial_spec = build_model(m.rho_spatial, data_inputs, inner_mod_data)
    
    hyper_dict = Dict(
        :rho_spatial_spec => rho_spatial_spec,
        :s_N => s_N,
        :t_N => t_N
    )
    
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

# Code Generator for SVAR
function _generate_manifold_code_fragments(m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    # Identify parameter names for the spatially varying rho
    rho_spatial_raw = Symbol("$(v.rho)_spatial_raw")
    rho_field = v.rho_field
    
    # 1. Priors
    priors = """
        # SVAR Implementation: $(key_str)
        # Reconstruct the rho field (mapped to -1, 1 via tanh)
        Q_rho = spec_registry["$(key_str)"].hyper.rho_spatial_spec.Q_template
        F_rho = cholesky(Symmetric(Matrix(Q_rho) + noise * I))
        $(rho_field) = tanh.(F_rho.U \\ $(rho_spatial_raw))
        
        # Initialize latent spatiotemporal field
        $(v.latent) = zeros(T, M.s_N, M.t_N)
        innov_matrix = reshape($(v.innov), M.s_N, M.t_N)
        
        # Evolve state for each spatial unit
        for s in 1:M.s_N
            $(v.latent)[s, :] = ar1_statespace($(rho_field)[s], 1.0, innov_matrix[s, :], T, M.t_N, noise)
        end

        $(v.latent) .*= $(v.sigma)
        for i in 1:N
            $(arch == "multivariate" ? "eta_latent[i, $(outcome_idx)]" : "eta[i]") += $(v.latent)[M.s_idx[i], M.t_idx[i]]
        end
    end
    """
    
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


# Code Generators for Advanced Manifolds

function _generate_manifold_code_fragments(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    
    h_dim = m.hidden_dim
    in_dim = spec.hyper.in_dim
    n_bins = m.nbins

    priors = """
    # Adaptive Basis Priors (Neural Transformation Weights)
    $(v.raw)_W1 ~ NamedDist(MvNormal(zeros(T, $(in_dim * h_dim)), I), :$(v.raw)_W1)
    $(v.raw)_b1 ~ NamedDist(MvNormal(zeros(T, $(h_dim)), I), :$(v.raw)_b1)
    $(v.raw)_W2 ~ NamedDist(MvNormal(zeros(T, $(h_dim * n_bins)), I), :$(v.raw)_W2)
    $(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))
    """

    update = """
    begin
        # Adaptive Basis transformation logic
        X_orig = spec_registry["$(key_str)"].hyper.coords
        
        W1 = reshape($(v.raw)_W1, $(in_dim), $(h_dim))
        b1 = $(v.raw)_b1
        W2 = reshape($(v.raw)_W2, $(h_dim), $(n_bins))

        # Layer 1: Learnable coordinate transformation
        # Using tanh for stable non-linear manifold warping
        H = tanh.(X_orig * W1 .+ b1')

        # Layer 2: Projection to basis space
        $(v.latent) = $(v.sigma) .* (H * W2)

        # Apply basis effect to linear predictor
        # This assumes one-to-one mapping or appropriate indexing from building phase
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= sum($(v.latent), dims=2)
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



# Threshold Autoregressive (TAR): Implements regime-switching temporal dynamics where model parameters depend on a covariate threshold logic.

function build_model(m::TAR, data_inputs::Dict, module_metadata::Dict)
    # TAR operates on temporal indices
    t_N = get(data_inputs, :t_N, 1)
    
    # Resolve the threshold covariate from the data
    data = data_inputs[:data]
    if !hasproperty(data, m.threshold_var)
        error("TAR threshold variable $(m.threshold_var) not found in data.")
    end
    
    threshold_data = data[!, m.threshold_var]

    hyper_dict = Dict(
        :threshold_var => m.threshold_var,
        :threshold_data => Float64.(threshold_data),
        :t_N => t_N
    )

    return (Q_template=nothing, scaling_factor=1.0, model_type=:tar, hyper=NamedTuple(hyper_dict))
end


function _generate_manifold_code_fragments(m::TAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"

    # Manually construct variable names for the regimes to avoid conflicts.
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

 

# LGCP Builder
function build_model(m::LGCP, data_inputs::Dict, module_metadata::Dict)
    # v4.3.0 (2026-07-22) - Spatiotemporal LGCP with Kronecker Solver and SVC support.
    # Rationale: This builder is updated to detect a temporal manifold in the main model
    #            specification. If found, it attaches the temporal spec to the LGCP's
    #            hyperparameters, enabling the code generator to build a spatiotemporal latent field.
    params = module_metadata[:params]
    
    # 1. Resolve underlying spatial model (e.g., ICAR or GP)
    inner_model_sym = Symbol(get(params, :model, :icar))
    
    scheme = get(data_inputs, :prior_scheme, :pcpriors)
    calling_mod = get(data_inputs, :calling_module, Main)
    inner_priors = resolve_hyperpriors(string(inner_model_sym), data_inputs[:hyperpriors], params, scheme, calling_mod)
    
    inner_m_obj = MANIFOLD_CONSTRUCTORS[inner_model_sym](inner_priors, params)
    inner_spec = build_model(inner_m_obj, data_inputs, Dict(:type => :spatial, :params => params))
    
    # 2. Resolve temporal model if present in the main model specification.
    temporal_spec_idx = findfirst(s -> s.domain == :temporal, data_inputs[:manifolds])

    # 3. Collect grid areas pre-computed by the processor.
    areas = get(data_inputs, :grid_areas, ones(data_inputs[:s_N]))

    hyper_dict = Dict(
        :inner_spec => inner_spec,
        :areas => Float64.(areas),
        :s_N => data_inputs[:s_N],
        :t_N => get(data_inputs, :t_N, 1)
    )

    if !isnothing(temporal_spec_idx)
        hyper_dict[:temporal_spec] = data_inputs[:manifolds][temporal_spec_idx]
    end

    return (Q_template=inner_spec.Q_template, scaling_factor=1.0, model_type=:lgcp, hyper=NamedTuple(hyper_dict))
end

function _generate_manifold_code_fragments(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # v4.3.0 (2026-07-22) - Spatiotemporal LGCP with Kronecker Solver and SVC support.
    # Rationale: This refactors the LGCP generator to support fully spatiotemporal latent fields
    #            and correctly incorporate observation-level covariate effects (including SVCs)
    #            into the grid-based intensity function.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)

    # The latent field can be purely spatial or spatiotemporal.
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
    end
    """

    return (priors=priors, update=update)
end


# Model Builder for Kriging

function build_model(m::Kriging, data_inputs::Dict, module_metadata::Dict)
    # Kriging requires continuous coordinates (e.g., s_x, s_y)
    # These are typically passed via the smooth() or spatial() call parameters
    coords = get(module_metadata[:params], :coords, nothing)
    
    if isnothing(coords)
        # Fallback: check if coordinates exist in the top-level data inputs
        if haskey(data_inputs, :coords)
            coords = data_inputs[:coords]
        else
            error("Kriging manifold requires coordinate data. Ensure s_x and s_y are provided.")
        end
    end

    hyper_dict = Dict(
        :coords => Float64.(coords),
        :kernel => m.kernel,
        :in_dim => size(coords, 2)
    )

    # Kriging does not use a sparse Q_template; it relies on a dense covariance matrix K
    return (Q_template=nothing, scaling_factor=1.0, model_type=:kriging, hyper=NamedTuple(hyper_dict))
end


# Code Generator for Kriging

function _generate_manifold_code_fragments(m::Kriging, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    key_str = string(spec.key)
    n_obs = size(spec.hyper.coords, 1)

    priors = """
    # Kriging Priors: Scale and Lengthscale
    $(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))
    $(v.ls) ~ NamedDist($(_distribution_to_string(m.lengthscale)), :$(v.ls))
    $(v.raw) ~ NamedDist(MvNormal(zeros(T, $n_obs), I), :$(v.raw))
    """

    update = """
    begin
        # Kriging (Gaussian Process) Logic
        X_coords = spec_registry["$(key_str)"].hyper.coords
        kernel_type = Symbol(spec_registry["$(key_str)"].hyper.kernel)

        # 1. Construct Covariance Matrix K
        # Uses a vectorized distance calculation and specified kernel function
        K_mat = evaluate_kernel_matrix(X_coords, $(v.sigma), $(v.ls), kernel_type, noise)

        # 2. Non-Centered Parameterization via Cholesky
        # f = sigma * L * z, where K = LL'
        F_krig = cholesky(Symmetric(K_mat))
        $(v.latent) = F_krig.L * $(v.raw)

        # 3. Apply to linear predictor
        $(arch == "multivariate" ? "eta_latent[:, $(outcome_idx)]" : "eta") .+= $(v.latent)
    end
    """

    return (priors=priors, update=update)
end
 



abstract type AbstractModelArchitecture end
struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end

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
    # Purpose: Finds parameter names in the MCMC chain based on the new naming scheme.
    # Rationale: Provides a robust way to locate parameters, trying outcome-specific names first.
    # Inputs:
    #   - p_names: A vector of all parameter names from the chain.
    #   - var, param: The components of the parameter name.
    #   - k: The optional outcome index for multivariate models.
    # Outputs: The full parameter name string, or an empty string if not found.
    base_name = "$(var)_$(param)"

    if !isnothing(k)
        specific_name = "$(base_name)_$(k)"
        if specific_name in p_names
            return specific_name
        end
    end

    if base_name in p_names
        return base_name
    end

    re_indexed = Regex("^" * escape_string(base_name) * "\\[")
    indexed_match = findfirst(n -> occursin(re_indexed, n), p_names)
    if !isnothing(indexed_match)
        return base_name
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
# SECTION 2: MANIFOLD-SPECIFIC EXTRACTION
# ==============================================================================

function extract_manifold(m_obj::Union{ICAR, Besag, RW1, RW2, Leroux, SAR, Cyclic, IID}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    
    domain = spec.domain
    for k in 1:outcomes_N
        var = string(spec.key)
        
        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        latent_name = _find_parameter_new(p_names, var, "raw", k)
        
        n_units = if domain == "spatial"; M.s_N; elseif domain == "temporal"; M.t_N; else M.u_N; end
        
        if isempty(sigma_name) || isempty(latent_name)
            @warn "Parameters for manifold $(spec.key) (domain $(domain), outcome $(k)) not found. Returning zero-matrix for effect."
            push!(structured_effects, zeros(Float64, n_units, n_samples))
            continue
        end

        Q_template = spec.Q_template
        F = cholesky(Symmetric(Matrix(Q_template) + M.noise * I))

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        raw_samples = get_params_vector(chain, latent_name, n_units)
        
        effect = zeros(Float64, n_units, n_samples)
        for s in 1:n_samples
            effect[:, s] = (F.U \ raw_samples[s, :]) .* sigma_samples[s, 1]
        end
        push!(structured_effects, effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 

function extract_manifold(m_obj::BYM2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    unstructured_effects = Vector{Matrix{Float64}}()
    noisy_effects = Vector{Matrix{Float64}}()

    domain = spec.domain
    for k in 1:outcomes_N
        var = string(spec.key)

        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        rho_name = _find_parameter_new(p_names, var, "rho", k)
        struct_name = _find_parameter_new(p_names, var, "struct", k)
        iid_name = _find_parameter_new(p_names, var, "iid", k)

        if isempty(sigma_name) || isempty(rho_name) || isempty(struct_name) || isempty(iid_name)
            @warn "Parameters for BYM2 manifold $(spec.key) (domain $(domain), outcome $(k)) not found. Returning zero-matrix."

            push!(structured_effects, zeros(Float64, M.s_N, n_samples))
            push!(unstructured_effects, zeros(Float64, M.s_N, n_samples))
            push!(noisy_effects, zeros(Float64, M.s_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)
        rho_samples = get_params_vector(chain, rho_name, 1)
        struct_samples = get_params_vector(chain, struct_name, M.s_N)
        iid_samples = get_params_vector(chain, iid_name, M.s_N)

        struct_effect = (struct_samples' .* sqrt.(rho_samples')) .* sigma_samples'
        unstruct_effect = (iid_samples' .* sqrt.(1.0 .- rho_samples')) .* sigma_samples'
        noisy_effect = struct_effect .+ unstruct_effect

        push!(structured_effects, struct_effect)
        push!(unstructured_effects, unstruct_effect)
        push!(noisy_effects, noisy_effect)
    end
    return (structured=structured_effects, unstructured=unstructured_effects, noisy=noisy_effects)
end



function extract_manifold(m_obj::RW1, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the RW1 manifold using state-space logic.
    # Rationale: This specialized method mirrors the state-space code generator for RW1,
    #            which samples innovations and computes a cumulative sum. This ensures
    #            consistency between model sampling and posterior reconstruction.
    # v1.5.1 (2026-07-21)
    structured_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var = string(spec.key)

        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        innov_name = _find_parameter_new(p_names, var, "innov", k)

        if isempty(sigma_name) || isempty(innov_name)
            @warn "Parameters for RW1 manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, M.t_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            latent_field_raw = cumsum(innovations_samples[j, :])
            # The raw field is already soft-centered by the model's prior.
            # We scale directly without applying a hard constraint.
            temporal_effect_k[:, j] = latent_field_raw .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::RW2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the RW2 manifold using state-space logic.
    # Rationale: This specialized method mirrors the state-space code generator for RW2,
    #            which uses a second-order difference equation. This ensures consistency
    #            between model sampling and posterior reconstruction.
    # v1.5.1 (2026-07-21)
    structured_effects = Vector{Matrix{Float64}}()

    for k in 1:outcomes_N
        var = string(spec.key)
        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        innov_name = _find_parameter_new(p_names, var, "innov", k)
        if isempty(sigma_name) || isempty(innov_name); @warn "Parameters for RW2 manifold $(spec.key) not found."; continue; end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            latent_field_raw = Vector{Float64}(undef, M.t_N)
            if M.t_N > 0; latent_field_raw[1] = innovations_samples[j, 1]; end
            if M.t_N > 1; latent_field_raw[2] = 2*latent_field_raw[1] + innovations_samples[j, 2]; end
            for i in 3:M.t_N; latent_field_raw[i] = 2*latent_field_raw[i-1] - latent_field_raw[i-2] + innovations_samples[j, i]; end 
            # The raw field is already soft-centered by the model's prior.
            # We scale directly without applying a hard constraint.
            temporal_effect_k[:, j] = latent_field_raw .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end


function extract_manifold(m_obj::AR1, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = string(spec.key)

        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        rho_name = _find_parameter_new(p_names, var, "rho", k)
        innov_name = _find_parameter_new(p_names, var, "innov", k)

        if isempty(sigma_name) || isempty(rho_name) || isempty(innov_name)
            @warn "Parameters for AR1 manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."

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

function extract_manifold(m::SVAR, spec::NamedTuple, chain, M::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Reconstructs the posterior samples for an SVAR manifold.
    # Rationale: This function mirrors the generative logic from the Turing model to reconstruct
    #            the full spatiotemporal latent field and the spatially varying rho field from
    #            the raw MCMC samples. This is necessary for visualization and prediction.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    # Extract samples from the chain
    sigma_samples = get_params_vector(chain, v.sigma, 1)
    rho_spatial_raw_samples = get_params_vector(chain, Symbol("$(v.rho)_spatial_raw"), M.s_N)
    innov_samples = get_params_vector(chain, v.innov, M.s_N * M.t_N)
    
    n_samples = size(sigma_samples, 1)
    s_N = M.s_N
    t_N = M.t_N
    
    # Reconstruct the spatially varying rho field for each sample
    Q_rho = spec.hyper.rho_spatial_spec.Q_template
    F_rho = cholesky(Symmetric(Matrix(Q_rho) + M.noise * I))
    
    rho_field_samples = Array{Float64, 2}(undef, s_N, n_samples)
    for i in 1:n_samples
        rho_field_samples[:, i] = tanh.(F_rho.U \ rho_spatial_raw_samples[i, :])
    end
    
    # Reconstruct the full spatiotemporal latent field for each sample
    latent_field_samples = Array{Float64, 3}(undef, s_N, t_N, n_samples)
    
    for i in 1:n_samples
        rho_s = rho_field_samples[:, i]
        sigma = sigma_samples[i, 1]
        innov_matrix = reshape(innov_samples[i, :], s_N, t_N)
        
        for s in 1:s_N
            latent_field_samples[s, :, i] = ar1_statespace(rho_s[s], sigma, innov_matrix[s, :], Float64, t_N, M.noise)
        end
    end
    
    # Map the spatiotemporal field to observation-level effects
    outcomes_N = (arch == "multivariate") ? M.outcomes_N : 1
    structured_effects = Vector{Matrix{Float64}}()

    s_idx_full = haskey(M, :s_idx) ? (isnothing(PS) || !haskey(PS, :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? (isnothing(PS) || !haskey(PS, :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)

    for k in 1:outcomes_N
        # This assumes SVAR is not shared if multivariate. If it can be, this logic needs adjustment.
        effect_k = zeros(Float64, N_tot, n_samples)
        for j in 1:n_samples
            for i in 1:N_tot
                effect_k[i, j] = latent_field_samples[s_idx_full[i], t_idx_full[i], j]
            end
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end



function extract_manifold(m_obj::AR2, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the AR2 manifold.
    # Rationale: This function implements the state-space reconstruction for an AR(2) process,
    #            mirroring the logic in the code generator to ensure consistency between
    #            model sampling and posterior summary.
    # v1.5.0 (2026-07-21)
    structured_effects = Vector{Matrix{Float64}}()
    noise_val = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = string(spec.key)

        sigma_name = _find_parameter_new(p_names, var, "sigma", k)
        rho1_name = _find_parameter_new(p_names, var, "rho1", k)
        rho2_name = _find_parameter_new(p_names, var, "rho2", k)
        innov_name = _find_parameter_new(p_names, var, "innov", k)

        if isempty(sigma_name) || isempty(rho1_name) || isempty(rho2_name) || isempty(innov_name)
            @warn "Parameters for AR2 manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, M.t_N, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho1_samples = get_params_vector(chain, rho1_name, 1)[:, 1]
        rho2_samples = get_params_vector(chain, rho2_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innov_name, M.t_N)

        temporal_effect_k = zeros(Float64, M.t_N, n_samples)
        for j in 1:n_samples
            temporal_field_j = Vector{Float64}(undef, M.t_N)
            if M.t_N > 0; temporal_field_j[1] = innovations_samples[j, 1]; end
            if M.t_N > 1; temporal_field_j[2] = rho1_samples[j] * temporal_field_j[1] + innovations_samples[j, 2]; end
            for i in 3:M.t_N
                temporal_field_j[i] = rho1_samples[j] * temporal_field_j[i-1] + rho2_samples[j] * temporal_field_j[i-2] + innovations_samples[j, i]
            end
            temporal_effect_k[:, j] = temporal_field_j .* sigma_samples[j]
        end
        push!(structured_effects, temporal_effect_k)
    end
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m_obj::Union{PSpline, BSpline, TPS, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    basis_key = Symbol(spec.var)
    
    if !haskey(M.basis_matrices, basis_key)
        @warn "Basis matrix for smooth manifold $(basis_key) not found. Returning zero-matrices."
        return (structured=[zeros(Float64, M.y_N, n_samples)], noisy=[zeros(Float64, M.y_N, n_samples)], coefficients=[zeros(Float64, 1, n_samples)])
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


 
function extract_manifold(m_obj::Harmonic, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the Harmonic manifold.
    # Rationale: This version is updated to match the corrected code generator, which samples
    #            `amplitude` and `phase` directly. This function reverses that process to
    #            reconstruct the effect for each posterior sample.
    # v1.8.0 (2026-07-21)
    structured_effects = Vector{Matrix{Float64}}()
    
    # If the period was a random variable, use its posterior mean for reconstruction.
    period_name = _find_parameter_new(p_names, string(spec.key), "period", nothing)
    period_samples = isempty(period_name) ? fill(m_obj.period, n_samples) : get_params_vector(chain, period_name, 1)[:, 1]

    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    for k in 1:outcomes_N
        var = string(spec.key)
        
        amplitude_name = _find_parameter_new(p_names, var, "amplitude", k)
        phase_name = _find_parameter_new(p_names, var, "phase", k)
        
        if isempty(amplitude_name) || isempty(phase_name)
            @warn "Parameters for Harmonic manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_tot, n_samples))
            continue
        end

        amplitude_samples = get_params_vector(chain, amplitude_name, 1)[:, 1]
        phase_samples = get_params_vector(chain, phase_name, 1)[:, 1]

        # Re-derive the internal coefficients from the posterior samples of amplitude and phase
        phase_rad_samples = 2.0 * pi .* phase_samples
        
        beta_cos_samples = amplitude_samples .* cos.(phase_rad_samples)
        beta_sin_samples = amplitude_samples .* sin.(phase_rad_samples)

        angle = (2.0 * pi ./ period_samples') .* u_idx_full
        effect = (beta_cos_samples' .* cos.(angle) .+ beta_sin_samples' .* sin.(angle))
        push!(structured_effects, effect)
    end

    return (structured=structured_effects, noisy=structured_effects)
end



function extract_manifold(m_obj::SVCManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot) 
    cov_var = m_obj.covariate
    is_intercept = (string(cov_var) == "1" || string(cov_var) == "intercept()")

    local x_svc_full
    if is_intercept
        x_svc_full = ones(Float64, N_tot)
    else
        if !hasproperty(M.data, cov_var)
            @warn "Covariate $(cov_var) for SVCManifold not found. Returning zero-matrices."
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
    inner_spec = (key=spec.key, domain=:spatial, var=spec.var, manifold_obj=inner_model)
    inner_effects = extract_manifold(inner_model, chain, M, n_samples, outcomes_N, p_names, inner_spec, PS, N_tot)

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

function extract_manifold(m_obj::Union{GP, FITC, SVGP, Nystrom}, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
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
            @warn "Parameters for GP manifold $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
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

function extract_manifold(m_obj::DynamicsManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the DynamicsManifold.
    # Rationale: This version correctly distinguishes between operators for advection (first-order, non-symmetric)
    #            and diffusion (second-order, symmetric). It uses the appropriate operators (A and L)
    #            retrieved from the manifold's specification and employs LU factorization for the non-symmetric
    #            advection and advection-diffusion cases, ensuring mathematical correctness.
    # v1.3.0 (2026-07-17) - Corrected from v1.2.1
    # Inputs: Standard `extract_manifold` arguments.
    # Outputs: A NamedTuple with the reconstructed spatiotemporal dynamic effect.

    structured_effects = Vector{Matrix{Float64}}()
    model_type = m_obj.model
    key = string(spec.var)
    
    # Retrieve the pre-computed operators from the manifold's specification.
    # The Laplacian L is used for diffusion, and the directed operator A for advection.
    L = spec.hyper.L_template
    A = spec.hyper.A_template
    noise = get(M, :noise, 1e-6)

    for k in 1:outcomes_N
        var = key

        if model_type == "advection" || model_type == "diffusion"
            param_name = model_type == "advection" ? "velocity" : "diffusion"
            rate_name = _find_parameter_new(p_names, key, param_name, k)
            sigma_name = _find_parameter_new(p_names, var, "sigma", k)
            innov_name = _find_parameter_new(p_names, var, "innov", k)

            if isempty(rate_name) || isempty(sigma_name) || isempty(innov_name)
                @warn "Parameters for Dynamics manifold $(key) (model: $(model_type), outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_tot, n_samples))
                continue
            end

            rate_samples = get_params_vector(chain, rate_name, 1)[:, 1]
            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innov_samples = get_params_vector(chain, innov_name, M.s_N * M.t_N)

            I_s = I(M.s_N)
            s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
            t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx
            effect_k = zeros(Float64, N_tot, n_samples)

            for j in 1:n_samples
                dyn_field = zeros(Float64, M.s_N, M.t_N)
                innov_matrix = reshape(innov_samples[j, :], M.s_N, M.t_N)
                
                local propagator
                if model_type == "diffusion"
                    # Diffusion uses the symmetric Laplacian operator L, allowing for efficient Cholesky factorization.
                    op_matrix = Symmetric(I_s - rate_samples[j] * L + noise * I_s)
                    propagator = cholesky(op_matrix)
                else # Advection
                    # Advection uses a non-symmetric directed operator A, requiring a general LU factorization.
                    op_matrix = I_s - rate_samples[j] * A + noise * I_s
                    propagator = lu(op_matrix)
                end

                # Initialize first time step
                dyn_field[:, 1] = innov_matrix[:, 1]

                # Evolve over time by solving the linear system at each step
                for t in 2:M.t_N 
                    dyn_field[:, t] = (propagator \ dyn_field[:, t-1]) + innov_matrix[:, t]
                end
                
                dyn_field .*= sigma_samples[j]

                for i in 1:N_tot
                    effect_k[i, j] = dyn_field[s_idx_full[i], t_idx_full[i]] # This is correct.
                end
            end
            push!(structured_effects, effect_k)

        elseif model_type == "advection_diffusion"
            v_name = _find_parameter_new(p_names, key, "velocity", k) 
            d_name = _find_parameter_new(p_names, key, "diffusion", k) 
            sigma_name = _find_parameter_new(p_names, key, "sigma", k) 
            innov_name = _find_parameter_new(p_names, key, "innov", k) 
 
            if isempty(v_name) || isempty(d_name) || isempty(sigma_name) || isempty(innov_name) 
                @warn "Parameters for advection-diffusion manifold $(key) not found. Returning zero-matrix." 
                push!(structured_effects, zeros(Float64, N_tot, n_samples)); continue 
            end 
 
            v_samples = get_params_vector(chain, v_name, 1)[:, 1] 
            d_samples = get_params_vector(chain, d_name, 1)[:, 1] 
            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1] 
            innov_samples = get_params_vector(chain, innov_name, M.s_N * M.t_N) 
 
            I_s = I(M.s_N) 
            s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx 
            t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx 
            effect_k = zeros(Float64, N_tot, n_samples) 
            for j in 1:n_samples 
                dyn_field = zeros(Float64, M.s_N, M.t_N) 
                innov_matrix = reshape(innov_samples[j, :], M.s_N, M.t_N)
                
                # The combined operator includes a non-symmetric advection part (A) and a symmetric diffusion part (L).
                # The resulting operator is non-symmetric, requiring LU factorization.
                op_matrix = I_s - v_samples[j] * A - d_samples[j] * L + noise * I_s
                propagator = lu(op_matrix)

                dyn_field[:, 1] = innov_matrix[:, 1] 
                for t in 2:M.t_N 
                    dyn_field[:, t] = (propagator \ dyn_field[:, t-1]) + innov_matrix[:, t] 
                end 
                dyn_field .*= sigma_samples[j] 
                for i in 1:N_tot 
                    effect_k[i, j] = dyn_field[s_idx_full[i], t_idx_full[i]] 
                end 
            end 
            push!(structured_effects, effect_k) 

        elseif model_type == "gompertz" || model_type == "logistic_basic"
            # This part of the function is unchanged.
            r_name = _find_parameter_new(p_names, key, "r", k)
            K_name = _find_parameter_new(p_names, key, "K", k)

            if isempty(r_name) || isempty(K_name)
                @warn "Parameters for Dynamics manifold $(key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_tot, n_samples))
                continue
            end

            r_samples = get_params_vector(chain, r_name, 1)
            K_samples = get_params_vector(chain, K_name, 1)
            
            pop_state_samples = zeros(Float64, M.t_N, n_samples)
            for j in 1:n_samples
                pop_state_j = Vector{Float64}(undef, M.t_N)
                pop_state_j[1] = log(K_samples[j] / 2.0) # Start at half carrying capacity
                for t in 2:M.t_N
                    growth = r_samples[j] * (log(K_samples[j]) - pop_state_j[t-1])
                    pop_state_j[t] = pop_state_j[t-1] + growth
                end
                pop_state_samples[:, j] = pop_state_j
            end
            
            t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx
            effect_k = zeros(Float64, N_tot, n_samples)
            for i in 1:N_tot
                effect_k[i, :] = pop_state_samples[t_idx_full[i], :]
            end
            push!(structured_effects, effect_k)
        end
    end
    return (structured=structured_effects, noisy=structured_effects)
end




function extract_manifold(m_obj::MixedManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Logic for reconstructing posterior random effects
    lhs_effects = m_obj.lhs
    n_terms = length(lhs_effects)
    var = string(spec.key)
    n_groups = get(spec.params, :n_cat, 0)

    if n_terms == 1
        # Extraction for simple uncorrelated effect
        structured_effects = Vector{Matrix{Float64}}()
        for k in 1:outcomes_N
            latent_name = _find_parameter_new(p_names, var, "latent", k)
            if isempty(latent_name)
                push!(structured_effects, zeros(Float64, n_groups, n_samples))
                continue
            end
            latent_samples = get_params_vector(chain, latent_name, n_groups)
            push!(structured_effects, latent_samples')
        end
        return (type=:simple, effects=structured_effects, lhs=lhs_effects[1], indices=spec.params.indices)
    else
        # Extraction for correlated multivariate effects
        correlated_effects = Dict{Symbol, Vector{Matrix{Float64}}}()
        noise_val = get(M, :noise, 1e-6)
        inner_m = m_obj.model

        for k in 1:outcomes_N
            l_corr_name = _find_parameter_new(p_names, var, "L_corr", k)
            sigma_effects_name = _find_parameter_new(p_names, var, "sigma_effects", k)
            raw_name = _find_parameter_new(p_names, var, "raw", k)

            if isempty(l_corr_name) || isempty(sigma_effects_name) || isempty(raw_name)
                continue
            end

            l_corr_samps = get_params_vector(chain, l_corr_name, n_terms * n_terms)
            sigma_eff_samps = get_params_vector(chain, sigma_effects_name, n_terms)
            raw_samps = get_params_vector(chain, raw_name, n_groups * n_terms)

            recon_matrix_k = zeros(n_groups, n_terms, n_samples)
            for s in 1:n_samples
                L_corr_s = reshape(l_corr_samps[s, :], n_terms, n_terms)
                L_eff_t = (Diagonal(sigma_eff_samps[s, :]) * L_corr_s)'
                
                L_grp_inv_t = if inner_m isa IID
                    sparse(I, n_groups, n_groups)
                else
                    # Requires specific hyperparameter access for non-IID group structure
                    cholesky(Symmetric(spec.hyper.inner_Q_template + noise_val * I)).U \ I
                end
                
                innov_mat = reshape(raw_samps[s, :], n_groups, n_terms)
                recon_matrix_k[:, :, s] = L_grp_inv_t * innov_mat * L_eff_t
            end

            for (i, term) in enumerate(lhs_effects)
                is_intercept_term = (term == "1" || term == "intercept()")
                term_key = is_intercept_term ? :intercept : Symbol("slope_$(term)")
                if !haskey(correlated_effects, term_key)
                    correlated_effects[term_key] = [zeros(0,0) for _ in 1:outcomes_N]
                end
                correlated_effects[term_key][k] = recon_matrix_k[:, i, :]
            end
        end
        return (type=:correlated, effects=correlated_effects, lhs=lhs_effects, indices=spec.params.indices)
    end
end



function extract_manifold(m_obj::Eigen, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs the effect of the Eigen (Bayesian PCA) manifold.
    # Rationale: This function reconstructs the principal component effects from the
    #            posterior samples of the Householder parameterization. This version corrects
    #            the order of operations to match the generative model.
    # v1.2.9 (2026-07-17) - Corrected from v1.2.1
   # Inputs: Standard `extract_manifold` arguments.
    # Outputs: A NamedTuple with the reconstructed effect of all principal components.
    structured_effects = Vector{Matrix{Float64}}()
    key = string(spec.key)
    var = string(spec.key)
    
    # Extract parameters from the chain
    v_raw_name = _find_parameter_new(p_names, var, "v_raw", nothing)
    d_raw_name = _find_parameter_new(p_names, var, "d_raw", nothing)
    pca_sds_name = _find_parameter_new(p_names, var, "pca_sds", nothing)
    pdef_sds_name = _find_parameter_new(p_names, var, "pdef_sds", nothing)
    factors_name = _find_parameter_new(p_names, var, "factors_flat", nothing)

    if isempty(v_raw_name) || isempty(d_raw_name) || isempty(pca_sds_name) || isempty(pdef_sds_name) || isempty(factors_name)
        @warn "Parameters for Eigen manifold $(key) not found. Returning zero-matrix."
        push!(structured_effects, zeros(Float64, N_tot, n_samples))
        return (structured=structured_effects, noisy=structured_effects)
    end

    n_vars = m_obj.n_vars
    n_factors = m_obj.n_factors
    ltri_indices = m_obj.ltri_indices

    v_raw_samples = get_params_vector(chain, v_raw_name, length(ltri_indices))
    d_raw_samples = get_params_vector(chain, d_raw_name, n_factors)
    pca_sds_samples = get_params_vector(chain, pca_sds_name, n_factors)
    pdef_sds_samples = get_params_vector(chain, pdef_sds_name, n_factors)
    factors_samples = get_params_vector(chain, factors_name, M.y_N * n_factors)

    total_effect = zeros(Float64, M.y_N, n_samples)

    for j in 1:n_samples
        # Reconstruct U (eigenvectors) from Householder reflectors
        v_mat_j = zeros(n_vars, n_factors)
        v_mat_j[ltri_indices] .= v_raw_samples[j, :]
        U_j = householder_to_eigenvector(v_mat_j, n_vars, n_factors)

        # Reconstruct D (eigenvalues)
        d_trans_j = exp.(d_raw_samples[j, :] .* pdef_sds_samples[j, :])
        D_mat_j = Diagonal(d_trans_j)

        # Reconstruct loadings matrix L
        L_j = U_j * D_mat_j

        # Reconstruct factors matrix F
        F_matrix_j = reshape(factors_samples[j, :], M.y_N, n_factors)

        # Corrected order of operations:
        # 1. Scale the raw factors by their standard deviations.
        F_scaled_j = F_matrix_j .* pca_sds_samples[j, :]'
        
        # 2. Then, compute the final effects by multiplying by the loadings matrix.
        eigen_effects_j = F_scaled_j * L_j'
        
        # Sum effects across all principal components
        total_effect[:, j] = sum(eigen_effects_j, dims=2)
    end

    if N_tot > M.y_N
        @warn "Prediction for Eigen manifold is not fully implemented and will be zero for out-of-sample points."
        total_effect = vcat(total_effect, zeros(Float64, N_tot - M.y_N, n_samples))
    end

    # The Eigen effect is univariate; it applies the same effect to all outcomes.
    # We replicate the single reconstructed effect for each outcome.
    for k in 1:outcomes_N
        push!(structured_effects, total_effect)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

function extract_manifold(m::TAR, spec::NamedTuple, chain, M::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Reconstructs the posterior samples for a TAR manifold.
    # Rationale: This function mirrors the generative logic from the Turing model to reconstruct
    #            the full regime-switching temporal latent field from the raw MCMC samples.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    # Define variable names consistent with the code generator
    rho1_name = Symbol("$(v.rho)_1")
    rho2_name = Symbol("$(v.rho)_2")
    sigma1_name = Symbol("$(v.sigma)_1")
    sigma2_name = Symbol("$(v.sigma)_2")
    thresh_raw_name = Symbol("$(v.raw)_thresh")

    # Extract samples from the chain
    rho1_samples = get_params_vector(chain, rho1_name, 1)[:,1]
    rho2_samples = get_params_vector(chain, rho2_name, 1)[:,1]
    sigma1_samples = get_params_vector(chain, sigma1_name, 1)[:,1]
    sigma2_samples = get_params_vector(chain, sigma2_name, 1)[:,1]
    thresh_raw_samples = get_params_vector(chain, thresh_raw_name, 1)[:,1]
    innov_samples = get_params_vector(chain, v.innov, M.t_N)

    n_samples = length(rho1_samples)
    t_N = M.t_N
    threshold_data = spec.hyper.threshold_data
    mean_threshold_data = mean(threshold_data)
    noise = M.noise

    latent_field_samples = Array{Float64, 2}(undef, t_N, n_samples)
    
    for i in 1:n_samples
        threshold_level = mean_threshold_data + thresh_raw_samples[i]
        innov = innov_samples[i, :]
        
        for t in 1:t_N
            regime_indicator = threshold_data[t] > threshold_level
            
            curr_rho = regime_indicator ? rho2_samples[i] : rho1_samples[i]
            curr_sigma = regime_indicator ? sigma2_samples[i] : sigma1_samples[i]
            
            if t == 1
                latent_field_samples[t, i] = (innov[t] * curr_sigma) / sqrt(1.0 - curr_rho^2 + noise)
            else
                latent_field_samples[t, i] = curr_rho * latent_field_samples[t-1, i] + innov[t] * curr_sigma
            end
        end
    end
    
    return (structured=[latent_field_samples], noisy=[latent_field_samples])
end

function extract_manifold(m::AdaptiveSmooth, spec::NamedTuple, chain, M::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Reconstructs the posterior samples for an AdaptiveSmooth manifold.
    # Rationale: This function mirrors the generative logic from the Turing model to reconstruct
    #            the full learned basis effect from the raw MCMC samples of the neural network weights.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    
    # Extract samples from the chain
    W1_samples = get_params_vector(chain, "$(v.raw)_W1", spec.hyper.in_dim * spec.hyper.hidden_dim)
    b1_samples = get_params_vector(chain, "$(v.raw)_b1", spec.hyper.hidden_dim)
    W2_samples = get_params_vector(chain, "$(v.raw)_W2", spec.hyper.hidden_dim * spec.hyper.nbins)
    sigma_samples = get_params_vector(chain, v.sigma, 1)

    n_samples = size(sigma_samples, 1)
    X_orig = spec.hyper.coords
    n_obs = size(X_orig, 1)
    
    latent_field_samples = Array{Float64, 2}(undef, n_obs, n_samples)
    
    for i in 1:n_samples
        W1 = reshape(W1_samples[i, :], spec.hyper.in_dim, spec.hyper.hidden_dim)
        b1 = b1_samples[i, :]
        W2 = reshape(W2_samples[i, :], spec.hyper.hidden_dim, spec.hyper.nbins)
        sigma = sigma_samples[i, 1]
        
        # Reconstruct the effect for this sample
        H = tanh.(X_orig * W1 .+ b1')
        latent_bases = (H * W2) .* sigma
        
        # Sum over the basis functions to get the final effect
        latent_field_samples[:, i] = sum(latent_bases, dims=2)
    end
    
    # Return the reconstructed field
    # For this manifold, structured and noisy are the same.
    return (structured=[latent_field_samples], noisy=[latent_field_samples])
end


function extract_manifold(m_obj::ComposedManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    # Purpose: Reconstructs effects from composed manifolds, with special handling for Kronecker product (spatiotemporal) interactions.
    # Rationale: This function is the designated reconstruction point for effects that are generated by combining multiple manifolds,
    #            such as Knorr-Held spatiotemporal interactions. This logic mirrors the model generation in the assembler.
    # Inputs: Standard `extract_manifold` arguments.
    # Outputs: A NamedTuple with the reconstructed effects.
    op = m_obj.operator 
    key = string(spec.key)

    function _find_spec_by_obj(obj, specs)
        idx = findfirst(s -> s.manifold_obj === obj, specs)
        return isnothing(idx) ? nothing : specs[idx]
    end

    if op == :kronecker_product && haskey(M, :model_st) && M.model_st != "none"
        # This block handles the reconstruction of a global spatiotemporal interaction term.
        spatial_comp_obj = m_obj.components[1] # By convention
        temporal_comp_obj = m_obj.components[2]

        spatial_spec = find_spec_by_obj(spatial_comp_obj, M.manifolds)
        temporal_spec = find_spec_by_obj(temporal_comp_obj, M.manifolds)

        if isnothing(spatial_spec) || isnothing(temporal_spec)
            @warn "Could not resolve components for Kronecker product '$(key)'. ST effect reconstruction skipped."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

        s_Q = spatial_spec.Q_template
        t_Q = temporal_spec.Q_template
        model_st = M.model_st
        noise = get(M, :noise, 1e-6)
        
        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
        C_t = cholesky(Symmetric(Matrix(t_Q) + noise * I))

        s_idx_full = !isnothing(PS) ? vcat(M.s_idx, PS.s_idx) : M.s_idx
        t_idx_full = !isnothing(PS) ? vcat(M.t_idx, PS.t_idx) : M.t_idx

        all_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]

        # This unified solver works for all Knorr-Held interaction types (I, II, III, IV)
        # because if a component is IID, its Q matrix is Identity, and its Cholesky factor is also Identity,
        # which makes the corresponding backslash solve an identity operation.
        for k in 1:outcomes_N
            st_sigma_samples = get_params_vector(chain, "st_interaction_sigma", outcomes_N)[:, k]
            st_raw_samples = if outcomes_N > 1
                st_raw_flat = get_params_vector(chain, "st_interaction_raw_flat", M.s_N * M.t_N * outcomes_N)
                st_raw_flat[:, (k-1)*M.s_N*M.t_N+1 : k*M.s_N*M.t_N]
            else
                get_params_vector(chain, "st_interaction_raw", M.s_N * M.t_N)
            end

            st_effect_k = zeros(Float64, N_tot, n_samples)

            for j in 1:n_samples
                st_innov_matrix = reshape(st_raw_samples[j, :], M.s_N, M.t_N)
                
                # Generalized solve: X = L_s^{-T} * Z * L_t^{-1}
                # This is implemented via backslash solves with the upper Cholesky factor:
                # tmp = L_s' \ Z  => tmp = L_s^{-T} Z
                # X = (L_t' \ tmp')' => X = ( (L_t^{-T} Z^T L_s^{-1})^T ) = L_s^{-T} Z L_t^{-1}
                tmp_spatial = C_s.U \ st_innov_matrix
                st_inter = (transpose(C_t.U \ transpose(tmp_spatial))) .* st_sigma_samples[j]

                for i in 1:N_tot; st_effect_k[i, j] = st_inter[s_idx_full[i], t_idx_full[i]]; end
            end

            all_effects[k] = st_effect_k
        end
        return (structured=all_effects, noisy=all_effects)

    elseif op == :pipe
        # Handles state-space models like `spatial |> smooth(time)` where the coefficients
        # of the dynamic manifold are themselves structured by the state manifold.
        if length(m_obj.components) != 2; error("Pipe operator reconstruction requires exactly two components: state |> dynamic."); end

        state_manifold_obj = m_obj.components[1] 
        dynamic_manifold_obj = get(spec.params, :dynamic_manifold_obj, nothing)

        if isnothing(dynamic_manifold_obj)
            @warn "Could not resolve dynamic manifold for piped manifold '$(key)'. Skipping."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

        basis_key = get(spec.params, :dynamic_basis_key, nothing)
        if isnothing(basis_key) || !haskey(M.basis_matrices, basis_key)
            @warn "Could not find basis matrix for dynamic component of piped manifold '$(key)'. Skipping."
            return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
        end

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
            var_name = "interact_$(key)"
            sigma_name = _find_parameter_new(p_names, var_name, "sigma", k)
            rho_name = _find_parameter_new(p_names, var_name, "rho", k)
            coeffs_raw_name = _find_parameter_new(p_names, var_name, "coeffs_raw", k)

            if isempty(sigma_name) || isempty(coeffs_raw_name); continue; end
     
            sigma_samples = get_params_vector(chain, sigma_name, 1)
            rho_samples = hasproperty(state_manifold_obj, :rho_prior) ? get_params_vector(chain, rho_name, 1) : nothing
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

    else
        @warn "Reconstruction for ComposedManifold with operator ':$op' is not implemented. Returning zero-effect for '$(key)'."
        return (structured=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N], noisy=[zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N])
    end
end


function extract_manifold(m_obj::CustomManifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    reconstruct_func = get(m_obj.params, :reconstruct_func, nothing)

    if !isnothing(reconstruct_func) && isa(reconstruct_func, Function)
        try
            return reconstruct_func(chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
        catch e
            @error "The custom reconstruction function for manifold '$(spec.key)' failed."
            rethrow(e)
        end
    else
        @warn "Reconstruction for custom manifold '$(spec.key)' is not defined. Returning a zero-effect. Please provide a `reconstruct_func` to the `custom()` module to enable posterior reconstruction."
        structured_effects = [zeros(Float64, N_tot, n_samples) for _ in 1:outcomes_N]
        return (structured=structured_effects, noisy=structured_effects)
    end
end

function extract_manifold(m_obj::Manifold, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
    @warn "No specific reconstruction logic for manifold type $(typeof(m_obj)). Returning zero effects."
    n_units = if spec.domain == :spatial; M.s_N; elseif spec.domain == :temporal; M.t_N; else 1; end
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
    # Purpose: Summarizes the posterior samples for all discovered manifold effects.
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

        spec_idx = findfirst(s -> s.key == key, M.manifolds)
        if !isnothing(spec_idx) && M.manifolds[spec_idx].manifold_obj isa MixedManifold
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
            mixed_effects_summaries[key] = (group_var=M.manifolds[spec_idx].var, summaries=summaries_final)
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

    registry = _discover_manifold_realizations(chain, M, PS, n_samples, p_names, 1, N_tot)
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

function _discover_manifold_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    # Purpose: Extracts all latent effects from the MCMC chain.
    # Rationale: Iterates through all specified manifolds and fixed effects, calling the appropriate
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
            fixed_effects_all = zeros(N_tot, n_samples, outcomes_N)
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
        registry[:fixed] = zeros(N_tot, n_samples, outcomes_N)
    end

    # Intercept
    if M.add_intercept
        intercept_samples = get_params_vector(chain, "intercept", outcomes_N)
        intercept_effects = zeros(N_tot, n_samples, outcomes_N)
        for k in 1:outcomes_N
            intercept_effects[:, :, k] .= intercept_samples[:, k]'
        end
        registry[:intercept] = intercept_effects
    else
        registry[:intercept] = zeros(N_tot, n_samples, outcomes_N)
    end

    # Manifolds
    for spec in M.manifolds
        effects = extract_manifold(spec.manifold_obj, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_tot)
        registry[spec.key] = effects
    end

    return NamedTuple(registry)
end

function _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    # Purpose: Assembles the full linear predictor (eta) from all discovered latent effects.
    # Rationale: Mirrors the model's additive structure, combining all components on the link scale.
    # Inputs: The registry of effects and model configuration.
    # Outputs: A 3D array of eta samples `[n_obs, n_samples, n_outcomes]`.
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    eta_latent = zeros(Float64, N_tot, n_samples, outcomes_N)

    eta_latent .+= registry.intercept
    eta_latent .+= registry.fixed

    s_idx_full = haskey(M, :s_idx) ? (isnothing(PS) || !haskey(PS, :s_idx) ? M.s_idx : vcat(M.s_idx, PS.s_idx)) : ones(Int, N_tot)
    t_idx_full = haskey(M, :t_idx) ? (isnothing(PS) || !haskey(PS, :t_idx) ? M.t_idx : vcat(M.t_idx, PS.t_idx)) : ones(Int, N_tot)
    u_idx_full = haskey(M, :u_idx) ? (isnothing(PS) || !haskey(PS, :u_idx) ? M.u_idx : vcat(M.u_idx, PS.u_idx)) : ones(Int, N_tot)

    for spec in M.manifolds
        key = spec.key
        if !haskey(registry, key); continue; end
        
        effects = registry[key]
        effect_set = hasproperty(effects, :noisy) ? effects.noisy : effects.structured
        if isempty(effect_set); continue; end

        for k in 1:outcomes_N
            if spec.domain in [:spatial, :temporal]
                effect_to_add = effect_set[k]
                idx_vec = spec.domain == :spatial ? s_idx_full : t_idx_full
                eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
            elseif spec.domain == :seasonal
                effect_to_add = effect_set[k]
                if spec.manifold_obj isa Harmonic # Harmonic basis is already expanded to N_tot
                    eta_latent[:, :, k] .+= effect_to_add
                else # Assumes Cyclic or other GMRF-like seasonal model
                    idx_vec = u_idx_full
                    eta_latent[:, :, k] .+= effect_to_add[idx_vec, :]
                end
            elseif spec.domain == :smooth || spec.domain == :interact
                eta_latent[:, :, k] .+= effect_set[k]
            elseif spec.domain == :mixed
                group_var_sym = Symbol(spec.var)
                train_indices = effects.indices
                idx_full = if isnothing(PS); train_indices;
                else; train_levels = unique(M.data[!, group_var_sym]); pred_levels = hasproperty(PS.data, group_var_sym) ? unique(PS.data[!, group_var_sym]) : []; all_levels = unique(vcat(train_levels, pred_levels)); level_map = Dict(v => i for (i, v) in enumerate(all_levels)); pred_indices = hasproperty(PS.data, group_var_sym) ? [level_map[v] for v in PS.data[!, group_var_sym]] : Int[]; vcat(train_indices, pred_indices);
                end

                if effects.type == :simple
                    effect_to_add = effects.effects[k]
                    if effects.lhs == "1"; eta_latent[:, :, k] .+= effect_to_add[idx_full, :]; else; cov_vec = isnothing(PS) ? M.data[!, Symbol(effects.lhs)] : vcat(M.data[!, Symbol(effects.lhs)], PS.data[!, Symbol(effects.lhs)]); eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec; end
                elseif effects.type == :correlated
                    for (term_name, term_effects) in pairs(effects.effects)
                        effect_to_add = term_effects[k]
                        if term_name == :intercept; eta_latent[:, :, k] .+= effect_to_add[idx_full, :]; else; cov_name = Symbol(replace(string(term_name), "slope_" => "")); cov_vec = isnothing(PS) ? M.data[!, cov_name] : vcat(M.data[!, cov_name], PS.data[!, cov_name]); eta_latent[:, :, k] .+= effect_to_add[idx_full, :] .* cov_vec; end
                    end
                end
            else
                if size(effect_set[k], 1) == N_tot
                    eta_latent[:, :, k] .+= effect_set[k]
                end
            end
        end
    end

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

    registry = _discover_manifold_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    eta_latent = _modular_eta_assembly(registry, M, PS, n_samples, outcomes_N)
    eta_final = _apply_multivariate_correlation(eta_latent, chain, outcomes_N)

    summarized_effects = _summarize_effects_registry(registry, M, outcomes_N, alpha)

    local p_denoised_summaries, p_noisy_summaries, raw_denoised, raw_noisy, all_log_lik

    is_multinomial = any(s -> s[:family] == "dirichlet_multinomial", M.likelihood_specs)

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

function _reconstruct(arch::MultifidelityArchitecture, mode::String, chain, M, PS, alpha)
    # Purpose: Main reconstruction entry point for multi-fidelity models.
    # Rationale: Handles the hierarchical reconstruction of a main model and its nested sub-models.
    # Inputs: Standard reconstruction arguments for a multi-fidelity model.
    # Outputs: A comprehensive NamedTuple with all summarized posterior statistics for the main model and its sub-models.
    n_samples = size(chain, 1)
    p_names = string.(names(chain))
    N_tot = isnothing(PS) ? M.y_N : M.y_N + PS.y_N
    outcomes_N = M.outcomes_N

    # 1. Reconstruct the main model's components (excluding nested effects)
    main_registry = _discover_manifold_realizations(chain, M, PS, n_samples, p_names, outcomes_N, N_tot)
    
    # 2. Assemble the main model's base eta
    eta_main = _modular_eta_assembly(main_registry, M, PS, n_samples, outcomes_N)

    # 3. Reconstruct sub-models' etas and add them to the main eta
    nested_results = Dict{Symbol, Any}()
    if haskey(M, :nested_manifolds)
        for (key, sub_M) in M.nested_manifolds
            sub_PS = if !isnothing(PS) && haskey(PS, :nested_prediction_sets)
                get(PS.nested_prediction_sets, key, nothing)
            else
                nothing
            end

            sub_outcomes_N = get(sub_M, :outcomes_N, 1)
            sub_N_tot = isnothing(sub_PS) ? sub_M.y_N : sub_M.y_N + sub_PS.y_N
            sub_registry = _discover_manifold_realizations(chain, sub_M, sub_PS, n_samples, p_names, sub_outcomes_N, sub_N_tot)
            eta_sub = _modular_eta_assembly(sub_registry, sub_M, sub_PS, n_samples, sub_outcomes_N)

            rho_name = "nested_$(key)_rho"
            rho_samples = get_params_vector(chain, rho_name, 1)[:, 1]

            if size(eta_sub, 1) != N_tot
                @warn "Size mismatch between main model observations ($N_tot) and nested model '$(key)' observations ($(size(eta_sub, 1))). Cannot apply nested effect."
                continue
            end
            
            if outcomes_N > 1 || sub_outcomes_N > 1
                @warn "Multi-fidelity connection between multivariate models is not fully supported. Assuming a 1-to-1 outcome mapping."
            end
            
            eta_main .+= reshape(rho_samples, 1, n_samples, 1) .* eta_sub

            sub_arch_raw = get(sub_M, :model_arch, "univariate")
            sub_arch_type = sub_arch_raw == "multivariate" ? MultivariateArchitecture() : UnivariateArchitecture()
            nested_results[key] = _reconstruct(sub_arch_type, mode, chain, sub_M, sub_PS, alpha)
        end
    end

    # 4. Apply correlation and generate predictions
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
    nsamples, nobs = size(log_lik)
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
    # v1.2.8 (2026-07-17)
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
    if haskey(M_train, :manifolds)
        ps_basis_registry = Dict{Symbol, Any}()
        smooth_specs = filter(s -> s.domain == :smooth, M_train.manifolds)
        
        for spec in smooth_specs
            key_sym = Symbol(spec.var)
            vars = get(spec.params, :positional_args, [])
            n_vars = length(vars)
            if haskey(M_train.basis_matrices, key_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                m_obj = spec.manifold_obj
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
    if haskey(M_train, :nested_manifolds) && !isempty(M_train.nested_manifolds)
        PS_dict[:nested_prediction_sets] = Dict{Symbol, Any}()
        for (key, sub_M) in M_train.nested_manifolds
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

            if haskey(sub_M, :manifolds)
                sub_ps_basis_registry = Dict{Symbol, Any}()
                sub_smooth_specs = filter(s -> s.domain == :smooth, sub_M.manifolds)
                for spec in sub_smooth_specs
                    v_sym = Symbol(spec.var)
                    vars = get(spec.params, :positional_args, [])
                    n_vars = length(vars)
                    if haskey(sub_M.basis_matrices, v_sym) && all(hasproperty(new_data, Symbol(v)) for v in vars)
                        m_obj = spec.manifold_obj
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
    # 3. Latent Manifold Reconstruction for Likelihood Registry
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

function compare_manifolds(loo_a_report, loo_b_report; model_names=["Model_A", "Model_B"])    
    # Purpose: A utility for formal model comparison between two fitted `bstm` models. It uses 
    #          their PSIS-LOO results to compute the difference in Expected Log Pointwise 
    #          Predictive Density (ELPD) and provides a statistical basis for model selection.
    # Inputs: loo_a_report, loo_b_report, model_names.
    # Outputs: A NamedTuple containing the comparison table, ELPD difference, and LOO objects.

    println("--- Starting BSTM Manifold Comparison ---")

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
    println("\n--- BSTM Manifold Selection Registry ---")
    println("Model A (", model_names[1], "): ELPD = ", round(elpd_a, digits=2), " | p_loo = ", round(p_loo_a, digits=2))
    println("Model B (", model_names[2], "): ELPD = ", round(elpd_b, digits=2), " | p_loo = ", round(p_loo_b, digits=2))
    diff_elpd = elpd_a - elpd_b
    println("\nELPD Delta (A - B): ", round(diff_elpd, digits=2))

    if abs(diff_elpd) > 4.0
        winning_model = diff_elpd > 0 ? model_names[1] : model_names[2]
        println("CONCLUSION: ", winning_model, " is statistically preferred based on predictive density.")
    else
        println("CONCLUSION: Competing manifold structures provide indistinguishable predictive density.")
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
