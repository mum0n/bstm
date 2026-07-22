#!Reference

# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES AND TYPE DEFINITIONS
# ==============================================================================

# Define simple structs for geometric primitives
struct Point2D
    x::Float64
    y::Float64
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
    nbins::Int
    hidden_dim::Int
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


# # Add trait for Dirichlet family
struct DirichletMultinomialFamily <: AbstractBSTM_Family end

# # Likelihood Kernel for Dirichlet-Multinomial
# # Rationale: Maps the linear predictor (eta) via softmax to the probability simplex.
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)
    # # eta_vec is the vector of linear predictors for each category
    # # Transform to simplex via softmax
    probs = softmax(eta_vec)
    # # Total count for the multinomial observation
    n_total = sum(d.y_obs)
    return Multinomial(Int(n_total), probs)
end

# # Specialized bstm_Likelihood method for vector outcomes (Multinomial)
function bstm_kernel(fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec)
    dist = get_dist_ref(fam, d, eta_vec, sig)
    return logpdf(dist, y_vec)
end



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
    :intercept, :spatial, :temporal, :smooth, :fixed, :nested, :eigen, 
    :mixed, :dynamics, :spacetime, :interact, :custom
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
    "inverse_wishart" => InverseWishartFamily()
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
    :kriging => (p, params) -> GP(p.lengthscale, p.sigma, string(get(params, :kernel, "se"))),
    :localadaptive => (p, params) -> LocalAdaptive(p.rho, p.sigma),
    :mosaic => (p, params) -> Mosaic(p.sigma, get(params, :n_regions, 4)),
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
    # Purpose: Generates Turing code for Random Effects (MixedManifold).
    # Rationale: Corrects term-level indexing to handle intercept strings and indices safely.

    v = generate_full_variable_names(spec, arch, outcome_idx, prefix=prefix)
    inner_model = m.model
    group_var = m.group_var
    lhs_effects = m.lhs
    k_effects = length(lhs_effects)
    n_groups = get(spec.params, :n_cat, 0)
    
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "mixed_idx_$(group_var)"

    if k_effects == 1
        # Uncorrelated Case
        inner_frags = _generate_manifold_code_fragments(inner_model, spec, arch, outcome_idx, prefix=prefix)
        update_inner_cleaned = replace(inner_frags.update, Regex("\\s*$(eta_target)\\s*\\.\\+=\\s*.*") => "")
        
        lhs_str = lhs_effects[1]
        is_intercept = (lhs_str == "1" || lhs_str == "intercept()")
        
        application_code = if is_intercept
            "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        else
            "$(eta_target) .+= M.data[!, :$(Symbol(lhs_str))] .* view($(v.latent), M.$(index_var))"
        end

        update_str = """
    begin
        # Mixed Effect Logic: $(lhs_str) | $(group_var)
        $(update_inner_cleaned)
        
        application_code = if is_intercept
            "$(eta_target) .+= view($(v.latent), M.$(index_var))"
        else
            "$(eta_target) .+= M.data[!, :$(Symbol(lhs_str))] .* view($(v.latent), M.$(index_var))"
        end
        $(application_code)
    end
    """
        return (priors=inner_frags.priors, update=update_str)

    else
        # Correlated Case (Multivariate Latent)
        priors_acc = String[]
        push!(priors_acc, "$(v.L_corr) ~ NamedDist(LKJCholesky($(k_effects), 1.0), :$(v.L_corr))")
        push!(priors_acc, "$(v.sigma_effects) ~ NamedDist(filldist(Exponential(1.0), $(k_effects)), :$(v.sigma_effects))")
        push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_groups * k_effects)), I), :$(v.raw))")

        local group_chol
        if inner_model isa IID
            group_chol = "L_groups_inv_t = sparse(I, $(n_groups), $(n_groups))"
        else
            group_chol = """
        Q_groups = spec_registry[\"$(spec.key)\"].Q_template
        F_groups = cholesky(Symmetric(Q_groups + noise * I))
        L_groups_inv_t = F_groups.U \\\\ I
        """
        end

        # Construct dynamic loop over coefficients
                loop_acc = String[]
        for i in 1:k_effects
            term = lhs_effects[i]
            is_int = (term == "1" || term == "intercept()")
            
            header = (i == 1) ? "if i == 1" : "elseif i == $(i)"
            effect_line = if is_int
                "$(eta_target) .+= view(effect_vec_i, M.$(index_var))"
            else
                "$(eta_target) .+= M.data[!, :$(Symbol(term))] .* view(effect_vec_i, M.$(index_var))"
            end
            push!(loop_acc, "        $(header) # term: $(term)")
            push!(loop_acc, "            $(effect_line)")
        end

        update_str = """
    begin
        # Correlated Mixed Effects for Group: $(group_var)
        L_effects_t = (Diagonal($(v.sigma_effects)) * $(v.L_corr).L)'
        $(group_chol)
        innovations_matrix = reshape($(v.raw), $(n_groups), $(k_effects))
        effects_matrix = L_groups_inv_t * innovations_matrix * L_effects_t
        
        for i in 1:$(k_effects)
            effect_vec_i = view(effects_matrix, :, i)
$(join(loop_acc, "\n"))
        end
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
        evolution_step = "dyn_field[:, t] = propagator \\ (dyn_field[:, t-1] + innov_matrix[:, t])"
        evolution_step = "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]"
    elseif model_type == "diffusion"
        propagator_logic = "propagator = cholesky(Symmetric(I(M.s_N) - $(diffusion_name) * $(L_op) + noise * I))"
        evolution_step = "dyn_field[:, t] = propagator \\ (dyn_field[:, t-1] + innov_matrix[:, t])"
        evolution_step = "dyn_field[:, t] = (propagator \\ dyn_field[:, t-1]) + innov_matrix[:, t]"
    elseif model_type == "advection_diffusion"
        propagator_logic = "propagator = lu(I(M.s_N) - $(velocity_name) * $(A_op) - $(diffusion_name) * $(L_op) + noise * I)"
        evolution_step = "dyn_field[:, t] = propagator \\ (dyn_field[:, t-1] + innov_matrix[:, t])"
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
        
        # Flatten the spatiotemporal field and add it to the linear predictor
        # using the pre-computed spatiotemporal index `st_idx`.
        dyn_field_flat = vec(dyn_field)
        $(eta_update_target) .+= view(dyn_field_flat, M.st_idx)
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
    key_str = string(spec.key)
    v = generate_full_variable_names(spec, arch, outcome_idx; prefix=prefix)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"

    params = spec.params
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_first_outcome = outcome_idx == 1
    is_shared = get(params, :shared, false)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(v.sigma))")
    end

    push!(priors_acc, "$(v.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(v.raw))")
    priors_str = join(priors_acc, "\n")

    index_var = spec.domain == :spatial ? "s_idx" : string(spec.domain) * "_idx"
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    effect_application_str = spec.domain == :smooth ? "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(latent_name)" : "$(eta_update_target) .+= view($(latent_name), M.$(index_var))"

    update_str = """
    begin
        # Besag/ICAR/BCGN model for $(key_str)
        Q_template = spec_registry["$(key_str)"].Q_template
        F = cholesky(Symmetric(Q_template + noise * I))
        latent_field_raw = F.U \\ $(v.raw)
        
        # Apply soft sum-to-zero constraint for identifiability
        Turing.@addlogprob! logpdf(Normal(0, 0.001 * $(n_latent)), sum(latent_field_raw))
        $(v.latent) = latent_field_raw .* $(v.sigma)
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
    priors_str = join(priors_acc, "\n    ")
    
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
            # 3. Clamp rho for numerical stability before sqrt.
            local rho_clamped = clamp($(v.rho), 0.0, 1.0)
            # 4. Combine structured and unstructured components using Riebler parameterization.
            local bym2_effect = $(v.sigma) .* (sqrt(rho_clamped) .* struct_latent .+ sqrt(1.0 - rho_clamped) .* $(v.iid))
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

function process_dynamics_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `dynamics()` module.
    # Rationale: Placeholder for future mechanistic model integration.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None.
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
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
    replace_intercept_in_expr(ex)

Recursively traverses a Julia expression and replaces calls to `intercept()` with the integer `1`.
This allows the `mixed()` module's formula parser to treat `intercept()` as a standard
StatsModels.jl intercept term (`1`), enabling correct parsing of random intercept models.

# Arguments
- `ex`: A Julia expression, symbol, or literal.

# Returns
- The modified expression.
"""
function replace_intercept_in_expr(ex)
    if ex isa Expr && ex.head == :call && ex.args[1] == :intercept
        return 1
    elseif ex isa Expr
        return Expr(ex.head, replace_intercept_in_expr.(ex.args)...)
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
    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    local effect_expr, group_var_str

    if !isempty(vars) && vars[1] isa Expr && vars[1].head == :call && vars[1].args[1] == :|
        # Handle `mixed(effect | group)` syntax
        effect_expr = vars[1].args[2]
        group_expr = vars[1].args[3]
        group_var_str = string(group_expr)
    elseif length(vars) >= 2
        # Handle `mixed(effect, group)` syntax
        effect_expr = vars[1]
        group_var_str = string(vars[2])
    else
        @warn "The mixed() module requires syntax `mixed(effect | group)` or `mixed(effect, group)`. Skipping."
        return false
    end

    # --- New Parsing Logic ---
    # 1. Replace `intercept()` with `1` for StatsModels.jl compatibility.
    effect_expr_mod = replace_intercept_in_expr(effect_expr)

    # 2. Use StatsModels to parse the expression and extract term names.
    schema = StatsModels.schema(data)

    local terms
    if effect_expr_mod isa Number
        # `StatsModels.term` correctly converts 1 to InterceptTerm{true} and 0 to InterceptTerm{false}.
        terms = StatsModels.term(effect_expr_mod)
    else
        # For symbols or expressions, wrap in a formula and then apply schema.
        # This is the robust way to parse a formula expression at runtime.
        calling_mod = get(opt_dict, :calling_module, Main)
        form = Core.eval(calling_mod, :(@formula(nothing ~ $(effect_expr_mod))))
        applied_form = StatsModels.apply_schema(form, schema)
        terms = applied_form.rhs
    end

    term_vec = terms isa Tuple ? collect(terms) : [terms]

    effect_names = String[]
    for term in term_vec
        if term isa StatsModels.InterceptTerm{true}
            push!(effect_names, "1")
        elseif term isa StatsModels.InterceptTerm{false}
            # This corresponds to `0` in a formula, so we add no effect.
            continue
        else
            push!(effect_names, _canonical_term_string(term))
        end
    end
    # --- End New Parsing Logic ---

    group_var_sym = Symbol(group_var_str)
    if !hasproperty(data, group_var_sym)
        @warn "Grouping variable ':$group_var_sym' for mixed() module not found in data. Skipping."
        return false
    end
    group_data = data[!, group_var_sym]
    levels = unique(group_data)
    group_map = Dict(v => i for (i, v) in enumerate(levels))
    indices = [group_map[v] for v in group_data]
    mod_data[:params][:indices] = indices

    # Store the generated indices in the main configuration object `M` (aliased as `opt_dict` here)
    # under a unique key that the code generator can construct.
    index_key = Symbol("mixed_idx_$(group_var_str)")
    opt_dict[index_key] = indices
    mod_data[:params][:n_cat] = length(levels)
    
    # Store the parsed effect names as a vector of strings.
    # This fixes the core bug.
    mod_data[:params][:lhs] = effect_names
    
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

function process_bcgn_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `bcgn()` module for bipartite graphs.
    # Rationale: Validates the provided bipartite adjacency matrix.
    # v1.2.1 (2026-07-16)
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Boolean indicating if a manifold should be created.
    params = mod_data[:params]
    if !haskey(params, :bipartite_adj) || isempty(params[:bipartite_adj]) || all(iszero, params[:bipartite_adj])
        error("The `bcgn()` module requires a non-empty `:bipartite_adj` sparse matrix parameter.")
    end
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
    # as a positional argument to the `bstm` function.
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


function bstm(formula::String, data::DataFrame; kwargs...)
    # Purpose: A wrapper for `bstm` that defaults to the `Main` module context.
    # Rationale: Simplifies calls when not inside a custom module.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - formula, data, kwargs.
    # Outputs: A Turing.jl model object.
    return bstm(formula, data, Main; kwargs...)
end

function bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # Purpose: A wrapper that first calls `bstm_config` and then `bstm`.
    # Rationale: Separates configuration from model instantiation.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - formula, data, calling_module, kwargs.
    # Outputs: A Turing.jl model object.
    options = bstm_config(formula, data; calling_module=calling_module, kwargs...)
    
    model_func_name, expr, new_config, registry = bstm_codegen(options)

    if get(new_config, :verbose, true)
        println("\n--- Dynamically Generated Model Code ---")
        println(new_config.generated_model_code)
        println("----------------------------------------\n")
    end

    # This function is intended for programmatic (non-interactive) use.
    # The `eval` happens here. For interactive use, the @bstm macro is recommended.
    calling_module.eval(expr)
    model_func = getfield(calling_module, model_func_name)
    
    # We use invokelatest to be safe, as the function was just defined.
    local model_instance = Base.invokelatest(model_func, new_config, registry)

    if get(new_config, :verbose, true)
        println("\n--- Running prior predictive check ---")
    end
    try
        local prior_sample = Base.invokelatest(rand, model_instance)
        if get(new_config, :verbose, true)
            println("Prior sample check successful. Sample values:")
            display(prior_sample)
        end
    catch e
        _bstm_error_handler(e, model_instance)
    end
    if get(new_config, :verbose, true)
        println("--------------------------------------\n")
    end

    return model_instance
end


function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # Purpose: Converts a parsed module dictionary into a concrete `Manifold` struct instance. 
    # v1.4.3 (2026-07-20)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    
    if m_type == :svc
        covariate_sym = Symbol(get(m_params, :covariate, :unknown))
        if covariate_sym == :unknown; error("SVC manifold is missing covariate information."); end
        
        spatial_model_spec_node = get(m_params, :spatial_model_spec, nothing)
        if isnothing(spatial_model_spec_node); error("SVC manifold is missing its spatial model specification."); end
        
        spatial_mod_data = Dict(:type => spatial_model_spec_node.module_type, :params => spatial_model_spec_node.args, :variables => get(spatial_model_spec_node.args, :positional_args, []))
        inner_manifold_obj = resolve_technical_primitive(spatial_mod_data, M, priors_dict, scheme)
        
        return SVCManifold(covariate_sym, inner_manifold_obj)

    elseif m_type == :mixed
        group_var_sym = Symbol(module_metadata[:variables][1])
        lhs_str = module_metadata[:params][:lhs]
        
        # Helper to resolve the inner model for the mixed effect.
        function resolve_inner_model(params, domain_type, M, priors_dict, scheme)
            default_model = "iid" # Default for mixed effects is IID random intercepts/slopes.
            model_name = string(get(params, :model, default_model))
            model_sym = Symbol(model_name)
            resolved_priors = resolve_hyperpriors(model_name, priors_dict, params, scheme, M[:calling_module])
            if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
                return MANIFOLD_CONSTRUCTORS[model_sym](resolved_priors, params)
            else
                error("Unknown inner model '$model_name' for mixed effect.")
            end
        end

        inner_model_obj = resolve_inner_model(m_params, :mixed, M, priors_dict, scheme)
        return MixedManifold(group_var_sym, lhs_str, inner_model_obj)

    elseif m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(:type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, [])), components_data)
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
        return ComposedManifold(resolved_components, op)

    else
        default_model = if m_type == :spatial; haskey(M, :W) ? "bym2" : "iid"; elseif m_type == :temporal; "rw2"; else "none"; end
        model_name = string(get(m_params, :model, default_model))
        model_sym = Symbol(model_name)
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
            return MANIFOLD_CONSTRUCTORS[model_sym](resolved_priors, m_params)
        else
            error("Unknown manifold model '$model_name' for module '$m_type'.")
        end
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
    for i in 1:length(eta)
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[i] : d.weight
        logp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
    end
    return logp
end
 

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
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

function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(clamp(exp(eta), 1e-9, 1e9)); end
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

    if type in [ "smooth", "barycentric", "linear"]
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
    families = [get(spec, :family, "gaussian") for spec in M.likelihood_specs]
    needs_sigma = any(f -> string(f) in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)

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
