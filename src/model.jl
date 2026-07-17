 
# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES AND TYPE DEFINITIONS
# ==============================================================================


abstract type Manifold end
abstract type ManifoldModel <: Manifold end
abstract type ManifoldOperator <: Manifold end

struct NoneManifold <: ManifoldModel end

struct IID <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ICAR <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct Besag <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct BYM2 <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct Leroux <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct SAR <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct RW1 <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct RW2 <: ManifoldModel; sigma_prior::UnivariateDistribution; end


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
    lengthscale_prior::UnivariateDistribution
end

struct Eigen <: ManifoldModel
    n_vars::Int
    n_factors::Int
    pca_sd_prior::UnivariateDistribution
    pdef_sd_prior::UnivariateDistribution
    ltri_indices::Vector{Int}
end
struct Moran <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct Spherical <: ManifoldModel; sigma_prior::UnivariateDistribution; range_prior::UnivariateDistribution; end
struct Barycentric <: ManifoldModel; sigma_prior::UnivariateDistribution; end

struct BCGN <: ManifoldModel; sigma_prior::UnivariateDistribution; bipartite_adj::AbstractMatrix; end
struct NetworkFlow <: ManifoldModel; sigma_prior::UnivariateDistribution; adjacency_matrix::AbstractMatrix; flow_direction::Symbol; end
struct LocalAdaptive <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct Mosaic <: ManifoldModel; sigma_prior::UnivariateDistribution; n_regions::Int; end
struct TensorProductSmooth <: ManifoldModel; sigma_prior::UnivariateDistribution; Q_template::AbstractMatrix; end

struct TPS <: ManifoldModel; nbins::Int; sigma_prior::UnivariateDistribution; end
struct BSpline <: ManifoldModel; nbins::Int; degree::Int; sigma_prior::UnivariateDistribution; end

struct AR1 <: ManifoldModel; rho_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
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
struct MixedManifold <: ManifoldOperator
    group_var::Symbol
    lhs::String
    model::ManifoldModel
end


abstract type ManifoldSupervisor <: Manifold end
struct NestedManifold <: ManifoldSupervisor
    var::Symbol
    formula::String
    data_source::Symbol
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
    r_nb::R
    sigma_y::S
    trial::T
    y_L::TL
    y_U::TU
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
    :wavelet => (p, params) -> Wavelet(get(params, :family, :db4), get(params, :nbins, 32), p.sigma_prior, p.lengthscale_prior),
    :eigen => (p, params) -> Eigen(get(params, :n_vars, 0), get(params, :n_factors, 1), p.pca_sd_prior, p.pdef_sd_prior, get(params, :ltri_indices, Int[])),
    :moran => (p, params) -> Moran(p.sigma_prior),
    :spherical => (p, params) -> Spherical(p.sigma_prior, p.range_prior),
    :barycentric => (p, params) -> Barycentric(p.sigma_prior),
    :bcgn => (p, params) -> BCGN(p.sigma_prior, get(params, :bipartite_adj, sparse(zeros(1,1)))),
    :networkflow => (p, params) -> NetworkFlow(p.sigma_prior, get(params, :adjacency_matrix, sparse(zeros(1,1))), get(params, :flow_direction, :bidirectional)),
    :kriging => (p, params) -> GP(p.lengthscale_prior, p.sigma_prior, string(get(params, :kernel, "se"))),
    :localadaptive => (p, params) -> LocalAdaptive(p.rho_prior, p.sigma_prior),
    :mosaic => (p, params) -> Mosaic(p.sigma_prior, get(params, :n_regions, 4)),
    :tensorproductsmooth => (p, params) -> TensorProductSmooth(p.sigma_prior, get(params, :Q_template, sparse(zeros(1,1)))),
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
            push!(terms, strip(current_term))
            current_term = ""
            i += sep_len
            continue
        end
        
        current_term *= char
        i += 1
    end
    if !isempty(strip(current_term))
        push!(terms, strip(current_term))
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
    val_str = strip(val_str)
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
        key = Symbol(strip(key_val[1]))
        val_str = String(strip(key_val[2]))
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
            arg_val = String(strip(current_arg))
            if !isempty(arg_val)
                _add_parsed_arg!(args_dict, positional_args, arg_val)
            end
            current_arg = ""
        else
            current_arg *= char
            if char == '(' || char == '['; depth += 1; elseif char == ')' || char == ']'; depth -= 1; end
        end
    end

    arg_val = String(strip(current_arg))
    if !isempty(arg_val)
        _add_parsed_arg!(args_dict, positional_args, arg_val)
    end

    if !isempty(positional_args)
        args_dict[:positional_args] = positional_args
    end

    return args_dict
end

function _parse_single_manifold_term(term_str::AbstractString)
    # Purpose: Parses a single module call string like "spatial(s_idx, model=bym2)".
    # Rationale: Extracts the module name and its arguments.
    # Assumptions: The input is a single, well-formed module call.
    # Inputs:
    #   - term_str: The module call string.
    # Outputs: A NamedTuple `(module_type, args)`.
    term_str = strip(term_str)
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

function resolve_hyperpriors(model_name::String, global_priors::Dict, local_params::Dict, scheme::Symbol, calling_mod::Module)
    # Purpose: Resolves hyperpriors for a given manifold based on a 3-level precedence.
    # Rationale: This version is updated to correctly evaluate `Expr` objects for priors,
    #            allowing users to specify distributions directly in the formula string.
    # v1.2.8 (2026-07-17)
    # Precedence:
    #   1. Local: A prior specified directly in the module call (e.g., `sigma_prior=...`).
    #   2. Global: A prior specified in the `hyperpriors` dictionary passed to `bstm()`.
    #   3. Scheme: The default prior from the selected scheme (`:pcpriors`, `:informative`, etc.).
    # Inputs:
    #   - model_name, global_priors, local_params, scheme, calling_mod.
    # Outputs: A NamedTuple containing the resolved prior distributions for the manifold.

    prior_defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else
        UNINFORMATIVE_PRIORS
    end

    possible_priors = [:sigma_prior, :rho_prior, :lengthscale_prior, :kappa_prior, :amplitude_prior, :phase_prior, :pca_sd_prior, :pdef_sd_prior, :range_prior]
    
    resolved = Dict{Symbol, Any}()

    for p_sym in possible_priors
        p_base_name = Symbol(replace(string(p_sym), "_prior" => ""))

        # 1. Check for local override in the module call.
        if haskey(local_params, p_sym)
            prior_val = local_params[p_sym]
            if prior_val isa Tuple # PC Prior constraint
                resolved[p_sym] = create_pc_prior(p_base_name, prior_val)
            elseif prior_val isa Expr
                try
                    resolved[p_sym] = Core.eval(calling_mod, prior_val)
                catch e
                    error("Could not evaluate `prior` argument `$(prior_val)` for manifold '$model_name'. Error: $e")
                end
            else # Assumed to be a Distribution object
                resolved[p_sym] = prior_val
            end
            continue
        end

        # 2. Check for global override in the `hyperpriors` dictionary.
        global_key_model = Symbol(model_name, "_", p_base_name)
        global_key_param = p_base_name
        
        if haskey(global_priors, global_key_model)
            resolved[p_sym] = global_priors[global_key_model]
            continue
        elseif haskey(global_priors, global_key_param)
            resolved[p_sym] = global_priors[global_key_param]
            continue
        end

        # 3. Fallback to the default scheme.
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
    term_str_stripped = strip(term_str)
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
        elseif hasproperty(node, :module_type) && node.module_type in BSTM_MODULE_KEYWORDS
            if node.module_type == :fixed
                if haskey(node.args, :positional_args)
                    append!(fixed_effects, string.(node.args[:positional_args]))
                end
            end
            key_parts = [string(node.module_type)]
            if haskey(node.args, :positional_args) && !isempty(node.args[:positional_args])
                pos_arg_str = join([string(a) for a in node.args[:positional_args]], "_")
                push!(key_parts, pos_arg_str)
            end
            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
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
    term = strip(term)
    m = match(r"likelihood\((.*)\)", term)
    specs = Dict{Symbol, Any}[]
    if !isnothing(m)
        inner_content = m.captures[1]
        args = split_terms_at_depth(inner_content, ",")
        if isempty(args); return specs; end
        outcome_var_str = strip(args[1])
        params_str = join(args[2:end], ",")
        params = _parse_arguments_string(params_str)
        for ov in [strip(s) for s in Base.split(outcome_var_str, '+')]; push!(specs, Dict(:var => ov, :params => params)); end
    else
        for ov in [strip(s) for s in Base.split(term, '+')]; push!(specs, Dict(:var => ov, :params => Dict())); end
    end
    return specs
end

function decompose_bstm_formula(formula_str::String)
    # Purpose: The main entry point for formula parsing.
    # Rationale: Decomposes the entire formula string into its constituent parts: outcomes, modules, and fixed effects.
    # Assumptions: Formula is in the format `lhs ~ rhs`.
    # v1.2.1 (2026-07-16)
    # Inputs:
    #   - formula_str: The complete model formula.
    # Outputs: A NamedTuple with `:outcomes`, `:modules`, `:fixed_effects`, `:has_intercept`, and `:intercept_prior`.
    parts = Base.split(formula_str, "~")
    lhs_str = strip(parts[1])
    rhs_str = strip(parts[2])
    outcome_specs = vcat([_parse_lhs_term(term) for term in split_terms_at_depth(lhs_str, "+")]...)

    rhs_normalized = replace(rhs_str, r"\s*-\s*" => " + -")
    rhs_terms = split_terms_at_depth(rhs_normalized, " + ")
    
    has_intercept = !("0" in rhs_terms || "-1" in rhs_terms)
    intercept_prior = nothing
    module_terms = String[]
    intercept_module_found = false

    for term in rhs_terms
        term_stripped = strip(term)
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
    # Purpose: Ensures all likelihood-related parameters (offsets, weights, etc.) are present in the
    #          model configuration `M` with the correct type (scalar or vector) and default values.
    # Rationale: This version avoids broadcasting scalars into vectors, preserving their scalar nature.
    #            This allows the model code generator to create more efficient and readable code by
    #            distinguishing between scalar and vector parameters at runtime.
    # v1.2.5 (2026-07-17)
    # Assumptions: M[:y_N] (number of observations) and M[:outcomes_N] (number of outcomes) are set.
    # Inputs:
    #   - M: The model configuration dictionary, which is mutated by this function.
    # Outputs: None.

    N = M[:y_N]
    K = M[:outcomes_N]
    is_multivariate = K > 1

    param_defaults = [
        (:y_L, -Inf),
        (:y_U, Inf),
        (:hurdle, -Inf),
        (:trials, 1),
        (:weights, 1.0),
        (:log_offset, 0.0)
    ]

    for (key, default_val) in param_defaults
        if !haskey(M, key)
            # Parameter not provided, store the scalar default.
            M[key] = default_val
        else
            val = M[key]
            if is_multivariate
                if val isa Real
                    # Keep as scalar, it will be broadcast at runtime if needed.
                    M[key] = val
                elseif val isa AbstractVector && length(val) == N
                    # This is a vector intended for all outcomes. It will be indexed at runtime.
                    # No change needed here.
                elseif val isa AbstractMatrix && size(val) == (N, K)
                    # Correctly specified matrix. No change needed.
                else
                    @warn "Likelihood parameter `:$key` has incorrect dimensions for multivariate model. Expected ($N, $K) or ($N,). Using default."
                    M[key] = default_val
                end
            else # Univariate case
                if val isa Real
                    # It's a scalar, leave it as is.
                    M[key] = val
                elseif !(val isa AbstractVector)
                    # Ensure it's a vector if it's not a scalar (e.g., a 1-column matrix)
                    M[key] = vec(val)
                end
                
                # Check vector length only if it is a vector
                if M[key] isa AbstractVector && length(M[key]) != N
                    @warn "Likelihood parameter `:$key` has incorrect length for univariate model. Expected length $N, got $(length(M[key])). Using default."
                    M[key] = default_val
                end
            end
        end
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
    for out_spec in decomposed_formula.outcomes; push!(all_vars, Symbol(out_spec[:var])); for key in [:offsets, :log_offsets, :weights, :trials, :y_L, :y_U]; if haskey(out_spec[:params], key); val = out_spec[:params][key]; if val isa Symbol; push!(all_vars, val); end; end; end; end
    for fe in decomposed_formula.fixed_effects; push!(all_vars, Symbol(fe)); end
    for (_, mod_data) in pairs(decomposed_formula.modules); if haskey(mod_data.args, :positional_args); for arg in mod_data.args[:positional_args]; if arg isa Symbol; push!(all_vars, arg); elseif arg isa Expr; _extract_symbols_from_expr!(all_vars, arg); end; end; end; end
    df_processed = deepcopy(data)
    vars_to_categorize = Set{Symbol}()
    for (_, mod_data_nt) in decomposed_formula.modules; if mod_data_nt.module_type == :fixed; if haskey(mod_data_nt.args, :positional_args); for var_sym in mod_data_nt.args[:positional_args]; if var_sym isa Symbol; if haskey(mod_data_nt.args, :contrast) || get(mod_data_nt.args, :model, nothing) == :categorical; push!(vars_to_categorize, var_sym); end; end; end; end; end; end
    for var_sym in vars_to_categorize; if hasproperty(df_processed, var_sym) && !(eltype(df_processed[!, var_sym]) <: CategoricalValue); df_processed[!, var_sym] = categorical(df_processed[!, var_sym]); end; end
    valid_vars_in_data = filter(v -> hasproperty(df_processed, v), all_vars)
    filtered_data = DataFrame(df_processed[completecases(df_processed, collect(valid_vars_in_data)), :])
    if nrow(filtered_data) < nrow(data); @warn "Missing values detected. $(nrow(data) - nrow(filtered_data)) rows were removed."; end
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
    M[:hyperpriors] = get(M, :hyperpriors, Dict{String, Any}())
    M[:prior_scheme] = get(M, :prior_scheme, :pcpriors)
    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        mod_data_dict = Dict(:type => mod_data_nt.module_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)
        
        create_manifold_for_module = true
        if !isnothing(processor!); create_manifold_for_module = processor!(M, mod_data_dict, M, M[:hyperpriors]); end
        
        if mod_type == :fixed || !create_manifold_for_module; continue; end
        
        manifold_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
        manifold_spec_built = build_model(manifold_obj, M)
        spec = (key=Symbol(key), domain=mod_data_dict[:type], var=join(mod_data_dict[:variables], "_"), manifold_obj=manifold_obj, params=mod_data_dict[:params], Q_template=manifold_spec_built.Q_template, scaling_factor=manifold_spec_built.scaling_factor)
        push!(M[:manifolds], spec)
    end
    _process_fixed_effects!(M, unique(decomposed_formula.fixed_effects))
    _process_fixed_effects_priors!(M)
    
    _precompute_static_manifolds!(M)

    _finalize_config!(M)
    return NamedTuple(M)
end


function _generate_manifold_code_fragments(m::ManifoldModel, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates code fragments for simple GMRF-style manifolds (IID, ICAR, etc.).
    # Rationale: Implements a standard non-centered parameterization for Gaussian Markov Random Fields.
    #            This version checks if a manifold is static and uses a pre-computed Cholesky factor
    # v1.2.1 (2026-07-16)
    #            to optimize sampling. It also handles multivariate parameter naming conventions.
 
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

    # Determine parameter names based on architecture (univariate vs. multivariate)
    # and whether parameters are shared across outcomes.
    local sigma_name_str, rho_name_str, latent_raw_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        rho_name_str = "$(prefixed_key)_rho_$(outcome_idx)"
        latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        rho_name_str = "$(prefixed_key)_rho"
        if is_multivariate
            latent_raw_name_str = "$(prefixed_key)_raw_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            latent_raw_name_str = "$(prefixed_key)_raw"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    sigma_name = Symbol(sigma_name_str)
    rho_name = Symbol(rho_name_str)
    latent_raw_name = Symbol(latent_raw_name_str)
    latent_name = Symbol(latent_name_str)

    priors_acc = String[]

    # Generate priors only once for shared parameters in multivariate models.
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        if hasproperty(m, :sigma_prior)
            push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(m.sigma_prior)), :$(sigma_name))")
        end
        if hasproperty(m, :rho_prior)
            push!(priors_acc, "$(rho_name) ~ NamedDist($(_distribution_to_string(m.rho_prior)), :$(rho_name))")
        end
    end

    # The raw latent effect is always sampled from a standard normal.
    push!(priors_acc, "$(latent_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(latent_raw_name))")
    priors_str = join(priors_acc, "\n")
    
    # Determine the correct indexing variable based on the manifold's domain.
    local index_var
    if spec.domain == :spatial
        index_var = "s_idx"
    elseif spec.domain == :temporal
        if typeof(m) <: Union{Cyclic, Harmonic}
            index_var = "u_idx"
        else
            index_var = "t_idx"
        end
    else
        index_var = string(spec.domain) * "_idx"
    end

    # Determine the target for updating the linear predictor.
    local eta_update_target
    if is_multivariate
        eta_update_target = "eta_latent[:, $(outcome_idx)]"
    else
        eta_update_target = "eta"
    end

    local effect_application_str
    if spec.domain == :smooth
        effect_application_str = "$(eta_update_target) .+= M.basis_matrices[:$(spec.var)] * $(latent_name)"
    else
        effect_application_str = "$(eta_update_target) .+= view($(latent_name), M.$(index_var))"
    end

    local update_str
    if get(spec, :is_static, false)
        update_str = """
        begin
            # Using pre-computed Cholesky factor for static manifold '$(key_str)'
            F = spec_registry["$(key_str)"].cholesky_factor
            $(latent_name) = $(sigma_name_str) .* (F.U \\ $(latent_raw_name))
            $(effect_application_str)
        end
        """
    else
        update_str = """
        begin
            Q_template = spec_registry["$(key_str)"].Q_template
            m_type = spec_registry["$(key_str)"].manifold_obj |> typeof |> Symbol
            rho_val = $(hasproperty(m, :rho_prior) ? rho_name_str : "nothing")
            
            Q_final = recompose_precision(m_type, Q_template, 1.0; extra_param=rho_val)
            F = cholesky(Symmetric(Q_final + noise * I))
            $(latent_name) = $(sigma_name_str) .* (F.U \\ $(latent_raw_name))
            $(effect_application_str)
        end
        """
    end
    
    return (priors=priors_str, update=update_str)
end



function _generate_manifold_code_fragments(m::AR1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # This is a specialized implementation for AR(1) processes.
    # It uses a state-space formulation for numerical stability and efficiency,
    # avoiding the construction and Cholesky decomposition of a dense precision matrix.

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

    # Determine parameter names
    local sigma_name_str, rho_name_str, innov_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        rho_name_str = "$(prefixed_key)_rho_$(outcome_idx)"
        innov_name_str = "$(prefixed_key)_innov_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        rho_name_str = "$(prefixed_key)_rho"
        if is_multivariate
            innov_name_str = "$(prefixed_key)_innov_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            innov_name_str = "$(prefixed_key)_innov"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    sigma_name = Symbol(sigma_name_str)
    rho_name = Symbol(rho_name_str)
    innov_name = Symbol(innov_name_str)
    latent_name = Symbol(latent_name_str)

    priors_acc = String[]

    # Generate priors only once for shared parameters
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(m.sigma_prior)), :$(sigma_name))")
        # The prior for rho is truncated to ensure stationarity and numerical stability.
        push!(priors_acc, "$(rho_name) ~ NamedDist(truncated($(_distribution_to_string(m.rho_prior)), -0.9999, 0.9999), :$(rho_name))")
    end

    # Sample standard normal innovations
    push!(priors_acc, "$(innov_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(innov_name))")
    priors_str = join(priors_acc, "\n")

    # Determine the target for updating the linear predictor
    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx" # AR1 is always temporal

    # The update block performs the sequential calculation
    update_str = """
    begin
        # AR1 state-space evolution for $(key_str)
        $(latent_name) = Vector{T}(undef, $(n_latent))
        
        rho_val = $(rho_name) # No clamp needed due to truncated prior

        # Initialize the first state and evolve
        $(latent_name)[1] = $(innov_name)[1] / sqrt(1.0 - rho_val^2 + noise)
        for t in 2:$(n_latent)
            $(latent_name)[t] = rho_val * $(latent_name)[t-1] + $(innov_name)[t]
        end
        
        # Scale by sigma and apply to eta
        $(latent_name) .*= $(sigma_name_str)
        $(eta_update_target) .+= view($(latent_name), M.$(index_var))
    end
    """
    
    return (priors=priors_str, update=update_str)
end


function _generate_manifold_code_fragments(m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for RW1 processes using a state-space formulation.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Determine parameter names
    local sigma_name_str, innov_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        innov_name_str = "$(prefixed_key)_innov_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        if is_multivariate
            innov_name_str = "$(prefixed_key)_innov_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            innov_name_str = "$(prefixed_key)_innov"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    sigma_name = Symbol(sigma_name_str)
    innov_name = Symbol(innov_name_str)
    latent_name = Symbol(latent_name_str)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(m.sigma_prior)), :$(sigma_name))")
    end
    push!(priors_acc, "$(innov_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(innov_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # RW1 state-space evolution for $(key_str)
        innovations = $(innov_name)
        latent_field_raw = cumsum(innovations)
        
        # Apply sum-to-zero constraint for identifiability
        if $(n_latent) > 0
            $(latent_name) = (latent_field_raw .- mean(latent_field_raw)) .* $(sigma_name_str)
            $(eta_update_target) .+= view($(latent_name), M.$(index_var))
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end


function _generate_manifold_code_fragments(m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Specialized implementation for RW2 processes using a state-space formulation.
    key_str = string(spec.key)
    prefixed_key = isempty(prefix) ? key_str : "$(prefix)_$(key_str)"
    
    n_latent = size(spec.Q_template, 1)
    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    # Determine parameter names
    local sigma_name_str, innov_name_str, latent_name_str
    if is_multivariate && !is_shared
        sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
        innov_name_str = "$(prefixed_key)_innov_$(outcome_idx)"
        latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
    else
        sigma_name_str = "$(prefixed_key)_sigma"
        if is_multivariate
            innov_name_str = "$(prefixed_key)_innov_$(outcome_idx)"
            latent_name_str = "$(prefixed_key)_latent_$(outcome_idx)"
        else
            innov_name_str = "$(prefixed_key)_innov"
            latent_name_str = "$(prefixed_key)_latent"
        end
    end

    sigma_name = Symbol(sigma_name_str)
    innov_name = Symbol(innov_name_str)
    latent_name = Symbol(latent_name_str)

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || outcome_idx == 1))
        push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(m.sigma_prior)), :$(sigma_name))")
    end
    push!(priors_acc, "$(innov_name) ~ NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(innov_name))")
    priors_str = join(priors_acc, "\n")

    eta_update_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    index_var = "t_idx"

    update_str = """
    begin
        # RW2 state-space evolution for $(key_str)
        innovations = $(innov_name)
        latent_field_raw = Vector{T}(undef, $(n_latent))
        
        if $(n_latent) > 0; latent_field_raw[1] = innovations[1]; end
        if $(n_latent) > 1; latent_field_raw[2] = 2*latent_field_raw[1] + innovations[2]; end

        for t in 3:$(n_latent)
            latent_field_raw[t] = 2*latent_field_raw[t-1] - latent_field_raw[t-2] + innovations[t]
        end
        
        # Apply sum-to-zero constraint for identifiability
        if $(n_latent) > 0
            $(latent_name) = (latent_field_raw .- mean(latent_field_raw)) .* $(sigma_name_str)
            $(eta_update_target) .+= view($(latent_name), M.$(index_var))
        end
    end
    """
    
    return (priors=priors_str, update=update_str)
end





function _generate_manifold_code_fragments(m::ComposedManifold, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    # Purpose: Generates code fragments for composed manifolds, such as state-space models (`|>`).
    # Rationale: This version is updated to prepend the manifold's domain type to parameter names,
    #            ensuring they are correctly identified by the `get_optimal_sampler` grouping logic.
    # v1.2.8 (2026-07-17)
    
    if m.operator == :pipe
        # This block handles state-space evolution, specifically for spatially varying curves.
        state_manifold = m.components[1]
        dynamic_manifold = get(spec.params, :dynamic_manifold_obj, nothing)

        key_str = string(spec.key)
        domain_str = string(spec.domain)
        if isempty(prefix)
            prefixed_key = "$(domain_str)_$(key_str)"
        else
            prefixed_key = "$(prefix)_$(domain_str)_$(key_str)"
        end
        
        is_multivariate = arch == "multivariate"
        is_first_outcome = outcome_idx == 1
        is_shared = get(spec.params, :shared, false)

        if is_multivariate
            eta_update_target = "eta_latent[:, $(outcome_idx)]"
        else
            eta_update_target = "eta"
        end

        # Define outcome-specific or shared parameter names
        local sigma_name_str, rho_name_str, coeffs_raw_name_str
        if is_multivariate && !is_shared
            sigma_name_str = "$(prefixed_key)_sigma_$(outcome_idx)"
            rho_name_str = "$(prefixed_key)_rho_$(outcome_idx)"
            coeffs_raw_name_str = "$(prefixed_key)_coeffs_raw_$(outcome_idx)"
        else
            sigma_name_str = "$(prefixed_key)_sigma"
            rho_name_str = "$(prefixed_key)_rho"
            coeffs_raw_name_str = "$(prefixed_key)_coeffs_raw"
        end
        sigma_name = Symbol(sigma_name_str)
        rho_name = Symbol(rho_name_str)
        coeffs_raw_name = Symbol(coeffs_raw_name_str)

        state_spec = spec.hyper.state_spec
        n_spatial = size(state_spec.Q_template, 1)
        
        n_basis = dynamic_manifold.nbins
        basis_key = get(spec.params, :dynamic_basis_key, nothing)
        if isnothing(basis_key); error("Could not find basis matrix key for piped manifold $(key_str)."); end

        priors_acc = String[]
        # Generate priors only once for shared parameters, or for each outcome for non-shared
        if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
            if hasproperty(state_manifold, :sigma_prior); push!(priors_acc, "$(sigma_name) ~ NamedDist($(_distribution_to_string(state_manifold.sigma_prior)), :$(sigma_name))"); end
            if hasproperty(state_manifold, :rho_prior); push!(priors_acc, "$(rho_name) ~ NamedDist($(_distribution_to_string(state_manifold.rho_prior)), :$(rho_name))"); end
        end

        push!(priors_acc, "$(coeffs_raw_name) ~ NamedDist(MvNormal(zeros(T, $(n_spatial * n_basis)), I), :$(coeffs_raw_name))")
        priors_str = join(priors_acc, "\n        ")

        is_state_static = get(state_spec, :is_static, false)

        local cholesky_block
        if is_state_static
            cholesky_block = """
    # Using pre-computed Cholesky factor for static state manifold in '$(prefixed_key)'
    F_spatial = spec_registry["$(key_str)"].hyper.state_spec.cholesky_factor
            """
        else
            cholesky_block = """
    Q_spatial_template = spec_registry["$(key_str)"].hyper.state_spec.Q_template
    state_m_type = spec_registry["$(key_str)"].hyper.state_spec.model_type
    rho_val = $(hasproperty(state_manifold, :rho_prior) ? rho_name_str : "nothing")
    Q_spatial = recompose_precision(state_m_type, Q_spatial_template, 1.0; extra_param=rho_val)
    F_spatial = cholesky(Symmetric(Q_spatial + noise * I))
            """
        end

        update_str = """
    begin
        $(cholesky_block)
        
        coeffs_raw_matrix = reshape($(coeffs_raw_name), $(n_spatial), $(n_basis))
        spatial_coeffs = $(sigma_name_str) .* (F_spatial.U \\ coeffs_raw_matrix)
        B_smooth = M.basis_matrices[:$(basis_key)]
        $(eta_update_target) .+= sum(B_smooth .* spatial_coeffs[M.s_idx, :], dims=2)
    end
        """
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
    noise = get(M, :noise, 1e-6)
    new_manifolds = []
    # Define manifold types that do not have dynamic structure parameters like `rho`.
    static_manifold_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Spherical, Barycentric, TensorProductSmooth]

    for spec_in in M[:manifolds]
        current_spec = spec_in
        m_obj = current_spec.manifold_obj

        # Handle nested static manifolds within a ComposedManifold
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
    for (k, v) in kwargs; M[k] = v; end
    M[:calling_module] = get(kwargs, :calling_module, Main)
    M[:manifolds] = []
    M[:basis_matrices] = Dict{Symbol, Any}()
    return M
end

function _process_lhs!(M::Dict, outcome_specs::Vector{<:Dict})
    # Purpose: Processes the left-hand side of the formula to set up outcome variables and likelihood parameters.
    # Rationale: Separates outcome setup from the latent process setup.
    # Assumptions: `outcome_specs` is a vector of dictionaries from `decompose_bstm_formula`.
    # Inputs:
    #   - M: The model configuration dictionary.
    #   - outcome_specs: Parsed information about the outcome(s).
    # Outputs: None (mutates `M`).
    outcomes = [Symbol(spec[:var]) for spec in outcome_specs]
    likelihood_specs = [spec[:params] for spec in outcome_specs]
    M[:outcomes] = outcomes
    if length(outcomes) > 1
        M[:model_arch] = "multivariate"
        M[:outcomes_N] = length(outcomes)
        M[:y_obs] = Matrix(M[:data][!, M[:outcomes]])
        M[:likelihood_specs] = likelihood_specs
    else
        M[:model_arch] = get(M, :model_arch, "univariate")
        M[:outcomes_N] = 1
        M[:y_obs] = M[:data][!, M[:outcomes][1]]
        M[:likelihood_specs] = likelihood_specs
    end
    representative_params = !isempty(likelihood_specs) ? likelihood_specs[1] : Dict()
    _resolve_obs_param!(M, representative_params, M[:data], [:offsets, :log_offsets], :log_offset)
    _resolve_obs_param!(M, representative_params, M[:data], [:weights], :weights)
    _resolve_obs_param!(M, representative_params, M[:data], [:trials], :trials)
    _resolve_obs_param!(M, representative_params, M[:data], [:y_L], :y_L)
    _resolve_obs_param!(M, representative_params, M[:data], [:y_U], :y_U)
    _resolve_obs_param!(M, representative_params, M[:data], [:hurdle], :hurdle)
    if get(representative_params, :zi, false) == true
        M[:use_zi] = true
    end
end

function _resolve_obs_param!(opt_dict, params, data, param_keys, target_key)
    # Purpose: Finds an observation-level parameter (like offsets or weights) in the data and adds it to the config.
    # Rationale: This version is updated to correctly handle scalar numeric inputs for parameters like y_L, y_U,
    #            and hurdle, which was a source of user-reported errors. It also improves the warning message.
    # v1.2.4 (2026-07-17)
    # Assumptions: `data` is a DataFrame.
    # Inputs:
    #   - opt_dict: The configuration dictionary to update.
    #   - params: The likelihood parameters from the formula.
    #   - data: The input DataFrame.
    #   - param_keys: A list of possible keys for the parameter (e.g., [:offsets, :log_offsets]).
    #   - target_key: The key to use in `opt_dict`.
    # Outputs: None (mutates `opt_dict`).
    for key in param_keys
        if haskey(params, key)
            val = params[key]
            if val isa Symbol && hasproperty(data, val)
                opt_dict[target_key] = data[!, val]
                opt_dict[Symbol("user_provided_", target_key)] = true
            elseif val isa AbstractVector
                opt_dict[target_key] = val
                opt_dict[Symbol("user_provided_", target_key)] = true
            elseif val isa Number
                opt_dict[target_key] = val
                opt_dict[Symbol("user_provided_", target_key)] = true
            else
                @warn "Observation parameter '$val' for '$target_key' is not a valid column name, vector, or scalar. Ignoring."
            end
            return
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
    custom_priors = get(M, :fixed_effects_priors, Dict{Symbol, Any}())
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
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr
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

function process_eigen_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `eigen()` module for Bayesian PCA.
    # Rationale: Sets up the necessary indices for the Householder transformation parameterization.
    # v1.2.1 (2026-07-16)
    # Assumptions: `variables` are present in `mod_data`.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `mod_data`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
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
    mod_data[:params][:n_vars] = n_vars
    return true
end

function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `mixed()` module for random effects.
    # Rationale: Creates group indices and sets up parameters for random intercepts or slopes.
    # v1.2.1 (2026-07-16)
    # Assumptions: `data` is present in `opt_dict` and grouping variable exists.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `mod_data`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    data = opt_dict[:data]
    vars = mod_data[:variables]
    
    effect_var_str = ""
    group_var_str = ""

    if !isempty(vars) && vars[1] isa Expr && vars[1].head == :call && vars[1].args[1] == :|
        # Handle `mixed(effect | group)` syntax
        effect_expr = vars[1].args[2]
        group_expr = vars[1].args[3]
        effect_var_str = string(effect_expr)
        group_var_str = string(group_expr)
    elseif length(vars) >= 2
        # Handle `mixed(effect, group)` syntax
        effect_var_str = string(vars[1])
        group_var_str = string(vars[2])
    else
        @warn "The mixed() module requires syntax `mixed(effect | group)` or `mixed(effect, group)`. Skipping."
        return false
    end
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
    mod_data[:params][:n_cat] = length(levels)
    mod_data[:params][:lhs] = effect_var_str
    mod_data[:variables] = [group_var_str]
    return true
end

function process_nested_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `nested()` module for multi-fidelity modeling.
    # Rationale: Recursively calls `bstm_config` to create a complete configuration for the sub-model.
    # v1.2.1 (2026-07-16)
    # Assumptions: The specified `data_source` exists in `opt_dict`.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `opt_dict`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    if !haskey(opt_dict, :nested_manifolds); opt_dict[:nested_manifolds] = Dict{Symbol, Any}(); end
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)

    if data_source_sym != :data
        error("Multi-fidelity models with a different `data_source` are not yet supported. The nested model must use the same data as the main model.")
    end

    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return false
    end

    sub_data = opt_dict[data_source_sym]
    sub_config = _initialize_config(sub_data, merge(opt_dict, Dict(:calling_module => get(opt_dict, :calling_module, Main))))
    sub_metadata = decompose_bstm_formula(sub_formula)
    _process_lhs!(sub_config, sub_metadata.outcomes)
    sub_config[:manifolds] = []
    for (key, mod_data_nt) in sub_metadata.modules
        mod_type = mod_data_nt.module_type
        processor! = get(MODULE_PROCESSORS, mod_type, nothing)
        isnothing(processor!) && continue
        mod_data_dict = Dict(:type => mod_data_nt.module_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)
        processor!(sub_config, mod_data_dict, sub_config, hyperpriors)
        manifold_obj = resolve_technical_primitive(mod_data_dict, sub_config, hyperpriors, get(opt_dict, :prior_scheme, :pcpriors))
        manifold_spec_built = build_model(manifold_obj, sub_config)
        spec = (key=Symbol(key), domain=mod_data_dict[:type], var=join(mod_data_dict[:variables], "_"), manifold_obj=manifold_obj, params=mod_data_dict[:params], Q_template=manifold_spec_built.Q_template, scaling_factor=manifold_spec_built.scaling_factor)
        push!(sub_config[:manifolds], spec)
    end
    _process_fixed_effects!(sub_config, sub_metadata.fixed_effects, sub_metadata.has_intercept)
    _process_fixed_effects_priors!(sub_config)
    _precompute_static_manifolds!(sub_config) # Precompute for sub-model as well
    _finalize_config!(sub_config) # Finalize after all processing
    opt_dict[:nested_manifolds][var] = NamedTuple(sub_config) # Convert to NamedTuple for consistency
    return false # Do not create a manifold object for the nested module itself
end

function process_interact_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes interaction modules created by operators like `⊗` and `⊕`.
    # Rationale: Stores the components and operator type for later use by the model assembler.
    # v1.2.1 (2026-07-16)
    # Assumptions: `mod_data` contains `:operator` and `:components`.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `opt_dict`).
    # Returns: A boolean indicating whether a standard manifold object should be created.
    op = get(mod_data, :operator, get(mod_data[:params], :operator, nothing))
    components = get(mod_data, :components, get(mod_data[:params], :components, []))
    if isnothing(op) || isempty(components)
        @warn "Interaction module found with no operator or components. Skipping."
        return false
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
            var1 = string(var1_sym)
            var2 = string(var2_sym)

            # Process each fixed() component to register its priors/contrasts and add it to the main effects list.
            process_fixed_module!(opt_dict, Dict(:type => :fixed, :variables => [var1_sym], :params => c1.args), registries, hyperpriors)
            process_fixed_module!(opt_dict, Dict(:type => :fixed, :variables => [var2_sym], :params => c2.args), registries, hyperpriors)

            # Add the interaction term string, which will be processed by StatsModels.jl.
            # This makes `fixed(a) ⊗ fixed(b)` equivalent to `a * b`.
            interaction_term = "$(var1)&$(var2)"
            if !haskey(opt_dict, :fixed_effects); opt_dict[:fixed_effects] = String[]; end
            push!(opt_dict[:fixed_effects], interaction_term)
            
            return false # This interaction is fully handled by the fixed effects design matrix.

        elseif op == :composition && length(components) == 2
            base_node, modifier_node = components[1], components[2]
            is_nonstationary_variance = base_node.module_type == :spatial && modifier_node.module_type == :smooth

            if is_nonstationary_variance
                # Process the modifier (smoother) to generate its basis matrix.
                modifier_vars = get(modifier_node.args, :positional_args, [])
                if isempty(modifier_vars); @warn "The modifier component (smooth) of a composition operator is missing variables. Skipping."; return false; end
                
                smooth_mod_data = Dict(:type => :smooth, :variables => modifier_vars, :params => modifier_node.args)
                process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:manifolds])
                
                basis_key = Symbol(join(modifier_vars, "_"))
                mod_data[:params][:modifier_basis_key] = basis_key

                # Resolve the manifold objects to get their properties.
                scheme = get(opt_dict, :prior_scheme, :pcpriors)
                base_manifold_obj = resolve_technical_primitive(Dict(:type => base_node.module_type, :params => base_node.args, :variables => get(base_node.args, :positional_args, [])), opt_dict, hyperpriors, scheme)
                modifier_manifold_obj = resolve_technical_primitive(smooth_mod_data, opt_dict, hyperpriors, scheme)
                
                # Inject the custom code fragments for this specific composition type.
                key_str = string(mod_data[:key])
                nbins = modifier_manifold_obj.nbins
                mod_data[:params][:priors] = """
                    $(key_str)_sv_sigma_smoother ~ NamedDist(Exponential(1.0), :$(key_str)_sv_sigma_smoother)
                    $(key_str)_sv_coeffs_smoother ~ NamedDist(MvNormal(zeros(T, $(nbins)), I), :$(key_str)_sv_coeffs_smoother)
                    $(key_str)_sv_icar_raw ~ NamedDist(MvNormal(zeros(T, M.s_N), I), :$(key_str)_sv_icar_raw)
                """
                mod_data[:params][:update] = """
                    begin
                        local Q_smoother_template = build_structure_template(:rw2, $(nbins)).matrix; local F_smoother = cholesky(Symmetric(Q_smoother_template + noise * I)); local smoother_latent = $(key_str)_sv_sigma_smoother .* (F_smoother.U \\ $(key_str)_sv_coeffs_smoother); local log_sigma_field = M.basis_matrices[:$(basis_key)] * smoother_latent; local spatially_varying_sigma = exp.(log_sigma_field); local Q_icar_template = spec_registry["$(key_str)"].hyper.base_spec.Q_template; local F_icar = cholesky(Symmetric(Q_icar_template + noise * I)); local icar_latent = F_icar.U \\ $(key_str)_sv_icar_raw; local final_effect = view(icar_latent, M.s_idx) .* spatially_varying_sigma; eta .+= final_effect;
                    end
                """
                mod_data[:params][:base_manifold_obj] = base_manifold_obj
                mod_data[:params][:modifier_manifold_obj] = modifier_manifold_obj
                return true
            end

        elseif op == :pipe && length(components) == 2
            state_node, dynamic_node = components[1], components[2]
            is_spatially_varying_curve = state_node.module_type == :spatial && dynamic_node.module_type == :smooth
    
            if is_spatially_varying_curve
                # This handles `spatial(...) |> smooth(...)`, creating spatially varying curves.
                # The coefficients of the smoother's basis functions are modeled as spatial fields.
    
                # 1. Process the dynamic (smooth) part to generate its basis matrix.
                dynamic_vars = get(dynamic_node.args, :positional_args, [])
                if isempty(dynamic_vars)
                    @warn "The dynamic component (smooth) of a pipe operator is missing variables. Skipping."
                    return false
                end
                
                smooth_mod_data = Dict(:type => :smooth, :variables => dynamic_vars, :params => dynamic_node.args)
                process_smooth_module!(opt_dict, smooth_mod_data, opt_dict[:basis_matrices], opt_dict[:manifolds])
    
                # 2. Store information needed by the code generator in the composed manifold's parameters.
                basis_key = Symbol(join(dynamic_vars, "_"))
                mod_data[:params][:dynamic_basis_key] = basis_key
    
                # 3. Resolve the dynamic manifold object to get its properties (e.g., nbins).
                # This is needed by the code generator to know how many spatial fields to create.
                scheme = get(opt_dict, :prior_scheme, :pcpriors)
                dynamic_manifold_obj = resolve_technical_primitive(smooth_mod_data, opt_dict, hyperpriors, scheme)
                mod_data[:params][:dynamic_manifold_obj] = dynamic_manifold_obj
    
                # 4. The state manifold (spatial) will be resolved by the main loop's call to `build_model`.
                # The `build_model` for ComposedManifold will attach the state spec.
                
                return true # Proceed to create the ComposedManifold object.
            end



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
            
            return true # Allow the main loop to create the new smooth manifold
        end
    end

    if op == :kronecker_product
        if length(components) == 2
            # Assuming spatial ⊗ temporal for index generation
            s_idx = get(opt_dict, :s_idx, nothing)
            t_idx = get(opt_dict, :t_idx, nothing)
            s_N = get(opt_dict, :s_N, nothing)
            if !isnothing(s_idx) && !isnothing(t_idx) && !isnothing(s_N)
                st_idx = [(t - 1) * s_N + s for (s, t) in zip(s_idx, t_idx)]
                mod_data[:params][:indices] = st_idx
            else
                @warn "Could not compute Kronecker product indices for '$(mod_data[:variables])'. Ensure spatial and temporal manifolds are defined."
            end

            # Determine Knorr-Held interaction type to make `⊗` equivalent to `spacetime()`
            if length(components) == 2
                # Infer component types by inspecting the parsed module data.
                c1_type = get(components[1], :module_type, :unknown)
                c2_type = get(components[2], :module_type, :unknown)

                spatial_node = c1_type == :spatial ? components[1] : (c2_type == :spatial ? components[2] : nothing)
                temporal_node = c1_type == :temporal ? components[1] : (c2_type == :temporal ? components[2] : nothing)

                if !isnothing(spatial_node) && !isnothing(temporal_node)
                    spatial_model_str = string(get(spatial_node.args, :model, :iid))
                    temporal_model_str = string(get(temporal_node.args, :model, :iid))

                    has_structured_space = spatial_model_str != "iid"
                    has_structured_time = temporal_model_str != "iid"

                    if has_structured_space && has_structured_time; opt_dict[:model_st] = "IV";
                    elseif !has_structured_space && has_structured_time; opt_dict[:model_st] = "II";
                    elseif has_structured_space && !has_structured_time; opt_dict[:model_st] = "III";
                    else opt_dict[:model_st] = "I";
                    end
                end
                return false # Handled by global ST block, do not create a separate manifold.
            end
        else
            @warn "Kronecker product with more than 2 components is not yet supported in process_interact_module!."
        end
    end
    return true # Create a standard ComposedManifold for other operators like ⊕, |>, etc.
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
    # Purpose: Processes the `localadaptive()` module call.
    # Rationale: This module implements a localized spatial model where effects are centered
    #            around cluster-specific means. This processor handles the clustering of
    # v1.2.1 (2026-07-16)
    #            spatial units based on their centroids.
    # Assumptions: The model requires spatial coordinates to perform clustering.
    # Inputs: opt_dict, mod_data, registries, hyperpriors.
    # Outputs: Boolean indicating if a manifold should be created.

    # Step 1: Ensure base spatial information (W, centroids) is available.
    # This leverages the existing spatial processor.
    process_spatial_module!(opt_dict, mod_data, registries, hyperpriors)

    # Step 2: Check for centroids, which are required for clustering.
    if !haskey(opt_dict, :centroids)
        error("The `localadaptive()` model requires centroids for clustering, but they were not found. Ensure spatial coordinates are provided to infer areal units and their centroids.")
    end
    
    centroids = opt_dict[:centroids]
    params = mod_data[:params]
    
    # Step 3: Perform K-means clustering on the centroids.
    n_clusters = get(params, :n_clusters, 5)
    
    if size(centroids, 1) < n_clusters
        @warn "Number of spatial units ($(size(centroids, 1))) is less than the requested number of clusters ($n_clusters). Adjusting n_clusters to $(size(centroids, 1))."
        n_clusters = size(centroids, 1)
    end
    # Convert the vector of tuples to a 2xN matrix for Clustering.jl
    centroids_matrix = hcat(collect.(centroids)...)
    kmeans_result = kmeans(centroids_matrix, n_clusters; maxiter=200, display=:none)
    
    opt_dict[:cluster_assignments] = assignments(kmeans_result)
    opt_dict[:n_clusters] = nclusters(kmeans_result)
    return true # Proceed with manifold creation.
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
    :custom => process_custom_module!,
    :bcgn => process_bcgn_module!,
    :networkflow => process_networkflow_module!
)


# ==============================================================================
# SECTION 6: MODEL BUILDING AND ASSEMBLY
# ==============================================================================

macro bstm(formula, data, kwargs...)
    # Purpose: The main user-facing macro for defining a bstm model.
    # Rationale: Provides a clean, unquoted formula syntax similar to other statistical packages.
    # Assumptions: `formula` is a valid expression, `data` is a DataFrame.
    # Inputs:
    #   - formula: The model formula expression.
    #   - data: The DataFrame containing the data.
    #   - kwargs: Additional keyword arguments.
    # Outputs: A Turing.jl model object.
    formula_str = string(formula)
    data_esc = esc(data)
    kwargs_esc = [esc(kw) for kw in kwargs]
    return :(bstm($formula_str, $data_esc, @__MODULE__; $(kwargs_esc...)))
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
    return bstm_dynamic_model(options)
end

function bstm(config::NamedTuple)
    # Purpose: The final user-facing wrapper that dispatches to the dynamic model generator.
    # Rationale: Provides a single entry point from a configuration object.
    # v1.2.1 (2026-07-16)
    # Assumptions: `config` is a valid model configuration.
    # Inputs:
    #   - config: The model configuration NamedTuple from `bstm_config`.
    # Outputs: A Turing.jl model object.
    return bstm_dynamic_model(config)
end


function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # Purpose: Converts a parsed module dictionary into a concrete `Manifold` struct instance.
    # Rationale: This version provides more sensible defaults for core modules (`spatial`, `temporal`, `smooth`)
    #            to prevent configuration errors when a model is not explicitly specified.
    # v1.2.8 (2026-07-17)
    # Assumptions: `module_metadata` is a valid dictionary from the parser.
    # Inputs:
    #   - module_metadata, M, priors_dict, scheme.
    # Outputs: A concrete `Manifold` struct instance.
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    if m_type == :transform
        inner_module_data = module_metadata[:inner_module]
        inner_manifold_obj = resolve_technical_primitive(inner_module_data, M, priors_dict, scheme)
        transform_fn = get(m_params, :fn, :identity)
        return TransformedManifold(inner_manifold_obj, transform_fn)
    elseif m_type == :interact
        if haskey(m_params, :operator) && haskey(m_params, :components)
            op = m_params[:operator]
            components_data = m_params[:components]
            components_metadata = map(components_data) do c_node
                if haskey(c_node, :module_type)
                    Dict(:type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, []))
                else
                    Dict(:type => :interaction, :params => Dict(:operator => c_node.op, :components => c_node.children), :variables => [])
                end
            end
            resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
            return ComposedManifold(resolved_components, op)
        else
            interaction_vars = [Symbol(v) for v in module_metadata[:variables]]
            inner_model_data = get(m_params, :model, Dict{Symbol, Any}())
            inner_manifold_obj = resolve_technical_primitive(inner_model_data, M, priors_dict, scheme)
            return VaryingInteractionManifold(interaction_vars, inner_manifold_obj)
        end
    elseif m_type == :svc
        cov_sym = Symbol(module_metadata[:variables][1])
        model_str = string(get(m_params, :model, "iid"))
        inner_priors = resolve_hyperpriors(model_str, priors_dict, m_params, scheme, M[:calling_module])
        constructor_func = MANIFOLD_CONSTRUCTORS[Symbol(model_str)]
        inner_manifold = constructor_func(inner_priors, m_params)
        return SVCManifold(cov_sym, inner_manifold)
    else
        default_model = if m_type == :spatial
            haskey(M, :W) ? "bym2" : "iid"
        elseif m_type == :temporal
            "rw2"
        elseif m_type == :smooth
            "pspline"
        else "none" end
        model_val = get(m_params, :model, default_model)
        model_name = if model_val isa Symbol; String(model_val); else; string(model_val); end
        model_sym = Symbol(model_name)
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme, M[:calling_module])
        if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
            constructor_func = MANIFOLD_CONSTRUCTORS[model_sym]
            return constructor_func(resolved_priors, m_params)
        else
            supported_keys = join(sort(collect(string.(keys(MANIFOLD_CONSTRUCTORS)))), ", ")
            error("Unknown manifold model '$model_name' for module '$m_type'. Supported models are: $supported_keys")
        end
    end
end


function build_structure_template(type::Symbol, n::Int; scale=true, W=nothing)
    # Purpose: A factory for creating precision matrix templates for various GMRF and other structured models.
    # Rationale: Centralizes the construction of common precision matrices like ICAR, RW1, RW2, etc.
    # v1.2.1 (2026-07-16)
    # Assumptions: `n` is the number of units, `W` is provided for graph-based models.
    # Inputs:
    #   - type: The symbol for the model type (e.g., :icar, :rw2).
    #   - n: The number of units.
    #   - scale: Whether to apply geometric mean scaling for identifiability.
    #   - W: The adjacency matrix for graph-based models.
    # Outputs: A NamedTuple `(matrix, scaling_factor)`.
    Q = nothing
    sf = 1.0

    if type == :iid || type == :none || type == :identity || type == :harmonic || type == :rff
        return (matrix = sparse(I(n)), scaling_factor = 1.0)
    elseif type in [:icar, :besag, :bym2, :leroux, :localadaptive]
        if isnothing(W); error("Adjacency matrix W required for manifold :$type"); end
        D_sp = spdiagm(0 => vec(sum(W, dims=2)))
        Q_raw = D_sp - W
        if scale
            evals = eigvals(Matrix(Q_raw))
            nz_ev = filter(x -> x > 1e-6, evals)
            sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev)))
            Q = Q_raw ./ sf
        else
            Q = Q_raw
        end
    elseif type == :sar
        if isnothing(W); error("Adjacency matrix W required for manifold :$type"); end
        Q = W 
        sf = 1.0 
    elseif type == :rw1
        Q_raw = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        Q_raw[1,1] = Q_raw[n,n] = 1.0
        if scale
            evals = eigvals(Matrix(Q_raw)); nz_ev = filter(x -> x > 1e-6, evals); sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev))); Q = Q_raw ./ sf;
        else
            Q = Q_raw
        end
    elseif type == :ar1
        Q = spdiagm(-1 => fill(1.0, n-1), 1 => fill(1.0, n-1))
        sf = 1.0
    elseif type == :rw2
        D_rw2 = spdiagm(-2 => ones(n-2), -1 => -2*ones(n-1), 0 => ones(n), 1 => -2*ones(n-1), 2 => ones(n-2))
        Q_raw = D_rw2' * D_rw2
        if scale
            evals = eigvals(Matrix(Q_raw)); nz_ev = filter(x -> x > 1e-6, evals); sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev))); Q = Q_raw ./ sf;
        else
            Q = Q_raw
        end
    elseif type == :cyclic
        Q_raw = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1), n-1 => [-1.0], -(n-1) => [-1.0])
        Q_raw[1,n] = Q_raw[n,1] = -1.0
        if scale
            evals = eigvals(Matrix(Q_raw)); nz_ev = filter(x -> x > 1e-6, evals); sf = isempty(nz_ev) ? 1.0 : exp(mean(log.(nz_ev))); Q = Q_raw ./ sf;
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

function build_model(m::Manifold, data_inputs::Dict)
    # Purpose: A generic builder that dispatches to a template-based builder.
    # Rationale: Provides a default behavior for manifold models.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs:
    #   - m: A Manifold object.
    #   - data_inputs: The model configuration dictionary.
    # Outputs: A NamedTuple with the manifold's technical specification.
    return _build_from_template(m, data_inputs, :spatial) # Default to spatial
end

function build_model(m::Union{IID, ICAR, Besag, BYM2, Leroux, SAR, LocalAdaptive}, data_inputs::Dict)
    # Purpose: Builder for spatial GMRF models.
    # Rationale: Dispatches to the template builder with a `:spatial` context.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_from_template(m, data_inputs, :spatial)
end

function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict)
    # Purpose: Builder for temporal GMRF models.
    # Rationale: Dispatches to the template builder with a `:temporal` context.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_from_template(m, data_inputs, :temporal)
end

function build_model(m::Union{GP, FITC, RFF, SVGP, Warp, Nystrom, Harmonic, Hyperbolic, ExponentialDecay}, data_inputs::Dict)
    # Purpose: Builder for continuous, spectral, and other advanced manifolds.
    # Rationale: These models do not rely on pre-computed templates in the same way as GMRFs, so they use a pass-through builder.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict)
    # Purpose: Builder for spline-based smoothers.
    # Rationale: Determines the appropriate underlying GMRF template (RW1 or RW2) based on the spline type and penalty order.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    n = m.nbins
    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Cyclic, data_inputs::Dict)
    # Purpose: Builder for the `Cyclic` manifold.
    # Rationale: Creates a circulant precision matrix for smooth periodic effects.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Eigen, data_inputs::Dict)
    # Purpose: Builder for the `Eigen` manifold.
    # Rationale: Creates an identity template, as the structure is handled dynamically within the model.
    # v1.2.1 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    n = get(data_inputs, :s_N, 1)
    template = build_structure_template(:eigen, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::DynamicsManifold, data_inputs::Dict)
    # Purpose: Builder for the `DynamicsManifold`.
    # Rationale: Provides a Laplacian template for physics-based models like advection/diffusion.
    # v1.2.1 (2026-07-16)
    # Assumptions: `W` is available in `data_inputs`.
    # Inputs/Outputs: See `build_model`.
    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    template = build_structure_template(:besag, n; W=W)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::TensorProductSmooth, data_inputs::Dict)
    # Purpose: Builder for the `TensorProductSmooth` manifold.
    # Rationale: This manifold is typically constructed programmatically by an interaction processor.
    # v1.2.1 (2026-07-16)
    #            It passes the pre-computed Q_template from the manifold object to the assembler.
    # Assumptions: `m.Q_template` is correctly populated.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); if !(fn in [:Q_template]) ; hyper_dict[fn] = getfield(m, fn); end; end
    return (Q_template=m.Q_template, scaling_factor=1.0, model_type=:tensorproductsmooth, hyper=NamedTuple(hyper_dict))
end

function build_model(m::ComposedManifold, data_inputs::Dict)
    # Purpose: Builder for composed manifolds.
    # Rationale: Handles the `pipe` operator by recursively building the state manifold's spec and attaching it.
    # v1.2.1 (2026-07-16)
    # Assumptions: For `pipe`, there are exactly two components.
    # Inputs/Outputs: See `build_model`.
    if m.operator == :pipe
        if length(m.components) != 2
            error("Pipe operator requires exactly two components: state |> dynamic.")
        end
        state_manifold = m.components[1]
        dynamic_manifold = m.components[2]
        
        state_spec = build_model(state_manifold, data_inputs)
        
        hyper_dict = Dict{Symbol, Any}()
        for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
        hyper_dict[:state_spec] = state_spec
        
        return (Q_template=nothing, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))
    elseif m.operator == :composition
        # For spatial ∘ smooth, we need the spec of the base (spatial) model.
        base_manifold = get(m.components, 1, nothing)
        if isnothing(base_manifold); error("Composition manifold is missing its base component."); end
        base_spec = build_model(base_manifold, data_inputs)
        hyper_dict = Dict(:base_spec => base_spec)
        return (Q_template=base_spec.Q_template, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    else
        # For other operators like ⊗, no special template is needed at this stage.
        return _build_pass_through_model(m, data_inputs)
    end
end

function _build_from_template(m::ManifoldModel, data_inputs::Dict, domain::Symbol)
    # Purpose: A generic builder for manifolds that use a pre-defined template.
    # Rationale: Reduces code duplication for common GMRF models.
    # v1.2.1 (2026-07-16)
    # Assumptions: The manifold type has a corresponding entry in `build_structure_template`.
    # Inputs:
    #   - m: The ManifoldModel object.
    #   - data_inputs: The model configuration dictionary.
    #   - domain: The domain of the manifold (:spatial or :temporal).
    # Outputs: A NamedTuple with the manifold's technical specification.
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
        if !(fn in [:Q_template]) ; hyper_dict[fn] = getfield(m, fn); end
    end
    return (Q_template = template.matrix, scaling_factor = template.scaling_factor, model_type = model_sym, hyper = NamedTuple(hyper_dict))
end

function _build_pass_through_model(m::ManifoldModel, data_inputs::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # Purpose: A generic builder for manifolds that do not require complex template generation.
    # Rationale: Used for models where the structure is defined by parameters (e.g., splines) or handled dynamically.
    # v1.2.1 (2026-07-16)
    #            This version ensures a default identity Q_template is created for basis-like models.
    # Assumptions: None.
    # Inputs:
    #   - m, data_inputs, and optional overrides.
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
        pca_sd_priors = $(m.pca_sd_prior)
        pdef_sd_priors = $(m.pdef_sd_prior)

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

    if !is_multivariate || !get(M, :spectral_orientation, true)  # false will hardly make any sense in Bayesian models
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

function bstm_text_assembler(M::NamedTuple)
    # Purpose: Dynamically generates the Turing `@model` function as a string.
    # Rationale: This refactored version uses helper functions to generate code blocks,
    # v1.2.1 (2026-07-16)
    #            improving readability and maintainability.
    # Assumptions: `M` is a complete and valid model configuration from `bstm_config`.
    # Inputs:
    #   - M: The model configuration NamedTuple. 
    # Outputs: A tuple containing the model string, the parsed model expression, and the specification registry.
    arch = get(M, :model_arch, "univariate")
    local is_multivariate, model_func_name
    if arch == "multivariate" 
        is_multivariate = true
        model_func_name = :bstm_text_generated_multivariate
    else
        is_multivariate = false
        model_func_name = :bstm_text_generated_univariate
    end

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
            if !isempty(strip(frag.priors)); push!(priors_acc, frag.priors); end
            if !isempty(strip(frag.update)); push!(updates_acc, frag.update); end
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
        if isempty(strip(text)) return "" end
        indent_str = "    " ^ level
        return indent_str * replace(strip(text), "\n" => "\n" * indent_str)
    end

    # Generate all code blocks
    likelihood_section = _generate_likelihood_section(M, is_multivariate)
    intercept_block = _generate_intercept_block(M, is_multivariate, eta_name)
    offset_block = _generate_offset_block(M, is_multivariate, eta_name)
    fixed_effects_block = _generate_fixed_effects_block(M, is_multivariate, eta_name) 
    st_interaction_block = _generate_st_interaction_block(M, main_spatial_spec, main_temporal_spec, is_multivariate)
    householder_priors, householder_update = _generate_householder_reflection_block(M, is_multivariate, eta_name)
    final_likelihood = _generate_final_likelihood_block(M, is_multivariate)
    nested_priors, nested_updates, nested_likelihoods = _generate_nested_model_block(M, is_multivariate, eta_name)

    # Indent all code blocks before interpolation
    priors_code = join([p for p in priors_acc if !isempty(strip(p))], "\n\n")
    updates_code = join([u for u in updates_acc if !isempty(strip(u))], "\n\n")
 
    model_string = """

@model function $(model_func_name)(M, spec_registry, ::Type{T}=Float64) where {T}
    noise = get(M, :noise, 1e-6)
    N = M.y_N
    K = $(outcomes_N)

$(_indent_block(likelihood_section))
$(_indent_block(priors_code))
$(_indent_block(householder_priors))
$(_indent_block(nested_priors))

    # --- Linear Predictor ---
    $(eta_name) = $(eta_init)

$(_indent_block(intercept_block))
$(_indent_block(offset_block))
$(_indent_block(fixed_effects_block))

    # --- Manifold Effects ---
$(_indent_block(updates_code))

$(_indent_block(householder_update))
$(_indent_block(nested_updates))
$(_indent_block(st_interaction_block))

    # --- Likelihood ---
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


# ==============================================================================
# SECTION 8: LIKELIHOOD IMPLEMENTATION
# ==============================================================================

function bstm_Likelihood(family_input::Union{String, Symbol}, y_obs;
    zi_state=nothing, censoring_state=nothing, weight=1.0,
    phi_zi=-Inf, r_nb=1.0, sigma_y=1.0, trial=1, 
    y_L=-Inf, y_U=Inf, hurdle=-Inf, extra_params=nothing
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
    if phi_zi > -Inf && h_val > -Inf
        @warn "Both zero-inflation (zi) and a hurdle model were specified. These are typically mutually exclusive. Prioritizing zero-inflation and ignoring the hurdle."
        h_val = -Inf
    end

    zi_trait = phi_zi > -Inf ? ZeroInflated() : NonZeroInflated()

    yL_val = isnothing(y_L) ? -Inf : y_L
    yU_val = isnothing(y_U) ? Inf : y_U

    censor_trait = if !isfinite(yL_val) && !isfinite(yU_val); Uncensored()
        elseif isfinite(yL_val) && !isfinite(yU_val); RightCensored()
        elseif !isfinite(yL_val) && isfinite(yU_val); LeftCensored()
        else IntervalCensored() end

    y_vec = y_obs isa AbstractVector ? y_obs : [y_obs]
    
    return bstm_Likelihood(f_trait, y_vec, zi_trait, censor_trait, weight, phi_zi, r_nb, sigma_y, trial, yL_val, yU_val, h_val, extra_params)
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

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end
function is_discrete_family(::AbstractBSTM_Family)
    return false
end

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
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
    
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        if y == 0.0
            if is_discrete_family(fam)
                logp_base_zero = logpdf(dist, 0.0)
                return logsumexp(log_phi, log_one_minus_phi + logp_base_zero)
            else
                return log_phi
            end
        else
            return log_one_minus_phi + logpdf(dist, y)
        end
    elseif d.hurdle > -Inf
        if y <= d.hurdle
            return -Inf 
        else
            return logpdf(dist, y) - logccdf(dist, d.hurdle)
        end
    else
        return logpdf(dist, y)
    end
end

function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zi::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for a left-censored observation.
    # Rationale: Correctly calculates the cumulative probability for standard, ZI, and hurdle models.
    # v1.2.1 (2026-07-16)
    # Assumptions: `d.y_U` is finite.
    # Inputs/Outputs: See `bstm_kernel` for uncensored.
    dist = get_dist_ref(fam, d, eta, sig)
    upper_bound = d.y_U[1]

    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        lp_base = logcdf(dist, upper_bound)
        if upper_bound >= 0.0
            return logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            return log_one_minus_phi + lp_base
        end
    elseif d.hurdle > -Inf
        if upper_bound <= d.hurdle
            return -Inf
        end
        log_prob_in_interval = _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, d.hurdle))
        log_normalizer = logccdf(dist, d.hurdle)
        return log_prob_in_interval - log_normalizer
    else
        return logcdf(dist, upper_bound)
    end
end

function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zi::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for a right-censored observation.
    # Rationale: Correctly calculates the complementary cumulative probability for all model types.
    # v1.2.1 (2026-07-16)
    # Assumptions: `d.y_L` is finite.
    # Inputs/Outputs: See `bstm_kernel` for uncensored.
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = d.y_L[1]
    adj_L = is_discrete_family(fam) ? lower_bound - 1.0 : lower_bound

    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        log_p_le_L = if lower_bound < 0.0
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return log1mexp(log_p_le_L)

    elseif d.hurdle > -Inf
        effective_lower_bound = max(lower_bound, d.hurdle)
        adj_eff_L = is_discrete_family(fam) ? effective_lower_bound - 1.0 : effective_lower_bound
        log_numerator = logccdf(dist, adj_eff_L)
        log_denominator = logccdf(dist, d.hurdle)
        return log_numerator - log_denominator
    else
        return logccdf(dist, adj_L)
    end
end

function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    # Purpose: Computes the log-probability for an interval-censored observation.
    # Rationale: Calculates the probability mass within the interval [y_L, y_U].
    # v1.2.1 (2026-07-16)
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = d.y_L[1]
    upper_bound = d.y_U[1]
    adj_L = is_discrete_family(fam) ? lower_bound - 1.0 : lower_bound

    if zi isa ZeroInflated
        log_p_le_U = if upper_bound < 0.0
            log(1-d.phi_zi) + logcdf(dist, upper_bound)
        else
            logsumexp(log(d.phi_zi), log(1-d.phi_zi) + logcdf(dist, upper_bound))
        end
        log_p_le_L = if lower_bound < 0.0
            log(1-d.phi_zi) + logcdf(dist, lower_bound)
        else
            logsumexp(log(d.phi_zi), log(1-d.phi_zi) + logcdf(dist, lower_bound))
        end
        return _stable_logsubexp(log_p_le_U, log_p_le_L)

    elseif d.hurdle > -Inf
        effective_lower_bound = max(lower_bound, d.hurdle)
        if upper_bound <= effective_lower_bound
            return -Inf
        end
        adj_eff_L = is_discrete_family(fam) ? effective_lower_bound - 1.0 : effective_lower_bound
        log_numerator = _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, adj_eff_L))
        log_denominator = logccdf(dist, d.hurdle)
        return log_numerator - log_denominator
    else
        return _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, adj_L))
    end
end


# ==============================================================================
# SECTION 9: UTILITY AND HELPER FUNCTIONS
# ==============================================================================

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
    # v1.2.1 (2026-07-16)
    # Assumptions: `param_name` is one of the recognized types (:sigma, :rho, etc.).
    # Inputs:
    #   - param_name: The base name of the parameter.
    #   - constraint: A tuple `(U, α)` or `(U, α, direction)`.
    # Outputs: A `Distribution` object.
    direction = :upper
    if length(constraint) == 2; U, α = constraint; elseif length(constraint) == 3; U, α, direction = constraint; else; error("PC prior constraint must be a tuple of (U, α) or (U, α, direction)."); end
    
    base_param_name = Symbol(replace(string(param_name), r"_prior$" => ""))

    if base_param_name == :sigma || endswith(string(base_param_name), "_sigma"); direction != :upper && error("PC prior for sigma only supports upper tail constraints."); λ = -log(α) / U; return Exponential(λ);
    elseif base_param_name == :rho || endswith(string(base_param_name), "_rho"); direction != :upper && error("PC prior for 'rho' only supports upper tail constraints."); λ = log(α) / log(1.0 - U); return Exponential(λ);
    elseif base_param_name == :lengthscale || endswith(string(base_param_name), "_lengthscale"); direction != :lower && error("PC prior for 'lengthscale' only supports lower tail constraints."); λ = -U * log(α); return Exponential(λ);
    elseif base_param_name == :kappa || endswith(string(base_param_name), "_kappa"); direction != :upper && error("PC prior for kappa only supports upper tail constraints."); λ = -log(α) / U; return Exponential(λ);
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
    final_rhs_string = strip(formula_rhs)

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
    println("\n--- Prior Sample Check ---")
    try
        prior_sample_chain = sample(m, Prior(), 1)
        println("Successfully drew 1 sample from the model's prior.")
        display(prior_sample_chain)
    catch e
        println("ERROR: Failed to draw a sample from the prior.")
        println("  Reason: ", e)
    end
    println("\n--- End Model Summary ---")
    return nothing
end

function _generate_model_pseudocode(m::DynamicPPL.Model)
    # Purpose: Reconstructs a pseudo-code representation of the Turing model definition.
    # Rationale: For inspection and clarity, as the actual model is built dynamically from a string.
    # v1.2.1 (2026-07-16)
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

function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{AbstractVector, Nothing} = nothing, kwargs...)
    # BSTM Smooth Basis Factory
    # v1.2.1 (2026-07-16)
    # Synopsis: A factory function that generates a 1D basis matrix for various smoothers. 
    # Rationale for v1.2.0:
    #     - Corrected knot generation for 'invdist' and 'kriging' to use a regular grid,
    #       ensuring stability for skewed data distributions.

    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9

    # Certain basis types require a regular grid for knot placement for stability.
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    # Knot generation logic
    local knots
    if knot_method == :custom && !isnothing(custom_knots)
        knots = custom_knots
    elseif knot_method == :range || use_regular_grid
        knots = collect(range(v_min, stop=v_max, length=nbins))
    else # :quantile or any other default
        knots = quantile(vals, range(0, 1, length=nbins))
    end

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        h = (v_max - v_min) / (nbins > 1 ? (nbins - 1) : 1)
        h = h > 0 ? h : 1.0

        for m in 1:nbins
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B[mask, m] .= 1.0 .- dist[mask]
        end

    elseif type == "tps"
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

function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; W=nothing, knot_method::Symbol = :quantile, custom_knots::Union{Tuple{AbstractVector, AbstractVector}, Nothing} = nothing, kwargs...)
    # BSTM Smooth Basis Factory
    # v1.2.1 (2026-07-16)
    # Synopsis: A factory function that generates a 2D basis matrix for various smoothers. 
    # Rationale for v1.2.0:
    #     - Corrected knot generation for 'tps', 'spherical', 'invdist', and 'kriging' to use a regular grid.

    n_obs = size(coords, 1)
    
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

    local kx, ky
    n_marginal = Int(floor(sqrt(nbins)))

    # Certain basis types require a regular grid for knot placement for stability.
    use_regular_grid = type in ["invdist", "kriging", "tps", "spherical"]

    if knot_method == :custom && !isnothing(custom_knots)
        if length(custom_knots) != 2 || length(custom_knots[1]) != n_marginal || length(custom_knots[2]) != n_marginal
            @warn "Custom knots for 2D smoother must be a Tuple{AbstractVector, AbstractVector} with each vector of length n_marginal ($n_marginal). Falling back to :quantile method."
            kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
            ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
        else
            kx = custom_knots[1]
            ky = custom_knots[2]
        end
    elseif knot_method == :quantile
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
    else # :range or if regular grid is required
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
    end

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)
        
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
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
        centers = [(x, y) for x in kx, y in ky][:]
        range_r = get(kwargs, :range, mean(c_std))
        for m in 1:min(nbins, length(centers))
            dx = coords[:, 1] .- centers[m][1]
            dy = coords[:, 2] .- centers[m][2]
            h = sqrt.(dx.^2 .+ dy.^2) ./ range_r
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # The 'moran' basis type was removed from smoothers. It is a spatial basis.
    elseif type == "moran"
        error("The 'moran' basis is not a valid smoother for covariates. Use a spatial manifold like `spatial(..., model=eigen)` instead.")

    elseif type == "invdist"
        B = zeros(Float64, n_obs, nbins)
        centers = [(x, y) for x in kx, y in ky][:]
        for m in 1:min(nbins, length(centers))
            dist_sq = (coords[:, 1] .- centers[m][1]).^2 .+ (coords[:, 2] .- centers[m][2]).^2
            B[:, m] .= 1.0 ./ (dist_sq .+ 1e-6)
        end

    elseif type == "kriging"
        B = zeros(Float64, n_obs, nbins)
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

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
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
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
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
        for m in 1:min(m_total, length(centers))
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

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

    if type in ["pspline", "bspline", "smooth", "barycentric", "linear"]
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
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
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

function evaluate_kernel_matrix(dist_sq::AbstractMatrix, param_val::Real, ls::Real, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    # v1.2.1 (2026-07-16)
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

"""
    recompose_precision(...)

Purpose: An internal factory function that constructs a final precision matrix from a template, 
         a scale parameter (`param_val`), and other manifold-specific parameters (e.g., correlation `rho`). 
         This function is central to defining GMRF priors within the model. v1.2.1 (2026-07-16)
Inputs: m_type, template_s, param_val, and other optional parameters.
Outputs: A symmetric precision matrix.
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

    if m_type == :none || m_type == :fixed
        return Symmetric(sparse(I(n_s)))
    end

    if m_type == :besag || m_type == :icar || m_type == :cyclic
        return Symmetric(template_s)
    end

    if m_type == :bym2
        rho = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(rho .* template_s + (1.0 - rho) .* sparse(I(n_s)))
    end

    if m_type == :ar1 || m_type == :rw1 || m_type == :rw2
        # This block is intentionally deprecated. These models should be implemented
        # via a state-space evolution for numerical stability, which is handled by a
        # specialized code generator. Calling this indicates a dispatch error.
        error("recompose_precision should not be called for $(m_type) models. Use the state-space implementation.")
    end

    if m_type == :leroux
        lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(lambda_val .* template_s + (1.0 - lambda_val) .* sparse(I(n_s)))
    end

    if m_type == :I || m_type == :II || m_type == :III || m_type == :IV
        if isnothing(template_t)
            Q_full = template_s
        else
            Q_full = kron(template_t, template_s)
        end
        return Symmetric(Q_full)
    end

    if m_type == :networkflow
        rho_net = isnothing(extra_param) ? 0.8 : extra_param
        W_net = !isnothing(directed_adj) ? directed_adj : template_s
        
        L_op = if flow_direction == :upstream
            I(n_s) - rho_net .* W_net'
        elseif flow_direction == :downstream
            I(n_s) - rho_net .* W_net
        else # :bidirectional
            W_symm = sign.(W_net + W_net')
            I(n_s) - rho_net .* W_symm
        end
        return Symmetric(L_op' * L_op)
    end

    if m_type == :sar || m_type == :dag || m_type == :proper_car
        rho_p = isnothing(extra_param) ? 0.8 : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(L_op' * L_op)
    end

    if m_type == :gp
        ls = isnothing(extra_param) ? 1.0 : extra_param
        K = (param_val^2) .* exp.(-(Matrix(template_s).^2) ./ (2 * ls^2 + noise))
        return inv(Symmetric(K))
    end

    if m_type == :spde
        kappa = isnothing(extra_param) ? 1.0 : extra_param
        L_spde = (kappa^2 .* I(n_s) + template_s)
        return Symmetric(L_spde' * L_spde)
    end

    if m_type == :rff || m_type == :fft || m_type == :bspline || m_type == :pspline || m_type == :tps
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
    if get(M, :use_zi, false)
        zi_prior_block = "lik_phi_zi ~ NamedDist(Beta(1,1), :lik_phi_zi)"
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

    corr_block = is_multivariate ? "L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)" : ""

    return """
    $(corr_block)
    $(sigma_block)
    $(zi_prior_block)
    """
end

function _generate_final_likelihood_block(M::NamedTuple, is_multivariate::Bool)
    # v1.2.3 (2026-07-17)
    # Rationale: This version makes the inclusion of `sigma_y` in the `bstm_Likelihood`
    #            call conditional. `sigma_y` is only included for likelihood families that require it
    #            (e.g., Gaussian), addressing user feedback about passing unnecessary parameters.
    
    families = [string(get(spec, :family, "gaussian")) for spec in M.likelihood_specs]
    any_needs_sigma = any(f -> f in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"], families)

    if is_multivariate
        kwargs_parts = String[]
        if any_needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma[k]"); end
        if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i, k])"); end
        if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[i, k]"); end
        if get(M, :user_provided_y_L, false); push!(kwargs_parts, "y_L=M.y_L[i, k]"); end
        if get(M, :user_provided_y_U, false); push!(kwargs_parts, "y_U=M.y_U[i, k]"); end
        if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=M.hurdle[i, k]"); end
        if get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end
        kwargs_str = join(kwargs_parts, ", ")

        return """
        eta = eta_latent * L_corr.L
        for k in 1:K
            family_k = M.likelihood_specs[k][:family]
            
            for i in 1:N
                d_lik = bstm_Likelihood(family_k, T(M.y_obs[i, k]); $(kwargs_str))
                Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
            end
        end
        """
    else # Univariate
        kwargs_parts = String[]
        if any_needs_sigma; push!(kwargs_parts, "sigma_y=y_sigma"); end
        if get(M, :user_provided_trials, false); push!(kwargs_parts, "trial=Int(M.trials[i])"); end
        if get(M, :user_provided_weights, false); push!(kwargs_parts, "weight=M.weights[i]"); end
        if get(M, :user_provided_y_L, false); push!(kwargs_parts, "y_L=M.y_L[i]"); end
        if get(M, :user_provided_y_U, false); push!(kwargs_parts, "y_U=M.y_U[i]"); end
        if get(M, :user_provided_hurdle, false); push!(kwargs_parts, "hurdle=M.hurdle[i]"); end
        if get(M, :use_zi, false); push!(kwargs_parts, "phi_zi=lik_phi_zi"); end
        kwargs_str = join(kwargs_parts, ", ")

        return """
        family = M.likelihood_specs[1][:family] # All outcomes share the same family in univariate case
        
        for i in 1:N
            d_lik = bstm_Likelihood(family, T(M.y_obs[i]); $(kwargs_str))
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
        end
        """
    end
end




function _generate_intercept_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.2.1 (2026-07-16)
    if !get(M, :add_intercept, false) return "" end
    
    intercept_prior_obj = get(M, :intercept_prior, Normal(0,5))
    local dist_str, update
    if is_multivariate
        dist_str = "filldist($(_distribution_to_string(intercept_prior_obj)), K)"
        update = "for k in 1:K; $(eta_name)[:, k] .+= intercept[k]; end"
    else
        dist_str = _distribution_to_string(intercept_prior_obj)
        update = "$(eta_name) .+= intercept"
    end
    
    return """
    intercept ~ NamedDist($(dist_str), :intercept)
    $(update)
    """
end

function _generate_offset_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.2.1 (2026-07-16)
    if !haskey(M, :log_offset) return "" end
    if is_multivariate
        return "$(eta_name) .+= M.log_offset"
    else
        return "$(eta_name) .+= M.log_offset"
    end
end



function _generate_fixed_effects_block(M::NamedTuple, is_multivariate::Bool, eta_name::String)
    # v1.2.8 (2026-07-17)
    # Rationale: This version generalizes the check for identical priors beyond just the Normal
    #            distribution. It now uses `filldist` for any set of identical priors, falling
    #            back to `Product` only when priors truly differ. This improves efficiency for
    #            cases where multiple fixed effects share the same non-Normal prior.
    if get(M, :Xfixed_N, 0) == 0 return "" end

    priors_vec = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
    
    all_same = false
    first_prior = nothing
    if !isempty(priors_vec)
        first_prior = priors_vec[1]
        # Check if all elements are identical to the first one.
        # This works for distribution objects as they are value types.
        if all(p -> p == first_prior, priors_vec)
            all_same = true
        end
    end

    if is_multivariate
        K = M.outcomes_N
        if all_same && !isnothing(first_prior)
            prior_str = _distribution_to_string(first_prior)
            return """
            Xfixed_beta_flat ~ NamedDist(filldist($(prior_str), M.Xfixed_N * K), :Xfixed_beta)
            $(eta_name) .+= M.Xfixed * reshape(Xfixed_beta_flat, M.Xfixed_N, K)
            """
        else
            full_priors_list = vcat([priors_vec for _ in 1:K]...)
            priors_str_list = [_distribution_to_string(p) for p in full_priors_list]
            priors_product_str = "Product([$(join(priors_str_list, ", "))])"
            return """
            Xfixed_beta_flat ~ NamedDist($(priors_product_str), :Xfixed_beta)
            $(eta_name) .+= M.Xfixed * reshape(Xfixed_beta_flat, M.Xfixed_N, K)
            """
        end
    else # Univariate case
        if all_same && !isnothing(first_prior)
            prior_str = _distribution_to_string(first_prior)
            return """
            Xfixed_beta ~ NamedDist(filldist($(prior_str), M.Xfixed_N), :Xfixed_beta)
            $(eta_name) .+= M.Xfixed * Xfixed_beta
            """
        else
            priors_str_list = [_distribution_to_string(p) for p in priors_vec]
            priors_product_str = "Product([$(join(priors_str_list, ", "))])"
            return """
            Xfixed_beta ~ NamedDist($(priors_product_str), :Xfixed_beta)
            $(eta_name) .+= M.Xfixed * Xfixed_beta
            """
        end
    end
end
 

function _generate_st_interaction_block(M::NamedTuple, main_spatial_spec, main_temporal_spec, is_multivariate::Bool)
    # v1.2.8 (2026-07-17)
    # Rationale: This version uses more descriptive parameter names (`st_interaction_...`)
    #            to ensure they are correctly identified and grouped by the `get_optimal_sampler` logic.

    model_st = get(M, :model_st, "none")
    if model_st == "none"
        return ""
    end
 
    if isnothing(main_spatial_spec) || isnothing(main_temporal_spec)
        @warn "Spacetime interaction specified, but could not find both a primary spatial and temporal manifold. Interaction term will be ignored."
        return ""
    end
 
    s_spec = main_spatial_spec
    t_spec = main_temporal_spec
    s_is_static = get(s_spec, :is_static, false)
    t_is_static = get(t_spec, :is_static, false)
    
    t_rho_name = if t_spec.manifold_obj isa AR1
        is_shared = get(t_spec.params, :shared, false) 
        local t_key_str
        if is_multivariate && !is_shared
            t_key_str = "$(t_spec.key)_rho_1"
        else
            t_key_str = "$(t_spec.key)_rho"
        end
        Symbol(t_key_str)
    else
        nothing
    end

    local transform_code
    if model_st == "I"
        transform_code = "st_inter_k = st_innov_matrix_k .* st_sigma_k"
    elseif model_st == "II"
        transform_code = "st_inter_k = transpose(C_t.U \\ transpose(st_innov_matrix_k)) .* st_sigma_k"
    elseif model_st == "III"
        transform_code = "st_inter_k = (C_s.U \\ st_innov_matrix_k) .* st_sigma_k"
    elseif model_st == "IV"
        if isnothing(t_rho_name)
            @warn "Type IV interaction specified, but main temporal effect is not AR1. Defaulting to Type III."
            transform_code = "st_inter_k = (C_s.U \\ st_innov_matrix_k) .* st_sigma_k"
        else
            transform_code = """
            t_rho_val = $(t_rho_name)
            st_inter_k[:, 1] = (C_s.U \\ st_innov_matrix_k[:, 1]) ./ sqrt(1.0 - t_rho_val^2 + noise)
            for t in 2:M.t_N; st_inter_k[:, t] = t_rho_val .* st_inter_k[:, t-1] .+ (C_s.U \\ st_innov_matrix_k[:, t]); end
            st_inter_k .*= st_sigma_k
            """
        end
    else
        return ""
    end
 
    if is_multivariate
        K = M.outcomes_N
        return """
    # Multivariate Spatiotemporal Interaction (Type $(model_st))
    st_interaction_sigma ~ NamedDist(filldist(Exponential(0.5), $(K)), :st_interaction_sigma)
    st_interaction_raw_flat ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N * $(K)), I), :st_interaction_raw_flat)

    let 
        st_innov_tensor = reshape(st_interaction_raw_flat, M.s_N, M.t_N, $(K))
        
        C_s = $(s_is_static ? "spec_registry[\"$(s_spec.key)\"].cholesky_factor" : "cholesky(Symmetric(spec_registry[\"$(s_spec.key)\"].Q_template + noise * I))")
        C_t = $(t_is_static ? "spec_registry[\"$(t_spec.key)\"].cholesky_factor" : "cholesky(Symmetric(spec_registry[\"$(t_spec.key)\"].Q_template + noise * I))")

        for k in 1:$(K)
            st_inter_k = zeros(T, M.s_N, M.t_N)
            st_innov_matrix_k = view(st_innov_tensor, :, :, k)
            st_sigma_k = st_interaction_sigma[k]

            $(transform_code)

            for i in 1:N; eta_latent[i, k] += st_inter_k[M.s_idx[i], M.t_idx[i]]; end
        end
    end
"""
    else # Univariate case
        univariate_transform_code = replace(transform_code, "st_sigma_k" => "st_interaction_sigma")
        univariate_transform_code = replace(univariate_transform_code, "st_innov_matrix_k" => "st_innov_matrix")
        univariate_transform_code = replace(univariate_transform_code, "st_inter_k" => "st_inter")

        return """
    # Univariate Spatiotemporal Interaction (Type $(model_st))
    st_interaction_sigma ~ NamedDist(Exponential(0.5), :st_interaction_sigma)
    st_interaction_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_interaction_raw)
 
    let 
        st_inter = zeros(T, M.s_N, M.t_N)
        st_innov_matrix = reshape(st_interaction_raw, M.s_N, M.t_N)
        
        C_s = $(s_is_static ? "spec_registry[\"$(s_spec.key)\"].cholesky_factor" : "cholesky(Symmetric(spec_registry[\"$(s_spec.key)\"].Q_template + noise * I))")
        C_t = $(t_is_static ? "spec_registry[\"$(t_spec.key)\"].cholesky_factor" : "cholesky(Symmetric(spec_registry[\"$(t_spec.key)\"].Q_template + noise * I))")

        $(univariate_transform_code)

        for i in 1:N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end
    end
"""
    end
end



function _generate_nested_model_block(M::NamedTuple, is_multivariate::Bool, main_eta_name::String)
    # Purpose: Generates the complete code block for all nested sub-models.
    # Rationale: This version generates more parsimonious code for sub-models by using
    #            Product distributions for fixed effects and conditional keyword arguments
    #            for the likelihood call, mirroring the optimizations in the main model assembler.
    # v1.2.7 (2026-07-17)
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
            all_same_normal = !isempty(priors_vec) && all(p -> p isa Normal && p.μ == priors_vec[1].μ && p.σ == priors_vec[1].σ, priors_vec)

            local prior_block
            if all_same_normal
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
            push!(all_nested_priors, "$(prefix)_intercept ~ NamedDist($(_distribution_to_string(prior_obj)), :$(prefix)_intercept)")
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
            $(sub_eta_name) = zeros(T, N)
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
            # Prior for a vector of linking parameters, one for each outcome.
            push!(all_nested_priors, "$(rho_name) ~ NamedDist(filldist(Normal(1.0, 0.5), $(K)), :$(rho_name))")
            
            # Apply the outcome-specific scaled sub-model effect to each latent predictor.
            push!(all_nested_updates, "for k in 1:$(K); $(main_eta_name)[:, k] .+= $(rho_name)[k] .* $(sub_eta_name); end")
        else
            # Univariate case: a single linking parameter.
            push!(all_nested_priors, "$(rho_name) ~ NamedDist(Normal(1.0, 0.5), :$(rho_name))")
            push!(all_nested_updates, "$(main_eta_name) .+= $(rho_name) .* $(sub_eta_name)")
        end

        # --- Generate Likelihood for Sub-Model ---
        kwargs_parts = String[]
        if sub_needs_sigma; push!(kwargs_parts, "sigma_y=$(sub_sigma_name)"); end
        
        param_keys = [:trials, :weights, :y_L, :y_U, :hurdle]
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
            for i in 1:N
                d_lik_sub = bstm_Likelihood(sub_family, [T(sub_M.y_obs[i])]; $(kwargs_str))
                Turing.@addlogprob! Distributions.logpdf(d_lik_sub, $(sub_eta_name)[i])
            end
        end
        """
        push!(all_nested_likelihoods, lik_loop)
    end

    return join(all_nested_priors, "\n\n"), join(all_nested_updates, "\n\n"), join(all_nested_likelihoods, "\n\n")
end

# to delete
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
