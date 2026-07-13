 
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

struct Eigen <: ManifoldModel; n_factors::Int; pca_sd_prior::UnivariateDistribution; pdef_sd_prior::UnivariateDistribution; ltri_indices::Vector{Int}; end
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
    :eigen => (p, params) -> Eigen(get(params, :n_factors, 1), p.pca_sd_prior, p.pdef_sd_prior, get(params, :ltri_indices, Int[])),
    :moran => (p, params) -> Moran(p.sigma_prior),
    :spherical => (p, params) -> Spherical(p.sigma_prior, p.range_prior),
    :barycentric => (p, params) -> Barycentric(p.sigma_prior),
    :kriging => (p, params) -> Kriging(p.lengthscale_prior, p.sigma_prior, string(get(params, :kernel, "se"))),
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

⊗(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
⊕(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)

otimes(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
oplus(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)


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

function _parse_single_manifold_term(term_str::String)
    # Purpose: Parses a single module call string like "spatial(s_idx, model=bym2)".
    # Rationale: Extracts the module name and its arguments.
    # Assumptions: The input is a single, well-formed module call.
    # Inputs:
    #   - term_str: The module call string.
    # Outputs: A NamedTuple `(module_type, args)`.
    term_str = strip(term_str)
    m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", term_str)
    
    if m === nothing
        error("Internal Parser Error: Expected a module call (e.g., 'name(...)'), but got non-module term '$term_str'.")
    end

    module_name = Symbol(m.captures[1])
    args_str = String(m.captures[2])
    args_dict = _parse_arguments_string(args_str)

    return (module_type = module_name, args = args_dict)
end 

function _parse_rhs_expression(term_str::String)
    # Purpose: Recursively parses the right-hand side of a formula, respecting operator precedence.
    # Rationale: Builds an Abstract Syntax Tree (AST) representing the model structure.
    # Assumptions: Operators are space-padded (e.g., " ⊗ ").
    # Inputs:
    #   - term_str: The RHS string or a substring of it.
    # Outputs: A nested NamedTuple representing the parsed structure.
    parts = split_terms_at_depth(term_str, " ⊕ ")
    if length(parts) > 1
        return (type=:operator, op=:direct_sum, children=[_parse_rhs_expression(p) for p in parts])
    end
    parts = split_terms_at_depth(term_str, " |> ")
    if length(parts) > 1
        return (type=:operator, op=:pipe, children=[_parse_rhs_expression(parts[1]), _parse_rhs_expression(join(parts[2:end], " |> "))])
    end
    parts = split_terms_at_depth(term_str, " ⊗ ")
    if length(parts) > 1
        return (type=:operator, op=:kronecker_product, children=[_parse_rhs_expression(p) for p in parts])
    end
    parts = split_terms_at_depth(term_str, " ∘ ")
    if length(parts) > 1
        return (type=:operator, op=:composition, children=[_parse_rhs_expression(p) for p in parts])
    end
    return _parse_single_manifold_term(term_str)
    try
        return _parse_single_manifold_term(term_str)
    catch e
        term_str_stripped = strip(term_str)
        if occursin(r"^[a-zA-Z_][a-zA-Z0-9_]*$", term_str_stripped)
            return (module_type = :fixed, args = Dict(:positional_args => [Symbol(term_str_stripped)]))
        else
            rethrow(e)
        end
    end
end

function _categorize_rhs_nodes!(nodes, modules)
    # Purpose: Traverses the parsed AST and categorizes nodes into a flat dictionary of modules.
    # Rationale: Simplifies processing by converting the tree structure into a key-value store.
    # Assumptions: `nodes` is a vector of parsed AST nodes.
    # Inputs:
    #   - nodes: The vector of AST nodes from `_parse_rhs_expression`.
    #   - modules: The dictionary to populate.
    # Outputs: None (mutates `modules`).
    for node in nodes
        if hasproperty(node, :type) && node.type == :operator
            key = "composed_$(length(modules)+1)"
            modules[key] = (module_type = :interact, args = Dict(:operator => node.op, :components => node.children))
        elseif hasproperty(node, :module_type) && node.module_type in BSTM_MODULE_KEYWORDS
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

function decompose_bstm_formula(formula_str::String)
    # Purpose: The main entry point for formula parsing.
    # Rationale: Decomposes the entire formula string into its constituent parts: outcomes, modules, and fixed effects.
    # Assumptions: Formula is in the format `lhs ~ rhs`.
    # Inputs:
    #   - formula_str: The complete model formula.
    # Outputs: A NamedTuple with `:outcomes`, `:modules`, `:fixed_effects`, `:has_intercept`, and `:intercept_prior`.
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])
    outcome_specs = Dict{Symbol, Any}[]
    lhs_terms = split_terms_at_depth(lhs, "+")
    for term in lhs_terms
        term = strip(term)
        m = match(r"likelihood\((.*)\)", term)
        if !isnothing(m)
            inner_content = m.captures[1]
            args = split_terms_at_depth(inner_content, ",")
            !isempty(args) || continue
            outcome_var = strip(args[1])
            params_str = join(args[2:end], ",")
            params = _parse_arguments_string(params_str)
            for ov in Base.split(outcome_var, '+')
                push!(outcome_specs, Dict(:var => strip(ov), :params => params))
            end
        else
            push!(outcome_specs, Dict(:var => term, :params => Dict()))
        end
    end
    rhs_normalized = replace(rhs, r"\s*-\s*" => " + -")
    rhs_terms = split_terms_at_depth(rhs_normalized, " + ")
    has_intercept = true
    intercept_prior = nothing
    fixed_effects = String[]
    module_terms = String[]
    intercept_module_found = false
    for term in rhs_terms
        term_stripped = strip(term)
        if startswith(term_stripped, "intercept(")
            if intercept_module_found
                @warn "Multiple intercept() modules found. Only the first will be evaluated for intercept control."
            else
                intercept_module_found = true
                intercept_data = _parse_single_manifold_term(term_stripped)
                pos_args = get(intercept_data.args, :positional_args, [])
                if !isempty(pos_args) && pos_args[1] == false
                    has_intercept = false
                else
                    has_intercept = true
                end
                if haskey(intercept_data.args, :prior)
                    intercept_prior = intercept_data.args[:prior]
                end
            end
            continue
        elseif term_stripped == "0" || term_stripped == "-1"
            has_intercept = false
        elseif term_stripped == "1"
            # Do nothing, intercept is on by default
        else
            # Check if the term corresponds to a known BSTM module keyword.
            # This correctly separates fixed effects (including transformations like log(x) or interactions x*z)
            # from the special bstm module calls.
            m = match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)", term_stripped)
            is_bstm_module = !isnothing(m) && Symbol(m.captures[1]) in BSTM_MODULE_KEYWORDS
            if !is_bstm_module
                push!(fixed_effects, term_stripped)
            else
                push!(module_terms, term_stripped)
            end
        end
    end
    top_level_nodes = [_parse_rhs_expression(term) for term in module_terms]
    modules = Dict{String, Any}()
    _categorize_rhs_nodes!(top_level_nodes, modules)
    return (outcomes=outcome_specs, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept, intercept_prior=intercept_prior)
end


# ==============================================================================
# SECTION 4: MODEL CONFIGURATION ENGINE
# ==============================================================================

function bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)
    # Purpose: The main configuration function that translates a formula and data into a detailed model specification.
    # Rationale: This is the central pipeline that orchestrates parsing, data processing, and manifold setup.
    # Assumptions: `formula` and `data` are provided correctly.
    # Inputs:
    #   - formula: The model formula string.
    #   - data: The input DataFrame.
    #   - calling_module: The module from which `@bstm` was called, for evaluating expressions.
    #   - kwargs: Additional keyword arguments (e.g., `W`, `hyperpriors`).
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
    M[:N_cov] = 0
    M[:add_intercept] = decomposed_formula.has_intercept
    if !isnothing(decomposed_formula.intercept_prior); M[:intercept_prior] = decomposed_formula.intercept_prior; end
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
    all_fixed_effects = get(M, :fixed_effects, String[])
    _process_fixed_effects!(M, get(M, :fixed_effects, String[]), decomposed_formula.has_intercept)
    _process_fixed_effects_priors!(M)
    _finalize_config!(M)
    return NamedTuple(M)
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
    # Rationale: A helper to avoid repetitive code in `_process_lhs!`.
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
            elseif val isa AbstractVector
                opt_dict[target_key] = val
            else
                @warn "Observation parameter ':$val' for '$target_key' not found in data or is not a vector. Ignoring."
            end
            return
        end
    end
end

function _process_fixed_effects!(M::Dict, fixed_effects_vars::Vector{String}, has_intercept::Bool)
    # Purpose: Creates the design matrix for all fixed effects.
    # Rationale: Consolidates fixed effects from bare terms and `fixed()` modules into a single design matrix.
    # Assumptions: `fixed_effects_vars` contains the names of fixed effect variables.
    # Inputs:
    #   - M: The model configuration dictionary.
    #   - fixed_effects_vars: A vector of fixed effect variable names.
    #   - has_intercept: A boolean indicating if an intercept should be included.
    # Outputs: None (mutates `M`).
    rhs = has_intercept ? "1" : "0"
    if !isempty(fixed_effects_vars)
        rhs_vars = join(fixed_effects_vars, " + ")
        rhs = (rhs == "0") ? rhs_vars : rhs * " + " * rhs_vars
    end
    Xfixed_named = create_fixed_design(rhs, M[:data]; contrasts=get(M, :contrasts, Dict()))
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
end 
 
function _process_fixed_effects_priors!(M::Dict) 
    # Purpose: Resolves and stores the prior distributions for each fixed effect coefficient.
    # Rationale: Allows for custom priors on specific coefficients, falling back to a default.
    # Assumptions: `M[:Xfixed_names]` and `M[:fixed_effects_priors]` are set.
    # Inputs:
    #   - M: The model configuration dictionary.
    # Outputs: None (mutates `M`).
    n_fixed = get(M, :Xfixed_N, 0) 
    if n_fixed == 0 
        M[:Xfixed_priors_vec] = UnivariateDistribution[] 
        return 
    end 
    coef_names = get(M, :Xfixed_names, Symbol[]) 
    custom_priors = get(M, :fixed_effects_priors, Dict{Symbol, Any}()) 
    intercept_prior_val = get(M, :intercept_prior, nothing) 
    default_prior = Normal(0, 5) 
    priors_vec = Vector{Union{UnivariateDistribution, Nothing}}(undef, n_fixed) 
    fill!(priors_vec, nothing) 
 
    # Pass 1: Apply specific and interaction priors
    for (var_sym, prior) in custom_priors 
        var_str = string(var_sym) 
        
        # Applying priors to compound terms like 'a*b' is ambiguous and not supported.
        # The documented approach is to set priors on individual terms ('a', 'b', 'a&b') separately.
        if occursin('*', var_str)
            @warn "Applying priors to compound terms like 'a*b' is not supported. The prior for '$var_str' will be ignored. Please set priors on individual terms (e.g., 'a', 'b', 'a&b') separately."
            continue
        end

        # Handle simple terms and explicit interactions like 'a&b'
        begin
            for i in 1:n_fixed 
                coef_name_str = string(coef_names[i]) 
                # Match exact name or categorical level name (e.g., region: R2)
                if coef_name_str == var_str || startswith(coef_name_str, var_str * ": ") 
                    priors_vec[i] = prior isa Tuple ? create_pc_prior(var_sym, prior) : prior 
                end 
            end 
        end 
    end
 
    # Pass 2: Fill in defaults for any remaining coefficients
    for i in 1:n_fixed 
        if isnothing(priors_vec[i]) 
            coef_name_str = string(coef_names[i]) 
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
        :prior_scheme => :pcpriors
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
    # Assumptions: `data` is present in `opt_dict`.
    # Inputs:
    #   - opt_dict, mod_data, basis_matrices_registry, manifolds_registry.
    # Outputs: None (mutates `opt_dict` and registries).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    data = opt_dict[:data]
    params = mod_data[:params]
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
    return true
end

function process_dynamics_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `dynamics()` module.
    # Rationale: Placeholder for future mechanistic model integration.
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
    return true
end

function process_mixed_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes the `mixed()` module for random effects.
    # Rationale: Creates group indices and sets up parameters for random intercepts or slopes.
    # Assumptions: `data` is present in `opt_dict` and grouping variable exists.
    # Inputs:
    #   - opt_dict, mod_data, registries, hyperpriors.
    # Outputs: None (mutates `mod_data`).
    # Returns: A boolean indicating whether a standard manifold object should be created for this module.
    data = opt_dict[:data]
    vars = mod_data[:variables]
    if length(vars) < 2
        @warn "The mixed() module requires at least two variables: mixed(effect_var | group_var). Skipping."
        return false
    end
    effect_var_str = vars[1]
    group_var_str = vars[2]
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
    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return false
    end
    sub_data = opt_dict[data_source_sym]
    sub_config = _initialize_config(sub_data, Dict(:calling_module => get(opt_dict, :calling_module, Main)))
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
    _finalize_config!(sub_config)
    opt_dict[:nested_manifolds][var] = sub_config
    return true
end

function process_interact_module!(opt_dict, mod_data, registries, hyperpriors)
    # Purpose: Processes interaction modules created by operators like `⊗` and `⊕`.
    # Rationale: Stores the components and operator type for later use by the model assembler.
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
    append!(opt_dict[:fixed_effects], string.(vars))
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
    kmeans_result = kmeans(centroids', n_clusters; maxiter=200, display=:none)
    
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
    # Assumptions: None.
    # Inputs:
    #   - formula, data, kwargs.
    # Outputs: A Turing.jl model object.
    return bstm(formula, data, Main; kwargs...)
end

function bstm(formula::String, data::DataFrame, calling_module::Module; kwargs...)
    # Purpose: A wrapper that first calls `bstm_config` and then `bstm`.
    # Rationale: Separates configuration from model instantiation.
    # Assumptions: None.
    # Inputs:
    #   - formula, data, calling_module, kwargs.
    # Outputs: A Turing.jl model object.
    options = bstm_config(formula, data; calling_module=calling_module, kwargs...)
    return bstm(options)
end

function bstm(config::NamedTuple)
    # Purpose: The final user-facing wrapper that dispatches to the dynamic model generator.
    # Rationale: Provides a single entry point from a configuration object.
    # Assumptions: `config` is a valid model configuration.
    # Inputs:
    #   - config: The model configuration NamedTuple.
    # Outputs: A Turing.jl model object.
    return bstm_dynamic_model(config)
end

function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    # Purpose: Converts a parsed module dictionary into a concrete `Manifold` struct instance.
    # Rationale: This is the bridge between the parsed formula and the typed manifold system.
    # Assumptions: `module_metadata` is a valid dictionary from the parser.
    # Inputs:
    #   - module_metadata: The parsed module data.
    #   - M: The main configuration dictionary.
    #   - priors_dict: The dictionary of global hyperpriors.
    #   - scheme: The prior scheme to use.
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
        inner_priors = resolve_hyperpriors(model_str, priors_dict, m_params, scheme)
        constructor_func = MANIFOLD_CONSTRUCTORS[Symbol(model_str)]
        inner_manifold = constructor_func(inner_priors, m_params)
        return SVCManifold(cov_sym, inner_manifold)
    else
        default_model = if m_type in [:spatial, :temporal, :smooth]; "none"; elseif m_type == :intercept || m_type == :fixed; "none"; else string(m_type); end
        model_val = get(m_params, :model, default_model)
        model_name = if model_val isa Symbol; String(model_val); else; string(model_val); end
        model_sym = Symbol(model_name)
        resolved_priors = resolve_hyperpriors(model_name, priors_dict, m_params, scheme)
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
        Q_rw1 = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        Q_rw1[1,1] = Q_rw1[n,n] = 1.0
        Q = Q_rw1
        sf = 1.0
    elseif type == :ar1
        Q = spdiagm(-1 => fill(1.0, n-1), 1 => fill(1.0, n-1))
        sf = 1.0
    elseif type == :rw2
        D_rw2 = spdiagm(-2 => ones(n-2), -1 => -2*ones(n-1), 0 => ones(n), 1 => -2*ones(n-1), 2 => ones(n-2))
        Q_rw2 = D_rw2' * D_rw2
        Q = Q_rw2
        sf = 1.0
    elseif type == :cyclic
        Q_cyc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1), n-1 => [-1.0], -(n-1) => [-1.0])
        Q_cyc[1,n] = Q_cyc[n,1] = -1.0
        Q = Q_cyc
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
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_from_template(m, data_inputs, :spatial)
end

function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict)
    # Purpose: Builder for temporal GMRF models.
    # Rationale: Dispatches to the template builder with a `:temporal` context.
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_from_template(m, data_inputs, :temporal)
end

function build_model(m::Union{GP, FITC, RFF, SVGP, Warp, Nystrom, Harmonic, Hyperbolic, ExponentialDecay, Kriging}, data_inputs::Dict)
    # Purpose: Builder for continuous, spectral, and other advanced manifolds.
    # Rationale: These models do not rely on pre-computed templates in the same way as GMRFs, so they use a pass-through builder.
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict)
    # Purpose: Builder for spline-based smoothers.
    # Rationale: Determines the appropriate underlying GMRF template (RW1 or RW2) based on the spline type and penalty order.
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
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Eigen, data_inputs::Dict)
    # Purpose: Builder for the `Eigen` manifold.
    # Rationale: Creates an identity template, as the structure is handled dynamically within the model.
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    n = get(data_inputs, :s_N, 1)
    template = build_structure_template(:eigen, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::DynamicsManifold, data_inputs::Dict)
    # Purpose: Builder for the `DynamicsManifold`.
    # Rationale: Provides a Laplacian template for physics-based models like advection/diffusion.
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
    #            It passes the pre-computed Q_template from the manifold object to the assembler.
    # Assumptions: `m.Q_template` is correctly populated.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); if !(fn in [:Q_template]) ; hyper_dict[fn] = getfield(m, fn); end; end
    return (Q_template=m.Q_template, scaling_factor=1.0, model_type=:tensorproductsmooth, hyper=NamedTuple(hyper_dict))
end

function build_model(m::ComposedManifold, data_inputs::Dict)
    # Purpose: Builder for composed manifolds.
    # Rationale: Handles the `pipe` operator by recursively building the state manifold's spec and attaching it.
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
    else
        # For other operators like ⊗, ⊕, no special template is needed at this stage.
        return _build_pass_through_model(m, data_inputs)
    end
end

function _build_from_template(m::ManifoldModel, data_inputs::Dict, domain::Symbol)
    # Purpose: A generic builder for manifolds that use a pre-defined template.
    # Rationale: Reduces code duplication for common GMRF models.
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


function bstm_text_assembler(M::NamedTuple)
    # Purpose: Dynamically generates the Turing `@model` function as a string.
    # Rationale: Allows for extreme flexibility in model specification via the formula interface.
    #            The model structure is built at runtime based on the configuration.
    # Assumptions: `M` is a complete and valid model configuration from `bstm_config`.
    # Inputs:
    #   - M: The model configuration NamedTuple.
    # Outputs: A tuple containing the parsed model expression and the specification registry.
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"
    
    model_func_name = if arch == "multivariate"
        :bstm_text_generated_multivariate
    elseif arch == "multifidelity"
        :bstm_text_generated_multifidelity
    else
        :bstm_text_generated_univariate
    end

    spec_registry = Dict{String, Any}()
    priors_acc = String[]
    updates_acc = String[]
    outcomes_N = get(M, :outcomes_N, 1)
    
    main_spatial_spec = nothing
    main_temporal_spec = nothing

    for spec in M.manifolds
        spec_registry[string(spec.key)] = spec
        calling_mod = get(M, :calling_module, Main)
        domain_fn = getfield(calling_mod, spec.domain)
        
        for k in 1:outcomes_N
            outcome_idx = arch == "multivariate" ? k : nothing
            frag = domain_fn(spec.manifold_obj, string(spec.key), arch, outcome_idx)
            
            push!(priors_acc, frag.priors)
            push!(updates_acc, frag.update)
        end

        if spec.domain == :spatial && isnothing(main_spatial_spec)
            main_spatial_spec = spec
        end
        if spec.domain == :temporal && isnothing(main_temporal_spec)
            main_temporal_spec = spec
        end
    end

    zi_prior_block = if get(M, :use_zi, false)
        "lik_phi_zi ~ NamedDist(Beta(1,1), :lik_phi_zi)"
    else
        "lik_phi_zi = 0.0"
    end

    likelihood_section = if is_multivariate
        """
        L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)
        y_sigma ~ NamedDist(filldist(Exponential(1.0), K), :y_sigma)
        $(zi_prior_block)
        """
    else
        """
        y_sigma_const ~ NamedDist(Exponential(1.0), :y_sigma_const)
        $(observation_volatility(M).priors)
        $(observation_volatility(M).calculation)
        $(zi_prior_block)
        """
    end

    eta_name = is_multivariate ? "eta_latent" : "eta"
    eta_init = is_multivariate ? "zeros(T, N, K)" : "zeros(T, N)"

    intercept_block = if get(M, :add_intercept, false)
        dist = is_multivariate ? "filldist(get(M, :intercept_prior, Normal(0,5)), K)" : "get(M, :intercept_prior, Normal(0, 5))"
        update = is_multivariate ? "for k in 1:K; $(eta_name)[:, k] .+= intercept[k]; end" : "$(eta_name) .+= intercept"
        """
        intercept ~ NamedDist($(dist), :intercept)
        $(update)
        """
    else "" end

    offset_block = if haskey(M, :log_offset)
        is_multivariate ? "$(eta_name) .+= get(M, :log_offset, zeros(T, N, 1))" : "$(eta_name) .+= get(M, :log_offset, zeros(T, N))"
    else "" end

    fixed_effects_block = if M.Xfixed_N > 0
        if is_multivariate
            beta_count = "M.Xfixed_N * K"
            update = "$(eta_name) .+= M.Xfixed * reshape(Xfixed_beta, M.Xfixed_N, K)"
            """
            Xfixed_beta ~ NamedDist(MvNormal(zeros($(beta_count)), 5.0 * I), :Xfixed_beta)
            $(update)
            """
        else
            """
            begin
                local Xfixed_beta = Vector{T}(undef, M.Xfixed_N)
                local priors = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
                for i in 1:M.Xfixed_N
                    Xfixed_beta[i] ~ NamedDist(priors[i], Symbol("Xfixed_beta[", i, "]"))
                end
                eta .+= M.Xfixed * Xfixed_beta
            end
            """
        end
    else "" end

    st_interaction_block = ""
    model_st = get(M, :model_st, "none")
    if model_st != "none"
        if isnothing(main_spatial_spec) || isnothing(main_temporal_spec)
            @warn "Spacetime interaction specified, but could not find both a primary spatial and temporal manifold. Interaction term will be ignored."
        else
            st_interaction_block = """
            st_sigma ~ NamedDist(Exponential(0.5), :st_sigma)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            
            let s_Q = spec_registry["$(main_spatial_spec.key)"].Q_template,
                t_Q = spec_registry["$(main_temporal_spec.key)"].Q_template,
                t_rho = if $(main_temporal_spec.manifold_obj isa AR1)
                            $(Symbol("temporal_$(main_temporal_spec.var)_rho"))
                        else
                            -999.0 
                        end

                st_inter = zeros(T, M.s_N, M.t_N)
                st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)

                if "$(model_st)" == "I"
                    st_inter = st_innov_matrix .* st_sigma
                elseif "$(model_st)" == "II"
                    C_t = cholesky(Symmetric(Matrix(t_Q) + noise * I))
                    st_inter = (C_t.U \\ st_innov_matrix')' .* st_sigma
                elseif "$(model_st)" == "III"
                    C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
                    st_inter = (C_s.U \\ st_innov_matrix) .* st_sigma
                elseif "$(model_st)" == "IV"
                    if t_rho == -999.0
                        @warn "Type IV interaction specified, but main temporal effect is not AR1. Defaulting to Type III."
                        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
                        st_inter = (C_s.U \\ st_innov_matrix) .* st_sigma
                    else
                        C_s = cholesky(Symmetric(Matrix(s_Q) + noise * I))
                        st_inter[:, 1] = (C_s.U \\ st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho^2 + noise)
                        for t in 2:M.t_N; st_inter[:, t] = t_rho .* st_inter[:, t-1] .+ (C_s.U \\ st_innov_matrix[:, t]); end
                        st_inter .*= st_sigma
                    end
                end

                for i in 1:N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end
            end
            """
        end
    end

    final_likelihood = if is_multivariate
        """
        eta = eta_latent * L_corr.L
        for k in 1:K
            family_k = M.likelihood_specs[k][:family]
            for i in 1:N
                y_L_ik = get(M, :y_L, -Inf) isa AbstractMatrix ? M.y_L[i,k] : (get(M, :y_L, -Inf) isa AbstractVector ? M.y_L[i] : get(M, :y_L, -Inf))
                y_U_ik = get(M, :y_U, Inf) isa AbstractMatrix ? M.y_U[i,k] : (get(M, :y_U, Inf) isa AbstractVector ? M.y_U[i] : get(M, :y_U, Inf))
                hurdle_ik = get(M, :hurdle, -Inf) isa AbstractMatrix ? M.hurdle[i,k] : (get(M, :hurdle, -Inf) isa AbstractVector ? M.hurdle[i] : get(M, :hurdle, -Inf))
                trials_ik = M.trials isa AbstractMatrix ? get(M.trials, (i, k), 1) : get(M.trials, i, 1)
                weight_ik = M.weights[i]
                d_lik = bstm_Likelihood(family_k, [T(M.y_obs[i, k])]; sigma_y=[y_sigma[k]], trial=[Int(trials_ik)], y_L=y_L_ik, y_U=y_U_ik, hurdle=hurdle_ik, phi_zi=lik_phi_zi, weight=weight_ik)
                Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
            end
        end
        """
    else
        """
        family = M.likelihood_specs[1][:family]
        for i in 1:N
            y_L_i = get(M, :y_L, -Inf) isa AbstractVector ? M.y_L[i] : get(M, :y_L, -Inf)
            y_U_i = get(M, :y_U, Inf) isa AbstractVector ? M.y_U[i] : get(M, :y_U, Inf)
            hurdle_i = get(M, :hurdle, -Inf) isa AbstractVector ? M.hurdle[i] : get(M, :hurdle, -Inf)
            weight_i = M.weights[i]
            d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma[i]], trial=[Int(get(M.trials, i, 1))], y_L=y_L_i, y_U=y_U_i, hurdle=hurdle_i, phi_zi=lik_phi_zi, weight=weight_i)
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
        end
        """
    end

    model_string = """
    @model function $(model_func_name)(M, spec_registry, ::Type{T}=Float64) where {T}
        noise = get(M, :noise, 1e-6)
        N = M.y_N
        K = $(outcomes_N)

        $(likelihood_section)

        $(join(priors_acc, "\n        "))

        $(eta_name) = $(eta_init)
        $(intercept_block)
        $(offset_block)
        $(fixed_effects_block)

        $(join(updates_acc, "\n        "))

        $(st_interaction_block)

        $(final_likelihood)
    end
    """
    
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
    # Assumptions: `formula_rhs` contains only fixed effects terms.
    # Inputs:
    #   - formula_rhs: The RHS of the formula string.
    #   - data: The input DataFrame.
    #   - contrasts: A dictionary specifying contrast coding for categorical variables.
    # Outputs: A NamedArray containing the design matrix.
    df_internal = copy(data)
    final_rhs_string = strip(formula_rhs)

    if isempty(final_rhs_string)
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[]))
    end

    if final_rhs_string == "1"
        return NamedArray(ones(size(df_internal, 1), 1), (1:size(df_internal, 1), [:Intercept]))
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

        return NamedArray(model_matrix_numeric, (1:size(model_matrix_numeric, 1), label_vector))

    catch design_error
        @warn "BSTM Registry: create_fixed_design expansion failed for: $final_rhs_string. Error: $design_error"
        return NamedArray(zeros(size(df_internal, 1), 0), (1:size(df_internal, 1), Symbol[]))
    end
end

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Purpose: Constructs an orthonormal loadings matrix (eigenvectors) from a matrix of Householder reflector vectors.
    # Rationale: Provides a differentiable and numerically stable way to parameterize an orthonormal matrix for Bayesian PCA.
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

    # The 'moran' basis type was removed from smoothers. It is a spatial basis.
    elseif type == "moran"
        error("The 'moran' basis is not a valid smoother for covariates. Use a spatial manifold like `spatial(..., model=eigen)` instead.")

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

    # The 'moran' basis type was removed from smoothers. It is a spatial basis.
    elseif type == "moran"
        error("The 'moran' basis is not a valid smoother for covariates. Use a spatial manifold like `spatial(..., model=eigen)` instead.")

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

    # The 'moran' basis type was removed from smoothers. It is a spatial basis.
    elseif type == "moran"
        error("The 'moran' basis is not a valid smoother for covariates. Use a spatial manifold like `spatial(..., model=eigen)` instead.")

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
    """v1.2.0 (2026-06-29 16:13:05)
    Purpose: An internal factory function that constructs a final precision matrix from a 
            template, a scale parameter (`param_val`), and other manifold-specific parameters 
            (e.g., correlation `rho`). This function is central to defining GMRF priors within
            the model.
    Inputs: m_type, template_s, param_val, and other optional parameters.
    Outputs: A symmetric precision matrix.
    """
    n_s = size(template_s, 1)
    scale_factor = 1.0 / (param_val^2 + noise)

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

    if m_type == :ar1
        rho = isnothing(extra_param) ? 0.8 : extra_param # Default to a reasonable correlation
        # The precision matrix for a stationary AR(1) process is constructed.
        # It is tridiagonal, with 1 on the main diagonal ends, 1+rho^2 elsewhere on the main
        # diagonal, and -rho on the first off-diagonals. This is built from the template adjacency matrix.
        Q_ar1_base = diagm(0 => fill(1.0 + rho^2, n_s)) - rho .* template_s
        Q_ar1_base[1, 1] = 1.0
        Q_ar1_base[n_s, n_s] = 1.0
        # The overall scaling is (1/sigma_marginal^2) / (1 - rho^2), where sigma_marginal is the marginal
        # standard deviation of the process (`param_val`). This correctly relates the marginal variance
        # to the innovation variance, which is what the precision matrix Q_ar1_base is based on.
        return Symmetric(scale_factor ./ (1.0 - rho^2 + noise) .* Q_ar1_base + noise * I)
    end

    if m_type == :leroux
        lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        return Symmetric(scale_factor .* (lambda_val .* template_s + (1.0 - lambda_val) .* I(n_s)) + noise * I)
    end

    if m_type == :I || m_type == :II || m_type == :III || m_type == :IV
        if isnothing(template_t)
            Q_full = template_s
        else
            Q_full = kron(template_t, template_s)
        end
        return Symmetric(scale_factor .* Q_full + noise * I)
    end

    if m_type == :networkflow
        rho_net = isnothing(extra_param) ? 0.8 : extra_param
        W_net = !isnothing(directed_adj) ? directed_adj : template_s
        
        # The operator for the SAR-like precision matrix depends on the flow direction.
        L_op = if flow_direction == :upstream
            # Upstream influence: a node is influenced by nodes that flow into it.
            # If W_ij=1 means flow from i to j, then W' captures this influence.
            I(n_s) - rho_net .* W_net' # Adjacency matrix represents flow from i to j, so transpose for upstream influence
        elseif flow_direction == :downstream
            # Downstream influence: a node is influenced by nodes it flows to.
            I(n_s) - rho_net .* W_net
        elseif flow_direction == :bidirectional
            # Bidirectional influence: a node is influenced by any neighbor, regardless of flow direction.
            # We create a symmetric, unweighted adjacency matrix where an edge exists if one existed in either direction.
            W_symm = sign.(W_net + W_net')
            I(n_s) - rho_net .* W_symm
        end

        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

    if m_type == :sar || m_type == :dag || m_type == :proper_car
        rho_p = isnothing(extra_param) ? 0.8 : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(scale_factor .* (L_op' * L_op) + noise * I)
    end

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

    return Symmetric(scale_factor .* template_s + noise * I)
end
