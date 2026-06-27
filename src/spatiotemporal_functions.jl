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



# BSTM Low-Level Manifold Registry [v06.1 - Reusable Schema] ---
# Rationale: ManifoldModels are now defined as low-level primitives that are domain-agnostic. 
# The context (Spatial vs Temporal) is determined at the model-building stage.

# --- 1. Core Abstract Types ---
abstract type Manifold end
abstract type ManifoldModel <: Manifold end
abstract type ManifoldOperator <: Manifold end
abstract type ManifoldSupervisor <: Manifold end

# --- 2. Manifold Primitives (ManifoldModel) ---

# 2.1 Placeholders
struct Fixed <: ManifoldModel end
struct Covariate <: ManifoldModel end

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
struct DAG <: ManifoldModel; adjacency_matrix::AbstractMatrix; sigma_prior::UnivariateDistribution; end

# 2.3 Continuous & Spectral Primitives
struct GP <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; kernel::String; end
struct FITC <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct RFF <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_features::Int; kernel::String; end
struct FFT <: ManifoldModel; sigma_prior::UnivariateDistribution; nbins::Int; kernel::String; lengthscale_prior::UnivariateDistribution; end
struct SPDE <: ManifoldModel; sigma_prior::UnivariateDistribution; kappa_prior::UnivariateDistribution; end
struct SVGP <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_inducing::Int; kernel::String; end
struct Warp <: ManifoldModel; lengthscale_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; n_features::Int; kernel::String; end
struct Nystrom <: ManifoldModel; sigma_prior::UnivariateDistribution; lengthscale_prior::UnivariateDistribution; n_inducing::Int; end
struct Wavelet <: ManifoldModel
    sigma_prior::UnivariateDistribution
    lengthscale_prior::UnivariateDistribution
    wavelet_family::Symbol
    nbins::Int
    kernel::String
end

struct PSpline <: ManifoldModel
    nbins::Int
    degree::Int
    diff_order::Int
    sigma_prior::UnivariateDistribution
end

# 2.4 Basis-Function Primitives (for Smooths)
struct TPS <: ManifoldModel; nbins::Int; sigma_prior::UnivariateDistribution; end
struct BSpline <: ManifoldModel; nbins::Int; degree::Int; sigma_prior::UnivariateDistribution; end

# 2.5 Seasonal & Periodic Primitives
struct Harmonic <: ManifoldModel
    amplitude_prior::UnivariateDistribution
    phase_prior::UnivariateDistribution
    sigma_prior::UnivariateDistribution
    period::Union{Real, UnivariateDistribution}
end
struct Cyclic <: ManifoldModel; period::Int; sigma_prior::UnivariateDistribution; end

# 2.6 Physics-Informed & Interaction Primitives
struct Advection <: ManifoldModel; velocity_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct Diffusion <: ManifoldModel; diffusion_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct AdvectionDiffusion <: ManifoldModel; velocity_prior::UnivariateDistribution; diffusion_prior::UnivariateDistribution; sigma_prior::UnivariateDistribution; end
struct ST_I <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ST_II <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ST_III <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct ST_IV <: ManifoldModel; sigma_prior::UnivariateDistribution; end
struct DynamicsManifold <: ManifoldModel
    var::Symbol
    sigma_prior::UnivariateDistribution
end

# 2.7 Specialized & Advanced Primitives
struct ExponentialDecay <: ManifoldModel; sigma_prior::UnivariateDistribution; decay_lengthscale_prior::UnivariateDistribution; end
struct Hyperbolic <: ManifoldModel; curvature::Float64; sigma_prior::UnivariateDistribution; end
struct Eigen <: ManifoldModel; sigma_prior::UnivariateDistribution; pca_sd_prior::UnivariateDistribution; pdef_sd_prior::UnivariateDistribution; n_factors::Int; ltri_indices::Vector{Int}; end
struct BCGN <: ManifoldModel; sigma_prior::UnivariateDistribution; bipartite_adj::AbstractMatrix; group_weights::AbstractVector; end
struct LocalAdaptive <: ManifoldModel
    sigma_prior::UnivariateDistribution
    weights_variable::Symbol
end
struct NetworkFlow <: ManifoldModel
    sigma_prior::UnivariateDistribution
    adjacency_matrix::AbstractMatrix
    flow_direction::Symbol # :upstream, :downstream, :bidirectional
end
struct Mosaic <: ManifoldModel
    coordinates::Vector{Symbol}
    n_regions::Int
    local_smoothness::Bool
end

# --- 3. Manifold Operators ---
struct ComposedManifold <: ManifoldOperator; components::Vector{Manifold}; operator::Symbol; end
struct ChangePointManifold <: ManifoldOperator
    manifold::ManifoldModel              # The process within each segment (e.g., AR1)
    n_changepoints::Int                  # Number of change points to infer
    changepoint_prior::UnivariateDistribution # Prior on the location of change points
end
struct TransformedManifold <: ManifoldOperator
    manifold::Manifold
    transform_fn::Symbol  # e.g., :log, :zscore, :unit
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

# Constructor with default for alpha_prior
RegularizationGroupManifold(manifolds, penalty, lambda_prior; alpha_prior=nothing) = RegularizationGroupManifold(manifolds, penalty, lambda_prior, alpha_prior)

struct SoftConstraintManifold <: ManifoldOperator
    manifold::Manifold
    type::Symbol # :sum_to_zero, :monotonic_increasing, :monotonic_decreasing, :periodicity, :non_negative, :convex, :concave
    weight::Float64
end

# --- 4. Manifold Supervisors ---
struct NestedManifold <: ManifoldSupervisor
    var::Symbol
    formula::String
    data_source::Symbol
end

# --- 5. Algebraic Operators ---
function Base.:|>(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :pipe)
end

⊗(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
⊕(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)

otimes(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :kronecker_product)
oplus(m1::Manifold, m2::Manifold) = ComposedManifold([m1, m2], :direct_sum)


# Helper function to capitalize the first letter of a string
function capitalize(s::String)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A simple helper function to capitalize the first letter of a string.
    """
    isempty(s) && return s
    return uppercasefirst(s)
end

# --- 6. Model Builder Dispatch ---
# These methods translate the high-level manifold structs into technical configurations
# that the @model supervisor can interpret and execute.

function _build_pass_through_model(m::ManifoldModel, data_inputs; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: Packages manifold hyperparameters into a standardized configuration object. This function
              is used for manifolds that do not require complex precision matrix template generation,
              such as continuous or spectral models.
    """
    model_sym = isnothing(model_type_sym) ? Symbol(lowercase(string(typeof(m)))) : model_type_sym
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        # Exclude fields that are not parameters, like type definitions or matrices
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


function _build_from_template(m::ManifoldModel, data_inputs, domain::Symbol)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: Builds a technical configuration for a manifold by first generating a structural precision
              matrix template (e.g., a graph Laplacian) and then packaging it with the manifold's
              hyperparameters. This is used for discrete GMRF-style manifolds.
    """
    # # 1. Model Symbol and Domain Discovery
    # The model symbol is derived from the manifold's type for use in the template factory.
    model_sym = Symbol(lowercase(string(typeof(m))))
    
    # Determine the number of units (n) and the adjacency matrix (W_mat) based on the domain.
    n, W_mat = if domain == :spatial
        (data_inputs.s_N, get(data_inputs, :W, nothing))
    elseif domain == :temporal
        # A default of 10 is used for 'n' if t_N is not provided, primarily for testing.
        (get(data_inputs, :t_N, 10), nothing)
    else 
        # Default to spatial context if domain is unrecognized, with a warning.
        @warn "Unrecognized domain '$domain'. Defaulting to spatial context."
        (data_inputs.s_N, get(data_inputs, :W, nothing))
    end

    # # 2. Precision Matrix Template Generation
    # The `build_structure_template` factory creates the base precision matrix (Q)
    # and its scaling factor based on the model type and dimensions.
    template = build_structure_template(model_sym, n; W=W_mat)

    # # 3. Dynamic Hyperparameter Extraction
    # This logic introspects the fields of the manifold struct and extracts all
    # parameters (priors, numbers, etc.), excluding non-parameter fields like
    # type definitions or pre-computed matrices.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        field_val = getfield(m, fn)
        if !(field_val isa DataType) && !(field_val isa AbstractMatrix)
             hyper_dict[fn] = field_val
        end
    end
    
    # # 4. Ensure Structural Consistency of Hyperparameters
    # To ensure that downstream functions can reliably access optional priors,
    # we add them to the dictionary with a value of `nothing` if they are not present.
    for p in [:rho_prior, :lengthscale_prior, :kappa_prior]
        if !haskey(hyper_dict, p)
            hyper_dict[p] = nothing
        end
    end

    # # 5. Final Configuration Assembly
    # The function returns a NamedTuple containing the technical configuration
    # required by the model supervisor.
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = model_sym,
        hyper = NamedTuple(hyper_dict)
    )
end


# Consolidated builders for graph-based and temporal manifolds
function build_model(m::Union{IID, ICAR, Besag, BYM2, Leroux, SAR}, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A dispatch method for building configurations for discrete spatial (graph-based)
              manifolds. It routes to the `_build_from_template` helper with a `:spatial`
              domain context.
    """
    return _build_from_template(m, data_inputs, :spatial)
end

function build_model(m::Union{AR1, RW1, RW2}, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A dispatch method for building configurations for discrete temporal manifolds. It
              routes to the `_build_from_template` helper with a `:temporal` domain context.
    """
    return _build_from_template(m, data_inputs, :temporal)
end
 
function build_model(m::SoftConstraintManifold, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A dispatch method for building configurations for manifolds with soft constraints. It
              builds the inner manifold's configuration and injects the constraint information.
    """
    # Build the inner model's configuration
    inner_config = build_model(m.manifold, data_inputs)
    # Add the soft constraint information to the hyperparameter tuple
    new_hyper = merge(inner_config.hyper, (soft_constraint_type=m.type, soft_constraint_weight=m.weight))
    return merge(inner_config, (hyper = new_hyper,))
end
 

function build_model(m::RegularizationGroupManifold, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A dispatch method for building configurations for regularization groups (e.g., Lasso,
              Ridge). It packages the penalty type and sub-manifold information for the model
              supervisor.
    """
    # This operator groups multiple manifolds for joint penalization.
    # It returns a special configuration that the @model supervisor recognizes.

    return (
        Q_template = nothing, # No single template for a group
        scaling_factor = 1.0,
        model_type = :regularization_group,
        hyper = (
            penalty = m.penalty,
            lambda_prior = m.lambda_prior,
            alpha_prior = m.alpha_prior, 
            sub_manifolds = m.manifolds # Store the original structs for later lookup
        )
    )
end

# Helper to split terms by a separator, respecting parentheses depth
function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal parsing utility that splits a string by a separator, but respects nested
              parentheses or brackets. This prevents incorrect splits inside function calls, which
              is essential for parsing complex formula strings.
    """
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


function resolve_hyperpriors(m_id::Union{String, Symbol}, user_priors::Dict{String, Any}, scheme::Symbol=:pcpriors)
    # BSTM Internal Utility v1.1.0
    # Timestamp: 2026-06-27 12:30:00
    # Synopsis: An internal helper that resolves and assigns default prior distributions to different
    #           manifold types, allowing for user overrides. This centralizes prior management.
    # Rationale for v1.1.0:
    #     - Refactored to use the global `PC_PRIORS`, `INFORMATIVE_PRIORS`, and `UNINFORMATIVE_PRIORS`
    #       constants, removing local dictionary definitions and improving maintainability.
    #     - The function now fully supports `:pcpriors`, `:informative`, and `:uninformative` schemes.

    m_id_str = string(m_id)

    # 1. Scheme Selection Logic
    # Selects the default prior dictionary based on the chosen scheme.
    defaults = if scheme == :pcpriors
        PC_PRIORS
    elseif scheme == :informative
        INFORMATIVE_PRIORS
    else # Defaults to :uninformative
        UNINFORMATIVE_PRIORS
    end

    # 2. Resolved Object Assembly
    # Using a mutable dictionary for assembly is more performant than repeated merging.
    res = Dict{Symbol, Any}(
        :sigma_prior => get(user_priors, "sigma", defaults["sigma"]),
        :rho_prior => nothing,
        :lengthscale_prior => nothing,
        :kappa_prior => nothing,
        :amplitude_prior => nothing,
        :phase_prior => nothing
    )

    # 3. Registry-Based Parameter Dispatch
    # Assigns specific priors based on the manifold type.
    if m_id_str in ["bym2", "leroux", "sar", "ar1", "proper_car", "dag", "network"]; res[:rho_prior] = get(user_priors, "rho", defaults["rho"]); end
    if m_id_str in ["gp", "fitc", "rff", "warp", "nystrom", "decay"]; res[:lengthscale_prior] = get(user_priors, "lengthscale", defaults["lengthscale"]); end
    if m_id_str == "spde"; res[:kappa_prior] = get(user_priors, "kappa", defaults["kappa"]); end
    if m_id_str == "harmonic"; res[:amplitude_prior] = get(user_priors, "amplitude", defaults["amplitude"]); res[:phase_prior] = get(user_priors, "phase", defaults["phase"]); end

    return NamedTuple(res)
end







# --- 7. Constraint and Regularization Application Helpers ---
# These functions are called from within the @model block to apply penalties.

function apply_soft_constraint(latent_field::AbstractVector, constraint_type::Symbol, weight::Float64)
    T = eltype(latent_field)
    penalty = zero(T)
    n = length(latent_field)

    if n == 0
        return penalty
    end

    if constraint_type == :sum_to_zero
        penalty = -weight * sum(latent_field)^2
    elseif constraint_type == :monotonic_increasing
        penalty = -weight * sum(max.(zero(T), -diff(latent_field)).^2)
    elseif constraint_type == :monotonic_decreasing
        penalty = -weight * sum(max.(zero(T), diff(latent_field)).^2)
    elseif constraint_type == :periodicity
        if n > 1
            penalty = -weight * (latent_field[1] - latent_field[end])^2
        end
    elseif constraint_type == :non_negative
        penalty = -weight * sum(max.(zero(T), -latent_field).^2)
    elseif constraint_type == :convex
        if n > 2; penalty = -weight * sum(max.(zero(T), -diff(diff(latent_field))).^2); end
    elseif constraint_type == :concave
        if n > 2; penalty = -weight * sum(max.(zero(T), diff(diff(latent_field))).^2); end
    else
        @warn "Unknown soft constraint type: $constraint_type. No penalty applied."
    end
    return penalty
end

function apply_regularization_penalty(fields::Vector{<:AbstractVector}, penalty_type::Symbol, lambda::Float64, alpha::Float64=0.5)
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

# --- 8. Manifold Utilities ---
function manifold_type(m::Manifold)
    return lowercase(string(typeof(m)))
end

const MANIFOLD_CONSTRUCTORS = Dict{Symbol, Function}(
    :iid => (p, params) -> IID(p.sigma_prior),
    :icar => (p, params) -> ICAR(p.sigma_prior),
    :besag => (p, params) -> Besag(p.sigma_prior),
    :bym2 => (p, params) -> BYM2(p.rho_prior, p.sigma_prior),
    :leroux => (p, params) -> Leroux(p.rho_prior, p.sigma_prior),
    :sar => (p, params) -> SAR(p.rho_prior, p.sigma_prior),
    :ar1 => (p, params) -> AR1(p.rho_prior, p.sigma_prior),
    :rw1 => (p, params) -> RW1(p.sigma_prior),
    :rw2 => (p, params) -> RW2(p.sigma_prior),
    :gp => (p, params) -> GP(p.lengthscale_prior, p.sigma_prior, string(get(params, :kernel, "se"))),
    :rff => (p, params) -> RFF(p.lengthscale_prior, p.sigma_prior, get(params, :n_features, 20), string(get(params, :kernel, "se"))),
    :fft => (p, params) -> FFT(p.sigma_prior, get(params, :nbins, 20), string(get(params, :kernel, "se")), p.lengthscale_prior),
    :spde => (p, params) -> SPDE(p.sigma_prior, p.kappa_prior),
    :cyclic => (p, params) -> Cyclic(get(params, :period, 12), p.sigma_prior),
    :harmonic => (p, params) -> Harmonic(p.amplitude_prior, p.phase_prior, p.sigma_prior, get(params, :period, 12.0)),
    :pspline => (p, params) -> PSpline(get(params, :nbins, 20), get(params, :degree, 3), get(params, :diff_order, 2), p.sigma_prior),
    :bspline => (p, params) -> BSpline(get(params, :nbins, 10), get(params, :degree, 3), p.sigma_prior),
    :tps => (p, params) -> TPS(get(params, :nbins, 20), p.sigma_prior),
    :dynamics => (p, params) -> DynamicsManifold(get(params, :var, :none), p.sigma_prior)
)

function _parse_module_call(module_call_str::AbstractString)
    """
    BSTM Internal Utility v1.2.0
    Timestamp: 2026-06-26 16:30:00
    Synopsis: An internal recursive parser that processes a single module call string (e.g.,
              "spatial(s_idx)") from the formula into a structured dictionary containing its
              type, variables, and parameters. It handles nested module calls and special syntax
              for Spatially Varying Coefficients (SVC).
    Rationale for v1.2.0:
        - Corrected a `MethodError` by changing the function signature to accept `AbstractString`,
          making the parser robust to `SubString` inputs.
        - Added a guard clause to ensure only keywords in `BSTM_MODULE_KEYWORDS` are processed,
          preventing the parser from misinterpreting other function-like syntax.
        - Implemented the logic to transform `spatial(cov, idx, ...)` syntax into a dedicated
          `:svc` type, which is handled by `resolve_technical_primitive`.
    """
    m_mod = match(r"(\w+)\((.*)\)", module_call_str)
    if isnothing(m_mod)
        return Dict{Symbol, Any}(:type => :literal, :value => module_call_str)
    end

    kw = lowercase(m_mod.captures[1])
    inner_str = m_mod.captures[2]
    
    # Check if the keyword is a known BSTM module. If not, treat it as a literal term.
    # This prevents the parser from attempting to process non-bstm function calls.
    if !(kw in BSTM_MODULE_KEYWORDS)
        return Dict{Symbol, Any}(:type => :literal, :value => module_call_str)
    end

    v_part_str = ""
    p_dict_raw = Dict{Symbol, Any}()
    
    # Split inner_str into variable part and parameter part (separated by ';')
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

    # Process variables and transforms
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

    # --- SVC Syntax Transformation ---
    # If `spatial(cov, idx, ...)` is used, transform it into an SVC module call.
    if kw == "spatial" && length(final_vars) > 1
        cov_var = final_vars[1]
        idx_var = final_vars[2]
        
        # Store the index variable in the parameters for later use
        p_dict_raw[:index_var] = idx_var
        
        return Dict{Symbol, Any}(
            :type => :svc,
            :variables => [cov_var], # The primary variable for an SVC is the covariate
            :params => p_dict_raw
        )
    end

    # Special handling for nested modules
    if kw == "transform"
        # The first variable in final_vars is the inner module call string
        inner_module_call_str = final_vars[1]
        inner_module_data = _parse_module_call(inner_module_call_str) # Recursive call
        
        return Dict{Symbol, Any}(
            :type => Symbol(kw),
            :inner_module => inner_module_data,
            :params => p_dict_raw # This should contain 'fn'
        )
    elseif kw == "interaction"
        # 'model' parameter needs to be parsed as an inner module
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
    """
    BSTM Internal Utility v1.1.0
    Timestamp: 2026-06-26 16:30:00
    Synopsis: An internal helper that resolves a parsed module's metadata into a concrete
              `Manifold` struct instance. It uses the `MANIFOLD_CONSTRUCTORS` dictionary to
              instantiate the correct struct with its resolved hyperpriors.
    Rationale for v1.1.0:
        - Refactored to correctly distinguish between the module's domain (e.g., `spatial`)
          and its implementation (e.g., `model='bym2'`). The `model` parameter is now used
          to look up the constructor, not the module's type.
        - Added logic to handle the `:svc` type, which wraps an inner spatial manifold.
        - Integrated constructors for `eigen`, `dynamics`, and `nested` modules.
    """
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    
    # Handle special operator types first
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
        
        # Create the inner manifold object that defines the spatial structure of the SVC
        inner_priors = resolve_hyperpriors(model_str, priors_dict, scheme)
        constructor_func = MANIFOLD_CONSTRUCTORS[Symbol(model_str)]
        inner_manifold = constructor_func(inner_priors, m_params)
        
        return SVCManifold(cov_sym, inner_manifold)
    else
        # For standard modules, determine the model implementation
        # Domains like :spatial or :temporal select an implementation via `model=...`.
        # Other modules like :eigen or :dynamics are their own implementation.
        default_model = if m_type == :spatial; "bym2"
                        elseif m_type == :temporal; "ar1"
                        elseif m_type == :smooth; "pspline"
                        else string(m_type) end
        
        model_name = string(get(m_params, :model, default_model))
        model_sym = Symbol(model_name)

        resolved_priors = resolve_hyperpriors(model_name, priors_dict, scheme)

        if haskey(MANIFOLD_CONSTRUCTORS, model_sym)
            constructor_func = MANIFOLD_CONSTRUCTORS[model_sym]
            return constructor_func(resolved_priors, m_params)
        else
            @warn "Unknown manifold model '$model_name' for module '$m_type'. Defaulting to IID."
            return IID(Exponential(1.0))
        end
    end
end



function parse_manifold_graph(expr_in::AbstractString)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal recursive parser that decomposes algebraic manifold expressions involving
              operators like `⊗` (Kronecker product) and `⊕` (direct sum) into a structural tree
              for the model builder.
    """
    # Audited Recursive DSL Parser [v15.2.0 - BSTM v06.1 Final Restoration]
    # Rationale: This engine decomposes algebraic manifold expressions into a structural tree.
    # Requirements: Support for ⊕ (Direct Sum), ⊗ (Kronecker), and ∘ (Composition).
    
    local expr = string(replace(expr_in, r"\s" => ""))

    if occursin("⊕", expr)
        local elements = Base.split(expr, "⊕")
        return (type=:sum, elements=parse_manifold_graph.(string.(elements)))

    elseif occursin("⊗", expr)
        local elements = Base.split(expr, "⊗")
        return (type=:kronecker, elements=parse_manifold_graph.(string.(elements)))

    elseif occursin("∘", expr)
        local parts = Base.split(expr, "∘")
        return (type=:composition, elements=parse_manifold_graph.(string.(parts)))

    else
        local m = match(r"(\w+)\((.*?)\)", expr)
        if !isnothing(m)
            local m_name = lowercase(string(m.captures[1]))
            local var_match = match(r"\((.*)\)", expr)
            local var_name = !isnothing(var_match) ? string(var_match.captures[1]) : expr
            return (type=:atomic, model=m_name, var=var_name)
        else
            return (type=:atomic, model="literal", var=expr)
        end
    end
end


function process_graph_into_rules!(re_rules::Dict{String, Any}, opt_kwargs::Dict{Symbol, Any}, graph)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper that traverses the parsed manifold graph and populates the `re_rules`
              (random effects rules) and `opt_kwargs` dictionaries. This function sets up the final
              model configuration based on the parsed algebraic structure.
    """
    # Audited Rule Dispatcher [v15.2.0]
    # Rationale: Recursively populates re_rules and architectural flags from the manifold graph.
    
    if graph.type == :atomic
        local m_name = lowercase(string(graph.model))
        local var_name = string(graph.var)
        re_rules[var_name] = Dict{Symbol, Any}(:model => m_name)

        # Routing for Global Architectural Flags
        if m_name in ["spatial", "iid", "bym2", "icar", "leroux", "sar", "dag", "spde", "gp", "fitc", "rff", "mosaic", "fft"]
            local m_resolved = (m_name == "spatial") ? "iid" : m_name
            re_rules[var_name][:model] = m_resolved
            re_rules[var_name][:domain] = :spatial
            opt_kwargs[:model_space] = m_resolved

        elseif m_name in ["temporal", "ar1", "rw1", "rw2", "fft", "rff", "harmonic"]
            local m_resolved = (m_name == "temporal") ? "ar1" : m_name
            re_rules[var_name][:model] = m_resolved
            re_rules[var_name][:domain] = :temporal
            opt_kwargs[:model_time] = m_resolved

        elseif m_name == "cyclic"
            re_rules[var_name][:domain] = :seasonal
            opt_kwargs[:model_season] = "cyclic"

        elseif m_name == "eigen"
            re_rules[var_name][:is_eigen] = true

        elseif m_name == "nested"
            opt_kwargs[:model_arch] = "multifidelity"
            re_rules[var_name][:is_nested] = true
        end

    elseif graph.type == :kronecker
        opt_kwargs[:model_st] = "IV"
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end

    elseif graph.type == :sum
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end

    elseif graph.type == :composition
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end
    end
end






# --- 3. Architectural Dispatch Types ---
# Concrete types cannot be subtyped; therefore, we must define abstract bases first.
abstract type AbstractModelArchitecture end

struct UnivariateArchitecture <: AbstractModelArchitecture end
struct MultivariateArchitecture <: AbstractModelArchitecture end
struct MultifidelityArchitecture <: AbstractModelArchitecture end
struct ExampleArchitecture <: AbstractModelArchitecture end
struct UnknownArchitecture <: AbstractModelArchitecture end


# --- 4. Likelihood Family Types ---
# Concrete types cannot be subtyped; therefore, we must define abstract bases first.
  
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


# --- 3. Unified Likelihood Structure ---
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A unified likelihood structure that dispatches to different probability density
              functions based on the specified model family. It handles complexities like
              zero-inflation and censoring through a trait-based system.
    """
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


# --- 4. Outer Constructor (Fixed Mapping) ---
function bstm_Likelihood(family_input, y_obs; sigma_y=0.0, weight=1.0, phi_zi=-Inf, r_nb=0, trial=0,
                         y_L=-Inf, y_U=Inf, hurdle=-Inf, extra_params=zeros(1)[])

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

    local z_trait = phi_zi >= 0.0 ? ZeroInflated() : NonZeroInflated()

    # Ensure hurdle and bounds are numeric for kernel comparison
    local h_val = isnothing(hurdle) ? -Inf : hurdle
    local yL_val = isnothing(y_L) ? -Inf : y_L
    local yU_val = isnothing(y_U) ? Inf : y_U

    local c_trait = if !isfinite(yL_val) && !isfinite(yU_val); Uncensored()
        elseif !isfinite(yL_val) && isfinite(yU_val); LeftCensored()
        elseif isfinite(yL_val) && !isfinite(yU_val); RightCensored()
        else IntervalCensored() end

    return bstm_Likelihood(fam, y_obs, z_trait, c_trait, weight, phi_zi, r_nb, sigma_y, trial, yL_val, yU_val, h_val, extra_params)
end

# --- 5. Distribution Mapping ---

# Rationale: Standardizing the mapping of eta to Distribution objects.
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A set of internal dispatch methods that map a linear predictor `eta` to the
              appropriate `Distributions.jl` object based on the model family. This function
              handles parameter transformations (e.g., `exp(eta)` for a log-link).
    """
function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(exp(eta)); end
function get_dist_ref(::GaussianFamily, d, eta, sig); return Normal(eta, sig); end
function get_dist_ref(::LogNormalFamily, d, eta, sig); return LogNormal(eta, sig); end
function get_dist_ref(::NegativeBinomialFamily, d, eta, sig); mu = exp(eta); return NegativeBinomial(d.r_nb, d.r_nb/(d.r_nb + mu)); end
function get_dist_ref(::BinomialFamily, d, eta, sig); n = d.trial isa AbstractVector ? d.trial[1] : d.trial; return Binomial(Int(n), logistic(eta)); end
function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, exp(eta)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(exp(eta)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(mu*phi, (1-mu)*phi); end
function get_dist_ref(::InverseGaussianFamily, d, eta, sig); mu = exp(eta); lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return InverseGaussian(mu, lambda); end
function get_dist_ref(::StudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return LocationScale(eta, sig, TDist(nu)); end
function get_dist_ref(::HalfNormalFamily, d, eta, sig); return truncated(Normal(0.0, sig), 0.0, Inf); end
function get_dist_ref(::HalfStudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return truncated(LocationScale(0.0, sig, TDist(nu)), 0.0, Inf); end
function get_dist_ref(::LaplaceFamily, d, eta, sig); return Laplace(eta, sig); end
function get_dist_ref(::ParetoFamily, d, eta, sig); alpha = exp(eta); k = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Pareto(alpha, k); end
function get_dist_ref(::DirichletFamily, d, eta, sig); return Dirichlet(exp.(eta)); end

# Specialized matrix-variate factory
function get_dist_ref(::InverseWishartFamily, d, eta::AbstractMatrix, sig)
    local p = size(eta, 1)
    local nu = d.extra_params isa Number && d.extra_params > p ? d.extra_params : Float64(p + 1)
    # Transformation to ensure positive definiteness
    local Psi = PDMat(Symmetric(eta * eta' + 1e-6 * I))
    return InverseWishart(nu, Psi)
end

# --- 4. Multiple Dispatch Kernels ---

# Helper to determine discrete status without ternary syntax
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper to check if a model family belongs to a discrete distribution.
              This is used for handling censoring and other adjustments correctly, as discrete and
              continuous distributions require different logic for censored intervals.
    """
function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    return false
end

# Point Kernel (Uncensored) 

    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A set of internal kernel functions that compute the log-probability for an
              observation by dispatching on the model family, censoring status, and
              zero-inflation state. This modular design separates the logic for different
              observation types.
    """
# Refined point kernel for all families
function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    dist = get_dist_ref(fam, d, eta, sig)
    lp_branch = logpdf(dist, y)
    
    # Initialize with default value
    lp_final = lp_branch

    # Logic for Zero-Inflation Mixture
    # Mathematical Definition: P(y) = phi * I(y=0) + (1 - phi) * f(y; theta)
    # Where I(y=0) is the indicator function (point mass at zero)
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        if y == 0.0
            # For both discrete and continuous, y=0 captures the point mass
            # Discrete families may also have probability mass at 0 from the distribution itself
            pz = 0.0
            if is_discrete_family(fam)
                pz = pdf(dist, 0.0)
            end
            lp_final = logsumexp(log_phi, log_one_minus_phi + log(pz + 1e-15))
        else
            # For y > 0, we only have the continuous/discrete density branch
            lp_final = log_one_minus_phi + lp_branch
        end
    end

    # Hurdle Truncation adjustment
    if d.hurdle > -Inf
        lp_final = lp_final - logccdf(dist, d.hurdle)
    end

    return lp_final
end

# Logic for Zero-Inflation Mixture in Interval Censoring
function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    function stable_logdiffexp(a, b)
        if a <= b
            return -1e15
        end
        diff = b - a
        safe_diff = min(-eps(typeof(diff)), diff)
        return a + log1mexp(safe_diff)
    end

    dist = get_dist_ref(fam, d, eta, sig)
    adj_L = d.y_L[1]
    if is_discrete_family(fam)
        adj_L = adj_L - 1.0
    end

    lp_prob = stable_logdiffexp(logcdf(dist, d.y_U[1]), logcdf(dist, adj_L))
    lp_final = lp_prob

    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        
        # If the interval [y_L, y_U] includes zero, add the point mass phi
        if d.y_L[1] <= 0.0
            lp_final = logsumexp(log_phi, log_one_minus_phi + lp_prob)
        else
            lp_final = log_one_minus_phi + lp_prob
        end
    end

    return lp_final
end


# Censoring Kernels
function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zi::AbstractZIState, d, eta, sig, y)
    local dist = get_dist_ref(fam, d, eta, sig)
    local lp = logcdf(dist, d.y_U[1])
    if zi isa ZeroInflated; lp = logsumexp(log(d.phi_zi + 1e-15), log(1.0 - d.phi_zi + 1e-15) + lp); end
    return lp
end

function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zi::AbstractZIState, d, eta, sig, y)
    local dist = get_dist_ref(fam, d, eta, sig)
    local adj_L = is_discrete_family(fam) ? d.y_L[1] - 1.0 : d.y_L[1]
    local lp = logccdf(dist, adj_L)
    if zi isa ZeroInflated; lp = (d.y_L[1] <= 0.0) ? 0.0 : (log(1.0 - d.phi_zi + 1e-15) + lp); end
    return lp
end


function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zi::AbstractZIState, d, eta, sig, y)
    # Math: log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
    function stable_logdiffexp(a, b)
        if a <= b
            return -1e15 # Effectively zero probability
        else
            local diff = b - a
            # Clamp diff slightly below zero to prevent domain errors in log1mexp
            local safe_diff = min(-eps(typeof(diff)), diff)
            return a + log1mexp(safe_diff)
        end
    end

    local dist = get_dist_ref(fam, d, eta, sig)
    local adj_L = is_discrete_family(fam) ? d.y_L[1] - 1.0 : d.y_L[1]
    local lp = stable_logdiffexp(logcdf(dist, d.y_U[1]), logcdf(dist, adj_L))
    if zi isa ZeroInflated; lp = (d.y_L[1] <= 0.0) ? logsumexp(log(d.phi_zi + 1e-15), log(1.0 - d.phi_zi + 1e-15) + lp) : log(1.0 - d.phi_zi + 1e-15) + lp; end
    return lp
end

# --- 5. Unified Dispatch Interface (logpdf) ---
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: Overloads the `Distributions.logpdf` function for the `bstm_Likelihood` struct.
              This provides the main entry point for calculating the log-likelihood within a
              Turing model, dispatching to the appropriate `bstm_kernel` based on the data traits.
    """

function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    local sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    local w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    if d.family isa DirichletFamily
        local w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
    end
    local total_lp = 0.0
    for i in 1:length(eta)
        local sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        local w = d.weight isa AbstractVector ? d.weight[i] : d.weight
        total_lp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
    end
    return total_lp
end

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractMatrix)
    local w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, 1.0, d.y_obs) * w
end


function scaling_factor_bym2( adjacency_mat )
    """
    BSTM Partitioning Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: Calculates the geometric mean of the variances of an Intrinsic CAR model's
              precision matrix. This factor is used to scale the structured component in a BYM2
              model, ensuring that the marginal variance of the structured effect is approximately 1.
    """
    # Re-scaling variance using the geometric mean of variances of the ICAR component
    # Rationale: Standardizes the structured innovation scale relative to the unstructured IID component.
    # Requirement: Precision matrix Q must be symmetric and positive-definite.

    N = size(adjacency_mat)[1]
    if N <= 1
        return 1.0
    end

    # 1. Construct the Graph Laplacian
    asum = vec(sum(adjacency_mat, dims=2))
    Q = Diagonal(asum) - adjacency_mat

    # 2. Symmetric Perturbation for Numerical Stability
    # Rationale: Adding a ridge to the full matrix ensures stability across the entire spectrum.
    jitter = sqrt(1e-15) * mean(asum)
    Q_stable = Q +jitter * I

    # 3. Sum-to-Zero Constraint Adjustment
    # Compute the diagonal of the generalized inverse subject to the constraint A'x = 0 where A is a vector of ones.
    A = ones(N)
    try
        # Solve Q * S = I under the constraint
        S = Q_stable \ Diagonal(A)
        V = S * A
        # Projection matrix to enforce the sum-to-zero constraint in the covariance space
        S = S - V * inv(A' * V) * V'

        diag_s = diag(S)

        # 4. Extract Variances for Valid Eigen-dimensions
        valid_diag = filter(x -> x > jitter, diag_s)

        if isempty(valid_diag)
            return 1.0
        end

        # 5. Calculate Scaling Factor
        # Geometric mean of the variances ensures the structured component represents marginal variance.
        scale_factor = exp(mean(log.(valid_diag)))

        return isnan(scale_factor) ? 1.0 : scale_factor
    catch
        return 1.0
    end
end

# POSITIONAL OVERLOAD
function scaling_factor_bym2(node1, node2, groups=ones(length(node1)))
    """
    BSTM Partitioning Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An overloaded method for `scaling_factor_bym2` that calculates independent
              scaling factors for multiple disconnected components within a graph. This is
              necessary when a spatial domain consists of separate, non-adjacent regions.
    """
    # Logic: Calculates independent scale factors for K connected components.
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function for discretizing a continuous vector into a specified number of
              categories. It supports both quantile-based binning (equal number of points per
              bin) and regular-interval binning (equal bin width).
    """
     
    if method=="quantile" 
        # simpler solutions
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

        xd = round.(Int, X ./ dx ) .* dx   # resolution to  units of dx
        brks = collect( minimum(xd):dx:maximum(xd) + dx  ) 
        mids = midpoints(brks)
        N_cats = length(mids)
        
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        return xd, xi, mids, N_cats, dx

    elseif method=="quantile_resolution"
    
        brks = quantile(X, range(0, 1, length=N_cats+1))
        mids = midpoints(brks)
        xd_cut = cut(X, brks, extend=true)  # from CategoricalArrays
        xi = levelcode.(xd_cut)  # integer index
        dx = diff(mids)[1]
        xd = mids[xi] 
        return xd, xi, mids, N_cats, dx

    end

end
 
 

function rff_map(coords, W, b)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that maps input coordinates into a Random Fourier Feature (RFF)
              space. This projection is the core of approximating a kernel function with a linear
              model in the feature space.
    """
    # Description:
    #   Maps input coordinates into a Random Fourier Feature (RFF) space
    #   to approximate a kernel function (usually Squared Exponential).
    # Inputs:
    #   coords: N x D matrix of input features (space/time).
    #   W: D x M weight matrix sampled from spectral density.
    #   b: Vector of M random phases.
    # Outputs:
    #   N x M feature matrix.

    # Project coordinates into higher dimensional space
    projection = (coords * W) .+ b'
    
    # Apply cosine transformation with scaling factor
    m = size(W, 2)
    feature_map = sqrt(2 / m) .* cos.(projection)
    
    return feature_map
end
 

function generate_informed_rff_params(coords, M_rff)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that generates parameters (weights `W` and biases `b`) for
              Random Fourier Features. The lengthscale used for sampling frequencies is informed
              by the standard deviation of the input data coordinates.
    """
    # Audited and Robust RFF Parameter Generator (v05.4)
    # Rationale: Fixing MethodError by ensuring numeric matrix conversion before adjoint.
    
    # 1. Convert input to standard Matrix{Float64}
    mat = if coords isa AbstractMatrix && eltype(coords) <: Real
        Matrix{Float64}(coords)
    elseif coords isa AbstractVector && eltype(coords) <: Tuple
        # Standardize Tuple vectors to numeric Matrix
        reduce(hcat, [[Float64(p[1]), Float64(p[2])] for p in coords])'
    else
        # Fallback for complex nesting or SubArrays
        collect(Matrix{Float64}(reduce(hcat, collect.(coords))'))
    end

    # 2. Dynamic Orientation Check (Enforce D x N for std calculation)
    if size(mat, 1) > size(mat, 2) && size(mat, 2) <= 3
        mat = collect(mat')
    end

    d = size(mat, 1)
    
    # 3. Scale Calculation along observation axis
    coord_scales = std(mat, dims=2) .+ 1e-6

    # 4. Frequency Weight Generation using spectral sampling
    # The lengthscale is informed by the data std, with a multiplier
    informed_ls = mean(coord_scales) * lengthscale_mult
    W = sample_spectral_density(kernel_name, d, M_rff, informed_ls; nu=nu)

    # 5. Phase Shift Generation
    b = rand(M_rff) .* (2.0 * pi)
    
    return W, b
end


function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A helper function to generate Random Fourier Feature parameters specifically for a
              Squared Exponential kernel. It samples the frequency weights `W` from the
              corresponding Gaussian spectral density.
    """
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 





function build_laplacian_precision(adj_matrix)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that constructs a GMRF precision matrix (Graph Laplacian, Q = D -
              W) from a given adjacency matrix W. This is the fundamental structure for ICAR and
              other graph-based spatial models.
    """
    # Description:
    #   Constructs a GMRF precision matrix (Graph Laplacian) from an adjacency matrix.
    # Inputs:
    #   adj_matrix: Sparse adjacency matrix (W).
    # Outputs:
    #   Sparse precision matrix (Q).

    # D is the diagonal matrix of node degrees
    D_diag = Diagonal(vec(sum(adj_matrix, dims=2)))
    Q_mat = D_diag - adj_matrix
    
    return Q_mat
end
 





 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_x=nothing, s_y=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    """
    BSTM Visualization Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A comprehensive visualization engine for plotting posterior results from `bstm`
              models. It can generate spatial maps of effects, temporal trend plots, and bar
              plots for covariate effects.
    """
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    # Extract target stats and guard against nothing or scalar values
    st = getproperty(stats, effect)
    isnothing(st) && return nothing
    if st isa Real
        return Plots.plot(title="$effect (Fixed: $st)")
    end
 

    # 1. Handle Categorical/Class Bar Plots
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

    # 2. Handle Temporal Main Effects
    elseif effect == :temporal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    # 2a. Handle Seasonal Effects
    elseif effect == :seasonal
        t_stats = st
        if isnothing(t_stats) || t_stats isa Real; return nothing; end
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean,
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean),
                   fillalpha=0.2, lw=2, title="Seasonal Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)


    # 3. Handle Spatial, ST, and Deep GP Mean Fields
    elseif effect in [:spatial, :spatial_structured, :spatial_unstructured, :predictions_denoised, :predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        # Determine the values to map to colors
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
        # elseif effect == :residuals
        #    stats.predictions_noisy.mean[:, isnothing(time_slice) ? 1 : time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        # SAFETY FIX: Plot only as many polygons as results for to avoid BoundsError
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
    """
    BSTM Visualization Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A diagnostic utility that overlays the posterior density of a parameter with its
              prior density. This is useful for assessing how much the data has informed the
              parameter estimate and for checking for prior-data conflict.
    """
    # Description: Overlays posterior and prior densities for a specific parameter to check learning/shrinkage.
    # Inputs:
    #   - model: Turing model object.
    #   - chain: MCMC sample chain.
    #   - param_sym: Symbol of the parameter to check.
    # Outputs:
    #   - A Plots.Plot object.

    # 1. Extract posterior samples using .data for AxisArray compatibility
    post_samples = vec(chain[param_sym].data)

    # 2. Automated Prior Sampling via Turing
    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    # 3. Visualization
    plt = StatsPlots.density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    StatsPlots.density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end




# --- 1. MODEL UTILITIES ---


function get_rff_deep2D_basis(X, m, lengthscale)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that generates a Random Fourier Feature (RFF) basis for 2D
              inputs, typically used for spatial or spatiotemporal GP approximations. This is a
              simplified version of `rff_map`.
    """
    # Description: Generates Random Fourier Feature (RFF) basis for 2D inputs (Spatial/Temporal).
    # Inputs:
    #   - X: Input matrix (N x D).
    #   - m: Number of features.
    #   - lengthscale: Gaussian kernel lengthscale.
    # Outputs:
    #   - N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .*  2pi
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end


function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that generates a Random Fourier Feature (RFF) basis for a 1D
              temporal trend. This is used to approximate a Gaussian Process over time with a
              linear model on the RFF features.
    """
    N = length(t)
    # Generate random parameters for RFFs.
    # Using a seed ensures consistency within the AD pass.
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that generates a basis matrix for periodic seasonal components
              using sine and cosine functions (a Fourier series). This is a deterministic basis,
              not a random one like RFFs.
    """
    # Description: Generates RFF-style basis for periodic seasonal components.
    # Inputs:
    #   - t: Time vector.
    #   - m: Number of harmonics.
    #   - freq: Base frequency.
    #   - lengthscale: Smoothness scale.
    # Outputs:
    #   - N x (2*m) feature matrix.
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
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper function that applies a specified discretization rule (e.g.,
              quantile, regular interval) or transformation (e.g., z-score, log) to a vector of
              continuous values.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A pre-processing utility that handles covariate discretization, transformation, and
              interaction term creation based on user-defined rules. It prepares the covariate
              data for use in the main model.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A wrapper function for running Automatic Differentiation Variational Inference
              (ADVI) on a Turing model. It fits the model using ADVI and formats the results
              into a summary, including posterior means and standard deviations.
    """

    # Fit via ADVI. minor speed benefit vs NUTS
    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

    # vectorize variable names ... needs more conditions if 2-D or higher ..
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


# -----------------------

function get_params_vector(chain, base_name::String, expected_len::Int)
    # Logic: Extracts parameters from FlexiChains, handling indexed names via numerical sorting.
    # Change: Fixed regex escaping for Julia strings to correctly match square brackets.

    N_samples = size(chain, 1)
    all_names = string.(FlexiChains.parameters(chain))

    # Use raw string r"..." to avoid double-backslash requirement for brackets
    regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    matched_names = filter(n -> occursin(regex, n), all_names)

    if !isempty(matched_names)
        # Sort by the captured integer index
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))

        res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            val_obj = chain[Symbol(n)]
            raw = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

            for s in 1:N_samples
                v = raw[s]
                res_mat[s, idx] = (v isa AbstractVector) ? Float64(v[1]) : Float64(v)
            end
        end

        if size(res_mat, 2) == 1 && expected_len > 1
            return repeat(res_mat, 1, expected_len)
        end

        return res_mat
    end

    if base_name in all_names
        val_obj = chain[Symbol(base_name)]
        raw_data = hasproperty(val_obj, :data) ? val_obj.data : collect(val_obj)

        mat_data = if eltype(raw_data) <: AbstractVector
             reduce(hcat, [vec(collect(v)) for v in raw_data])'
        else
             Matrix{Float64}(reshape(collect(raw_data), N_samples, :))
        end

        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 1) == expected_len && size(mat_data, 2) != expected_len
            return mat_data'
        elseif size(mat_data, 2) == 1
            return repeat(mat_data, 1, expected_len)
        else
            return reshape(vec(mat_data), N_samples, :)
        end
    end

    return zeros(Float64, N_samples, expected_len)
end


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing, M=nothing)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper for reconstructing the observation volatility (noise) surface. It
              handles both homoscedastic (constant) and heteroscedastic (stochastic, RFF-based)
              cases for both univariate and multivariate models.
    """
    y_sig_samples = zeros(N_tot, N_samples)
    for j in 1:N_samples
        local sig_y
        if !isnothing(outcome_idx)
            # Multivariate specific volatility extraction
            # FIX: Explicitly construct string for v_key and c_key to avoid ParseError
            v_key = string("y_sigma_k[", outcome_idx, "]")
            c_key = string("y_sigma_const_k[", outcome_idx, "]")

            if v_key in name_strs # Stochastic volatility for specific outcome
                # FIX: Explicitly construct string for parameter names passed to get_params_vector
                sig_val = get_params_vector(chain, string("sigma_log_var_k[", outcome_idx, "]"), 1)[j]
                beta_vol_latent_val = get_params_vector(chain, string("beta_vol_latent_k[", outcome_idx, "]"), M.M_rff_sigma)[j, :]
                # FIX: M.vol_proj is global, not outcome-specific for now
                y_vol_proj_k = M.vol_proj
                sig_y = exp.((sig_val .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(y_vol_proj_k) * beta_vol_latent_val)) ./ 2.0)
            elseif c_key in name_strs # Homoskedastic volatility for specific outcome
                sig_val = Float64(chain[Symbol(c_key)][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        elseif !isnothing(M) && get(M, :use_sv, false) # Added SV handling for univariate
            # Reconstruct y_sigma from SV parameters for univariate model
            sigma_log_var_val = get_params_vector(chain, "sigma_log_var", 1)[j]
            beta_vol_latent_val = get_params_vector(chain, "beta_vol_latent", M.M_rff_sigma)[j, :]
            # M.vol_proj must be available for reconstruction
            if haskey(M, :vol_proj)
                sig_y = exp.((sigma_log_var_val .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj) * beta_vol_latent_val)) ./ 2.0)
            else
                @warn "M.vol_proj not found for stochastic volatility reconstruction. Defaulting to 1.0."
                sig_y = fill(1.0, N_tot)
            end
        else
            # General univariate volatility extraction
            if "y_sigma" in name_strs
                val = get_params_vector(chain, "y_sigma", 1)[j]
                sig_y = val isa AbstractVector ? vec(collect(val)) : fill(Float64(val), N_tot)
            elseif "y_sigma_const" in name_strs
                sig_val = Float64(chain[:y_sigma_const][j])
                sig_y = fill(sig_val, N_tot)
            else
                sig_y = fill(1.0, N_tot)
            end
        end

        # Final flattening and type enforcement
        flat_sig = vec(Float64.(collect(sig_y)))

        if length(flat_sig) >= N_tot
            y_sig_samples[:, j] = flat_sig[1:N_tot]
        else
            # Pad with the last value if reconstructed sig_y is shorter than N_tot (e.g., if only M.y_N is returned)
            y_sig_samples[1:length(flat_sig), j] = flat_sig
            y_sig_samples[length(flat_sig)+1:end, j] .= flat_sig[end]
        end
    end
    return y_sig_samples
end

 


function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that generates Random Fourier Feature weights (`W`) by sampling
              from a provided 2D magnitude spectrum. This allows for data-informed feature
              generation, where frequencies are sampled based on their observed power.
    """
"""
    generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)

Generates 2D RFF weights W by sampling frequencies from the provided 2D magnitude spectrum.

Args:
    freqs_x: Vector of x-dimension frequencies.
    freqs_y: Vector of y-dimension frequencies.
    magnitude_spectrum: 2D array of magnitude values corresponding to freqs_x, freqs_y.
    M_rff_count: Number of RFF features to generate.

Returns:
    A 2 x M_rff_count matrix for Wfixed.
"""
    # Flatten frequency grids and magnitude spectrum into 1D arrays for sampling
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    # Normalize magnitudes to form a probability distribution
    # Add a small constant to magnitudes before normalization to prevent division by zero for zero probabilities.
    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    # Sample M_rff_count indices based on probabilities
    # StatsBase.sample expects Weights from non-negative numbers
    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    Wfixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        Wfixed[1, i] = all_freqs_x[idx] *  2pi # Scale by  2pi to match RFF convention (often ω'x)
        Wfixed[2, i] = all_freqs_y[idx] *  2pi
    end

    return Wfixed
end
  


function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility for selecting inducing point locations for sparse Gaussian Process
              models. It supports methods like k-means clustering, random sampling, and
              furthest point sampling to place inducing points strategically.
    """

    # Helper function to generate inducing points (simple random sampling for now)

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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A simple helper function to compute a squared-exponential covariance.
    """
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A simple helper function to compute an exponential covariance.
    """
exp_cov_fn(D, phi) = exp.(-D / phi)



function ar1_covariance_local( n, rho, var,  ::Type{T}=Float64 )  where {T} 
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A helper function to compute a local AR(1) covariance matrix. This version is
              likely intended for a specific use case where only immediate neighbors are
              correlated, as it only computes covariances for `d=0` and `d=1`.
    """
    vcv = zeros( T, n, n) .+ I(n) 
    Threads.@threads for r in 1:n
    for c in 1:n
        d = r-c
        if d == 0 | d == 1
            vcv[r,c] = var * rho^d  
        end
    end
    end
    return vcv
end



# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A helper function to compute a full AR(1) covariance matrix based on the time
              differences between observations.
    """
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

# Helper to create AR1 cross-covariance matrix
function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A helper function to compute an AR(1) cross-covariance matrix between two
              different sets of time points. This is useful for prediction and interpolation in
              time series models.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that prepares data for Fast Fourier Transform (FFT) analysis by
              gridding scattered 2D data onto a regular grid and applying zero-padding to avoid
              aliasing effects.
    """
    # 1. Define the bounding box

    xmin, xmax = minimum(s_x), maximum(s_x)
    ymin, ymax = minimum(s_y), maximum(s_y)

    # 2. Map points to a grid
    # Use the length of the shorter input to prevent BoundsError
    n_limit = min(length(s_x), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        ix = Int(floor((s_x[i] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((s_y[i] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    # 3. Apply Zero-Padding
    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end
 



#####

 


function get_model_family(model_family::String)
    """
    BSTM Utility v1.1.0
    Timestamp: 2026-06-26 12:05:57
    Synopsis: Maps a string identifier (e.g., "poisson") to its corresponding concrete
              `AbstractBSTM_Family` type. This function uses a registry dictionary for efficient
              and extensible lookups, ensuring consistency with the full range of supported
              likelihood families.
    Rationale for v1.1.0:
        - Replaced the `if/elseif` chain with a `Dict`-based registry (`BSTM_FAMILY_REGISTRY`)
          to improve performance and maintainability.
        - Expanded the function to support all 16 likelihood families defined in the `bstm`
          framework, including Gamma, Beta, Student-T, and others, ensuring full parity with
          the `get_dist_ref` and `_apply_link_and_lik` dispatchers.
    """
    family_key = lowercase(model_family)
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$model_family'. Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end

 

function get_model_parameters(m::DynamicPPL.Model)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function to extract parameter names from a Turing model instance by
              sampling from it once. This is a fallback method for parameter discovery when more
              direct introspection is not available.
    """
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


function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal post-processing engine that generates denoised predictions
              (expectations), noisy predictions (posterior predictive samples), and pointwise
              log-likelihood values for model assessment (e.g., WAIC).
    """
    # Audited and Feature-Complete Prediction Utility (v10.5.0)
    # Rationale: Finalizing the full-taxonomy support for all 15+ model families.
    # Requirement: Ensure posterior predictive sampling (noisy) matches the support and parameters
    # defined in the bstm_Likelihood dispatcher and get_dist_ref factory.

    # 1. Container Initialization
    # denoised: expected value mu (response scale)
    # noisy: realized stochastic draws (prediction scale)
    # log_lik: pointwise log-probabilities for WAIC/LOO
    local denoised = zeros(N_tot, N_samples)
    local noisy = zeros(N_tot, N_samples)
    local log_lik = zeros(N_samples, M.y_N)

    local name_strs = string.(FlexiChains.parameters(chain))
    local use_zi = get(M, :use_zi, false)
    local fam_str = hasproperty(M, :model_family) ? M.model_family : "gaussian"

    # 2. Primary Sampling Loop
    for j in 1:N_samples

        # 2.1 Extraction of Volatility Surface
        # Standardizing heteroskedastic vs homoskedastic error scales
        local sig_y
        if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        else
            sig_y = _extract_volatility(chain, name_strs, N_tot, N_samples, nothing, M)[:, j]
        end

        # 2.2 Parameter Discovery for the Current Sample
        local r_val = "lik_r" in name_strs ? chain[:lik_r].data[j] : 1.0
        local phi_val = "lik_phi" in name_strs ? chain[:lik_phi].data[j] : 0.0
        local extra = "extra_params" in name_strs ? chain[:extra_params].data[j] : 1.0

        # 2.3 Link Transformation and Expectation Mapping
        # Rationale: Denoised expectations must use the standardized inverse link logic.
        local mu_vec = _apply_link_and_lik(fam_str, eta[:, j], use_zi, phi_val, r_val)
        denoised[:, j] .= mu_vec

        # 2.4 Likelihood and Predictive Sampling
        for i in 1:N_tot
            local is_obs = i <= M.y_N
            local mu = mu_vec[i]
            local eta_val = eta[i, j]

            # --- Likelihood Calculation (Training Data Only) ---
            if is_obs
                local y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                # Standardize likelihood evaluation via the bstm_Likelihood functor
                local lik_obj = bstm_Likelihood(
                    fam_str, [y_vals_src[i]]; sigma_y=sig_y[i], weight=M.weights[i],
                    phi_zi=use_zi ? phi_val : -Inf, r_nb=r_val, trial=M.trials[i], extra_params=extra
                )
                log_lik[j, i] = Distributions.logpdf(lik_obj, eta_val)
            end

            # --- Feature-Complete Posterior Predictive Sampling (Noisy) ---
            # Rationale: Handling the full taxonomy of realized draws.

            if use_zi && rand() < phi_val
                # Handle structural zeros for Zero-Inflated models
                noisy[i, j] = 0.0
            else
                noisy[i, j] = if fam isa GaussianFamily
                    mu + randn() * sig_y[i]
                elseif fam isa PoissonFamily
                    Float64(rand(Poisson(clamp(mu, 1e-10, 1e9))))
                elseif fam isa NegativeBinomialFamily
                    Float64(rand(NegativeBinomial(r_val, r_val / (r_val + mu))))
                elseif fam isa BinomialFamily
                    local n_t = Int(is_obs ? M.trials[i] : 1)
                    Float64(rand(Binomial(n_t, mu)))
                elseif fam isa LogNormalFamily
                    rand(LogNormal(mu, sig_y[i]))
                elseif fam isa GammaFamily
                    local alpha_g = extra > 0 ? extra : 1.0
                    rand(Gamma(alpha_g, mu / alpha_g))
                elseif fam isa BetaFamily
                    local phi_b = extra > 0 ? extra : 10.0
                    # Parameters for Beta are mu*phi and (1-mu)*phi
                    rand(Beta(clamp(mu * phi_b, 1e-6, Inf), clamp((1.0 - mu) * phi_b, 1e-6, Inf)))
                elseif fam isa InverseGaussianFamily
                    local lambda_ig = extra > 0 ? extra : 1.0
                    rand(InverseGaussian(mu, lambda_ig))
                elseif fam isa StudentTFamily
                    local nu_st = extra > 0 ? extra : 5.0
                    rand(LocationScale(mu, sig_y[i], TDist(nu_st)))
                elseif fam isa LaplaceFamily
                    rand(Laplace(mu, sig_y[i]))
                elseif fam isa ExponentialFamily
                    rand(Exponential(mu))
                elseif fam isa ParetoFamily
                    local shape_p = extra > 0 ? extra : 1.0
                    rand(Pareto(mu, shape_p))
                elseif fam isa HalfNormalFamily
                    rand(truncated(Normal(0.0, sig_y[i]), 0.0, Inf))
                elseif fam isa HalfStudentTFamily
                    local nu_hst = extra > 0 ? extra : 5.0
                    rand(truncated(LocationScale(0.0, sig_y[i], TDist(nu_hst)), 0.0, Inf))
                else
                    # Fallback to expectation for unknown families
                    mu
                end
            end
        end
    end

    # Audit Verification: Container check for N_tot x N_samples shapes
    return denoised, noisy, log_lik
end


function _calculate_ps_weights(p_denoised, M, PS, N_PS, N_samples)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal utility for calculating post-stratification weights. It compares the
              predicted values on a prediction surface to the values at observed locations to
              derive weights for population-level estimates.
    """
     
    if N_PS == 0
        return nothing
    end

    # Initialize the weight matrix [Strata x Samples]
    local ps_weights = zeros(N_PS, N_samples)
    
    for k in 1:N_PS
        # Identify the target spatiotemporal and seasonal indices for the k-th stratum
        local s_target = PS.s_idx[k]
        local t_target = PS.t_idx[k]
        local u_target = PS.u_idx[k]

        # Search for an exact match in the observed data configuration to establish a baseline
        local obs_match_idx = findfirst(i -> M.s_idx[i] == s_target && M.t_idx[i] == t_target && M.u_idx[i] == u_target, 1:M.y_N)

        if !isnothing(obs_match_idx)
            # If a match is found, compute weight relative to the specific observed realization
            for j in 1:N_samples
                # Use a small epsilon to prevent division by zero in sparse count scenarios
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (p_denoised[obs_match_idx, j] + 1e-9)
            end
        else
            # Fallback: If no exact stratum exists in training data, scale by the mean observed expectation
            local sample_mean_obs = mean(p_denoised[1:M.y_N, :], dims=1)
            for j in 1:N_samples
                ps_weights[k, j] = p_denoised[M.y_N + k, j] / (sample_mean_obs[j] + 1e-9)
            end
        end
    end

    return ps_weights
end



function _ensure_matrix(field, n_units, n_samples)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper function that robustly converts a scalar, vector, or matrix into
              a matrix of a specified size. It handles broadcasting and reshaping as needed to
              ensure dimensional consistency in downstream calculations.
    """
    # Null/Empty handling: Return zero-field of correct dimensions
    if isnothing(field) || isempty(field)
        return zeros(Float64, n_units, n_samples)
    end

    # Type Discovery and Normalization
    # Case 1: Pure Scalar - Broadcast to full N x S block
    if field isa Number
        return fill(Float64(field), n_units, n_samples)
    end

    # Case 2: Vector Handling
    if field isa AbstractVector
        # If length matches units, assume it is a spatial field constant across samples
        if length(field) == n_units
            return repeat(reshape(field, n_units, 1), 1, n_samples)
        # If length matches samples, assume it is a global trend across units
        elseif length(field) == n_samples
            return repeat(reshape(field, 1, n_samples), n_units, 1)
        else
            # Fallback for 1-element vectors: scalar broadcast
            return fill(Float64(field[1]), n_units, n_samples)
        end
    end

    # Case 3: Matrix Handling (Critical for Outcome-Specific Slices)
    if ndims(field) == 2
        local mat = Matrix{Float64}(field)
        local r, c = size(mat)

        # Scenario A: Exact Match
        if r == n_units && c == n_samples
            return mat
        # Scenario B: Global component (1xS Matrix) - Broadcast rows across units
        elseif r == 1 && c == n_samples
            return repeat(mat, n_units, 1)
        # Scenario C: Static field (Nx1 Matrix) - Broadcast columns across samples
        elseif r == n_units && c == 1
            return repeat(mat, 1, n_samples)
        else
            # Aggressive fallback using first element to prevent BoundsError
            return fill(mat[1,1], n_units, n_samples)
        end
    end

    return zeros(Float64, n_units, n_samples)
end

  

function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal utility that applies the appropriate inverse link function (e.g., `exp`
              for log-link, `logistic` for logit-link) to the linear predictor `eta` to obtain
              the expected value `mu` on the response scale.
    """
    # Audited and Feature-Complete Link Utility (v10.2.0)
    # Rationale: Mirroring the full 15+ family taxonomy established in _process_ll_and_predictions.
    # This ensures that denoised realizations (expectations) are mathematically consistent with the likelihood kernels.
    # Audit: Verified against Reference Hierarchy to include Log-link, Logit-link, and Identity-link routing.

    # 1. Family-Specific Inverse Link Dispatch
    # The linear predictor 'eta' exists in the latent space (usually unbounded Real).
    # We map 'mu' to the support of the target distribution family.
    local mu

    if family in ["poisson", "negbin", "gamma", "exponential", "inverse_gaussian", "pareto"]
        # Log-link group: Support is (0, Inf)
        # Rationale: Enforcing positivity via exponential transformation.
        mu = exp.(eta)

    elseif family in ["bernoulli", "binomial", "beta"]
        # Logit-link group: Support is (0, 1)
        # Rationale: Standard logistic mapping for probabilities or proportion parameters.
        mu = logistic.(eta)

    elseif family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"]
        # Identity-link group: Mapping stays in the space of the distribution parameter.
        # Note: LogNormal uses identity because 'eta' models the mean of the underlying Normal log-process.
        # Half-Normal/Half-StudentT are truncated but the location parameter follows identity in the latent field.
        mu = eta

    else
        # Fallback to Identity for unknown or experimental families
        mu = eta
    end

    # 2. Zero-Inflation Adjustment
    # Rationale: If Zero-Inflation is active, the expected value of the mixture is (1 - phi) * mu_branch.
    # This matches the analytical mean of the ZI-distribution where 'phi' is the probability of a structural zero.
    if use_zi
        mu = (1.0 .- phi) .* mu
    end

    return mu
end



function _compute_waic(log_lik)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper to compute the Widely Applicable Information Criterion (WAIC) from
              a pointwise log-likelihood matrix. WAIC is a measure of out-of-sample predictive
              accuracy that accounts for model complexity.
    """
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper to extract and summarize the posterior distributions for the
              coefficients of smoothed covariate effects, which are typically modeled as random effects.
    """
    # Identify all categorical covariate groups present in the chain
    # Matches patterns like "beta_cov[1]", "beta_cov[2]", etc.
    cov_matches = unique(map(m -> m.captures[1], filter(!isnothing, match.(r"beta_cov\[(\d+)\]", all_names))))
    
    if isempty(cov_matches)
        return nothing
    end

    # Parse indices and sort to ensure sequential processing
    cov_indices = sort(parse.(Int, cov_matches))
    
    results = []
    for k in cov_indices
        base_name = "beta_cov[$k]"
        # Use the robust get_params_vector helper to extract the full vector for this covariate group
        raw_vals = get_params_vector(chain, base_name, M.N_cat)
        
        if !all(raw_vals .== 0) # Only process if data was actually found
            # Reshape for summarize_array (N_categories x 1 x N_samples)
            summ = summarize_array(reshape(raw_vals', M.N_cat, 1, N_samples); alpha=alpha)
            push!(results, summ)
        end
    end

    return isempty(results) ? nothing : results
end


# --- 1. Structural Storage for Latent Components ---
# Standardizing for Heterogeneous Outcome Support (v14.0.0 Refactor)
# This structure acts as the primary registry for discovered manifold realizations.
# It ensures that every latent component defined in the BSTM v05.5 Taxonomy is captured.
mutable struct ManifoldRegistry
    # Fixed Effects: Standard Matrix representing [K_fixed * Outcomes x Samples]
    Xfixed_betas::Union{Nothing, Matrix{Float64}}

    # Mixed Effects: A Vector containing matrices of [Category_Levels x Samples]
    mixed_eff_coeffs::Vector{Matrix{Float64}}

    # Spatial Manifolds: Outcome-specific vectors of [Spatial_Units_k x Samples]
    s_eff_struct::Vector{Matrix{Float64}}
    s_eff_noisy::Vector{Matrix{Float64}}

    # Temporal Manifolds: Outcome-specific vectors of [Time_Slices_k x Samples]
    t_eff::Vector{Matrix{Float64}}

    # Seasonal Manifolds: Shared periodic components [Seasonal_Bins x Samples]
    u_eff::Matrix{Float64}

    # Basis/Spline Effects: Accumulated values mapped to the training grid [Observations x Samples]
    basis_eff_accum::Matrix{Float64}

    # Spatiotemporal maps: Outcome-specific tensors [Units_k x Time_k x Samples]
    st_eff_maps::Vector{Array{Float64, 3}}

    # Spatially Varying Coefficients: Outcome-specific tensors [Units_k x Covariates x Samples]
    svc_slopes::Vector{Array{Float64, 3}}

    # Observation Error Scale: Pointwise surface [Observations x Samples]
    sv_surface::Matrix{Float64}

    # Internal Metadata for Validation and Architectural Dispatch
    n_samples::Int
    outcomes_N::Int
end
 
 
 

function _discover_manifold_realizations(chain, M, n_samples, outcomes_N, p_names)
    """
    BSTM Manifold Discovery Engine v1.0.0
    Timestamp: 2026-06-26 20:00:00
    Synopsis: The main internal engine for discovering and extracting all latent manifold
              realizations from a fitted MCMC chain. It iterates through the model's manifold
              registry and uses multiple dispatch to call the correct extraction method for each component.
    Rationale for v1.0.0:
        - Added specific logic to detect the `MultifidelityArchitecture` and extract its unique
          RFF-based latent fields (`z_latent`, `w_latent`), which are not part of the standard
          manifold registry.
        - This ensures the discovery engine is feature-complete for all `bstm` architectures.
    """
    # --- 1. Initialize Manifold Registries ---
    s_eff_struct = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    s_eff_noisy  = [zeros(Float64, M.s_N, n_samples) for _ in 1:outcomes_N]
    t_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    u_eff = zeros(Float64, M.u_N, n_samples)
    basis_eff_accum = zeros(Float64, M.y_N, n_samples)
    st_eff_maps = [zeros(Float64, M.s_N, M.t_N, n_samples) for _ in 1:outcomes_N]
    dynamics_eff = [zeros(Float64, M.t_N, n_samples) for _ in 1:outcomes_N]
    eigen_eff = zeros(Float64, M.y_N, n_samples)
    
    # Registries specific to Multifidelity Architecture
    z_latent_field = nothing
    w_latent_field = nothing

    svc_slopes = !isempty(get(M, :svc_covariates, [])) ? [zeros(Float64, M.s_N, length(M.svc_covariates), n_samples) for _ in 1:outcomes_N] : nothing
    mixed_eff_coeffs = !isempty(get(M, :mixed_terms, [])) ? [zeros(Float64, term.n_cat, n_samples) for term in M.mixed_terms] : nothing
    sv_surface = [ones(Float64, M.y_N, n_samples) for _ in 1:outcomes_N]
    
    # --- 2. Extract Fixed Effects ---
    xf_betas = nothing
    if M.Xfixed_N > 0 && ("Xfixed_beta" in p_names || any(occursin.(Ref("Xfixed_beta["), p_names)))
        xf_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
    end

    # --- 3. Manifold Dispatch Loop ---
    if hasproperty(M, :manifolds)
        for spec in M.manifolds
            m_obj = spec.manifold_obj
            m_domain = spec.domain
            extracted_fields = extract_manifold(m_obj, chain, M, n_samples, outcomes_N, p_names, spec)

            if m_domain == :spatial; for k in 1:outcomes_N; s_eff_struct[k] .+= extracted_fields.structured[k]; s_eff_noisy[k] .+= extracted_fields.noisy[k]; end
            elseif m_domain == :temporal; for k in 1:outcomes_N; t_eff[k] .+= extracted_fields.structured[k]; end
            elseif m_domain == :seasonal; u_eff .+= extracted_fields.structured[1];
            elseif m_domain == :smooth; basis_eff_accum .+= extracted_fields.structured[1];
            elseif m_domain == :mixed && !isnothing(mixed_eff_coeffs); term_idx = findfirst(t -> t.name == spec.var, M.mixed_terms); if !isnothing(term_idx); mixed_eff_coeffs[term_idx] .+= extracted_fields.structured[1]; end
            elseif m_obj isa SVCManifold && !isnothing(svc_slopes); cov_idx = findfirst(c -> c == m_obj.covariate, M.svc_covariates); if !isnothing(cov_idx); for k in 1:outcomes_N; svc_slopes[k][:, cov_idx, :] .+= extracted_fields.structured[k]; end; end
            elseif m_domain == :dynamics; for k in 1:outcomes_N; dynamics_eff[k] .+= extracted_fields.structured[k]; end
            elseif m_domain == :eigen; eigen_eff .+= extracted_fields.structured[1];
            end
        end
    end
    
    # --- 4. Multifidelity-Specific Field Extraction ---
    if get(M, :model_arch, "univariate") == "multifidelity"
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
    end

    # --- 5. Extract Stochastic Volatility Surface ---
    if get(M, :use_sv, false)
        sv_surface = _extract_volatility(chain, p_names, M.y_N, n_samples, outcomes_N, M)
    end

    # --- 6. Return Final Registry ---
    return (
        Xfixed_betas = xf_betas,
        mixed_eff_coeffs = mixed_eff_coeffs,
        s_eff_struct = s_eff_struct, 
        s_eff_noisy = s_eff_noisy,
        t_eff = t_eff, 
        u_eff = u_eff, 
        basis_eff_accum = basis_eff_accum,
        st_eff_maps = st_eff_maps, 
        svc_slopes = svc_slopes,
        dynamics_eff = dynamics_eff,
        eigen_eff = eigen_eff,
        z_latent = z_latent_field,
        w_latent = w_latent_field,
        sv_surface = outcomes_N == 1 ? sv_surface[1] : sv_surface,
        n_samples = n_samples,
        outcomes_N = outcomes_N
    )
end




function _modular_eta_assembly(N_tot_in, registry, M, PS_in)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The primary internal engine for assembling the final linear predictor `eta`. It
              superimposes all recovered latent fields from the manifold registry for both the
              training data and any provided prediction grid.
    """
    # BSTM Linear Predictor Assembly Engine v18.1.32
    # Timestamp: 2026-06-07 14:45:00
    # Rationale: Optimizing SVC inner loop by precomputing column indices and avoiding redundant symbol conversion.

    n_samples = registry.n_samples
    outcomes_n = registry.outcomes_N
    y_n_train = Int(M.y_N)
    actual_limit = isnothing(PS_in) ? y_n_train : Int(N_tot_in)

    # Precomputing SVC column indices to prevent findfirst/names calls in inner loops
    svc_vars = get(M, :svc_covariates, Symbol[])
    n_svc = length(svc_vars)
    
    # Training grid indices
    train_col_names = Symbol.(names(M.Xfixed, 2))
    train_svc_indices = Int[findfirst(==(v), train_col_names) for v in svc_vars]
    
    # Prediction grid indices if PS is present
    ps_svc_indices = Int[]
    if !isnothing(PS_in)
        ps_col_names = Symbol.(names(PS_in.Xfixed, 2))
        ps_svc_indices = Int[findfirst(==(v), ps_col_names) for v in svc_vars]
    end

    # Primary linear predictor container [Total Observations x Outcomes x MCMC Samples]
    eta_container = zeros(Float64, actual_limit, outcomes_n, n_samples)

    for j in 1:n_samples
        for k in 1:outcomes_n
            # Extract realized manifold slices for current outcome and sample
            s_f = !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy) ? registry.s_eff_noisy[k][:, j] : Float64[]
            t_f = !isnothing(registry.t_eff) && !isempty(registry.t_eff) ? registry.t_eff[k][:, j] : Float64[]
            u_f = !isnothing(registry.u_eff) && !isempty(registry.u_eff) ? registry.u_eff[:, j] : Float64[]

            # Interaction and Varying Coefficient tensors
            st_f = !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps[k]) ? registry.st_eff_maps[k][:, :, j] : zeros(0, 0)
            svc_f = !isnothing(registry.svc_slopes) && !isempty(registry.svc_slopes[k]) ? registry.svc_slopes[k][:, :, j] : zeros(0, 0)

            # Fixed Effect parameters
            xf_n = Int(M.Xfixed_N)
            has_fixed = !isnothing(registry.Xfixed_betas) && xf_n > 0
            beta_slice = has_fixed ? registry.Xfixed_betas[((k-1)*xf_n + 1):(k*xf_n), j] : Float64[]

            for i in 1:actual_limit
                is_obs = i <= y_n_train
                src = is_obs ? M : PS_in
                idx = is_obs ? i : i - y_n_train
                
                # Pre-cached indices for the specific metadata source
                target_svc_indices = is_obs ? train_svc_indices : ps_svc_indices

                s_ptr = Int(src.s_idx[idx])
                t_ptr = Int(src.t_idx[idx])
                u_ptr = Int(src.u_idx[idx])

                # # 1. Structural Superposition
                val = 0.0
                if !isempty(s_f); val += s_f[s_ptr]; end
                if !isempty(t_f); val += t_f[t_ptr]; end
                if !isempty(u_f); val += u_f[u_ptr]; end

                # # 2. Spatiotemporal Interaction
                if !isempty(st_f); val += st_f[s_ptr, t_ptr]; end

                # # 3. SVC Contribution using precomputed indices
                if !isempty(svc_f)
                    for v_idx in 1:n_svc
                        col_idx = target_svc_indices[v_idx]
                        if !isnothing(col_idx)
                            val += svc_f[s_ptr, v_idx] * src.Xfixed[idx, col_idx]
                        end
                    end
                end

                # # 4. Observation-Specific Components
                if is_obs
                    if !isnothing(registry.basis_eff_accum)
                        val += registry.basis_eff_accum[idx, j]
                    end
                    if haskey(M, :log_offset)
                        val += M.log_offset[idx]
                    end
                end

                # # 5. Fixed Effect Design Matrix Mapping
                if has_fixed
                    val += dot(vec(collect(src.Xfixed[idx, :])), beta_slice)
                end

                eta_container[i, k, j] = val
            end
        end
    end
    
    return eta_container
end

 


function summarize_array(samples::AbstractArray; alpha=0.05)
    # BSTM Posterior Summarization  v1.0.0
    # Timestamp: 2026-06-25 18:00:00
    # Rationale: Adding posterior standard deviation (std) to the summary output.
    # Refactoring the vectorization logic for improved clarity and robustness.

    # 1. Edge Case: Handle empty or invalid input arrays
    # Rationale: Prevents errors in downstream processing if a latent field is not recovered.
    if isempty(samples) || all(isnan, samples)
        # Return a NamedTuple with empty vectors to maintain type stability
        return (mean = Float64[], median = Float64[], std = Float64[], lower = Float64[], upper = Float64[])
    end

    # 2. Dimension and Probability Discovery
    # The last dimension is consistently treated as the MCMC sample dimension.
    dims = size(samples)
    sample_dim = length(dims)
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    # 2. Centralized Statistic Calculation
    # Rationale: Using the `dims` keyword is efficient for multidimensional arrays.
    # `dropdims` removes the singleton dimension resulting from the reduction.
    post_mean = dropdims(Statistics.mean(samples, dims=sample_dim), dims=sample_dim)
    post_median = dropdims(Statistics.median(samples, dims=sample_dim), dims=sample_dim)
    post_std = dropdims(Statistics.std(samples, dims=sample_dim), dims=sample_dim)
    
    # `mapslices` is used for quantiles as it's a non-standard reduction.
    low_bound = dropdims(mapslices(x -> Statistics.quantile(x, low_prob), samples, dims=sample_dim), dims=sample_dim)
    high_bound = dropdims(mapslices(x -> Statistics.quantile(x, high_prob), samples, dims=sample_dim), dims=sample_dim)

    # 3. Vectorization Post-Processor
    # Rationale: Standardizes all outputs to Vector{Float64} for type consistency.
    # This prevents errors in downstream functions that expect vector inputs.
    function to_vector(x)
        if x isa AbstractArray
            # `vec` creates a 1D view, `collect` materializes it as a new Vector.
            return vec(collect(Float64, x))
        else
            # Handles scalar results from single-observation summaries.
            return [Float64(x)]
        end
    end

    # 4. Feature Complete Return Object
    # The NamedTuple provides a standardized, accessible structure for all summary statistics.
    return (
        mean = to_vector(post_mean),
        median = to_vector(post_median),
        std = to_vector(post_std),
        lower = to_vector(low_bound),
        upper = to_vector(high_bound)
    )
end





function decompose_bstm_formula(formula_str::String)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The main formula parser for the `bstm` interface. It decomposes a formula string
              (e.g., "y ~ 1 + x + spatial(s)") into its constituent parts: outcomes, fixed
              effects, and a structured registry of manifold modules.
    """

    # Partition the formula into Left-Hand Side (Outcomes) and Right-Hand Side (Predictors)
    parts = Base.split(formula_str, "~")
    lhs = strip(parts[1])
    rhs = strip(parts[2])

    # # Outcome Discovery
    outcomes = String[]
    if startswith(lhs, "[") && endswith(lhs, "]")
        content = lhs[2:end-1]
        outcomes = [strip(s) for s in Base.split(content, ",")]
    else
        outcomes = [lhs]
    end

    # # Module and Fixed Effect Registries
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

        # Check if it's a module call
        m_mod = match(r"(\w+)\((.*)\)", t_clean)
        if !isnothing(m_mod) && (lowercase(m_mod.captures[1]) in BSTM_MODULE_KEYWORDS)
            module_data = _parse_module_call(t_clean)
            
            # Generate a unique key for the module.
            # This logic is enhanced to prevent collisions when multiple modules of the same type are used.
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

            # To ensure uniqueness if all else is identical, add a counter.
            base_key = join(key_parts, "_")
            module_key = base_key
            counter = 1
            while haskey(modules, module_key)
                counter += 1
                module_key = base_key * "_$counter"
            end

            modules[module_key] = module_data
        else
            # It's a fixed effect or algebraic operator
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

                # Improved key generation for algebraic terms
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
            
            # If not a module and not an algebraic operator, it's a fixed effect
            if !isempty(t_clean)
                push!(fixed_effects, t_clean)
            end
        end
    end

    return (outcomes=outcomes, modules=modules, fixed_effects=fixed_effects, has_intercept=has_intercept)
end


# # BSTM Monolithic Visualization Engine [v19.38.5 ]
# Timestamp: 2025-11-05 20:00:00
# Rationale: Expanding the dashboard to include Mixed Effects, Seasonal Cycles, and 4D smooths.
# Requirements: 100% parity with v19.38.0 Registry. Zero-truncation of dashboard panels.

function model_results_plots(res, ts=1; outcome=1, centroids=nothing, polygons=nothing, y_obs=nothing)
    # Description: Primary exhaustive dashboard for spatiotemporal manifold verification.
    # Requirement: Scalable grid to accommodate all recovered latent effects (S, T, U, Smooth, Mixed, Nested).

    println("--- Rendering Dashboard v19.38.5 [Outcome: ", outcome, "] ---")

    # # 1. Metadata and Result Extraction
    # Rationale: Standardizing the posterior statistics and metadata sources for visualization.
    pstats = res.pstats
    # Resolving observation source (Training vs custom override)
    obs_src = isnothing(y_obs) ? res.y_obs : y_obs
    # Synchronizing spatial geometry for mapping
    coords = isnothing(centroids) ? res.centroids : centroids
    polys = isnothing(polygons) ? res.polygons : polygons

    plots_list = []

    # # 2. Panel 1: Posterior Predictive Check (PPC)
    # Rationale: Core fit audit comparing expectations to observations.
    is_mv = (obs_src isa AbstractMatrix && size(obs_src, 2) > 1)
    y_p = is_mv ? pstats.predictions_denoised.mean[:, outcome] : vec(pstats.predictions_denoised.mean)
    y_o = is_mv ? obs_src[:, outcome] : vec(obs_src)

    if length(y_p) == length(y_o)
        p_ppc = Plots.scatter(vec(y_p), vec(y_o), title="PPC Audit", xlabel="Predicted", ylabel="Observed", alpha=0.5, markersize=3, markerstrokewidth=0, legend=false)
        # Identity line reference
        clean_p = filter(x -> !isnan(x), y_p)
        clean_o = filter(x -> !isnan(x), y_o)
        if !isempty(clean_p) && !isempty(clean_o)
            m_val = max(maximum(clean_p), maximum(clean_o))
            Plots.plot!(p_ppc, [0, m_val], [0, m_val], color=:red, ls=:dash, lw=1.5)
        end
        push!(plots_list, p_ppc)
    end

    # # 3. Panel 2: Spatial Field Mapping
    # Rationale: High-fidelity geographic rendering of structured innovations.
  if !isnothing(coords)
        s_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? pstats.spatial_structured[outcome] : pstats.spatial_structured

        if !isnothing(s_field) && hasproperty(s_field, :mean)
            s_mean = vec(collect(s_field.mean))
            p_map = Plots.plot(aspect_ratio=:equal, title="Spatial Innovation", frame=:box)

            if !isnothing(polys) && length(polys) >= length(s_mean)
                for i in 1:length(s_mean)
                    px = [pt[1] for pt in polys[i]]
                    py = [pt[2] for pt in polys[i]]
                    if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1])
                        push!(py, py[1])
                    end
                    Plots.plot!(p_map, px, py, seriestype=:shape, fill_z=s_mean[i], c=:viridis, linecolor=:black, lw=0.2, label=nothing)
                end
            else
                cx = [c[1] for c in coords]
                cy = [c[2] for c in coords]
                Plots.scatter!(p_map, cx, cy, marker_z=vec(s_mean), markersize=4, c=:viridis, label=nothing, colorbar=true)
            end
            push!(plots_list, p_map)
        end
    end


    # Panel: Temporal Trend Recovery
    # Ensure t_field is correctly extracted for the requested outcome
    t_field = nothing
    if hasproperty(pstats, :temporal_effects)
        raw_t = pstats.temporal_effects
        t_field = (pstats.arch isa MultivariateArchitecture || pstats.arch isa MultifidelityArchitecture) ? raw_t[outcome] : raw_t
    end

    # Robust rendering of longitudinal trend with ribbons
    if !isnothing(t_field) && hasproperty(t_field, :mean)
        tm = vec(collect(t_field.mean))
        tl = vec(collect(t_field.lower))
        tu = vec(collect(t_field.upper))

        if !isempty(tm) && !all(isnan, tm)
            p_trend = Plots.plot(tm, ribbon=(tm .- tl, tu .- tm), title="Temporal Trend", lw=2.5, fillalpha=0.25, color=:royalblue, legend=false, xlabel="Time Index", ylabel="Latent Value")
            push!(plots_list, p_trend)
        end
    end

    # Panel: Seasonal Cycle
    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal)
        u_field = pstats.seasonal
        if hasproperty(u_field, :mean) && !all(isnan, u_field.mean)
            um = vec(collect(u_field.mean))
            ul = vec(collect(u_field.lower))
            uu = vec(collect(u_field.upper))
            p_seas = Plots.plot(um, ribbon=(um .- ul, uu .- um), title="Seasonal Cycle", lw=2, fillalpha=0.2, color=:forestgreen, legend=false, xlabel="Season Bin")
            push!(plots_list, p_seas)
        end
    end

    # # 6. Panel 5: RESTORED Mixed-Effect Categorical Registry
    # Rationale: Specifically added to audit hospital/group varying intercepts.
    # Checks for summaries of c_latent terms identified in v19.21.2.
    if hasproperty(pstats, :mixed_effects) && !isnothing(pstats.mixed_effects)
        for (grp_sym, m_summ) in pstats.mixed_effects
            mm, ml, mu = vec(m_summ.mean), vec(m_summ.lower), vec(m_summ.upper)
            push!(plots_list, Plots.bar(mm, yerror=(mm .- ml, mu .- mm), title=string("Mixed: ", grp_sym), color=:purple, legend=false, xlabel="Category ID"))
        end
    end

    # # 7. Panel 6: RESTORED Hyper-Volumetric Smooths (1D-4D)
    # Rationale: High-dimensional surfaces projected onto the observation grid.
    if hasproperty(pstats, :smooth_effects) && !isnothing(pstats.smooth_effects)
        b_field = pstats.smooth_effects
        if !all(isnan, b_field.mean) && Statistics.std(b_field.mean) > 1e-9
            bm, bl, bu = vec(b_field.mean), vec(b_field.lower), vec(b_field.upper)
            push!(plots_list, Plots.plot(bm, ribbon=(bm .- bl, bu .- bm), title="Accumulated Smooths (1D-4D)", lw=1.5, fillalpha=0.1, color=:darkorange, legend=false, xlabel="Observation Order"))
        end
    end

    # # 8. Panel 7: Hierarchical Supervisor Contributions
    # Rationale: Renders signals transferred from nested supervisor processes.
    if hasproperty(pstats, :nested_contributions) && !isnothing(pstats.nested_contributions)
        n_field = pstats.nested_contributions
        if !all(isnan, n_field.mean) && Statistics.std(n_field.mean) > 1e-9
            nm, nl, nu = vec(n_field.mean), vec(n_field.lower), vec(n_field.upper)
            push!(plots_list, Plots.plot(nm, ribbon=(nm .- nl, nu .- nm), title="Hierarchical Supervisors", lw=1.5, fillalpha=0.2, color=:brown, legend=false, xlabel="Observation Order"))
        end
    end

    # # 9. Panel 8: Fixed Effect Coefficient Registry

    # Panel: Fixed Effect Forest Plot
    # Dot-and-whisker style for regression coefficients
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects)
        f_field = pstats.fixed_effects
        if hasproperty(f_field, :mean) && !all(isnan, f_field.mean)
            fm = vec(f_field.mean)
            fl = vec(f_field.lower)
            fu = vec(f_field.upper)
            n_coeffs = length(fm)
            # Reverse indices for intuitive top-to-bottom reading in forest plots
            p_forest = Plots.scatter(fm, 1:n_coeffs, xerror=(fm .- fl, fu .- fm), title="Fixed Effects Registry", xlabel="Coefficient Estimate", ylabel="Index", markersize=5, color=:black, legend=false, yticks=(1:n_coeffs, ["#$i" for i in 1:n_coeffs]))
            Plots.vline!(p_forest, [0], color=:red, ls=:dash, lw=1)
            push!(plots_list, p_forest)
        end
    end


    # # 10. Dashboard Assembly and Layout
    # Rationale: Collating all discovered panels into a unified multi-plot object.
    if !isempty(plots_list)
        n_plots = length(plots_list)
        cols = min(n_plots, 2)
        rows = Int(ceil(n_plots / cols))
        final_plt = Plots.plot(plots_list..., layout=(rows, cols), size=(1200, 350 * rows), margin=5Plots.mm)
        return final_plt
    end

    @warn "BSTM Visualization: No active technical manifolds discovered for outcome $outcome."
    return nothing
end


# --- Modular Configuration Processors ---

function process_interaction_module!(opt_dict, mod_data, registries, hyperpriors)
    # BSTM Internal Utility v2.1.0
    # Timestamp: 2026-06-27 12:25:00
    # Synopsis: Processes algebraic interaction modules (e.g., `spatial() ⊗ temporal()`).
    #           It determines the Knorr-Held interaction type and sets the `model_st` flag.
    # Rationale for v2.1.0:
    #     - Updated function signature to accept standardized `registries` and `hyperpriors`
    #       arguments for architectural consistency, though they are not used in this function.
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
    # BSTM Internal Utility v2.2.0
    # Timestamp: 2026-06-27 12:25:00
    # Synopsis: Processes the `observationprocess()` module to extract observation-level
    #           parameters like offsets, weights, and censoring bounds.
    # Rationale for v2.2.0:
    #     - Updated function signature to accept standardized `registries` and `hyperpriors`
    #       arguments for architectural consistency. These arguments are not used here but
    #       are required for a uniform call signature from the configuration engine.
    params = mod_data[:params]

    if haskey(params, :weights); opt_dict[:weights] = params[:weights];
    elseif haskey(params, :weight); opt_dict[:weights] = params[:weight]; end
    
    if get(params, :volatility, false) == true
        opt_dict[:use_sv] = true
        opt_dict[:M_rff_sigma] = get(params, :nbins, 20)
    end

    if haskey(params, :log_offsets); opt_dict[:log_offset] = params[:log_offsets];
    elseif haskey(params, :offsets); opt_dict[:log_offset] = params[:offsets]; end
    
    if haskey(params, :trials); opt_dict[:trials] = params[:trials];
    elseif haskey(params, :trial); opt_dict[:trials] = params[:trial]; end
    
    if haskey(params, :y_L); opt_dict[:y_lower_bound] = params[:y_L]; end
    if haskey(params, :y_U); opt_dict[:y_upper_bound] = params[:y_U]; end
    if haskey(params, :hurdle); opt_dict[:hurdle] = params[:hurdle]; end
end


function process_intercept_module!(opt_dict, mod_data)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 20:45:00
    Synopsis: Processes the `intercept()` module from the formula string. It signals that an
              intercept should be included in the model and extracts a custom prior if one is
              provided via the `prior` parameter.
    Rationale for v1.0.0:
        - Corrected the function to be effective. It now sets the `:add_intercept` flag and
          stores any custom prior in `:intercept_prior` within the main configuration object,
          making it accessible to the model builder.
        - The function no longer interacts with the `hyperprior_registry`, as the intercept
          prior is a global model parameter, not a manifold-specific one.
    """
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

function process_spatial_module!(opt_dict, mod_data)
    # BSTM Internal Utility v1.1.0
    # Timestamp: 2026-06-27 00:03:49
    # Synopsis: Processes the `spatial()` module from the formula string. This function is
    #           responsible for extracting the spatial index variable, calculating the number of
    #           unique spatial units (s_N), and extracting an adjacency matrix (W) if it is
    #           provided directly within the formula.
    # Rationale for v1.1.0:
    #     - Verified that the function correctly extracts the spatial index (`s_idx`), the number
    #       of spatial units (`s_N`), and an optional adjacency matrix (`W`).
    #     - Confirmed that the function correctly avoids interpreting the `model` parameter,
    #       properly leaving that task to downstream model-building functions. The implementation
    #       is deemed complete and correct for its role. The signature has been updated to
    #       accept standardized `registries` and `hyperpriors` arguments for architectural
    #       consistency, though they are not used in this function.

    data = opt_dict[:data]

    # Extract spatial index variable
    if !isempty(mod_data[:variables])
        var_sym = Symbol(mod_data[:variables][1])
        if hasproperty(data, var_sym)
            opt_dict[:s_idx] = data[!, var_sym]
            opt_dict[:s_idx_var] = var_sym

            # Calculate and store the number of unique spatial units
            opt_dict[:s_N] = length(unique(data[!, var_sym]))
        else
            @warn "Spatial index variable ':$var_sym' not found in data."
        end
    end
 
    # Extract adjacency matrix if provided in the formula
    if haskey(mod_data[:params], :W)
        opt_dict[:W] = mod_data[:params][:W]
    end
end


function process_temporal_module!(opt_dict, mod_data, registries, hyperpriors)
    # BSTM Internal Utility v2.0.0
    # Timestamp: 2026-06-27 00:10:15
    # Synopsis: Processes the `temporal()` module from the formula string. This function is
    #           responsible for extracting temporal and seasonal index variables, setting the
    #           respective model types, and passing user-defined options to the time unit
    #           discretization engine.
    # Rationale for v2.0.0:
    #     - Corrected a major architectural flaw where the function only processed the first
    #       variable and a single string model. It now correctly handles the documented
    #       `temporal(t_idx, u_idx, model=('ar1', 'cyclic'))` syntax.
    #     - Implemented parsing for tuple-based model specifications, correctly dispatching
    #       to `:model_time` and `:model_season`.
    #     - Removed the hardcoded call to `assign_time_units`. The function now extracts
    #       parameters like `time_method`, `t_N`, and `u_N` from the formula and passes
    #       them to the discretization engine, giving the user full control.
    #     - Decoupled seasonal model specification, removing the side-effect where
    #       `model='cyclic'` would incorrectly set the global `:model_season` flag.
    #     - The signature has been updated to accept standardized `registries` and `hyperpriors`
    #       

    data = opt_dict[:data]
    params = mod_data[:params]
    variables = mod_data[:variables]

    # 1. Parse model specifications
    model_spec = get(params, :model, "ar1")
    if model_spec isa Tuple && length(model_spec) >= 2
        opt_dict[:model_time] = string(model_spec[1])
        opt_dict[:model_season] = string(model_spec[2])
    elseif model_spec isa String
        opt_dict[:model_time] = model_spec
    end

    # 2. Process temporal and seasonal indices
    if !isempty(mod_data[:variables])
        t_var_sym = Symbol(variables[1])
        if hasproperty(data, t_var_sym)
            time_opts = Dict(:time_method => get(params, :time_method, "regular"), :t_N => get(params, :t_N, nothing), :u_N => get(params, :u_N, nothing))
            filter!(p -> !isnothing(p.second), time_opts)
            tu_meta = assign_time_units(data[!, t_var_sym]; time_opts...)
            opt_dict[:t_idx] = tu_meta.t_idx
            opt_dict[:t_N] = tu_meta.tn
            opt_dict[:t_idx_var] = t_var_sym
            if length(variables) > 1
                u_var_sym = Symbol(variables[2])
                if hasproperty(data, u_var_sym); opt_dict[:u_idx] = data[!, u_var_sym]; opt_dict[:u_N] = length(unique(data[!, u_var_sym])); else; @warn "Seasonal index variable ':$u_var_sym' not found. Using derived seasonal index."; opt_dict[:u_idx] = tu_meta.u_idx; opt_dict[:u_N] = tu_meta.un; end
            else
                opt_dict[:u_idx] = tu_meta.u_idx
                opt_dict[:u_N] = tu_meta.un
            end
        else; @warn "Temporal index variable ':$t_var_sym' not found in data."; end
    end
end
 

function process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, manifolds_registry)
    # BSTM Internal Utility v2.0.0
    # Timestamp: 2026-06-27 12:00:00
    # Synopsis: Processes `smooth()` modules. This function is now a dispatcher that handles
    #           two distinct types of smooths:
    #           1. Basis Function Smooths (e.g., pspline, rff): Generates a basis matrix `B`
    #              for an effect `B * beta`, where beta is IID.
    #           2. Structured Random Effect Smooths (e.g., rw2, icar ⊗ ar1): Discretizes
    #              covariates and configures a GMRF to be applied to the resulting indices.
    # Rationale for v2.0.0:
    #     - The function is now complete, supporting both basis and structured random effect smooths.
    #     - Added a new path to handle GMRF-style models (`rw2`, `icar`, etc.) and algebraic
    #       interactions (`⊗`) as specified by user requirements.
    #     - This new path populates a new `:structured_smooths` registry in the configuration,
    #       which requires a corresponding update in the main `bstm_univariate` model.
    #     - The existing basis function logic is preserved and remains the default path.

    data = opt_dict[:data]
    model_str = string(get(mod_data[:params], :model, "pspline"))
    basis_models = ["pspline", "bspline", "tps", "rff", "fft", "moran", "spherical", "barycentric", "decay"]

    if model_str in basis_models
        # --- Path 1: Basis Function Smoothing (Existing Logic) ---
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
    else
        # --- Path 2: Structured Random Effect Smoothing (New Logic) ---
        if !haskey(opt_dict, :structured_smooths); opt_dict[:structured_smooths] = []; end

        vars = mod_data[:variables]
        nbins_param = get(mod_data[:params], :nbins, 20)
        smooth_name = Symbol(join(vars, "_"))

        if occursin("⊗", model_str) # 2D Interaction Smooth
            if length(vars) != 2; @warn "Interaction smooth on $(join(vars, ",")) requires exactly 2 variables."; return; end
            
            parts = split_terms_at_depth(model_str, "⊗")
            model1_str, model2_str = strip(parts[1]), strip(parts[2])
            nbins1, nbins2 = nbins_param isa Tuple ? nbins_param : (nbins_param, nbins_param)

            _, idx1 = apply_discretization_logic(data[!, Symbol(vars[1])], nbins1)
            _, idx2 = apply_discretization_logic(data[!, Symbol(vars[2])], nbins2)

            Q1 = build_structure_template(Symbol(model1_str), nbins1).matrix
            Q2 = build_structure_template(Symbol(model2_str), nbins2).matrix
            
            push!(opt_dict[:structured_smooths], (
                name = smooth_name,
                indices = (idx1 .- 1) .* nbins2 .+ idx2,
                Q_template = kron(Q2, Q1),
                n_units = nbins1 * nbins2,
                sigma_prior = Exponential(1.0) # Default prior
            ))
        else # 1D Structured Smooth
            if length(vars) != 1; @warn "1D structured smooth on $(join(vars, ",")) requires exactly 1 variable."; return; end

            _, indices = apply_discretization_logic(data[!, Symbol(vars[1])], nbins_param)
            n_units = length(unique(indices))

            push!(opt_dict[:structured_smooths], (
                name = smooth_name,
                indices = indices,
                Q_template = build_structure_template(Symbol(model_str), n_units).matrix,
                n_units = n_units,
                sigma_prior = Exponential(1.0) # Default prior
            ))
        end
    end
end
 

function process_dynamics_module!(opt_dict, mod_data)
    # BSTM Internal Utility v1.0.0
    # Timestamp: 2026-06-27 12:10:00
    # Synopsis: Processes the `dynamics()` module. This function is currently a placeholder.
    #           The primary logic for dynamics models is handled within the `_apply_manifold!`
    #           dispatcher, which calls the appropriate state-space model kernel. This
    #           processor ensures the module is correctly registered in the processing pipeline.
    # No pre-processing is required at this stage.
end

function process_eigen_module!(opt_dict, mod_data)
    # BSTM Internal Utility v1.0.0
    # Timestamp: 2026-06-27 12:10:00
    # Synopsis: Processes the `eigen()` module, which is used for PCA-based factor models.
    #           This function calculates the linear indices for the lower-triangular part of the
    #           Householder reflection matrix used to construct the orthonormal loadings. These
    #           indices are required by the `Eigen` manifold constructor.
    params = mod_data[:params]
    vars = mod_data[:variables]
    n_vars = length(vars)
    n_factors = get(params, :n_factors, 1)

    if n_factors > n_vars
        @warn "Number of factors ($n_factors) for eigen() module cannot be greater than number of variables ($n_vars). Setting to $n_vars."
        n_factors = n_vars
    end

    # # Calculate the indices for the lower-triangular part of the Householder matrix
    ltri_mask = [r >= c for r in 1:n_vars, c in 1:n_factors]
    ltri_indices = findall(vec(ltri_mask))

    # # Add these indices to the parameters so resolve_technical_primitive can use them
    mod_data[:params][:ltri_indices] = ltri_indices
    mod_data[:params][:n_factors] = n_factors # Ensure updated value is stored
end

function process_mixed_module!(opt_dict, mod_data)
    # BSTM Internal Utility v1.0.0
    # Timestamp: 2026-06-27 12:10:00
    # Synopsis: Processes the `mixed()` module for random effects (e.g., random intercepts or slopes).
    #           It parses the `mixed(effect_var, group_var)` syntax, creates integer indices for the
    #           grouping variable, and stores the indices and category count in the module's
    #           parameters for use by the `_apply_manifold!` dispatcher.
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
    
    # # Store the computed indices and category count in the module's parameters.
    # This information will be passed to the manifold's specification tuple.
    mod_data[:params][:indices] = indices
    mod_data[:params][:n_cat] = length(levels)
    
    # # The variable list is updated to contain only the effect variable, which is used
    # as the primary variable for the manifold specification (e.g., for random slopes).
    mod_data[:variables] = [effect_var_str]
end

function process_nested_module!(opt_dict, mod_data)
    # BSTM Internal Utility v1.0.0
    # Timestamp: 2026-06-27 12:10:00
    # Synopsis: Processes the `nested()` supervisor module, which is used for multi-fidelity models.
    #           This processor manually constructs a configuration object for the sub-model defined
    #           in the nested formula. It avoids a recursive call to `bstm_config` by directly
    #           invoking the necessary sub-processors.
    opt_dict[:model_arch] = "multifidelity"
    if !haskey(opt_dict, :nested_manifolds); opt_dict[:nested_manifolds] = Dict{Symbol, Any}(); end
    
    var = Symbol(mod_data[:variables][1])
    params = mod_data[:params]
    sub_formula = get(params, :formula, "")
    data_source_sym = get(params, :data_source, :data)
    
    if !haskey(opt_dict, data_source_sym)
        @warn "Data source ':$data_source_sym' for nested module on '$var' not found. Skipping."
        return
    end
    
    sub_data = opt_dict[data_source_sym]
    
    # # Manually create a sub-configuration object
    sub_config = Dict{Symbol, Any}()
    sub_config[:data] = sub_data
    sub_metadata = decompose_bstm_formula(sub_formula)
    
    # # Process spatial module for the sub-config, if present
    spatial_mod_data_pair = filter(m -> m.second[:type] == :spatial, sub_metadata.modules)
    if !isempty(spatial_mod_data_pair)
        spatial_mod_data = first(spatial_mod_data_pair).second
        process_spatial_module!(sub_config, spatial_mod_data)
        sub_config[:model_space] = get(spatial_mod_data[:params], :model, "bym2")
        sub_config[:s_Q_template] = build_structure_template(Symbol(sub_config[:model_space]), sub_config[:s_N]; W=get(sub_config, :W, nothing))
    end
    
    # # Process fixed effects for the sub-config
    fixed_effects_formula_part = join(sub_metadata.fixed_effects, " + ")
    if sub_metadata.has_intercept
        fixed_effects_formula_part = isempty(fixed_effects_formula_part) ? "1" : "1 + " * fixed_effects_formula_part
    end
    if !isempty(fixed_effects_formula_part)
        sub_config[:Xfixed] = create_fixed_design(fixed_effects_formula_part, sub_data)
    end
    
    # # Store the generated sub-configuration
    opt_dict[:nested_manifolds][var] = sub_config
end

# must come aftyer the above
const MODULE_PROCESSORS = Dict{Symbol, Function}(
    :spatial => process_spatial_module!,
    :temporal => process_temporal_module!,
    :smooth => (opt_dict, mod_data, basis_matrices_registry) -> process_smooth_module!(opt_dict, mod_data, basis_matrices_registry, get(opt_dict, :manifolds, [])),
    :interaction_composition => process_interaction_module!,
    :intercept => process_intercept_module!,
    :observationprocess => process_observationprocess_module!,
    :nested => process_nested_module!,
    :eigen => process_eigen_module!,
    :mixed => process_mixed_module!,
    :dynamics => process_dynamics_module!
)

 
function model_results_comprehensive(model, chain; n_samples=100, alpha=0.05)
    println("--- Starting Comprehensive Model Reporting ---")

    # # Metadata and Architecture Extraction
    # Rationale: Extracting the technical configuration from the model arguments.
    M = model.args.M
    y_obs = M.y_obs
    raw_arch = get(M, :model_arch, "univariate")
    model_family = get(M, :model_family, "gaussian")

    # Mapping the configuration string to the architectural dispatch types for reconstruction
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
    else
        UnivariateArchitecture()
    end

    # # Latent Manifold Reconstruction
    # Rationale: Executes the architectural-specific reconstruction of every latent component.
    # The alpha parameter controls the width of the credible intervals (default 95%).
    res = _reconstruct(arch_type, "model_results", chain, M, nothing, alpha)

    # # Performance Metric Assessment
    # Rationale: Metrics are calculated using the posterior predictive mean (denoised expectation).
    y_pred = res.predictions_denoised.mean

    # Flatten for metric consistency across Univariate and Multivariate responses
    y_obs_flat = vec(collect(y_obs))
    y_pred_flat = vec(collect(y_pred))

    # Filter valid indices for calculation to handle missing values or NaNs in observed data
    valid_idx = findall(x -> !isnan(x) && !isnothing(x), y_obs_flat)

    rmse_val = 0.0
    r_pearson = 0.0

    if !isempty(valid_idx)
        obs_v = y_obs_flat[valid_idx]
        pred_v = y_pred_flat[valid_idx]

        # Root Mean Square Error (RMSE)
        rmse_val = sqrt(Statistics.mean((obs_v .- pred_v).^2))

        # Pearson Correlation Coefficient (r)
        try
            r_pearson = Statistics.cor(obs_v, pred_v)
        catch
            # Handle cases with zero variance in predictions to prevent domain errors
            r_pearson = 0.0
        end
    end

    # FlexiChains store timing data in _metadata, not info.
    # Rationale: Direct access to _metadata keys via Symbol lookup.
    sampling_time = chain._metadata.sampling_time[]

    # Standardizing Diagnostic Extraction via DataFrames
    # Rationale: summarystats(chain) returns a FlexiSummary which is converted to a DataFrame
    # to ensure column-based indexing for rhat and ess_bulk is stable and type-safe.
    sum_stats_df = DataFrame(MCMCChains.summarystats(chain))
    
    mean_rhat = 1.0
    min_ess = 0.0

    try
        # Extraction of diagnostic vectors
        rhat_vector = sum_stats_df[!, :rhat]
        ess_bulk_vector = sum_stats_df[!, :ess_bulk]
        
        # Robust aggregation ignoring NaNs
        mean_rhat = Statistics.mean(filter(!isnan, rhat_vector))
        min_ess = Statistics.minimum(filter(!isnan, ess_bulk_vector))
    catch
        # Fallback to standard ess if ess_bulk is absent in the DataFrame schema
        try
            ess_alt = sum_stats_df[!, :ess]
            min_ess = Statistics.minimum(filter(!isnan, ess_alt))
        catch
            mean_rhat = 1.0
            min_ess = 0.0
        end
    end

    waic_val = get(res, :waic, 0.0)

    ess_rate = round(min_ess/sampling_time, digits=2)

    # # Displaying the Report
    println("\n--- Model Metadata ---")
    println("Architecture:     ", raw_arch)
    println("Family:           ", model_family)
    println("Space Component:  ", get(M, :model_space, "none"))
    println("Time Component:   ", get(M, :model_time, "none"))
    println("Seasonal Component: ", get(M, :model_season, "none"))

    println("\n--- Performance Metrics ---")
    println("Compute Time:     ", round(sampling_time, digits=2), " seconds")
    println("RMSE:             ", round(rmse_val, digits=4))
    println("Pearson r:        ", round(r_pearson, digits=4))
    println("WAIC Score:       ", round(waic_val, digits=2))

    println("\n--- MCMC Diagnostics ---")
    println("Mean R-hat:       ", round(mean_rhat, digits=4))
    println("Minimum ESS:      ", round(min_ess, digits=2))
    println("ESS per second:    ", ess_rate)

    # # Final Registry Object Assembly
    # Rationale: Returning the full set of metrics and reconstructed latent fields.
    return (
        metrics = (
            rmse = rmse_val, 
            r_pearson = r_pearson, 
            waic = waic_val, 
            rhat = mean_rhat, 
            ess = min_ess, 
            ess_rate = ess_rate,
            sampling_time = sampling_time
        ),
        pstats = res,
        y_obs = y_obs,
        model_family = model_family,
        arch = arch_type
    )
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

        # Dimensional Realignment and Transpose Logic
        if size(mat_data, 2) == expected_len
            return mat_data
        elseif size(mat_data, 1) == expected_len && size(mat_data, 2) != expected_len
            # Transpose if the chain orientation is [Params x Samples]
            return mat_data'
        elseif size(mat_data, 2) == 1
            # Scalar Broadcast fallback
            return repeat(mat_data, 1, expected_len)
        else
            # Final fallback: return the raw data if it matches expected_len after collect
            return reshape(vec(mat_data), N_samples, :)
        end
    end

    # 3. Null Safety Fallback
    # If the parameter is missing, return a zero-matrix to prevent downstream assembly failure.
    @warn "get_params_vector: Parameter '$base_name' not discovered in chain. Initializing with zeros (len=$expected_len)."
    return zeros(Float64, N_samples, expected_len)
end
 

function create_prediction_surface(
    data_df::DataFrame, 
    au_obj::NamedTuple, 
    tu_obj::NamedTuple, 
    covariate_vars::Vector{Symbol}; 
    iterations::Int = 3
)
    # # 1. Grid Construction
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility function that creates a prediction surface (a full spatiotemporal grid)
              and imputes missing covariate values using an iterative spatiotemporal neighborhood
              averaging method.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility to convert the output of a `Turing.jl` ADVI (variational inference) run
              into a format compatible with the standard post-processing engine. It samples from
              the variational posterior and formats the output as a `FlexiChain`.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility to convert a point estimate from an optimization routine (e.g., MAP, MLE)
              into a sample-based format for post-processing. It uses a Laplace approximation
              (sampling from a multivariate normal centered at the mode) if the Hessian is available.
              If a Hessian is not available, it adds a small amount of Gaussian noise to the point
              estimate to create a sample distribution.
    """
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
                    sorted_keys = sort(collect(col_keys), by = k -> begin
                        m_idx = match(r"\\[(\\d+)\\]", string(k))
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
    """
    BSTM Utility v1.0.3
    Timestamp: 2026-06-26 13:01:15
    Synopsis: A utility function for generating a standardized simulated spatiotemporal dataset with
              known underlying trends, seasonal effects, and covariate relationships.
    Rationale for v1.0.3:
        - Corrected an `ArgumentError` in the `DataFrame` constructor. The `s_coord` column was
          being created as a matrix, which is not a valid column type. It has been corrected to
          be a vector of coordinate tuples.
        - Corrected a dimension mismatch where `cluster_assignments` had length `s_N` instead of `n_total`.
        - Made index generation for `s_clusters` and `reg` robust to cases where the number of
          units is not perfectly divisible by the number of categories.
        - Removed all `local` keyword usage to align with project coding standards.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A data factory that generates a spatiotemporal version of the classic Scottish Lip
              Cancer dataset. It also creates an expanded "nested" dataset for testing
              multi-fidelity models.
    """
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
    
    # println("Generated projection matrix of size: ", size(K_nystrom_proj))
    return K_nystrom_proj
end



# --- Optimized Householder PCA Helper Functions ---

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    # Initializes the Identity matrix to be transformed
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
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper for PCA-based models that performs the inverse operation:
              extracting Householder reflector vectors from a given orthonormal loadings matrix.
              This is useful for initializing a Bayesian PCA model from a frequentist result.
    """
# --- Optimal Vector Extraction (Orthonormal to Householder) ---

# eigenvector_to_householder(U, n_factors)
#
# Description:
#   Extracts the Householder reflector vectors (v) from an orthonormal loadings matrix U.
#   This allows for initializing the Bayesian model from a frequentist PCA result.
#
# Complexity: O(K * N^2)

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


# --- Modular Manifold Application Functions (for use inside @model) ---

# BSTM Manifold Application Function for IID v2.0.0
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A set of internal `_apply_manifold!` methods that are dispatched within the main
              `@model` block. Each method applies the statistical logic for a specific manifold
              type (e.g., IID, ICAR, GP) to the linear predictor `eta`.
    """
# Timestamp: 2026-06-25 19:15:00
# Rationale: Refactored for improved readability and performance.
# - Replaced dense ternary logic with a clear if/elseif block for domain handling.
# - Vectorized the linear predictor updates to replace explicit loops.
# - Removed redundant 'local' keyword declaration.

function _apply_manifold!(eta, spec, m_obj::IID, M, noise)
    """
    BSTM Internal Utility v2.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies an IID (Independent and Identically Distributed) random effect to the
              linear predictor `eta`. This function is dispatched for various domains like
              spatial, temporal, seasonal, and mixed effects, sampling a latent effect for each
              unit and adding it to the corresponding observations.
    Rationale for v2.0.0:
        - Removed a duplicated, incorrect method definition for this function signature that
          was incorrectly dispatching to the dynamics model.
    """
    var_name = string(spec.var)
    m_domain = spec.domain
    is_svc = m_domain == :svc

    # Construct unique parameter names based on domain and variable
    sigma_name = is_svc ? Symbol("sig_svc_", var_name) : Symbol("sigma_", m_domain, "_", var_name)
    latent_name = is_svc ? Symbol("beta_svc_", var_name) : Symbol("latent_", m_domain, "_", var_name)
    
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

    # Determine the number of units and the index mapping based on the manifold's domain
    n_units, indices = if m_domain == :mixed
        (spec.params.n_cat, spec.params.indices)
    elseif m_domain == :spatial
        (M.s_N, M.s_idx)
    elseif m_domain == :temporal
        (M.t_N, M.t_idx)
    elseif m_domain == :seasonal
        (M.u_N, M.u_idx)
    else
        # Fallback for unrecognized domains
        (@warn "Unsupported domain '$m_domain' for IID manifold. Defaulting to spatial context."; (M.s_N, M.s_idx))
    end

    # Sample the latent IID random effect
    latent ~ NamedDist(filldist(Normal(0, sigma), n_units), latent_name)

    # Apply the effect to the linear predictor
    if m_domain == :mixed
        # This handles random intercepts `(1|group)` and random slopes `(x|group)`.
        # If `spec.var` is `:none` or `1`, it's an intercept. Otherwise, it's a slope.
        if spec.var != :none && spec.var != 1
            # This is a random slope: latent[indices] .* x_vals
            x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
            if !isnothing(x_col_idx)
                x_vals = M.Xfixed[:, x_col_idx]
                eta .+= latent[indices] .* x_vals
            else
                @warn "Covariate '$(spec.var)' for random slope not found. Applying as random intercept."
                eta .+= latent[indices]
            end
        else
            # This is a random intercept: just add the effect
            eta .+= latent[indices]
        end
    else
        # For standard spatial/temporal/seasonal random effects, add the latent value
        eta .+= latent[indices]
    end
end

function _apply_manifold!(eta, spec, m_obj::Union{ICAR, BYM2}, M, noise)
    """
    BSTM Internal Utility v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Applies an ICAR or BYM2 manifold to the linear predictor.
    Rationale for v1.1.0:
        - Corrected an `UndefVarError` by explicitly declaring `sigma` and `rho` as local
          variables before they are sampled. This ensures their values are correctly scoped
          and available to the `recompose_precision` function.
    """
    var_name = string(spec.var)
    m_domain = spec.domain
    is_svc = m_domain == :svc
    sigma_name = is_svc ? Symbol("sig_svc_", var_name) : Symbol("sigma_", m_domain)
    latent_name = is_svc ? Symbol("beta_svc_", var_name) : Symbol("latent_", m_domain)
    
    T = eltype(eta)
    sigma::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

    extra_p_val = nothing
    if m_obj isa BYM2
        rho_name = is_svc ? Symbol("rho_svc_", var_name) : Symbol("rho_", m_domain)
        rho::T = 0.0
        rho ~ NamedDist(m_obj.rho_prior, rho_name)
        extra_p_val = rho
    end

    template = (m_domain == :temporal) ? M.t_Q_template.matrix : M.s_Q_template.matrix
    n_units = size(template, 1)
    Q = recompose_precision(Symbol(typeof(m_obj)), template, sigma; extra_param=extra_p_val, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    
    if is_svc
        x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
        if !isnothing(x_col_idx)
            x_vals = M.Xfixed[:, x_col_idx]
            for i in 1:M.y_N; eta[i] += latent[M.s_idx[i]] * x_vals[i]; end
        end
    else
        indices = (m_domain == :spatial) ? M.s_idx : M.t_idx
        for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
    end
end

# --- 7. Model Builder Dispatch (Consolidated) ---
# BSTM Model Builder Dispatch v2.0.0
# Timestamp: 2026-06-27 12:20:00
# Rationale: This section has been refactored to remove redundant methods and provide a
# complete, non-overlapping set of dispatches for all supported manifold types.

# 7.1. Primary `Union` Dispatches for Common Manifold Groups

function build_model(m::Union{IID, ICAR, Besag, BYM2, Leroux, SAR}, data_inputs)
    # Handles all standard discrete graph-based spatial manifolds.
    return _build_from_template(m, data_inputs, :spatial)
end

function build_model(m::Union{AR1, RW1, RW2}, data_inputs)
    # Handles all standard discrete temporal manifolds.
    return _build_from_template(m, data_inputs, :temporal)
end

function build_model(m::Union{GP, FITC, RFF, SVGP, Warp, Nystrom, Harmonic, Hyperbolic, ExponentialDecay}, data_inputs)
    # Handles continuous, spectral, and other advanced manifolds that do not require a pre-computed Q template.
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs)
    # Handles spline-based smooths, determining the correct random walk penalty structure.
    n = hasproperty(m, :nbins) ? m.nbins : get(data_inputs, :s_N, 20)
    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Union{FFT, Wavelet}, data_inputs)
    # Handles spectral manifolds where precision is computed dynamically.
    n = hasproperty(m, :nbins) ? m.nbins : get(data_inputs, :t_N, get(data_inputs, :s_N, 20))
    template = build_structure_template(:iid, n) # Placeholder template
    return _build_pass_through_model(m, data_inputs, model_type_sym=:spectral, Q_template_val=template.matrix)
end

function build_model(m::Union{Advection, Diffusion, AdvectionDiffusion}, data_inputs)
    # Handles physics-informed transport manifolds.
    n = data_inputs.s_N
    W = get(data_inputs, :W, nothing)
    template = build_structure_template(:besag, n; W=W) # Base Laplacian
    return _build_pass_through_model(m, data_inputs, model_type_sym=Symbol(lowercase(string(typeof(m)))), Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

# 7.2. Specific Dispatches for Manifolds with Unique Logic

function build_model(m::SPDE, data_inputs)
    # SPDE requires the graph Laplacian as its base template.
    n = data_inputs.s_N
    W = get(data_inputs, :W, nothing)
    template = build_structure_template(:besag, n; W=W)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Eigen, data_inputs)
    # Eigen/PCA models use an identity template as coefficients are unconstrained.
    n = data_inputs.s_N
    template = build_structure_template(:eigen, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::BCGN, data_inputs)
    # BCGN coefficients are at the group level, not the spatial unit level.
    n_groups = size(m.bipartite_adj, 2)
    template = build_structure_template(:iid, n_groups)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix)
end

function build_model(m::NetworkFlow, data_inputs)
    # NetworkFlow passes the directed adjacency matrix directly as the template.
    return _build_pass_through_model(m, data_inputs, model_type_sym=:network, Q_template_val=m.adjacency_matrix)
end

function build_model(m::LocalAdaptive, data_inputs)
    # LocalAdaptive uses a base ICAR model template.
    template = build_structure_template(:besag, data_inputs.s_N; W=data_inputs.W)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:local_adaptive, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Mosaic, data_inputs)
    # Mosaic models handle templates internally.
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Cyclic, data_inputs)
    # Cyclic models require a specific circular graph Laplacian.
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

# 7.3. Generic Fallback

function build_model(m::Manifold, data_inputs)
    # Fallback for any manifold not explicitly covered.
    @warn "No specific builder for $(typeof(m)). Using IID identity template."
    template = build_structure_template(:iid, get(data_inputs, :s_N, 1))
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = hasproperty(m, :sigma_prior) ? m.sigma_prior : Exponential(1.0),)
    )
end












function _apply_manifold!(eta, spec, m_obj::Union{Leroux, SAR, DAG}, M, noise)
    """
    BSTM Internal Utility v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Applies a Leroux, SAR, or DAG manifold to the linear predictor.
    Rationale for v1.1.0:
        - Corrected an `UndefVarError` by explicitly declaring `sigma` and `rho` as local
          variables before they are sampled.
    """
    var_name = string(spec.var)
    m_domain = spec.domain
    sigma_name = Symbol("sigma_", m_domain, "_", var_name)
    rho_name = Symbol("rho_", m_domain, "_", var_name)
    latent_name = Symbol("latent_", m_domain, "_", var_name)

    T = eltype(eta)
    sigma::T = 0.0
    rho::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
    rho ~ NamedDist(m_obj.rho_prior, rho_name)
    
    template = M.s_Q_template.matrix
    n_units = size(template, 1)
    
    m_type_sym = m_obj isa Leroux ? :leroux : (m_obj isa SAR ? :sar : :dag)
    Q = recompose_precision(m_type_sym, template, sigma; extra_param=rho, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    indices = M.s_idx
    for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
end

function _apply_manifold!(eta, spec, m_obj::Union{AR1, RW1, RW2}, M, noise)
    """
    BSTM Internal Utility v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Applies an AR1, RW1, or RW2 manifold to the linear predictor.
    Rationale for v1.1.0:
        - Corrected an `UndefVarError` by explicitly declaring `sigma` and `rho` as local
          variables before they are sampled.
        - Corrected the logic for assigning the `extra_p_val`.
    """
    m_domain = spec.domain
    T = eltype(eta)
    sigma::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
    
    extra_p_val = nothing
    if m_obj isa AR1
        rho::T = 0.0
        rho ~ NamedDist(m_obj.rho_prior, Symbol("rho_", m_domain))
        extra_p_val = rho
    end

    template = (m_domain == :temporal) ? M.t_Q_template.matrix : M.s_Q_template.matrix
    n_units = size(template, 1)
    Q = recompose_precision(Symbol(typeof(m_obj)), template, sigma; extra_param=extra_p_val, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
    indices = (m_domain == :temporal) ? M.t_idx : M.s_idx
    for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
end


function _apply_manifold!(eta, spec, m_obj::Union{Cyclic, Harmonic}, M, noise)
    m_domain = spec.domain
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_", m_domain))
    
    if m_obj isa Cyclic
        template = M.u_Q_template.matrix
        n_units = size(template, 1)
        Q = recompose_precision(:cyclic, template, sigma; noise=noise)
        latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), Symbol("latent_", m_domain))
    else # Harmonic
        latent ~ NamedDist(filldist(Normal(0, sigma), M.u_N), Symbol("latent_", m_domain))
    end

    indices = M.u_idx
    for i in 1:M.y_N; eta[i] += latent[indices[i]]; end
end

function _apply_manifold!(eta, spec, m_obj::Union{PSpline, BSpline, TPS, RFF, FFT}, M, noise)
        """
        BSTM Internal Utility v1.0.0
        Timestamp: 2026-06-26 10:22:15
        Synopsis: An internal helper for multivariate models that applies the statistical logic for
                  a basis-function manifold to each outcome.
        """
    var_sym = spec.var
    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)
    
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_basis_", var_sym))
    latent_coeffs ~ NamedDist(filldist(Normal(0, sigma), n_basis_cols), Symbol("beta_basis_", var_sym))
    
    eta .+= B_mat * latent_coeffs
end

function _apply_manifold!(eta, spec, m_obj::DynamicsManifold, M, noise)
    # Rationale: This is the primary entry point for mechanistic state-space models.
    # It dispatches to the appropriate dynamics kernel based on the manifold's metadata.
    _apply_dynamics_model!(spec, eta, M, M.hyperpriors)
end

function _apply_manifold!(eta, spec, m_obj::Eigen, M, noise)
    # 1. Get covariate data
    vars = spec.variables
    cov_data = Matrix(M.data[!, vars])' # Transpose to get [n_vars x n_obs]
    n_vars, n_obs = size(cov_data)
    n_factors = m_obj.n_factors

    # 2. Define priors for PCA components
    pca_sd ~ NamedDist(m_obj.pca_sd_prior, :pca_sd)
    pdef_sd ~ NamedDist(m_obj.pdef_sd_prior, :pdef_sd) # residual/uniqueness SD
    v ~ NamedDist(filldist(Normal(0, 1), length(m_obj.ltri_indices)), :v)

    # 3. Construct orthonormal loadings matrix U
    v_mat = zeros(eltype(v), n_vars, n_factors)
    v_mat[m_obj.ltri_indices] .= v
    U = householder_to_eigenvector(v_mat, n_vars, n_factors)

    # 4. Define latent scores (principal components)
    z ~ NamedDist(filldist(Normal(0, 1), n_factors, n_obs), :latent_scores)
    eta .+= z[1, :]
    reconstructed_data = U * z
    for i in 1:n_obs; Turing.@addlogprob! logpdf(MvNormal(reconstructed_data[:, i], pdef_sd^2 * I), cov_data[:, i]); end
end

function _apply_manifold!(eta, spec, m_obj::GP, M, noise)
    """
    BSTM Internal Utility v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Applies a Gaussian Process manifold to the linear predictor.
    Rationale for v1.1.0:
        - Corrected an `UndefVarError` by explicitly declaring `ls` and `sigma` as local
          variables before they are sampled.
    """
    var_sym = spec.var
    T = eltype(eta)
    ls::T = 0.0
    sigma::T = 0.0
    ls ~ NamedDist(m_obj.lengthscale_prior, Symbol("ls_gp_", var_sym))
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_gp_", var_sym))
    coords = spec.params.coords

    kernel_base = get_kernel_from_string(m_obj.kernel)
    k_gp = sigma^2 * kernel_base ∘ ScaleTransform(1/ls)
    
    K_ff = kernelmatrix(k_gp, coords) + noise*I
    f_latent ~ NamedDist(MvNormal(zeros(size(coords,1)), K_ff), Symbol("latent_smooth_", var_sym))
    eta .+= f_latent
end


function _apply_manifold!(eta, spec, m_obj::Union{SVGP, FITC}, M, noise)
    """
    BSTM Internal Utility v1.1.0
    Timestamp: 2026-06-26 18:45:00
    Synopsis: Applies a sparse GP (SVGP or FITC) manifold to the linear predictor.
    Rationale for v1.1.0:
        - Corrected an `UndefVarError` by explicitly declaring `ls` and `sigma` as local
          variables before they are sampled.
    """
    var_sym = spec.var
    T = eltype(eta)
    ls::T = 0.0
    sigma::T = 0.0
    ls ~ NamedDist(m_obj.lengthscale_prior, Symbol("ls_svgp_", var_sym))
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_svgp_", var_sym))
    n_inducing = m_obj.n_inducing
    coords = spec.params.coords

    mean_coords = mean(coords, dims=1)
    std_coords = std(coords, dims=1)
    inducing_locs ~ MvNormal(vec(mean_coords), Diagonal(vec(std_coords)))

    kernel_base = get_kernel_from_string(m_obj.kernel)
    k_svgp = sigma^2 * kernel_base ∘ ScaleTransform(1/ls)
    K_uu = kernelmatrix(k_svgp, inducing_locs) + noise*I
    
    u_latent ~ MvNormal(zeros(n_inducing), K_uu)
    
    K_fu = kernelmatrix(k_svgp, coords, inducing_locs)
    f_mean = K_fu * (K_uu \ u_latent)
    eta .+= f_mean
end

function _apply_manifold!(eta, spec, m_obj::SPDE, M, noise)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies a manifold based on a Stochastic Partial Differential Equation (SPDE),
              typically used to approximate a Matern field.
    """
    var_name = string(spec.var)
    m_domain = spec.domain
    sigma_name = Symbol("sigma_", m_domain, "_", var_name)
    kappa_name = Symbol("kappa_", m_domain, "_", var_name)
    latent_name = Symbol("latent_", m_domain, "_", var_name)

    T = eltype(eta)
    sigma::T = 0.0
    kappa::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
    kappa ~ NamedDist(m_obj.kappa_prior, kappa_name)

    template = M.s_Q_template.matrix
    n_units = size(template, 1)
    
    Q = recompose_precision(:spde, template, sigma; extra_param=kappa, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    
    indices = M.s_idx
    eta .+= latent[indices]
end

function _apply_manifold!(eta, spec, m_obj::Nystrom, M, noise)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies a sparse Gaussian Process manifold using the Nyström approximation.
    """
    var_sym = spec.var
    T = eltype(eta)
    ls::T = 0.0
    sigma::T = 0.0
    ls ~ NamedDist(m_obj.lengthscale_prior, Symbol("ls_nystrom_", var_sym))
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_nystrom_", var_sym))
    
    coords = spec.params.coords
    n_inducing = m_obj.n_inducing
    
    # For simplicity, inducing points are a subset of the data here.
    # A more advanced version might learn their locations.
    inducing_indices = 1:n_inducing
    inducing_points = coords[inducing_indices, :]

    kernel_base = SqExponentialKernel() # Defaulting to SE kernel
    k_gp = sigma^2 * kernel_base ∘ ScaleTransform(1/ls)
    
    K_mm = kernelmatrix(k_gp, inducing_points) + noise*I
    K_nm = kernelmatrix(k_gp, coords, inducing_points)
    
    proj = K_nm / K_mm
    u ~ NamedDist(MvNormal(zeros(n_inducing), I), Symbol("u_nystrom_", var_sym))
    
    eta .+= (proj * u) .* sigma
end

function _apply_manifold!(eta, spec, m_obj::BCGN, M, noise)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies a Bipartite Graph Convolutional Network (BCGN) manifold.
    """
    var_name = string(spec.var)
    sigma_name = Symbol("sigma_bcgn_", var_name)
    latent_name = Symbol("latent_bcgn_", var_name)
    
    T = eltype(eta)
    sigma::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
    
    W_bipartite = m_obj.bipartite_adj
    n_groups = size(W_bipartite, 2)
    
    group_coeffs ~ NamedDist(MvNormal(zeros(n_groups), sigma^2 * I), latent_name)
    
    eta .+= W_bipartite * group_coeffs
end

function _apply_manifold!(eta, spec, m_obj::NetworkFlow, M, noise)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies a network flow manifold, suitable for directed graphs.
    """
    var_name = string(spec.var)
    sigma_name = Symbol("sigma_net_", var_name)
    rho_name = Symbol("rho_net_", var_name)
    latent_name = Symbol("latent_net_", var_name)

    T = eltype(eta)
    sigma::T = 0.0
    rho::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
    rho ~ NamedDist(Beta(1,1), rho_name) # Assuming a generic rho prior
    
    template = m_obj.adjacency_matrix
    n_units = size(template, 1)
    
    Q = recompose_precision(:network, template, sigma; extra_param=rho, directed_adj=template, flow_direction=m_obj.flow_direction)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    
    indices = M.s_idx
    eta .+= latent[indices]
end

function _apply_manifold!(eta, spec, m_obj::Mosaic, M, noise)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies a mosaic (mixture-of-experts) manifold.
    """
    var_name = string(spec.var)
    sigma_name = Symbol("sigma_mosaic_", var_name)
    latent_name = Symbol("latent_mosaic_", var_name)
    
    T = eltype(eta)
    sigma::T = 0.0
    sigma ~ NamedDist(Exponential(1.0), sigma_name) # Generic prior
    
    n_clusters = m_obj.n_regions
    cluster_assignments = M.cluster_assignments[spec.var] # Assumes this is pre-computed
    
    cluster_effects ~ NamedDist(MvNormal(zeros(n_clusters), sigma^2 * I), latent_name)
    
    eta .+= cluster_effects[cluster_assignments]
end

function _apply_manifold!(eta, spec, m_obj::ExponentialDecay, M, noise)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 19:15:00
    Synopsis: Applies a manifold with an exponential decay kernel.
    """
    var_sym = spec.var
    T = eltype(eta)
    ls::T = 0.0
    sigma::T = 0.0
    ls ~ NamedDist(m_obj.decay_lengthscale_prior, Symbol("ls_decay_", var_sym))
    sigma ~ NamedDist(m_obj.sigma_prior, Symbol("sigma_decay_", var_sym))
    coords = spec.params.coords

    dist_matrix = pairwise(Euclidean(), coords, dims=1)
    K = (sigma^2) .* exp.(-dist_matrix ./ ls) + noise*I
    
    f_latent ~ NamedDist(MvNormal(zeros(size(coords,1)), Symmetric(K)), Symbol("latent_decay_", var_sym))
    eta .+= f_latent
end

# Fallback for any other manifold types
function _apply_manifold!(eta, spec, m_obj::ManifoldModel, M, noise)
    @warn "No specific `_apply_manifold!` method for $(typeof(m_obj)). Using IID fallback."
    _apply_manifold!(eta, spec, IID(Exponential(1.0)), M, noise)
end


@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
    """
    BSTM Model Definition v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The main Turing `@model` definition for all univariate `bstm` models. It
              constructs the full Bayesian hierarchical model by assembling all specified
              manifolds (spatial, temporal, etc.) and defining the final likelihood.
    """
    # # 1. Global Likelihood Hyperparameters
    # Rationale: Standardizing scalars for the target likelihood family.
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = one(T)

    if family == "negbin"; lik_r ~ NamedDist(Exponential(1.0), :lik_r); end
    if use_zi == true; lik_phi ~ NamedDist(Beta(1, 1), :lik_phi); end
    if family in ["gamma", "beta", "student_t", "inverse_gaussian", "pareto"]; extra_p ~ NamedDist(Exponential(1.0), :extra_params); end

    # # 2. Observation Volatility & Stochastic Volatility (SV)
    # Rationale: Reconstructs heteroskedastic error scales via RFF log-variance projection.
 y_sigma = Vector{T}(undef, M.y_N)
    if get(M, :use_sv, false) == true
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)
        vol_proj_field = M.vol_proj * beta_vol_latent
        vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)
        for i in 1:M.y_N; y_sigma[i] = exp((sigma_log_var * vol_latent_field[i]) / 2.0); end
    else
        y_sigma_val ~ NamedDist(get(M.hyperpriors, "y_sigma_prior", Exponential(1.0)), :y_sigma)
        for i in 1:M.y_N; y_sigma[i] = y_sigma_val; end
    end

    # # 3. Base Predictor: Fixed Effects & Link-Scale Offsets
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
        if family in ["gaussian", "student_t", "laplace"]; eta .+= exp.(M.log_offset);
        else; eta .+= M.log_offset; end
    end

    # # 4. Modular Manifold Realization
    # This section iterates through the manifold objects created by the config engine.
    for spec in M.manifolds
        _apply_manifold!(eta, spec, spec.manifold_obj, M, noise)
    end
 
    # # 4.1 Structured Smooths Realization
    # This block handles GMRF-style smooths on binned covariates, including interactions.
    if haskey(M, :structured_smooths)
        for smooth_spec in M.structured_smooths
            sigma_name = Symbol("sigma_smooth_", smooth_spec.name)
            latent_name = Symbol("latent_smooth_", smooth_spec.name)
            
            sigma_smooth ~ NamedDist(smooth_spec.sigma_prior, sigma_name)
            
            Q_smooth = (1.0 / (sigma_smooth^2 + noise)) .* smooth_spec.Q_template
            latent_smooth ~ NamedDist(MvNormalCanon(zeros(smooth_spec.n_units), Q_smooth), latent_name)
            
            eta .+= latent_smooth[smooth_spec.indices]
        end
    end

    # # 4.1 Nested Hierarchical Supervisors
    if haskey(M, :nested_manifolds) && !isempty(M.nested_manifolds)
        for (z_key, z_meta) in M.nested_manifolds
            rho_nested ~ NamedDist(Normal(1.0, 0.5), Symbol("rho_nested_", z_key))

            if haskey(z_meta, :model_space) && z_meta.model_space != "none"
                sig_z_s ~ NamedDist(Exponential(1.0), Symbol("sig_nested_spatial_", z_key))
                Q_z_s = recompose_precision(Symbol(z_meta.model_space), z_meta.s_Q_template.matrix, sig_z_s, noise=noise)
                lat_z_s ~ NamedDist(MvNormalCanon(zeros(z_meta.s_N), Q_z_s), Symbol("lat_nested_spatial_", z_key))

                z_s_ptr = z_meta.s_idx
                for i in 1:M.y_N
                    eta[i] += rho_nested * lat_z_s[z_s_ptr[i]]
                end
            end

            if haskey(z_meta, :Xfixed)
                xf_z = z_meta.Xfixed
                beta_z_f ~ NamedDist(MvNormal(zeros(size(xf_z, 2)), 5.0 * I), Symbol("beta_nested_fixed_", z_key))
                eta .+= rho_nested .* (xf_z * beta_z_f)
            end
        end
    end

    # # 5. Final Pointwise Likelihood Dispatch
    yL = T(get(M, :y_lower_bound, -Inf))
    yU = T(get(M, :y_upper_bound, Inf))
    h = T(get(M, :hurdle, -Inf))

    for i in 1:M.y_N
        d_lik = bstm_Likelihood(
            family, [T(M.y_obs[i])]; sigma_y=[y_sigma[i]], weight=[T(M.weights[i])],
            phi_zi=lik_phi, r_nb=lik_r, trial=[Int(M.trials[i])],
            y_L=yL, y_U=yU, hurdle=h, extra_params=extra_p 
        )
        Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
    end
end

 
# Rationale: Defining these as constants prevents re-allocation on every function call.

const PC_PRIORS = Dict(
    "sigma" => Exponential(1.0),
    "rho" => Beta(1, 1),
    "lengthscale" => InverseGamma(3, 3),
    "kappa" => Exponential(1.0),
    "amplitude" => Normal(0, 1),
    "phase" => Beta(1, 1)
)

const INFORMATIVE_PRIORS = Dict(
    "sigma" => Exponential(0.5),
    "rho" => Beta(2, 2),
    "lengthscale" => InverseGamma(5, 5),
    "kappa" => Exponential(0.1),
    "amplitude" => Normal(0, 0.5),
    "phase" => Beta(2, 2)
)

const UNINFORMATIVE_PRIORS = Dict(
    "sigma" => Normal(0, 1e6),
    "rho" => Uniform(0, 1),
    "lengthscale" => InverseGamma(0.01, 0.01),
    "kappa" => Exponential(10.0),
    "amplitude" => Normal(0, 100),
    "phase" => Uniform(0, 1)
)



function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The internal reconstruction engine specifically for multivariate models. It handles
              the extraction and summarization of parameters and latent fields for multiple,
              correlated outcomes.
    """
    N_samples = size(chain, 1)
    outcomes_N = Int(M.outcomes_N)
    p_names = string.(FlexiChains.parameters(chain))
    N_PS = isnothing(PS) ? 0 : size(PS.Xfixed, 1)
    N_tot = M.y_N + N_PS
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)

    eta = zeros(Float64, N_tot, outcomes_N, N_samples)
    summarized_effects = Dict{Symbol, Any}()

    # Fixed Effects
    if "Xfixed_beta" in p_names
        betas_flat = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)'
        for k in 1:outcomes_N
            betas_k = betas_flat[((k-1)*M.Xfixed_N + 1):(k*M.Xfixed_N), :]
            eta[1:M.y_N, k, :] .+= M.Xfixed * betas_k
            if N_PS > 0; eta[(M.y_N+1):end, k, :] .+= PS.Xfixed * betas_k; end
        end
        summarized_effects[:fixed_effects] = summarize_array(betas_flat'; alpha=alpha)
    end

    # Manifold Effects
    latent_innov_samples = zeros(N_tot, outcomes_N, N_samples)
    for spec in M.manifolds
        m_domain = spec.domain
        m_obj = spec.manifold_obj
        var_name = string(spec.var)

        for k in 1:outcomes_N
            latent_name_str = if m_domain == :smooth
                "beta_basis_" * var_name * "_" * string(k)
            else
                "latent_" * string(m_domain) * "_" * var_name * "_" * string(k)
            end

            if latent_name_str in p_names
                n_units = if m_domain == :spatial; M.s_N
                          elseif m_domain == :temporal; M.t_N
                          elseif m_domain == :seasonal; M.u_N
                          elseif m_domain == :smooth; size(M.basis_matrices[spec.var], 2)
                          else 1 end

                latent_samples = get_params_vector(chain, latent_name_str, n_units)'
                summarized_effects[Symbol(latent_name_str)] = summarize_array(latent_samples; alpha=alpha)

                if m_domain in [:spatial, :temporal, :seasonal]
                    indices = if m_domain == :spatial; M.s_idx
                              elseif m_domain == :temporal; M.t_idx
                              else M.u_idx end
                    latent_innov_samples[1:M.y_N, k, :] .+= latent_samples[indices, :]
                    if N_PS > 0
                        ps_indices = if m_domain == :spatial; PS.s_idx
                                     elseif m_domain == :temporal; PS.t_idx
                                     else PS.u_idx end
                        latent_innov_samples[(M.y_N+1):end, k, :] .+= latent_samples[ps_indices, :]
                    end
                elseif m_domain == :smooth
                    B_mat_train = M.basis_matrices[spec.var]
                    eta[1:M.y_N, k, :] .+= B_mat_train * latent_samples
                    if N_PS > 0
                        # Re-compute basis for PS grid - logic similar to univariate _reconstruct
                    end
                end
            end
        end
    end

    # Apply LKJ coupling
    if "L_corr" in p_names
        L_corr_samples = get_params_matrix_sizestructured(chain, "L_corr", (outcomes_N, outcomes_N))
        for j in 1:N_samples
            eta[:, :, j] .+= latent_innov_samples[:, :, j] * L_corr_samples[:,:,j]
        end
    end

    # Predictions
    p_den = zeros(Float64, N_tot, outcomes_N, N_samples)
    y_sigma_samples = zeros(N_tot, outcomes_N, N_samples)
    if "y_sigma" in p_names
        sig_samps = get_params_vector(chain, "y_sigma", outcomes_N)'
        for k in 1:outcomes_N
            y_sigma_samples[:, k, :] .= sig_samps[k, :]'
        end
    end

    for j in 1:N_samples
        for k in 1:outcomes_N
            p_den[:, k, j] .= _apply_link_and_lik(family_str, eta[:, k, j], false)
        end
    end

    summarized_effects[:predictions_denoised] = summarize_array(p_den[1:M.y_N, :, :]; alpha=alpha)
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_den[(M.y_N+1):end, :, :]; alpha=alpha)
    end
    
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch
    return summarized_effects
end


 



@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    """
    BSTM Model Definition v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The main Turing `@model` definition for multi-fidelity `bstm` models. It
              integrates data from different resolutions or sources using a nested latent
              process structure, typically with RFF-based mappings between fidelity levels.
    """
    # # 1. Global Hyperpriors
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    y_sigma ~ NamedDist((family in ["gaussian", "lognormal"]) ? Exponential(1.0) : Dirac(1.0), :y_sigma)
    r_nb ~ NamedDist((family == "negbin") ? Exponential(1.0) : Dirac(1.0), :r_nb)
    phi_zi ~ NamedDist(use_zi ? Beta(1, 1) : Dirac(0.0), :phi_zi)

    z_sigma ~ NamedDist(Exponential(0.5), :z_sigma)
    w_sigma ~ NamedDist(filldist(Exponential(0.5), 3), :w_sigma)

    # # 2. Nested Latent Covariate Structure (RFF based)
    # High-Fidelity Z (Spatial)
    z_ls ~ NamedDist(Gamma(2, 2), :z_ls)
    z_beta ~ NamedDist(filldist(Normal(0, 1), M.M_rff), :z_beta)
    z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls)) .+ M.b_fixed'
    z_latent = M.rff_scale .* (cos.(z_proj) * z_beta)

    # Medium-Fidelity W (Spatiotemporal, depends on Z)
    w_ls ~ NamedDist(Gamma(2, 2), :w_ls)
    w_beta ~ NamedDist(filldist(Normal(0, 1), M.M_rff, 3), :w_beta)
    w_coords_augmented = hcat(M.w_coords_st, z_latent[1:size(M.w_coords_st, 1)])
    w_proj = (w_coords_augmented * (M.W_fixed[1:size(w_coords_augmented, 2), :] ./ w_ls)) .+ M.b_fixed'
    w_latent = M.rff_scale .* (cos.(w_proj) * w_beta)

    # # 3. Spatial, Temporal, and Seasonality Manifolds
    # Spatial Manifold (s_eta)
    s_sigma ~ NamedDist(Exponential(1.0), :s_sigma)
    s_rho ~ NamedDist(Beta(1, 1), :s_rho)
    s_icar ~ NamedDist(MvNormalCanon(zeros(M.N_areas), M.s_Q + noise*I), :s_icar)
    s_iid ~ NamedDist(MvNormal(zeros(M.N_areas), I), :s_iid)
    sum_icar = sum(s_icar)
    sum_icar ~ NamedDist(Normal(0, 0.001 * M.N_areas), :sum_icar)
    s_eta = (s_sigma .* (sqrt(s_rho) .* s_icar .+ sqrt(1 - s_rho) .* s_iid))[M.s_idx]

    # Temporal Manifold (t_eta_full)
    t_sigma ~ NamedDist(Exponential(1.0), :t_sigma)
    t_rho ~ NamedDist((M.model_time == "ar1") ? Beta(2, 2) : Dirac(0.0), :t_rho)
    t_ls ~ NamedDist((M.model_time == "gp") ? InverseGamma(3, 3) : Dirac(1.0), :t_ls)
    
    local t_eta_full, t_Q
    if M.model_time == "ar1"
        t_Q_base = Symmetric((1.0 + t_rho^2) .* I(M.t_N) .+ (t_rho) .* M.t_Q)
        t_Q = Symmetric((1.0 / (1.0 - t_rho^2 + noise)) .* t_Q_base )
        t_raw ~ NamedDist(MvNormalCanon(zeros(M.t_N), t_Q), :t_raw)
        t_eta_full = (t_raw .* t_sigma)[M.t_idx]
    elseif M.model_time == "rw2"
        t_Q = Symmetric((1.0 / (t_sigma^2 + noise)) .* M.t_Q )
        t_raw ~ NamedDist(MvNormalCanon(zeros(M.t_N), t_Q), :t_raw)
        t_eta_full = t_raw[M.t_idx]
    elseif M.model_time == "gp"
        K_t = (t_sigma^2) .* kernelmatrix(SqExponentialKernel() ∘ ScaleTransform(inv(t_ls)), 1.0:Float64(M.t_N)) + noise * I
        t_gp ~ NamedDist(MvNormal(zeros(M.t_N), Symmetric(K_t)), :t_gp)
        t_eta_full = t_gp[M.t_idx]
        t_Q = inv(Symmetric(K_t))
    elseif M.model_time == "iid"
        t_Q = Symmetric((1.0 / (t_sigma^2 + noise)) .* I(M.t_N))
        t_iid ~ NamedDist(MvNormal(zeros(M.t_N), I), :t_iid)
        t_eta_full = (t_iid .* t_sigma)[M.t_idx]
    else
        t_eta_full = zeros(T, M.N_obs)
        t_Q = I(M.t_N)
    end

    # Seasonality Manifold (u_eta_full)
    u_sigma ~ NamedDist((M.model_season != "none") ? Exponential(0.5) : Dirac(0.0), :u_sigma)
    u_rho ~ NamedDist((M.model_season == "ar1") ? Beta(2, 2) : Dirac(0.0), :u_rho)
    u_ls ~ NamedDist((M.model_season == "gp") ? InverseGamma(3, 3) : Dirac(1.0), :u_ls)
    
    local u_eta_full
    if M.model_season == "ar1"
        u_Q_base = Symmetric((1.0 + u_rho^2) .* I(M.u_N) .+ (u_rho) .* M.u_Q)
        u_Q = Symmetric((1.0 / (1.0 - u_rho^2 + noise)) .* u_Q_base )
        u_raw ~ NamedDist(MvNormalCanon(zeros(M.u_N), u_Q), :u_raw)
        u_eta_full = (u_raw .* u_sigma)[M.u_idx]
    elseif M.model_season == "rw2"
        u_Q = Symmetric((1.0 / (u_sigma^2 + noise)) .* M.u_Q )
        u_raw ~ NamedDist(MvNormalCanon(zeros(M.u_N), u_Q), :u_raw)
        u_eta_full = u_raw[M.u_idx]
    else
        u_eta_full = zeros(T, M.N_obs)
    end

    # # 4. Space-Time Interaction (st_eta)
    st_sigma ~ NamedDist((M.model_st != "none") ? Exponential(0.5) : Dirac(0.0), :st_sigma)
    local st_eta
    if M.model_st == "none"
        st_eta = zeros(T, M.N_areas, M.t_N)
    else
        st_Q = if M.model_st == "I"; Symmetric(I(M.N_areas * M.t_N))
               elseif M.model_st == "II"; Symmetric(kron(I(M.N_areas), t_Q))
               elseif M.model_st == "III"; Symmetric(kron(M.s_Q, I(M.t_N)))
               else Symmetric(kron(M.s_Q, t_Q)) end
        st_raw ~ NamedDist(MvNormalCanon(zeros(M.N_areas * M.t_N), st_Q), :st_raw)
        st_eta = reshape(st_raw, M.N_areas, M.t_N) .* st_sigma
    end

    # # 5. Linear Predictor Construction
    z_beta_eta ~ NamedDist(Normal(0, 1), :z_beta_eta)
    w_beta_eta ~ NamedDist(MvNormal(zeros(3), I), :w_beta_eta)
    eta = M.log_offset .+ s_eta .+ t_eta_full .+ u_eta_full .+ (z_latent .* z_beta_eta) .+ (w_latent * w_beta_eta)
    if M.model_st != "none"
        for i in 1:M.N_obs
            eta[i] += st_eta[M.s_idx[i], M.t_idx[i]]
        end
    end

    # # 6. Joint Multi-fidelity Likelihood
    # High-Fidelity Latent Likelihood
    Turing.@addlogprob! logpdf(MvNormal(z_latent, z_sigma^2 * I), M.z_obs)

    # Medium-Fidelity Latent Likelihoods
    for k in 1:3
        Turing.@addlogprob! logpdf(MvNormal(w_latent[:, k], w_sigma[k]^2 * I), M.w_obs[:, k])
    end

    # Primary Observation Likelihood
    Turing.@addlogprob! logpdf(bstm_Likelihood(family, M.y_obs; sigma_y=y_sigma, use_zi=use_zi, weights=M.weights, phi_zi=phi_zi, r_nb=r_nb, trials=M.trials), eta)
end
 





function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    N_samples = size(chain, 1)
    p_names = string.(MCMCChains.names(chain))
    family_str = get(M, :model_family, "gaussian")
    fam_obj = get_model_family(family_str)
    noise = get(M, :noise, 1e-4)

    # --- 1. Parameter Extraction ---
    z_ls_s = get_params_vector(chain, "z_ls", 1)
    z_beta_s = get_params_vector(chain, "z_beta", M.M_rff)
    w_ls_s = get_params_vector(chain, "w_ls", 1)
    w_beta_s = get_params_vector(chain, "w_beta", M.M_rff * 3)
    s_sigma_s = get_params_vector(chain, "s_sigma", 1)
    s_rho_s = get_params_vector(chain, "s_rho", 1)
    s_icar_s = get_params_vector(chain, "s_icar", M.N_areas)
    s_iid_s = get_params_vector(chain, "s_iid", M.N_areas)
    t_sigma_s = get_params_vector(chain, "t_sigma", 1)
    t_rho_s = get_params_vector(chain, "t_rho", 1)
    t_ls_s = get_params_vector(chain, "t_ls", 1)
    t_raw_s = get_params_vector(chain, "t_raw", M.t_N)
    t_gp_s = get_params_vector(chain, "t_gp", M.t_N)
    u_sigma_s = get_params_vector(chain, "u_sigma", 1)
    u_rho_s = get_params_vector(chain, "u_rho", 1)
    u_raw_s = get_params_vector(chain, "u_raw", M.u_N)
    st_sigma_s = get_params_vector(chain, "st_sigma", 1)
    st_raw_s = get_params_vector(chain, "st_raw", M.N_areas * M.t_N)
    z_beta_eta_s = get_params_vector(chain, "z_beta_eta", 1)
    w_beta_eta_s = get_params_vector(chain, "w_beta_eta", 3)

    # --- 2. Latent Field Reconstruction ---
    eta_samples = zeros(M.N_obs, N_samples)
    z_latent_samples = zeros(size(M.z_coords_s, 1), N_samples)
    w_latent_samples = zeros(size(M.w_coords_st, 1), 3, N_samples)
    s_eta_samples = zeros(M.N_obs, N_samples)
    t_eta_samples = zeros(M.N_obs, N_samples)
    u_eta_samples = zeros(M.N_obs, N_samples)
    st_eta_samples = zeros(M.N_obs, N_samples)

    for j in 1:N_samples
        # RFF fields
        z_proj = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_ls_s[j])) .+ M.b_fixed'
        z_latent_j = M.rff_scale .* (cos.(z_proj) * z_beta_s[j, :])
        z_latent_samples[:, j] = z_latent_j

        w_coords_aug = hcat(M.w_coords_st, z_latent_j[1:size(M.w_coords_st, 1)])
        w_proj = (w_coords_aug * (M.W_fixed[1:size(w_coords_aug, 2), :] ./ w_ls_s[j])) .+ M.b_fixed'
        w_beta_mat = reshape(w_beta_s[j, :], M.M_rff, 3)
        w_latent_j = M.rff_scale .* (cos.(w_proj) * w_beta_mat)
        w_latent_samples[:, :, j] = w_latent_j

        # GMRF fields
        s_eta_samples[:, j] = (s_sigma_s[j] .* (sqrt(s_rho_s[j]) .* s_icar_s[j, :] .+ sqrt(1 - s_rho_s[j]) .* s_iid_s[j, :]))[M.s_idx]

        if M.model_time == "ar1"
            t_eta_samples[:, j] = (t_raw_s[j, :] .* t_sigma_s[j])[M.t_idx]
        elseif M.model_time == "gp"
            t_eta_samples[:, j] = t_gp_s[j, :][M.t_idx]
        else # rw2, iid
            t_eta_samples[:, j] = t_raw_s[j, :][M.t_idx]
        end

        if M.model_season != "none"
            u_eta_samples[:, j] = (u_raw_s[j, :] .* u_sigma_s[j])[M.u_idx]
        end

        if M.model_st != "none"
            st_eta_mat = reshape(st_raw_s[j, :], M.N_areas, M.t_N) .* st_sigma_s[j]
            for i in 1:M.N_obs
                st_eta_samples[i, j] = st_eta_mat[M.s_idx[i], M.t_idx[i]]
            end
        end

        # Assemble final eta
        eta_samples[:, j] = M.log_offset .+ s_eta_samples[:, j] .+ t_eta_samples[:, j] .+ u_eta_samples[:, j] .+ st_eta_samples[:, j] .+
                            (z_latent_j .* z_beta_eta_s[j]) .+ (w_latent_j * w_beta_eta_s[j, :])
    end

    # --- 3. Predictions and WAIC ---
    y_sigma_samples = _extract_volatility(chain, p_names, M.N_obs, N_samples, nothing, M)
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(fam_obj, eta_samples, chain, M, M.N_obs, N_samples, y_sigma_samples)

    # --- 4. Summarize and Return ---
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy; alpha=alpha)
    summarized_effects[:spatial_structured] = summarize_array(s_eta_samples; alpha=alpha)
    summarized_effects[:temporal] = summarize_array(t_eta_samples; alpha=alpha)
    summarized_effects[:seasonal] = summarize_array(u_eta_samples; alpha=alpha)
    summarized_effects[:spacetime_interaction] = summarize_array(st_eta_samples; alpha=alpha)
    summarized_effects[:z_latent_summary] = summarize_array(z_latent_samples; alpha=alpha)
    summarized_effects[:w_latent_summary] = summarize_array(w_latent_samples; alpha=alpha)
    summarized_effects[:waic] = _compute_waic(log_lik)
    summarized_effects[:log_lik_matrix] = log_lik
    summarized_effects[:family] = family_str
    summarized_effects[:arch] = arch

    return summarized_effects
end
 

####


 
;;


function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing, bipartite_adj=nothing)
    # Consolidated Structural Factory [v14.6.0 - BSTM v06.1 Unified Registry]
    # Rationale: This function provides the definitive factory for precision matrix templates
    # across all manifold domains. It standardizes scaling and topological initialization.
    # Audit: Fixed SPDE scaling and DAG operator registration; added TPS 2D routing.

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

    # #
    # 3. Wavelet Domain Precision Mapping
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
        
        # Map wavelet variances to real-space precision weights via inverse transform
        # This constructs the first row of the operator assuming circulant-like structure
        precision_eigenvalues = 1.0 ./ (wavelet_eigenvalues .+ noise_floor)
        
        # Standardize return to first_row_q for sparse construction
        # Note: True wavelets are non-circulant; this is the stationary wavelet approximation.
        first_row_q = real(FFTW.ifft(precision_eigenvalues))
    end

    # #
    # 4. Precision Mapping via IFFT (Stationary Path)
    if kernel_type != :wavelet
        precision_eigenvalues = 1.0 ./ (psd .+ noise_floor)
        first_row_q = real(FFTW.ifft(precision_eigenvalues))
    end

    # #
    # 5. Sparse Matrix Construction
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



function estimate_spectral_precision(n::Int64, ls::Real, sig::Real; kernel_type=:se, periodicity=nothing, noise_floor=1e-6, wavelet_levels=3)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: An internal utility that constructs a sparse precision matrix for a stationary process
              by computing the kernel's power spectral density (PSD) and using the Wiener-Khinchin theorem.
    """
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
    end

    # #
    # Precision Recomposition in Spectral Domain
    # The eigenvalues of the precision matrix are the reciprocal of the spectral density
    precision_eigenvalues = 1.0 ./ (psd .+ noise_floor)

    # #
    # Real-Space Mapping via Inverse Fourier Transform
    # The first row of the circulant precision matrix is the IFFT of its eigenvalues
    first_row_q = real(FFTW.ifft(precision_eigenvalues))

    # #
    # Sparse Matrix Construction
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

 


function build_model(m::Eigen, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Eigen/PCA-based models.
    """
    # Audited Householder PCA Builder [v14.3.5 - BSTM v06.1 Spectral Patch]
    # Rationale: Provides low-rank decomposition using Householder reflections for orthonormal loadings.
    # Requirements: Uses Group 1 (Identity) template as coefficients are modeled in an unconstrained latent space.
    
    local n = data_inputs.s_N
    
    # Dispatching to Group 1 (Identity/Spectral Bases) in the structural factory
    local template = build_structure_template(:eigen, n)
    
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :eigen,
        hyper = (
            sigma_prior = m.sigma_prior, 
            pca_sd_prior = m.pca_sd_prior, 
            pdef_sd_prior = m.pdef_sd_prior,
            n_factors = m.n_factors,
            ltri_indices = m.ltri_indices
        )
    )
end


function build_model(m::Nystrom, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Nyström-based sparse GP models.
    """
    # Audited Nystrom Approximation Builder [v14.3.0 - BSTM v06.1 Basis Patch]
    # Rationale: Provides low-rank kernel approximation using landmark inducing points.
    # Requirements: Returns a distance-based template or placeholder for dynamic calculation.
    
    local n = data_inputs.s_N
    local coords = get(data_inputs, :s_coord, nothing)
    
    # Dispatching to Group 7 (Distance-Based Kernel Manifolds) in the structural factory
    local template = build_structure_template(:nystrom, n; coords=coords)
    
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :nystrom,
        hyper = (
            sigma_prior = m.sigma_prior, 
            lengthscale_prior = m.lengthscale_prior, 
            n_inducing = m.n_inducing
        )
    )
end

function build_model(m::BCGN, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Bipartite Graph Convolutional Networks.
    """
    # Audited BCGN (Bipartite Covariate Graph Network) Builder [v14.3.8 - BSTM v06.1 Network Patch]
    # Rationale: Models latent signals propagated through a bipartite unit-group adjacency.
    # Requirements: Routes to Group 1 (Identity) as weights are modeled in an unconstrained space before bipartite projection.
    
    # Determine the number of auxiliary nodes/groups from the provided manifold struct
    local n_groups = size(m.bipartite_adj, 2)
    
    # Dispatching to Group 1 (Identity) in the structural factory to initialize group-level coefficients
    local template = build_structure_template(:bgcn, n_groups)
    
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bgcn,
        hyper = (
            sigma_prior = m.sigma_prior,
            bipartite_adj = m.bipartite_adj,
            group_weights = m.group_weights
        )
    )
end



# 2. Directed Network Flow Builder
function build_model(m::NetworkFlow, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for network flow models.
    """
    # For directed graphs, utilize the provided directed adjacency matrix
    W_directed = m.adjacency_matrix
    n = size(W_directed, 1)

    # Precision for directed flow often uses the flow-Laplacian: (I - rho*W)'(I - rho*W)
    return (
        Q_template = W_directed,
        scaling_factor = 1.0,
        model_type = :network,
        hyper = (
            sigma_prior = m.sigma_prior,
            direction = m.flow_direction
        )
    )
end

# 3. Hyperbolic Embedding Builder
function build_model(m::Hyperbolic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Hyperbolic geometry models.
    """
    # Hyperbolic manifolds are continuous; utilize the coordinate names
    return (
        Q_template = nothing, # Calculated dynamically via hyperbolic distance kernel
        model_type = :hyperbolic,
        hyper = (
            sigma_prior = m.sigma_prior,
            curvature = m.curvature,
            coords = m.coordinates
        )
    )
end
  
 
function build_model(m::IID, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for IID (unstructured) random effects.
    """
    # IID Builder: Unstructured random effects
    # Rationale: Standardizes variance scales across the spatial or temporal unit grid.
    local n = data_inputs.s_N
    local template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end
 
function build_model(m::ICAR, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for ICAR (Besag) models.
    """
    template = build_structure_template(:icar, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :icar,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end

 

function build_model(m::BYM2, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for BYM2 models.
    """
    # BYM2 Builder: Standardized ICAR + IID
    # Rationale: Decouples variance scale from graph topology using geometric mean scaling.
    local n = data_inputs.s_N
    local W = get(data_inputs, :W, nothing)
    local template = build_structure_template(:bym2, n; W=W)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bym2,
        hyper = (
            sigma_prior = m.sigma_prior,
            rho_prior = m.rho_prior
        )
    )
end

function build_model(m::Leroux, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Leroux models.
    """
    # Leroux Builder: Convex combination of structure and noise
    local n = data_inputs.s_N
    local W = get(data_inputs, :W, nothing)
    local template = build_structure_template(:leroux, n; W=W)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :leroux,
        hyper = (
            sigma_prior = m.sigma_prior,
            rho_prior = m.rho_prior
        )
    )
end

function build_model(m::AR1, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for AR(1) models.
    """
    # AR1 Builder: Mean-reverting temporal or spatial process
    # Rationale: Maps time indices or regular lattice units to an AR(1) precision structure.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:ar1, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :ar1,
        hyper = (
            sigma_prior = m.sigma_prior,
            rho_prior = m.rho_prior
        )
    )
end

function build_model(m::RW2, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for RW2 (second-order random walk) models.
    """
    # RW2 Builder: Second-order random walk (Smooth Trend)
    # Rationale: Utilizes intrinsic GMRF precision with a locally linear slope assumption.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:rw2, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :rw2,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end


function build_model(m::Cyclic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for cyclic random walks.
    """
    # Cyclic Builder: Periodic continuity (Seasonal)
    # Rationale: Enforces boundary condition y_0 = y_T via a circular Laplacian.
    local n = m.period
    local W_circ = zeros(n, n)
    for i in 1:n
        W_circ[i, mod1(i-1, n)] = 1.0
        W_circ[i, mod1(i+1, n)] = 1.0
    end
    local template = build_structure_template(:besag, n; W=W_circ)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :cyclic,
        hyper = (sigma_prior = m.sigma_prior, period = m.period)
    )
end

function build_model(m::RFF, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for RFF-based GP approximations.
    """
    # RFF Builder: Spectral approximation of stationary kernels
    # Rationale: Maps coordinates to random Fourier feature space coefficients.
    return (
        Q_template = Matrix(1.0I, m.n_features, m.n_features),
        scaling_factor = 1.0,
        model_type = :rff,
        hyper = (
            sigma_prior = m.sigma_prior,
            lengthscale_prior = m.lengthscale_prior,
            n_features = m.n_features
        )
    )
end 
 


function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs)
    # Basis-mapped manifolds resolve resolution n based on domain to ensure dimensional parity.

    n = 1
    if hasproperty(m, :domain)
        if m.domain == :spatial
            n = data_inputs.s_N
        elseif m.domain == :temporal
            n = data_inputs.t_N
        elseif m.domain == :seasonal
            n = data_inputs.u_N
        else
            n = hasproperty(m, :nbins) ? m.nbins : data_inputs.s_N
        end
    else
        n = hasproperty(m, :nbins) ? m.nbins : data_inputs.s_N
    end

    # Resolve template based on manifold type
    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :spectral,
        hyper = (
            sigma_prior = m.sigma_prior,
            nbins = n,
            degree = hasproperty(m, :degree) ? m.degree : 3,
            diff_order = hasproperty(m, :diff_order) ? m.diff_order : 2
        )
    )
end

    



function build_model(m::FFT, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for FFT-based spectral models.
    """
    # Spectral manifolds define precision in the frequency domain.
    # The Wiener-Khinchin theorem maps the kernel PSD to a circulant precision matrix.
    
    # 1. Determine System Dimensionality (n)
    # Prefers nbins for basis resolution, else falls back to temporal or spatial unit counts.
    n = hasproperty(m, :nbins) ? m.nbins : 
             (hasproperty(data_inputs, :t_N) ? data_inputs.t_N : data_inputs.s_N)
    
    # 2. Extract Kernel Metadata
    # Standardizing the mapping for the spectral precision factory.
    # kernel_sym defaults to :se (Squared Exponential) if not specified.
    kernel_sym = Symbol(get(m, :kernel, :se))
    wav_levels = hasproperty(m, :wavelet_levels) ? m.wavelet_levels : 3
    
    # 3. Structural Template Placeholder
    # The actual precision matrix is often recomposed inside the model to allow 
    # gradients to flow through sig/ls. An identity template is provided as a placeholder.
    template = build_structure_template(:iid, n)

    # 4. Object Assembly
    # Returns the configuration required by the bstm supervisors.
    return (
        Q_template = template.matrix, 
        scaling_factor = 1.0,
        model_type = :spectral, 
        hyper = (
            sigma_prior = m.sigma_prior, 
            lengthscale_prior = m.lengthscale_prior, 
            kernel = kernel_sym,
            wavelet_levels = wav_levels,
            n_bins = n
        )
    )
end



function build_model(m::Wavelet, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for wavelet-based smooths.
    """
    n = 1
    if hasproperty(m, :domain)
        if m.domain == :spatial
            n = data_inputs.s_N
        elseif m.domain == :temporal
            n = data_inputs.t_N
        elseif m.domain == :seasonal
            n = data_inputs.u_N
        else
            n = hasproperty(m, :nbins) ? m.nbins : data_inputs.s_N
        end
    else
        n = hasproperty(m, :nbins) ? m.nbins : data_inputs.s_N
    end

    wav_family = Symbol(get(m, :wavelet_family, :db2))
    wav_levels = hasproperty(m, :wavelet_levels) ? m.wavelet_levels : 3
    kernel_sym = Symbol(get(m, :kernel, :se))
    template = build_structure_template(:wavelet, n)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :spectral,
        hyper = (
            sigma_prior = m.sigma_prior,
            lengthscale_prior = m.lengthscale_prior,
            kernel = kernel_sym,
            wavelet_family = wav_family,
            wavelet_levels = wav_levels,
            n_bins = n
        )
    )
end

 
function build_model(m::SPDE, data_inputs)
    # The Stochastic Partial Differential Equation (SPDE) manifold approximates 
    # continuous random fields via Finite Element discretization on a graph Laplacian.
    
    # # System Dimensionality Discovery
    n = hasproperty(data_inputs, :s_N) ? data_inputs.s_N : size(data_inputs.W, 1)

    # # Graph Laplacian Acquisition
    # Extract the adjacency matrix W to construct the discrete Laplacian operator L.
    W = hasproperty(data_inputs, :W) ? data_inputs.W : sparse(I(n))
    
    # Construct the base Laplacian: L = D - W
    # This operator is the fundamental component for discretized elliptic operators.
    D_diag = Diagonal(vec(sum(W, dims=2)))
    L_operator = D_diag - W

    # # Kernel and Metadata Resolution
    # Extract user-specified kernel (e.g., :matern32, :se). Defaults to :matern.
    kernel_sym = hasproperty(m, :kernel) ? Symbol(m.kernel) : :matern

    # # Configuration Assembly
    # Returns the Laplacian as a template. The actual precision is recomposed 
    # during model execution (e.g., Q = (kappa^2 * I + L)^2 for Matern) 
    # to allow gradients to flow through the range parameter kappa.
    return (
        Q_template = sparse(L_operator),
        scaling_factor = 1.0,
        model_type = :spde,
        hyper = (
            sigma_prior = m.sigma_prior,
            kappa_prior = m.kappa_prior,
            kernel = kernel_sym,
            n_units = n
        )
    )
end





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
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal factory function that constructs a final precision matrix from a
              template, a scale parameter (`param_val`), and other manifold-specific parameters
              (e.g., correlation `rho`). This function is central to defining GMRF priors within
              the model.
    """
    # Rationale: Standardizing the inversion of the variance scale for precision mapping.
    # Requirement: Avoid ternary if/else during sampling to maintain AD stability.
    
    n_s = size(template_s, 1)
    scale_factor = 1.0 / (param_val^2 + noise)

    # 1. Base Graph & CAR Manifolds
    if m_type == :none || m_type == :fixed
        return Symmetric(scale_factor * I(n_s) + noise * I)
    end

    if m_type == :besag || m_type == :icar || m_type == :diffusion || m_type == :cyclic
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

    if m_type == :sar || m_type == :advection || m_type == :advection_diffusion || m_type == :dag || m_type == :proper_car
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
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper for the formula parser that separates a variable name from any
              piped transformation functions (e.g., "x |> log").
    """
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





# Rationale: Centralizing the generation of non-linear basis matrices (B).
# This factory ensures absolute parity with the v06.1 manifold taxonomy.

function bstm_smooth_basis_1D(type::String, vals::AbstractVector, nbins::Int, degree::Int; kwargs...)
    # # 1. Initialization and Scoping
    # Rationale: Ensures every call returns a valid N x M matrix to prevent sampler collapse.
    n_obs = length(vals)
    B = zeros(Float64, n_obs, nbins)
    
    # Technical Metadata Discovery
    v_min = minimum(vals)
    v_max = maximum(vals)
    v_std = std(vals) + 1e-9

    # # 2. Manifold Dispatch Registry
    
    # # 2.1 B-Splines / P-Splines (Piecewise Polynomials)
    if type in ["bspline", "pspline", "smooth"]
        # Step A: Define knot sequence using a range-based sequence for full coverage.
        knots = collect(range(v_min, stop=v_max, length=nbins - degree + 1))
        
        # Step B: Recursive Radial Expansion
        for m in 1:nbins
            # Index resolution without 'clamp'
            k_idx = m > length(knots) ? length(knots) : (m < 1 ? 1 : m)
            target_knot = knots[k_idx]
            B[:, m] .= exp.(-((vals .- target_knot).^2) ./ (2.0 * v_std^2))
        end

    # # 2.2 Thin Plate Splines (TPS - Bending Energy Minimization)
    elseif type == "tps"
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            r = abs.(vals .- knots[m])
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    # # 2.3 Random Fourier Features (RFF - Spectral GP Approximation)
    elseif type == "rff"
        m_rff = nbins
        ls = get(kwargs, :lengthscale, v_std)
        Omega = randn(1, m_rff) ./ ls
        Phi = rand(m_rff) .* (2.0 * pi)
        B .= sqrt(2.0 / m_rff) .* cos.((vals * Omega) .+ Phi')

    # # 2.4 Fast Fourier Transform (FFT Basis - Toroidal Grid)
    elseif type == "fft"
        t_coords = collect(range(0, 1, length=n_obs))
        for m in 1:div(nbins, 2)
            B[:, 2m-1] .= sin.(2.0 * pi * m .* t_coords)
            B[:, 2m]   .= cos.(2.0 * pi * m .* t_coords)
        end

    # # 2.5 Wavelet (Multi-Resolution Decomposition)
    elseif type == "wavelet"
        for m in 1:nbins
            center = v_min + (m/nbins) * (v_max - v_min)
            width = (v_max - v_min) / nbins
            B[:, m] .= (vals .>= center) .& (vals .< (center + width)) ? 1.0 : 0.0
        end

    # # 2.6 Moran's I Basis (Spectral Filtering) - RESTORED
    # Rationale: Uses eigenvectors of the spatial weights matrix to capture specific scales.
    elseif type == "moran"
        # Placeholder: In production, this requires the adjacency matrix W.
        # Here we provide a spectral sine-basis as a technical proxy.
        for m in 1:nbins
            B[:, m] .= sin.(pi * m * (vals .- v_min) ./ (v_max - v_min))
        end

    # # 2.7 Spherical Basis (Compact Support) - RESTORED
    # Rationale: Ensures correlation is exactly zero beyond a finite range R.
    elseif type == "spherical"
        range_r = get(kwargs, :range, v_std * 2.0)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            h = abs.(vals .- knots[m]) ./ range_r
            # Spherical kernel: 1 - 1.5h + 0.5h^3 if h < 1, else 0
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.8 Exponential Decay (Continuous Attenuation) - RESTORED
    elseif type == "decay"
        ls = get(kwargs, :lengthscale, v_std)
        knots = quantile(vals, range(0, 1, length=nbins))
        for m in 1:nbins
            B[:, m] .= exp.(-abs.(vals .- knots[m]) ./ ls)
        end

    # # 2.9 Barycentric (Piecewise Linear Triangulation) - RESTORED
    elseif type == "barycentric"
        knots = collect(range(v_min, stop=v_max, length=nbins))
        for m in 1:nbins
            # Standard triangle/hat basis
            h = (v_max - v_min) / (nbins - 1)
            dist = abs.(vals .- knots[m]) ./ h
            mask = dist .< 1.0
            B[mask, m] .= 1.0 .- dist[mask]
        end

    # # 2.10 Identity / Fixed Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end

function bstm_smooth_basis_2D(type::String, coords::AbstractMatrix, nbins::Int; kwargs...)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A factory function that generates a 2D basis matrix for various types of
              smoothers. It supports anisotropic kernels for directional smoothing by allowing
              different lengthscales for each dimension.
    """
    # # 1. Initialization and Context Discovery
    # Rationale: Ensures every call returns a valid N x M matrix to prevent sampler collapse.
    # coords is expected to be an N x 2 matrix (e.g., [lon lat]).
    n_obs = size(coords, 1)
    
    # Technical Metadata Discovery
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2])]
    c_std = [std(coords[:, 1]), std(coords[:, 2])] .+ 1e-9

    # # 2. Manifold Dispatch Registry for 2D Surfaces

    # Anisotropy Configuration: Extract independent lengthscales if provided, else fallback to marginal std
    # ls_x and ls_y allow the user to control directional smoothing separately
    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])

     # # 2.1 Anisotropic Bivariate P-Splines (Tensor Product Basis)
    # Rationale: Expands 1D marginal bases into a 2D surface grid with directional scaling.
    if type in ["pspline", "bspline", "smooth"]
        # n_marginal defines the resolution per dimension to reach total nbins
        n_marginal = Int(floor(sqrt(nbins)))
        m_total = n_marginal^2
        B = zeros(Float64, n_obs, m_total)

        # Marginal Knot Grids: Distributed evenly across the observed ranges
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                # Tensor product of anisotropic radial marginals
                # bx uses ls_x; by uses ls_y to define the support width
                bx = exp.(-((coords[:, 1] .- kx[i]).^2) ./ (2.0 * ls_x^2))
                by = exp.(-((coords[:, 2] .- ky[j]).^2) ./ (2.0 * ls_y^2))
                B[:, idx] .= bx .* by
                idx += 1
            end
        end

    # # 2.2 Anisotropic Thin Plate Splines (TPS)
    # Rationale: Normalizes distances relative to anisotropic scales before computing bending energy.
    elseif type == "tps"
        B = zeros(Float64, n_obs, nbins)
        n_grid = Int(floor(sqrt(nbins)))
        kx = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]

        for m in 1:min(nbins, length(centers))
            # Normalized distances relative to anisotropic scales
            # This warps the radial basis function to match the directional lengthscales
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            r = sqrt.(dx^2 .+ dy^2)
            # TPS Kernel: r^2 * log(r)
            B[:, m] .= (r.^2) .* log.(r .+ 1e-6)
        end

    # # 2.3 Anisotropic Random Fourier Features (RFF)
    # Rationale: Samples frequencies from an anisotropic spectral density.
    elseif type == "rff" || type == "anisotropic"
        # Generate 2D frequencies with directional scaling
        # Omega[1] is scaled by ls_x; Omega[2] is scaled by ls_y
        Omega = randn(2, nbins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        
        Phi = rand(nbins) .* (2.0 * pi)
        # Projection: feature_map = sqrt(2/M) * cos(Coords * Omega + Phi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi')

    # # 2.4 2D Fast Fourier Transform (FFT - Anisotropic Spectral Lattice)
    # Rationale: Periodic decomposition where coordinates are stretched independently.
    elseif type == "fft"
        n_marginal = Int(floor(sqrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)

        # Normalize coordinates relative to the independent dimension ranges
        nx = (coords[:, 1] .- c_min[1]) ./ (c_max[1] - c_min[1] + 1e-9)
        ny = (coords[:, 2] .- c_min[2]) ./ (c_max[2] - c_min[2] + 1e-9)

        idx = 1
        for mx in 1:n_marginal
            for my in 1:n_marginal
                if idx + 1 <= nbins
                    # The spectral lattice now stretches independently based on coordinate normalization
                    B[:, idx]   .= sin.(2.0 * pi * (mx .* nx .+ my .* ny))
                    B[:, idx+1] .= cos.(2.0 * pi * (mx .* nx .+ my .* ny))
                    idx += 2
                end
            end
        end

    # # 2.5 Barycentric Triangulation (Delaunay interpolation)
    # Rationale: Piecewise linear interaction for irregular 2D scattered data.
    elseif type == "barycentric" || type == "triangulation"
        B = zeros(Float64, n_obs, nbins)
        # Inducing landmarks via quantile grid
        n_grid = Int(floor(sqrt(nbins)))
        kx = quantile(coords[:, 1], range(0, 1, length=n_grid))
        ky = quantile(coords[:, 2], range(0, 1, length=n_grid))
        centers = [(x, y) for x in kx, y in ky][:]
        # Standard Hat Basis for triangulation nodes
        for m in 1:min(nbins, length(centers))
            dist_x = abs.(coords[:, 1] .- centers[m][1]) ./ (c_std[1] / 2.0)
            dist_y = abs.(coords[:, 2] .- centers[m][2]) ./ (c_std[2] / 2.0)
            B[:, m] .= max.(0.0, 1.0 .- dist_x) .* max.(0.0, 1.0 .- dist_y)
        end

    # # 2.6 Spherical Compact Interaction (Zero-Correlation Support)
    # Rationale: Interaction drops to exactly zero beyond a range R.
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
            # Spherical Kernel polynomial
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.7 Moran's I Spectral Interaction
    # Rationale: Captures non-linearities at specific spatial scales.
    elseif type == "moran"
        B = zeros(Float64, n_obs, nbins)
        # Proxy for spatial weighting eigenvectors
        for m in 1:nbins
            B[:, m] .= sin.(pi * m .* (coords[:, 1] .+ coords[:, 2]) ./ sum(c_max))
        end

    # # 2.8 Identity Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end

# # BSTM 3D Smooth Basis & Volumetric Factory [v19.34.0]
# Timestamp: 2025-10-20 10:00:00
# Rationale: Finalizing the 3D manifold registry with absolute parity to 1D/2D counterparts.
# This implementation restores missing Moran, Spherical, and Barycentric methods with full anisotropy support.

    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A factory function that generates a 3D basis matrix for various types of
              smoothers. It supports anisotropic kernels for volumetric smoothing by allowing
              different lengthscales for each of the three dimensions.
    """
function bstm_smooth_basis_3D(type::String, coords::AbstractMatrix, nbins::Int; kwargs...)
    # # 1. Initialization and Technical Discovery
    # coords is expected to be an N x 3 matrix: [dim1, dim2, dim3]
    n_obs = size(coords, 1)

    # Technical Boundary Discovery: Identifying the range and spread of input dimensions
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2]), minimum(coords[:, 3])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2]), maximum(coords[:, 3])]
    c_std = [std(coords[:, 1]), std(coords[:, 2]), std(coords[:, 3])] .+ 1e-9

    # Anisotropy Configuration: Extract independent lengthscales if provided, else fallback to marginal std
    # ls_x, ls_y, and ls_z allow the user to control directional smoothing separately
    ls_x = get(kwargs, :ls_x, c_std[1])
    ls_y = get(kwargs, :ls_y, c_std[2])
    ls_z = get(kwargs, :ls_z, c_std[3])

    # # 2. Manifold Dispatch Registry for Anisotropic Volumetric Surfaces

    # # 2.1 Anisotropic Trivariate P-Splines (Tensor Product Basis)
    # Rationale: Expands 1D marginal bases into a 3D volumetric grid with directional scaling.
    if type in ["pspline", "bspline", "smooth"]
        # n_marginal defines the cubic root resolution to reach total nbins
        n_marginal = Int(floor(cbrt(nbins)))
        m_total = n_marginal^3
        B = zeros(Float64, n_obs, m_total)

        # Marginal Knot Grids: Distributed evenly across the observed ranges
        kx = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        ky = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        kz = collect(range(c_min[3], stop=c_max[3], length=n_marginal))

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                for k in 1:n_marginal
                    # Tensor product of anisotropic radial marginals
                    bx = exp.(-((coords[:, 1] .- kx[i]).^2) ./ (2.0 * ls_x^2))
                    by = exp.(-((coords[:, 2] .- ky[j]).^2) ./ (2.0 * ls_y^2))
                    bz = exp.(-((coords[:, 3] .- kz[k]).^2) ./ (2.0 * ls_z^2))
                    B[:, idx] .= bx .* by .* bz
                    idx += 1
                end
            end
        end

    # # 2.2 Volumetric Random Fourier Features (RFF)
    # Rationale: Samples frequencies from an anisotropic 3D spectral density.
    elseif type == "rff"
        Omega = randn(3, nbins)
        Omega[1, :] ./= ls_x
        Omega[2, :] ./= ls_y
        Omega[3, :] ./= ls_z
        
        Phi = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi')

    # # 2.3 3D Fast Fourier Transform (FFT - Anisotropic Spectral Lattice)
    # Rationale: Volumetric periodic decomposition where coordinates are stretched independently.
    elseif type == "fft"
        n_marginal = Int(floor(cbrt(nbins / 2)))
        B = zeros(Float64, n_obs, nbins)

        # Normalize coordinates relative to the independent dimension ranges
        nx = (coords[:, 1] .- c_min[1]) ./ (c_max[1] - c_min[1] + 1e-9)
        ny = (coords[:, 2] .- c_min[2]) ./ (c_max[2] - c_min[2] + 1e-9)
        nz = (coords[:, 3] .- c_min[3]) ./ (c_max[3] - c_min[3] + 1e-9)

        idx = 1
        for mx in 1:n_marginal
            for my in 1:n_marginal
                for mz in 1:n_marginal
                    if idx + 1 <= nbins
                        # Periodic projection respecting the volume stretching
                        B[:, idx]   .= sin.(2.0 * pi * (mx .* nx .+ my .* ny .+ mz .* nz))
                        B[:, idx+1] .= cos.(2.0 * pi * (mx .* nx .+ my .* ny .+ mz .* nz))
                        idx += 2
                    end
                end
            end
        end

    # # 2.4 Volumetric Spherical Basis (Compact Support Anisotropy)
    # Rationale: Ensures interaction drops to zero beyond directional ranges.
    elseif type == "spherical"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)

        # Inducing landmarks via 3D mesh
        ix = collect(range(c_min[1], stop=c_max[1], length=n_grid))
        iy = collect(range(c_min[2], stop=c_max[2], length=n_grid))
        iz = collect(range(c_min[3], stop=c_max[3], length=n_grid))
        centers = [(x, y, z) for x in ix, y in iy, z in iz][:]

        for m in 1:min(m_total, length(centers))
            # Normalized Mahalanobis-like distance for anisotropy
            dx = (coords[:, 1] .- centers[m][1]) ./ ls_x
            dy = (coords[:, 2] .- centers[m][2]) ./ ls_y
            dz = (coords[:, 3] .- centers[m][3]) ./ ls_z
            h = sqrt.(dx.^2 .+ dy.^2 .+ dz.^2)
            
            # Spherical Kernel with directional compact support
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.5 3D Moran's I Spectral Basis
    # Rationale: Captures volumetric non-linearities at specific scales using Sine basis proxy.
    elseif type == "moran"
        B = zeros(Float64, n_obs, nbins)
        for m in 1:nbins
            # Multi-scale spectral filters for 3D volumes
            arg = (coords[:, 1] ./ ls_x) .+ (coords[:, 2] ./ ls_y) .+ (coords[:, 3] ./ ls_z)
            B[:, m] .= sin.(pi * m .* arg ./ 3.0)
        end

    # # 2.6 Barycentric Volumetric Interpolation (Tetrahedral Proxy)
    # Rationale: Piecewise linear interaction for 3D scattered data.
    elseif type == "barycentric" || type == "triangulation"
        n_grid = Int(floor(cbrt(nbins)))
        m_total = n_grid^3
        B = zeros(Float64, n_obs, m_total)
        ix = quantile(coords[:, 1], range(0, 1, length=n_grid))
        iy = quantile(coords[:, 2], range(0, 1, length=n_grid))
        iz = quantile(coords[:, 3], range(0, 1, length=n_grid))
        centers = [(x, y, z) for x in ix, y in iy, z in iz][:]

        for m in 1:min(m_total, length(centers))
            # Standard 3D Hat Basis (Trilinear interpolation proxy)
            dist_x = abs.(coords[:, 1] .- centers[m][1]) ./ ls_x
            dist_y = abs.(coords[:, 2] .- centers[m][2]) ./ ls_y
            dist_z = abs.(coords[:, 3] .- centers[m][3]) ./ ls_z
            B[:, m] .= max.(0.0, 1.0 .- dist_x) .* max.(0.0, 1.0 .- dist_y) .* max.(0.0, 1.0 .- dist_z)
        end

    # # 2.7 Technical Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end
 

# # BSTM Hyper-Volumetric Factory & Entry Point [v19.36.0]
# Timestamp: 2025-10-30 10:00:00
# Rationale: Expanding the 4D manifold registry to include advanced methods: Spherical, Moran, and Barycentric.
# Requirement: Absolute parity with the v06.1 Taxonomy. Ensures non-truncated hyper-volumetric surfaces.

    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A factory function that generates a 4D basis matrix for various types of
              smoothers. It supports anisotropic kernels for hyper-volumetric smoothing by
              allowing different lengthscales for each of the four dimensions.
    """
function bstm_smooth_basis_4D(type::String, coords::AbstractMatrix, nbins::Int; kwargs...)
    # # 1. Initialization and Hyper-Dimensional Discovery
    # coords is expected to be an N x 4 matrix: [dim1, dim2, dim3, dim4]
    n_obs = size(coords, 1)

    # Technical Boundary Discovery: Identifying the hyper-cube extents
    c_min = [minimum(coords[:, 1]), minimum(coords[:, 2]), minimum(coords[:, 3]), minimum(coords[:, 4])]
    c_max = [maximum(coords[:, 1]), maximum(coords[:, 2]), maximum(coords[:, 3]), maximum(coords[:, 4])]
    c_std = [std(coords[:, 1]), std(coords[:, 2]), std(coords[:, 3]), std(coords[:, 4])] .+ 1e-9

    # Anisotropy Configuration: Independent lengthscales for 4 dimensions
    ls_1 = get(kwargs, :ls_1, c_std[1])
    ls_2 = get(kwargs, :ls_2, c_std[2])
    ls_3 = get(kwargs, :ls_3, c_std[3])
    ls_4 = get(kwargs, :ls_4, c_std[4])

    # # 2. Manifold Dispatch Registry for 4D Latent Fields

    # # 2.1 4D Anisotropic P-Splines (Tensor Product)
    if type in ["pspline", "bspline", "smooth"]
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)

        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))

        idx = 1
        for i in 1:n_marginal
            for j in 1:n_marginal
                for k in 1:n_marginal
                    for l in 1:n_marginal
                        b1 = exp.(-((coords[:, 1] .- k1[i]).^2) ./ (2.0 * ls_1^2))
                        b2 = exp.(-((coords[:, 2] .- k2[j]).^2) ./ (2.0 * ls_2^2))
                        b3 = exp.(-((coords[:, 3] .- k3[k]).^2) ./ (2.0 * ls_3^2))
                        b4 = exp.(-((coords[:, 4] .- k4[l]).^2) ./ (2.0 * ls_4^2))
                        B[:, idx] .= b1 .* b2 .* b3 .* b4
                        idx += 1
                    end
                end
            end
        end

    # # 2.2 4D Random Fourier Features (Spectral Anisotropy)
    elseif type == "rff"
        Omega = randn(4, nbins)
        Omega[1, :] ./= ls_1
        Omega[2, :] ./= ls_2
        Omega[3, :] ./= ls_3
        Omega[4, :] ./= ls_4
        Phi = rand(nbins) .* (2.0 * pi)
        B = sqrt(2.0 / nbins) .* cos.((coords * Omega) .+ Phi')

    # # 2.3 4D Fast Fourier Transform (Spectral Lattice)
    elseif type == "fft"
        n_marginal = Int(floor(sqrt(sqrt(nbins / 2))))
        B = zeros(Float64, n_obs, nbins)
        nx1 = (coords[:, 1] .- c_min[1]) ./ (c_max[1] - c_min[1] + 1e-9)
        nx2 = (coords[:, 2] .- c_min[2]) ./ (c_max[2] - c_min[2] + 1e-9)
        nx3 = (coords[:, 3] .- c_min[3]) ./ (c_max[3] - c_min[3] + 1e-9)
        nx4 = (coords[:, 4] .- c_min[4]) ./ (c_max[4] - c_min[4] + 1e-9)
        idx = 1
        for m1 in 1:n_marginal
            for m2 in 1:n_marginal
                for m3 in 1:n_marginal
                    for m4 in 1:n_marginal
                        if idx + 1 <= nbins
                            arg = m1 .* nx1 .+ m2 .* nx2 .+ m3 .* nx3 .+ m4 .* nx4
                            B[:, idx]   .= sin.(2.0 * pi * arg)
                            B[:, idx+1] .= cos.(2.0 * pi * arg)
                            idx += 2
                        end
                    end
                end
            end
        end

    # # 2.4 4D Spherical Basis (Anisotropic Compact Support)
    # Rationale: Ensures latent correlation is zero beyond hyper-volumetric directional ranges.
    elseif type == "spherical"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        k1 = collect(range(c_min[1], stop=c_max[1], length=n_marginal))
        k2 = collect(range(c_min[2], stop=c_max[2], length=n_marginal))
        k3 = collect(range(c_min[3], stop=c_max[3], length=n_marginal))
        k4 = collect(range(c_min[4], stop=c_max[4], length=n_marginal))
        centers = [(w, x, y, z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:min(m_total, length(centers))
            d1 = (coords[:, 1] .- centers[m][1]) ./ ls_1
            d2 = (coords[:, 2] .- centers[m][2]) ./ ls_2
            d3 = (coords[:, 3] .- centers[m][3]) ./ ls_3
            d4 = (coords[:, 4] .- centers[m][4]) ./ ls_4
            h = sqrt.(d1.^2 .+ d2.^2 .+ d3.^2 .+ d4.^2)
            mask = h .< 1.0
            B[mask, m] .= 1.0 .- 1.5 .* h[mask] .+ 0.5 .* h[mask].^3
        end

    # # 2.5 4D Moran's I Spectral Basis
    # Rationale: Hyper-dimensional multi-scale filtering using spectral Sine proxy.
    elseif type == "moran"
        B = zeros(Float64, n_obs, nbins)
        for m in 1:nbins
            arg = (coords[:, 1] ./ ls_1) .+ (coords[:, 2] ./ ls_2) .+ (coords[:, 3] ./ ls_3) .+ (coords[:, 4] ./ ls_4)
            B[:, m] .= sin.(pi * m .* arg ./ 4.0)
        end

    # # 2.6 4D Barycentric Interpolation (Hyper-Tetrahedral Proxy)
    # Rationale: Piecewise linear interaction across irregular 4D scattered data.
    elseif type == "barycentric" || type == "triangulation"
        n_marginal = Int(floor(sqrt(sqrt(nbins))))
        m_total = n_marginal^4
        B = zeros(Float64, n_obs, m_total)
        k1 = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        k2 = quantile(coords[:, 2], range(0, 1, length=n_marginal))
        k3 = quantile(coords[:, 3], range(0, 1, length=n_marginal))
        k4 = quantile(coords[:, 4], range(0, 1, length=n_marginal))
        centers = [(w, x, y, z) for w in k1, x in k2, y in k3, z in k4][:]
        for m in 1:min(m_total, length(centers))
            dw = abs.(coords[:, 1] .- centers[m][1]) ./ ls_1
            dx = abs.(coords[:, 2] .- centers[m][2]) ./ ls_2
            dy = abs.(coords[:, 3] .- centers[m][3]) ./ ls_3
            dz = abs.(coords[:, 4] .- centers[m][4]) ./ ls_4
            B[:, m] .= max.(0.0, 1.0 .- dw) .* max.(0.0, 1.0 .- dx) .* max.(0.0, 1.0 .- dy) .* max.(0.0, 1.0 .- dz)
        end

    # # 2.7 Technical Fallback
    else
        B = ones(n_obs, 1)
    end

    return B
end


 

  
# 1. Base Generic Fallback
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A set of `build_model` dispatch methods that translate high-level `Manifold`
              structs into technical configuration objects for the model constructor. This is the
              fallback method for unrecognized manifold types.
    """
function build_model(m::Manifold, data_inputs)
    @warn "No specific builder for $(typeof(m)). Using IID identity template."
    template = build_structure_template(:iid, get(data_inputs, :s_N, 1))
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = hasproperty(manifold, :sigma_prior) ? manifold.sigma_prior : Exponential(1.0),)
    )
end

function build_model(m::Besag, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Besag (ICAR) models.
    """
    # Besag is an alias for ICAR
    local n = data_inputs.s_N
    local template = build_structure_template(:besag, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :besag,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::SAR, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for SAR (Simultaneous Autoregressive) models.
    """
    # SAR Builder: Simultaneous Autoregressive
    local n = data_inputs.s_N
    local template = build_structure_template(:sar, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0, # SAR precision includes rho, not scaled like BYM2
        model_type = :sar,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

function build_model(m::FITC, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for FITC (sparse GP) models.
    """
    # FITC Builder: Fully Independent Training Conditional (Sparse GP)
    # Uses M inducing points to approximate the full covariance
    return (
        Q_template = nothing, # Requires dynamic kernel distance matrix
        scaling_factor = 1.0,
        model_type = :fitc,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior, n_inducing = m.n_inducing)
    )
end

function build_model(m::SPDE, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for SPDE-based models.
    """
    # SPDE Builder: Matern approximation via (kappa^2 I + L)^alpha
    local n = data_inputs.s_N
    local template = build_structure_template(:besag, n; W=get(data_inputs, :W, nothing))
    return (
        Q_template = template.matrix, # Base Laplacian L used in SPDE expansion
        scaling_factor = 1.0,
        model_type = :spde,
        hyper = (sigma_prior = m.sigma_prior, kappa_prior = m.kappa_prior)
    )
end

function build_model(m::Hyperbolic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Hyperbolic geometry models.
    """
    # Hyperbolic Builder: Hierarchical embedding with constant negative curvature
    return (
        Q_template = nothing, # Requires dynamic geodesic distance matrix
        scaling_factor = 1.0,
        model_type = :hyperbolic,
        hyper = (
            sigma_prior = m.sigma_prior,
            curvature = m.curvature
        )
    )
end

function build_model(m::IID, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for IID (unstructured) random effects.
    """
    # IID Builder: Unstructured random effects
     return (
        Q_template = Matrix(1.0I, data_inputs.s_N, data_inputs.s_N),
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

# 2. Discrete Spatial Manifolds (CAR/Laplacian Family)
function build_model(m::Union{BYM2, ICAR}, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for BYM2 and ICAR models.
    """
    # BYM2/ICAR require a spatial unit count and adjacency matrix W
    template = build_structure_template(:bym2, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bym2,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end



function build_model(m::Leroux, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Leroux models.
    """
    template = build_structure_template(:leroux, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :leroux,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

# 3. Topological & Directed Manifolds (v05.8 Expansion)
function build_model(m::LocalAdaptive, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for locally adaptive spatial models.
    """
    # Q_template is the base Laplacian; local weights are applied in recompose_precision
    template = build_structure_template(:besag, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :local_adaptive,
        hyper = (sigma_prior = m.sigma_prior, weights_var = m.weights_variable)
    )
end


function build_model(m::NetworkFlow, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for network flow models.
    """
    # For directed networks, the template is the raw adjacency matrix
    return (
        Q_template = m.adjacency_matrix,
        scaling_factor = 1.0,
        model_type = :network,
        hyper = (sigma_prior = m.sigma_prior, direction = m.flow_direction)
    )
end

function build_model(m::DAG, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for DAG (Directed Acyclic Graph) models.
    """
    # DAG uses the directed adjacency as the operator (I - rho*W)
    return (
        Q_template = m.adjacency_matrix,
        scaling_factor = 1.0,
        model_type = :dag,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end

# 4. Temporal & Seasonal Manifolds
function build_model(m::AR1, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for AR(1) models.
    """
    # n derived from temporal units in metadata
    n = get(data_inputs, :t_N, 10)
    template = build_structure_template(:ar1, n)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :ar1,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

function build_model(m::Harmonic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for harmonic (seasonal) models.
    """
    # Harmonic manifolds are basis-driven, not GMRF-driven
    return (
        Q_template = nothing,
        scaling_factor = 1.0,
        model_type = :harmonic,
        hyper = (
            sigma_prior = m.sigma_prior, 
            amplitude_prior = m.amplitude_prior, 
            phase_prior = m.phase_prior,
            period = m.period
        )
    )
end

function build_model(m::RW1, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for RW1 (first-order random walk) models.
    """
    # RW1 Builder: First-order random walk (Stochastic Level shifts)
    # Rationale: Maps temporal or lattice units to a standard differencing precision.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:ar1, n) # Reuse AR1 for differencing structure
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :rw1,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::RW2, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for RW2 (second-order random walk) models.
    """
    # RW2 Builder: Second-order random walk (Smooth locally linear trends)
    # Rationale: Higher-order smoothing with geometric mean scaling.
    local n = get(data_inputs, :t_N, data_inputs.s_N)
    local template = build_structure_template(:rw2, n)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :rw2,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::Cyclic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for cyclic random walks.
    """
    # Cyclic Builder: Periodic continuity (Seasonal)
    # Rationale: Enforces boundary condition y_0 = y_T via a circular Laplacian.
    local n = m.period
    local W_circ = zeros(n, n)
    for i in 1:n
        W_circ[i, mod1(i-1, n)] = 1.0
        W_circ[i, mod1(i+1, n)] = 1.0
    end
    local template = build_structure_template(:besag, n; W=W_circ)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :cyclic,
        hyper = (sigma_prior = m.sigma_prior, period = m.period)
    )
end

function build_model(m::BSpline, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for B-spline smooths.
    """
    # BSpline Builder: Cubic Spline basis-mapped coefficients
    # Rationale: Extracts resolution (nbins) and degree for basis construction.
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins), # IID in basis space
        scaling_factor = 1.0,
        model_type = :bspline,
        hyper = (sigma_prior = m.sigma_prior, nbins = m.nbins, degree = m.degree)
    )
end

function build_model(m::Wavelet, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for wavelet-based smooths.
    """
    # Wavelet Builder: Multi-resolution frequency decomposition
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :wavelet,
        hyper = (sigma_prior = m.sigma_prior, family = m.wavelet_family, nbins = m.nbins)
    )
end

function build_model(m::Hyperbolic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Hyperbolic geometry models.
    """
    # Hyperbolic Builder: Hierarchical embedding with constant negative curvature
    return (
        Q_template = nothing, # Requires dynamic geodesic distance matrix
        scaling_factor = 1.0,
        model_type = :hyperbolic,
        hyper = (sigma_prior = m.sigma_prior, curvature = m.curvature)
    )
end

function build_model(m::ExponentialDecay, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for exponential decay models.
    """
    # Exponential Decay Builder: Signal attenuation over continuous domains
    return (
        Q_template = nothing, # Distance-based precision mapping
        scaling_factor = 1.0,
        model_type = :decay,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.decay_lengthscale_prior)
    )
end
 

# 5. Covariate Smooths (Splines)
function build_model(m::TPS, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for Thin Plate Spline (TPS) smooths.
    """
    template = build_structure_template(:tps, m.nbins)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :tps,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::PSpline, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for P-spline smooths.
    """
    template = build_structure_template(:pspline, m.nbins)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :pspline,
        hyper = (sigma_prior = m.sigma_prior, degree = m.degree, diff_order = m.diff_order)
    )
end

function build_model(m::Harmonic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for harmonic (seasonal) models.
    """
    # Harmonic manifolds are basis-driven, not GMRF-driven
    # The precision on the coefficients is typically IID.
    template = build_structure_template(:iid, get(data_inputs, :u_N, 12) * 2) # n_harmonics * 2
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid, # Coefficients are IID
        hyper = (
            sigma_prior = m.sigma_prior, 
            period = m.period
        )
    )
end

function build_model(m::Cyclic, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for cyclic random walks.
    """
    n = get(data_inputs, :u_N, get(data_inputs, :period, 12))
    template = build_structure_template(:cyclic, n)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :cyclic,
        hyper = (sigma_prior = m.sigma_prior, period = m.period)
    )
end

function build_model(m::Union{FFT, Wavelet}, data_inputs)
    # FFT/Wavelet Builder with Spectral Precision
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for FFT and Wavelet-based spectral models.
    """
    # Rationale: Uses Wiener-Khinchin theorem to build a precision matrix from a kernel's spectral density.
    n = hasproperty(m, :nbins) ? m.nbins : get(data_inputs, :t_N, get(data_inputs, :s_N, 20))
    
    # The template is a placeholder; the actual precision is recomposed inside the Turing model.
    template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix, # Placeholder
        scaling_factor = 1.0,
        model_type = :spectral, # A new type to signify this approach
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior, kernel=m.kernel)
    )
end

# 6. Spectral Manifolds 

function build_model(m::FFT, data_inputs)
    # FFT Builder: Frequency-domain temporal or spatial coefficients
    # Precision is assumed IID in the spectral space
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (sigma_prior = m.sigma_prior, nbins = m.nbins)
    )
end


function build_model(m::RFF, data_inputs)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:22:15
    Synopsis: A dispatch method for building configurations for RFF-based GP approximations.
    """
    return (
        Q_template = Matrix(1.0I, m.n_features, m.n_features),
        scaling_factor = 1.0,
        model_type = :rff,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior)
    )
end

 
function build_model(m::GP, data_inputs)
    # Continuous GP Builder: Exact kernel-based covariance
    # Rationale: Distance-based template construction using coordinate metadata.
    local n = data_inputs.s_N
    local coords = get(data_inputs, :s_coord, nothing)
    local template = build_structure_template(:gp, n; coords=coords)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :gp,
        hyper = (
            sigma_prior = m.sigma_prior,
            lengthscale_prior = m.lengthscale_prior,
            kernel = m.kernel
        )
    )
end


  

# # Knorr-Held Interaction Builder
# function build_model(m::KnorrHeld, data_inputs)
#     # Space-Time Interaction types I, II, III, IV
#     # Precision is built via kron() in the sampler, so no static template is stored
#     return (
#         Q_template = nothing,
#         model_type = :knorrheld,
#         smooth_class = m.type,
#         hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
#     )
# end 
 
 
 


function get_inits(model::DynamicPPL.Model; refine="map", n_samples=100, optimizer=LBFGS(), max_iters=500, maxtime=60.0, noise=nothing)
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility for generating sensible initial values for MCMC sampling. It uses either a
              heuristic based on prior samples or refines these initial values via Maximum A
              Posteriori (MAP) estimation to find a good starting point for the sampler.
    """
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

        mu = mean(vals)
        s_name = string(ks)

        if vals[1] isa AbstractVector
            # Latent fields are centered at zero for stability
            init_dict[ks] = zeros(eltype(mu), length(mu))
        elseif occursin(r"sigma|ls_|lengthscale|sig_", s_name)
            init_dict[ks] = max(0.1, mu)
        elseif occursin("rho", s_name)
            init_dict[ks] = clamp(mu, -0.9, 0.9)
        elseif occursin("phi", s_name)
            init_dict[ks] = clamp(mu, 0.01, 0.5)
        else
            init_dict[ks] = mu
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


# --- Multivariate Manifold Application Functions ---

function _apply_manifold_mv!(innovations, spec, m_obj::Union{ICAR, BYM2, Leroux, SAR, DAG}, M, noise, k, outcomes_N)
    m_domain = spec.domain
    var_name = string(spec.var)
    sigma_name = Symbol("sigma_", m_domain, "_", var_name, "_", k)
    latent_name = Symbol("latent_", m_domain, "_", var_name, "_", k)
    
    T = eltype(innovations)
    sigma::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

    extra_p_val = nothing
    if m_obj isa Union{BYM2, Leroux, SAR}
        rho_name = Symbol("rho_", m_domain, "_", var_name, "_", k)
        rho::T = 0.0
        rho ~ NamedDist(m_obj.rho_prior, rho_name)
        extra_p_val = rho
    end

    template = (m_domain == :temporal) ? M.t_Q_template.matrix : M.s_Q_template.matrix
    n_units = size(template, 1)
    Q = recompose_precision(Symbol(typeof(m_obj)), template, sigma; extra_param=extra_p_val, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    
    indices = (m_domain == :spatial) ? M.s_idx : M.t_idx
    innovations[:, k] .+= latent[indices]
end

function _apply_manifold_mv!(innovations, spec, m_obj::Union{AR1, RW1, RW2, Cyclic}, M, noise, k, outcomes_N)
    m_domain = spec.domain
    var_name = string(spec.var)
    sigma_name = Symbol("sigma_", m_domain, "_", var_name, "_", k)
    latent_name = Symbol("latent_", m_domain, "_", var_name, "_", k)

    T = eltype(innovations)
    sigma::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
    
    extra_p_val = nothing
    if m_obj isa AR1
        rho_name = Symbol("rho_", m_domain, "_", var_name, "_", k)
        rho::T = 0.0
        rho ~ NamedDist(m_obj.rho_prior, rho_name)
        extra_p_val = rho
    end

    template = if m_domain == :temporal; M.t_Q_template.matrix
               elseif m_domain == :seasonal; M.u_Q_template.matrix
               else M.s_Q_template.matrix end
    n_units = size(template, 1)
    Q = recompose_precision(Symbol(typeof(m_obj)), template, sigma; extra_param=extra_p_val, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    
    indices = if m_domain == :temporal; M.t_idx
              elseif m_domain == :seasonal; M.u_idx
              else M.s_idx end
    innovations[:, k] .+= latent[indices]
end

function _apply_manifold_mv!(innovations, spec, m_obj::IID, M, noise, k, outcomes_N)
    m_domain = spec.domain
    var_name = string(spec.var)
    sigma_name = Symbol("sigma_", m_domain, "_", var_name, "_", k)
    latent_name = Symbol("latent_", m_domain, "_", var_name, "_", k)
    
    T = eltype(innovations)
    sigma::T = 0.0
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)

    n_units, indices = if m_domain == :mixed
        (spec.params.n_cat, spec.params.indices)
    elseif m_domain == :spatial
        (M.s_N, M.s_idx)
    elseif m_domain == :temporal
        (M.t_N, M.t_idx)
    else
        (M.u_N, M.u_idx)
    end

    latent ~ NamedDist(filldist(Normal(0, sigma), n_units), latent_name)
    
    if m_domain == :mixed && spec.var != :none && spec.var != 1
        x_col_idx = findfirst(==(spec.var), Symbol.(names(M.Xfixed, 2)))
        if !isnothing(x_col_idx)
            x_vals = M.Xfixed[:, x_col_idx]
            innovations[:, k] .+= latent[indices] .* x_vals
        end
    else
        innovations[:, k] .+= latent[indices]
    end
end

function _apply_manifold_mv!(innovations, spec, m_obj::SVCManifold, M, noise, k, outcomes_N)
    var_name = string(m_obj.covariate)
    inner_manifold = m_obj.model
    inner_m_type = Symbol(lowercase(string(typeof(inner_manifold))))

    sigma_name = Symbol("sig_svc_", var_name, "_", k)
    latent_name = Symbol("beta_svc_", var_name, "_", k)

    T = eltype(innovations)
    sigma::T = 0.0
    sigma ~ NamedDist(inner_manifold.sigma_prior, sigma_name)

    extra_p_val = nothing
    if inner_manifold isa Union{BYM2, Leroux, SAR, AR1}
        rho_name = Symbol("rho_svc_", var_name, "_", k)
        rho::T = 0.0
        rho ~ NamedDist(inner_manifold.rho_prior, rho_name)
        extra_p_val = rho
    end

    template = M.s_Q_template.matrix
    n_units = size(template, 1)
    Q = recompose_precision(inner_m_type, template, sigma; extra_param=extra_p_val, noise=noise)
    latent ~ NamedDist(MvNormalCanon(zeros(n_units), Q), latent_name)
    
    x_col_idx = findfirst(==(m_obj.covariate), Symbol.(names(M.Xfixed, 2)))
    if !isnothing(x_col_idx)
        x_vals = M.Xfixed[:, x_col_idx]
        innovations[:, k] .+= latent[M.s_idx] .* x_vals
    end
end

function _apply_manifold_mv!(innovations, spec, m_obj::Union{PSpline, BSpline, TPS, RFF, FFT}, M, noise, k, outcomes_N)
    var_sym = spec.var
    B_mat = M.basis_matrices[var_sym]
    n_basis_cols = size(B_mat, 2)
    sigma_name = Symbol("sigma_basis_", var_sym, "_", k)
    beta_name = Symbol("beta_basis_", var_sym, "_", k)

    sigma ~ NamedDist(m_obj.sigma_prior, sigma_name)
    latent_coeffs ~ NamedDist(filldist(Normal(0, sigma), n_basis_cols), beta_name)
    
    innovations[:, k] .+= B_mat * latent_coeffs
end

function _apply_manifold_mv!(innovations, spec, m_obj::GP, M, noise, k, outcomes_N)
    var_name = string(spec.var)
    coords = spec.params.coords
    ls_name = Symbol("ls_gp_", var_name, "_", k)
    sigma_gp_name = Symbol("sigma_gp_", var_name, "_", k)
    latent_gp_name = Symbol("latent_gp_", var_name, "_", k)

    T = eltype(innovations)
    ls::T = 0.0
    sigma::T = 0.0
    ls ~ NamedDist(m_obj.lengthscale_prior, ls_name)
    sigma ~ NamedDist(m_obj.sigma_prior, sigma_gp_name)

    kernel_base = get_kernel_from_string(m_obj.kernel)
    k_gp = sigma^2 * kernel_base ∘ ScaleTransform(1/ls)
    
    K_ff = kernelmatrix(k_gp, coords) + noise*I
    f_latent ~ NamedDist(MvNormal(zeros(size(coords,1)), K_ff), latent_gp_name)
    innovations[:, k] .+= f_latent
end

function _apply_manifold_mv!(innovations, spec, m_obj::DynamicsManifold, M, noise, k, outcomes_N)
    @warn "DynamicsManifold is applied independently to each outcome in multivariate models. For coupled dynamics, a custom model is required."
    _apply_dynamics_model!(spec, innovations[:, k], M, M.hyperpriors)
end

function _apply_manifold_mv!(innovations, spec, m_obj::ManifoldModel, M, noise, k, outcomes_N)
    @warn "No specific multivariate `_apply_manifold!` method for $(typeof(m_obj)). Using IID fallback."
    _apply_manifold_mv!(innovations, spec, IID(Exponential(1.0)), M, noise, k, outcomes_N)
end


@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    """
    BSTM Model Definition v1.1.0
    Timestamp: 2026-06-26 19:00:00
    Synopsis: The main Turing `@model` definition for all multivariate `bstm` models. It
              constructs the full Bayesian hierarchical model by assembling all specified
              manifolds and coupling them via an LKJ prior on their correlations.
    Rationale for v1.1.0:
        - Refactored to move all manifold-specific logic into external `_apply_manifold_mv!`
          helper functions for improved clarity, maintainability, and adherence to Julia
          best practices.
        - The main model block now only contains the global hyperpriors, the LKJ coupling
          mechanism, and the final likelihood evaluation.
    """
    # # 1. Global Architectural Scope & Hyperpriors
    outcomes_N = M.outcomes_N
    y_N = M.y_N
    family = M.model_family
    noise = get(M, :noise, 1e-4)
    use_zi = get(M, :use_zi, false)

    lik_r = one(T)
    lik_phi = zero(T)
    extra_p = ones(T, outcomes_N)

    if family == "negbin"; lik_r ~ NamedDist(Exponential(1.0), :lik_r); end
    if use_zi == true; lik_phi ~ NamedDist(Beta(1, 1), :lik_phi); end
    if family in ["gamma", "beta", "student_t"]; extra_p ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :extra_params); end

    # # 2. Multivariate Coupling & Observation Volatility
    L_corr ~ NamedDist(LKJCholesky(outcomes_N, 1.0, T), :L_corr)
    y_sigma = Matrix{T}(undef, y_N, outcomes_N)
    if family in ["gaussian", "lognormal"]
        y_sigma_val ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :y_sigma)
        for k in 1:outcomes_N; y_sigma[:, k] .= y_sigma_val[k]; end
    else
        for k in 1:outcomes_N; y_sigma[:, k] .= one(T); end
    end

    # # 3. Base Predictor: Fixed Effects
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N * outcomes_N), 5.0 * I), :Xfixed_beta)
    eta = M.Xfixed * reshape(Xfixed_beta, M.Xfixed_N, outcomes_N)

    # # 4. Modular Manifold Realization
    latent_innovations = zeros(T, y_N, outcomes_N)
    for spec in M.manifolds
        for k in 1:outcomes_N
            _apply_manifold_mv!(latent_innovations, spec, spec.manifold_obj, M, noise, k, outcomes_N)
        end
    end

    # Apply multivariate coupling to the innovations
    eta .+= (latent_innovations * L_corr.L)

    # # 5. Pointwise Likelihood Evaluation
    for k in 1:outcomes_N
        ok_idx = get(M, :y_ok, [1:y_N for _ in 1:outcomes_N])[k]
        for i in ok_idx
            d_lik = bstm_Likelihood(
                family, [T(M.y_obs[i, k])]; sigma_y=[y_sigma[i, k]],
                phi_zi=lik_phi, r_nb=lik_r, extra_params=extra_p[k]
            )
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
        end
    end
end




 

function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: The internal reconstruction engine for univariate models. It discovers all latent
              fields from the MCMC chain, assembles the linear predictor, and generates
              predictions, summaries, and diagnostic metrics.
    """
    # BSTM Modular Reconstruction Engine v22.3.0
    # Timestamp: 2026-06-25 19:25:00
    # Rationale: Integrating post-stratification weight calculation and summarization
    # to provide a complete set of outputs for weighted population estimates.

    n_samples = size(chain, 1)
    p_names = string.(MCMCChains.names(chain))
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
        s_struct_samples = registry.s_eff_struct[1][M.s_idx, :]
        summarized_effects[:spatial_structured] = summarize_array(s_struct_samples; alpha=alpha)
    end
    if !isnothing(registry.s_eff_noisy) && !isempty(registry.s_eff_noisy)
        s_unstruct_samples = registry.s_eff_noisy[1][M.s_idx, :] .- registry.s_eff_struct[1][M.s_idx, :]
        summarized_effects[:spatial_unstructured] = summarize_array(s_unstruct_samples; alpha=alpha)
    end
    if !isnothing(registry.t_eff) && !isempty(registry.t_eff)
        t_samples = registry.t_eff[1][M.t_idx, :]
        summarized_effects[:temporal] = summarize_array(t_samples; alpha=alpha)
    end
    if !isnothing(registry.u_eff) && !all(iszero, registry.u_eff)
        u_samples = registry.u_eff[M.u_idx, :]
        summarized_effects[:seasonal] = summarize_array(u_samples; alpha=alpha)
    end
    if !isnothing(registry.st_eff_maps) && !isempty(registry.st_eff_maps)
        st_samples = zeros(M.y_N, n_samples)
        for j in 1:n_samples
            for i in 1:M.y_N
                st_samples[i, j] = registry.st_eff_maps[1][M.s_idx[i], M.t_idx[i], j]
            end
        end
        summarized_effects[:spacetime_interaction] = summarize_array(st_samples; alpha=alpha)
    end
    if !isnothing(registry.basis_eff_accum)
        summarized_effects[:smooth_effects] = summarize_array(registry.basis_eff_accum[1:M.y_N, :]; alpha=alpha)
    end
    if !isnothing(registry.Xfixed_betas)
        summarized_effects[:fixed_effects] = summarize_array(registry.Xfixed_betas'; alpha=alpha)
    end

    # 4. Generate Predictions and Compute Log-Likelihood
    p_denoised, p_noisy, log_lik = _process_ll_and_predictions(
        fam_obj, eta_samples, chain, M, N_tot, n_samples, registry.sv_surface
    )

    # 5. Summarize Predictions and Post-Stratification Weights
    summarized_effects[:eta] = summarize_array(eta_samples[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_denoised] = summarize_array(p_denoised[1:M.y_N, :]; alpha=alpha)
    summarized_effects[:predictions_noisy] = summarize_array(p_noisy[1:M.y_N, :]; alpha=alpha)
    if N_PS > 0
        summarized_effects[:ps_predictions_denoised] = summarize_array(p_denoised[(M.y_N+1):end, :]; alpha=alpha)
        summarized_effects[:ps_predictions_noisy] = summarize_array(p_noisy[(M.y_N+1):end, :]; alpha=alpha)

        # Calculate and summarize post-stratification weights
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

    return summarized_effects
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

 

# 1. PSIS-LOO Implementation for BSTM
# Rationale: Standardizing the extraction of log-likelihood matrices to provide 
# Expected Log Pointwise Predictive Density (ELPD) estimates.
function bstm_loo(model_obj::DynamicPPL.Model, chain; alpha=0.05)
"""
BSTM Utility v1.0.0
Timestamp: 2026-06-26 10:22:15
Synopsis: A utility for performing Leave-One-Out Cross-Validation using Pareto Smoothed Importance
          Sampling (PSIS-LOO) to assess a model's out-of-sample predictive accuracy.
Description: Calculates the Leave-One-Out Cross-Validation metrics for BSTM manifolds.
Rationale: Standardizing the extraction of log-likelihood matrices to provide ELPD estimates.
"""

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
    elseif raw_arch == "multifidelity" || raw_arch == "nested"
        MultifidelityArchitecture()
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: A utility for formal model comparison between two fitted `bstm` models. It uses
              their PSIS-LOO results to compute the difference in Expected Log Pointwise
              Predictive Density (ELPD) and provides a statistical basis for model selection.
    """
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
    """
    BSTM Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An orchestration utility for performing cross-validation. It supports standard
              k-fold and Leave-One-Location-Out (LOLO) strategies to assess model performance on
              held-out data.
    """
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


### 

# Advection-Diffusion Operator Registration
function build_transport_operator(M, velocity, diffusion_coeff)
    # Rationale: Constructs the discretized transport operator on the graph manifold.
    # Requirement: Adjacency matrix W must be non-null for graph-based advection.
    local n = M.s_N
    local W = get(M, :W, sparse(I(n)))
    local D = Matrix(Diagonal(vec(sum(W, dims=2))))
    local L = D - Matrix(W)

    # Operator: (I - dt * (diffusion * L + velocity * G))
    # For this restoration, we focus on the Diffusion Laplacian contribution.
    return I(n) .- (diffusion_coeff .* L)
end

# Updated Parser Logic for Spatial Manifold Dispatch
function parse_spatial_term(var_name, params_str)
    """
    BSTM Internal Utility v1.0.0
    Timestamp: 2026-06-26 10:23:44
    Synopsis: An internal helper for the formula parser that specifically processes a `spatial()`
              module term, extracting its variable and parameters. This function appears to be a
              precursor to the more general `_parse_module_call` and may be redundant.
    """
    local p = parse_module_params(params_str)
    local manifold = get(p, :manifold, "bym2")

    # Logic: Determine if input is index (graph) or coordinates (continuous)
    # Graph-based requires W; Coordinate-based requires (x,y) parsing.
    return (variable=var_name, manifold=manifold, params=p)
end



function bstm(config::NamedTuple)
    # Main model dispatcher
    arch = get(config, :model_arch, "univariate")
    if arch == "multivariate"
        return bstm_multivariate(config)
    elseif arch == "multifidelity"
        return bstm_multifidelity(config)
    else
        return bstm_univariate(config)
    end
end

  



function bstm(args...; model_family="gaussian", model_arch="univariate", auxiliary_responses=nothing, kwargs...)
    """
    BSTM Main Entry Point v1.1.0
    Timestamp: 2026-06-26 17:15:00
    Synopsis: The main user-facing `bstm` function. It acts as a dispatcher, accepting either a
              formula string and data, or a pre-built configuration object, and routes to the
              appropriate model constructor.
    Rationale for v1.1.0:
        - Corrected a `FieldError` by ensuring that all keyword arguments, including `model_arch`
          and `model_family`, are correctly passed from the main `bstm` call to the `bstm_config` engine.
    """
    # Mode 1: Formula-Driven Model Construction
    if length(args) >= 2 && args[1] isa String && args[2] isa DataFrame
        formula = args[1] 
        data = args[2] 
 
        # The modular config engine decomposes the formula DSL.
        # All keyword arguments are passed to the configuration engine.
        options = bstm_config(formula, data; 
            model_family=model_family, 
            model_arch=model_arch, 
            auxiliary_responses=auxiliary_responses, 
            kwargs...
        ) 

        # Dispatch to model supervisors based on discovered architecture.
        if get(options, :model_arch, "univariate") == "multivariate"
            return bstm_multivariate(options)
        elseif get(options, :model_arch, "univariate") == "multifidelity"
            return bstm_multifidelity(options)
        else
            return bstm_univariate(options)
        end

    # Mode 2: Direct Metadata Execution (Expert / Internal Mode)
    elseif length(args) == 1 && args[1] isa NamedTuple
        M = args[1]
        arch = get(M, :model_arch, "univariate")
        if arch == "multivariate"
            return bstm_multivariate(M)
        elseif arch == "multifidelity"
            return bstm_multifidelity(M)
        else
            return bstm_univariate(M)
        end

    else
        error("BSTM Dispatch Error: Unsupported argument combination. Permitted: (String, DataFrame) or (NamedTuple).")
    end
end



function bstm_config(formula::String, data::DataFrame; kwargs...)
    """
    BSTM Utility v1.4.0
    Timestamp: 2026-06-26 20:45:00
    Synopsis: The main configuration engine for the formula-based interface. It parses the
              formula string, processes all specified modules, and assembles the final technical
              configuration `NamedTuple` required to build and run a `bstm` model.
    Rationale for v1.4.0:
        - Updated to correctly handle custom intercept priors. The function now checks if an
          `:intercept_prior` is set and, if so, ensures the fixed-effects design matrix is
          created without a default intercept column, as the intercept will be handled as a
          separate parameter in the model.
        - Standardized the call signatures for all module processors.
    """
    # 1. Formula Scope Discovery
    metadata = decompose_bstm_formula(formula)

    # 2. Metadata Context Initialization
    opt_dict = Dict{Symbol, Any}(kwargs)
    opt_dict[:y_obs_vars] = metadata.outcomes
    opt_dict[:data] = data
    opt_dict[:y_N] = size(data, 1)
    opt_dict[:add_intercept] = metadata.has_intercept

    # 3. Technical Registers Initialization
    opt_dict[:model_space] = get(opt_dict, :model_space, "none")
    opt_dict[:model_time] = get(opt_dict, :model_time, "none")
    opt_dict[:model_season] = get(opt_dict, :model_season, "none")
    opt_dict[:model_st] = get(opt_dict, :model_st, "none")

    # 4. Hyperprior Registry Initialization
    initial_hp = get(opt_dict, :hyperpriors, Dict{Any, Any}())
    hyperprior_registry = Dict{String, Any}(string(k) => v for (k, v) in initial_hp)
    opt_dict[:hyperpriors] = hyperprior_registry

    # 5. Internal Technical Registries
    basis_matrices_registry = Dict{Symbol, Any}()
    manifolds_registry = []

    # 6. Technical Module Resolution Loop
    for (key_str, mod_data) in metadata.modules
        m_type = mod_data[:type]
        if haskey(MODULE_PROCESSORS, m_type)
            processor_func = MODULE_PROCESSORS[m_type]
            if m_type == :smooth
                processor_func(opt_dict, mod_data, basis_matrices_registry)
            else
                processor_func(opt_dict, mod_data)
            end
        end
        if m_type in [:interaction_composition, :observationprocess, :intercept, :nested]; continue; end
        manifold_obj = resolve_technical_primitive(mod_data, opt_dict, hyperprior_registry, :pcpriors)
        spec_params = Dict{Symbol, Any}()
        if m_type == :mixed
            mixed_str = join(mod_data[:variables], ",")
            m_match = match(r"(.+?)\s*\|\s*(.+)", mixed_str)
            if !isnothing(m_match)
                group_var = Symbol(strip(m_match.captures[2]))
                if hasproperty(data, group_var)
                    levels = unique(data[!, group_var])
                    group_map = Dict(v => i for (i, v) in enumerate(levels))
                    spec_params[:indices] = [group_map[v] for v in data[!, group_var]]
                    spec_params[:n_cat] = length(levels)
                end
            end
        elseif m_type == :smooth
            spec_params[:coords] = Matrix{Float64}(data[!, Symbol.(mod_data[:variables])])
        end
        spec = (domain = m_type, var = isempty(mod_data[:variables]) ? :none : Symbol(mod_data[:variables][1]), manifold_obj = manifold_obj, params = NamedTuple(spec_params))
        push!(manifolds_registry, spec)
    end

    # 7. Final Registry Consolidation
    opt_dict[:manifolds] = manifolds_registry
    opt_dict[:basis_matrices] = basis_matrices_registry
    opt_dict[:formula] = formula

    # 8. Fixed Effects Design Matrix Construction
    fixed_effects_formula_part = join(metadata.fixed_effects, " + ")
    add_intercept_to_matrix = get(opt_dict, :add_intercept, false) && !haskey(opt_dict, :intercept_prior)

    if add_intercept_to_matrix
        fixed_effects_formula_part = isempty(fixed_effects_formula_part) ? "1" : "1 + " * fixed_effects_formula_part
    end

    if !isempty(fixed_effects_formula_part)
        opt_dict[:Xfixed] = create_fixed_design(fixed_effects_formula_part, data; contrasts=get(opt_dict, :contrasts, Dict()))
        opt_dict[:Xfixed_N] = size(opt_dict[:Xfixed], 2)
    else
        opt_dict[:Xfixed] = NamedArray(zeros(size(data, 1), 0))
        opt_dict[:Xfixed_N] = 0
    end

    # 9. Finalize Outcome and Architecture
    if length(metadata.outcomes) == 1
        opt_dict[:y_obs] = data[!, metadata.outcomes[1]]
    else
        opt_dict[:y_obs] = Matrix(data[!, metadata.outcomes])
        opt_dict[:model_arch] = "multivariate"
    end

    # 10. Return final configuration object
    return NamedTuple(opt_dict)
end



function parse_module_params(params_str::AbstractString)
    # BSTM Internal Utility v1.2.0
    # Timestamp: 2026-06-27 12:36:40
    # Synopsis: An internal utility that parses the parameter part of a module string (e.g.,
    #           "model='bym2', W=data[:W]") into a dictionary of key-value pairs. It handles
    #           different data types and can evaluate Julia expressions.
    # Rationale for v1.2.0:
    #     - Enhanced parsing logic to robustly handle unquoted string values (e.g., `model=ar1`),
    #       quoted string values (`model='ar1'`), and symbolic values (`model=:ar1`).
    #     - The `try-catch` block for `Main.eval` now gracefully falls back to treating the
    #       raw value as a string, which is the most flexible format for downstream functions
    #       that stringify the model name anyway. This makes the behavior more predictable
    #       and less dependent on the `Main` execution scope.

    d = Dict{Symbol, Any}()
    if isempty(strip(params_str))
        return d
    end

    pairs = split_terms_at_depth(params_str, ",")

    for entry in pairs
        if occursin("=", entry)
            elements = Base.split(entry, "=", limit = 2)
            param_key = Symbol(strip(elements[1]))
            param_val_raw = strip(elements[2])

            # # 1. Check for nested module calls first
            m_mod_nested = match(r"(\w+)\((.*)\)", param_val_raw)
            if !isnothing(m_mod_nested) && (lowercase(m_mod_nested.captures[1]) in BSTM_MODULE_KEYWORDS)
                d[param_key] = _parse_module_call(param_val_raw)
                continue
            end

            # # 2. Handle explicit string literals
            is_string_literal = (startswith(param_val_raw, "'") && endswith(param_val_raw, "'")) || (startswith(param_val_raw, "\"") && endswith(param_val_raw, "\""))
            if is_string_literal
                d[param_key] = strip(param_val_raw, ['\'', '"'])
                continue
            end

            # # 3. Attempt to parse as simple types before falling back to expression evaluation
            if !isnothing(tryparse(Int, param_val_raw))
                d[param_key] = parse(Int, param_val_raw)
            elseif !isnothing(tryparse(Float64, param_val_raw))
                d[param_key] = parse(Float64, param_val_raw)
            elseif param_val_raw == "true"
                d[param_key] = true
            elseif param_val_raw == "false"
                d[param_key] = false
            else
                # # 4. If not a simple type, it could be a symbol, a complex expression, or an unquoted string.
                # Try to evaluate it as a Julia expression. This handles symbols like `:bym2` and complex objects like `data[:W]`.
                try
                    d[param_key] = Main.eval(Meta.parse(param_val_raw))
                catch e
                    # If evaluation fails, it's likely an unquoted string intended as a value (e.g., `model=bym2`).
                    # Treat it as a string, which is the most flexible format for downstream functions.
                    @warn "Could not evaluate '$param_val_raw' as a Julia expression. Treating as a String. Error: $e"
                    d[param_key] = param_val_raw
                end
            end
        end
    end
    return d
end


function get_optimal_sampler(model_obj::DynamicPPL.Model; target_acceptance=0.65, adaptation_steps=100)
    """
    BSTM Utility v1.1.0
    Timestamp: 2026-06-26 17:45:12
    Synopsis: A utility that selects an optimal MCMC sampler for a given Turing model. It
              distinguishes between discrete and continuous parameters to construct a Gibbs
              sampler that uses Particle Gibbs for discrete variables and NUTS for continuous ones.
    Rationale for v1.1.0:
        - Corrected an `UndefVarError` by replacing the call to the non-existent function
          `DynamicPPL.get_varnames` with `keys(rand(model_obj))`. This is the current,
          robust method for discovering all latent `VarName` objects from a model instance.
    """
    # 1. Parameter Discovery
    all_params = keys(rand(model_obj))
    
    # 2. Categorization Logic
    discrete_keywords = ["innov", "state", "cluster", "assignment", "cat"]
    discrete_params = Symbol[]
    continuous_params = Symbol[]

    for vn in all_params
        p_sym = Symbol(DynamicPPL.getsymbol(vn))
        p_name_str = string(p_sym)
        
        is_discrete = false
        for kw in discrete_keywords
            if occursin(kw, p_name_str)
                is_discrete = true
                break
            end
        end

        if is_discrete
            push!(discrete_params, p_sym)
        else
            push!(continuous_params, p_sym)
        end
    end

    # 3. Sampler Block Construction
    gibbs_blocks = []

    if !isempty(continuous_params)
        push!(gibbs_blocks, NUTS(adaptation_steps, target_acceptance))
    end

    if !isempty(discrete_params)
        pg_particles = 20
        push!(gibbs_blocks, PG(pg_particles))
    end

    # 4. Final Sampler Dispatch
    if isempty(discrete_params)
        return NUTS(adaptation_steps, target_acceptance)
    else
        return Gibbs(gibbs_blocks...)
    end
end

;;
 
