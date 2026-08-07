

# ==============================================================================
# SECTION 1: CORE DATA STRUCTURES AND TYPE DEFINITIONS
# ==============================================================================


abstract type Component end

abstract type ComponentOperator <: Component end

# All component models must inherit from ComponentModel
abstract type ComponentModel <: Component end

struct None <: ComponentModel end


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



# ==============================================================================
# SECTION 2: CONSTANTS, REGISTRIES, AND OPERATOR OVERLOADS
# ==============================================================================

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



const COMPONENT_TYPE_REGISTRY = Dict{Symbol, Type{<:ComponentModel}}(
    :none => NoneComponent,
    :iid => IID,
    :icar => ICAR,
    :besag => Besag,
    :bym2 => BYM2,
    :leroux => Leroux,
    :sar => SAR,
    :proper_car => SAR,
    :dag => DAG,
    :ar1 => AR1,
    :ar2 => AR2,
    :rw1 => RW1,
    :rw2 => RW2,
    :fitc => FITC,
    :svgp => SVGP,
    :nystrom => Nystrom,
    :warp => Warp,
    :hyperbolic => Hyperbolic,
    :decay => ExponentialDecay,
    :exponentialdecay => ExponentialDecay,
    :gp => GP,
    :rff => RFF,
    :fft => FFT,
    :spde => SPDE,
    :cyclic => Cyclic,
    :harmonic => Harmonic,
    :pspline => PSpline,
    :bspline => BSpline,
    :tps => TPS,
    :wavelet => Wavelet,
    :eigen => Eigen,
    :moran => Moran,
    :spherical => Spherical,
    :barycentric => Barycentric,
    :bcgn => BCGN,
    :networkflow => NetworkFlow,
    :svar => SVAR,
    :kriging => Kriging,
    :localadaptive => LocalAdaptive,
    :tar => TAR,
    :dynamics => Dynamics,
    :custom => Custom,
    :composed => Composed,
    :nonstationary_variance => NonStationaryVariance,
    :adaptivesmooth => AdaptiveSmooth,
    :lgcp => LGCP,
    :lgammap => LogGammaCoxProcess,
    :sncp => ShotNoiseCoxProcess
)




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
    MODEL_TO_STRUCTURE_MAP

v1.9.1 (2026-08-07) - This dictionary maps `ComponentModel` types to their primary
structure type (e.g., `:spatial`, `:temporal`). This is used for internal dispatch
and configuration.

The mapping for `Besag` has been added.
"""
const MODEL_TO_STRUCTURE_MAP = Dict{Type{<:ComponentModel}, Symbol}(
    IID => :any, # IID can be spatial, temporal, mixed, etc.
    ICAR => :spatial,
    Besag => :spatial,
    BYM2 => :spatial,
    Leroux => :spatial,
    SAR => :spatial,
    DAG => :spatial,
    RW1 => :temporal,
    RW2 => :temporal,
    AR1 => :temporal,
    AR2 => :temporal,
    Cyclic => :temporal,
    Harmonic => :temporal,
    SPDE => :spatial,
    GP => :smooth, # GP can be spatial, temporal, or smooth
    RFF => :smooth, # RFF can be spatial, temporal, or smooth
    FFT => :smooth, # FFT can be spatial, temporal, or smooth
    PSpline => :smooth,
    BSpline => :smooth,
    TPS => :smooth,
    Wavelet => :smooth,
    Eigen => :any, # Eigen can be applied to any set of variables
    Moran => :spatial,
    Spherical => :spatial, # Spherical is a continuous spatial kernel
    Barycentric => :spatial, # Barycentric is a continuous spatial kernel
    BCGN => :spatial,
    NetworkFlow => :spatial,
    LocalAdaptive => :spatial,
    TAR => :temporal,
    TensorProductSmooth => :smooth,
    Dynamics => :spacetime, # Dynamics components are inherently spatiotemporal
    Custom => :any, # Custom components can be anything
    LGCP => :spatial, # Log-Gaussian Cox Process is spatial
    LogGammaCoxProcess => :spatial, # Log-Gamma Cox Process is spatial
    ShotNoiseCoxProcess => :spatial, # Shot-Noise Cox Process is spatial
    SVAR => :spacetime, # Spatially Varying Autoregressive is spatiotemporal
    SVC => :spatial, # Spatially Varying Coefficient is spatial
    TVC => :temporal, # Temporally Varying Coefficient is temporal
    Composed => :any, # Composed components are context-dependent
    NonStationaryVariance => :spatial, # Non-stationary variance is spatial
    AdaptiveSmooth => :smooth, # Adaptive smooth is a smooth component
    Kriging => :smooth, # Kriging is a smooth component
    None => :none # Placeholder
)

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


function Base.:|>(m1::Component, m2::Component)
    return Composed([m1, m2], :pipe)
end

composition(m1::Component, m2::Component) = Composed([m1, m2], :composition)
∘(m1::Component, m2::Component) = Composed([m1, m2], :composition)

otimes(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)
⊗(m1::Component, m2::Component) = Composed([m1, m2], :kronecker_product)

# ==============================================================================
# SECTION 3: FORMULA PARSING ENGINE
# ==============================================================================

function split_terms_at_depth(input::AbstractString, sep::AbstractString)
    # Purpose: Splits a string by a separator, but only when the separator is not inside parentheses or brackets.
    # Rationale: This version uses a robust iteration pattern that correctly handles multi-byte characters
    #            (like `∘` or `⊗`) and avoids StringIndexError.
    # v1.1.0 (2026-08-01)
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

    # --- Explicitly handle aliases and missing models ---
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


function _add_parsed_arg!(args_dict::Dict{Symbol, Any}, positional_args::Vector{Any}, arg_val::AbstractString)
    # Purpose: A helper to add a parsed argument to either the keyword or positional argument list.
    # Rationale: Centralizes the logic for distinguishing between `key=value` and positional arguments.
    #            This version is updated to accept `AbstractString` to handle both `String` and `SubString`
    #            types, resolving a `MethodError`.
    # v1.0.1 (2026-08-02)
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
    # Rationale: This version is updated to correctly handle commas within string literals,
    #            preventing incorrect splitting of arguments.
    # v1.1.0 (2026-08-01)
    args_dict = Dict{Symbol, Any}()
    positional_args = []
    current_arg = IOBuffer()
    depth = 0
    in_string = false
    string_char = ' '

    for char in args_str
        if char == '"' || char == '\''
            if !in_string
                in_string = true
                string_char = char
            elseif char == string_char
                in_string = false
            end
        end

        if char == ',' && depth == 0 && !in_string
            arg_val = strip(String(take!(current_arg)))
            if !isempty(arg_val)
                _add_parsed_arg!(args_dict, positional_args, arg_val)
            end
        else
            write(current_arg, char)
            if !in_string
                if char == '(' || char == '['; depth += 1;
                elseif char == ')' || char == ']'; depth -= 1; end
            end
        end
    end

    arg_val = strip(String(take!(current_arg)))
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


function decompose_bstm_formula(formula_str::String, data::DataFrame)
    # Purpose: Decomposes a formula string into its constituent parts.
    # Rationale: This version is updated to handle formulas that do not contain a right-hand
    #            side (RHS). If the `~` operator is missing, it now defaults the RHS to "1"
    #            (an intercept-only model), preventing a `BoundsError` during parsing.
    # v1.0.1 (2026-08-01)
    # Inputs:
    #   - formula_str: The model formula string.
    #   - data: The input DataFrame, used for AST rewriting.
    # Outputs: A NamedTuple containing the parsed formula components.
    parts = Base.split(formula_str, "~")
    lhs_str = Base.strip(parts[1])
    
    # If there is no '~', default the RHS to "1" (intercept only).
    rhs_str = if length(parts) > 1
        Base.strip(parts[2])
    else
        "1" 
    end

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


"""
    bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)

Constructs the complete model configuration by parsing the formula, processing data,
and assembling component specifications using the explicit component interface.
"""
function bstm_config(formula::String, data::DataFrame; calling_module::Module=Main, kwargs...)
    df_processed = deepcopy(data)
    decomposed_formula = decompose_bstm_formula(formula, df_processed)

    M = _initialize_config(df_processed, merge(Dict(kwargs), Dict(:calling_module => calling_module)))
    M[:formula] = formula
    
    _process_lhs!(M, decomposed_formula.outcomes)
    
    is_multivariate = get(M, :model_arch, "univariate") == "multivariate"
    if is_multivariate
        for (key, mod_data_nt) in decomposed_formula.modules
            model_name = get(mod_data_nt.args, :model, :none)
            if mod_data_nt.module_type == :dynamics && model_name in [:leslie_matrix, :delay_difference, :generalized_lotka_volterra, :generalized_leslie_matrix]
                M[:is_multivariate_dynamics] = true
                M[:multivariate_dynamics_key] = key
                break
            end
        end
    end

    _precompute_likelihood_params!(M)

    M[:add_intercept] = decomposed_formula.has_intercept
    if !isnothing(decomposed_formula.intercept_prior)
        prior_val = decomposed_formula.intercept_prior
        if prior_val isa Expr
            try; M[:intercept_prior] = Core.eval(calling_module, prior_val);
            catch e; error("Could not evaluate `prior` argument `$(prior_val)` in intercept() module. Error: $e"); end
        else; M[:intercept_prior] = prior_val; end
    end

    # --- NEW MAIN COMPONENT PROCESSING LOOP ---
    for (key, mod_data_nt) in decomposed_formula.modules
        mod_type = mod_data_nt.module_type
        mod_data_dict = Dict(:key => key, :type => mod_type, :variables => get(mod_data_nt.args, :positional_args, []), :params => mod_data_nt.args)

        if mod_type == :fixed
            if !haskey(M, :fixed_effects_from_modules); M[:fixed_effects_from_modules] = String[]; end
            for var in mod_data_dict[:variables]; push!(M[:fixed_effects_from_modules], string(var)); end
            if haskey(mod_data_dict[:params], :prior); M[:fixed_effects_priors][Symbol(mod_data_dict[:variables][1])] = mod_data_dict[:params][:prior]; end
            continue
        end

        if mod_type == :random
            params = mod_data_dict[:params]
            if !haskey(params, :structure)
                args_for_inference = copy(params)
                args_for_inference[:vars] = get(mod_data_dict, :variables, [])
                params[:structure] = _infer_structure_from_args(args_for_inference)
            end
            mod_data_dict[:type] = params[:structure]
        end

        model_name = get(mod_data_dict[:params], :model, :iid)
        if mod_type == :interact; model_name = :composed; end
        if !haskey(COMPONENT_TYPE_REGISTRY, model_name)
            error("Model type ':$model_name' not found in COMPONENT_TYPE_REGISTRY.")
        end
        component_type = COMPONENT_TYPE_REGISTRY[model_name]

        # For Composed, we need the child objects to dispatch get_datastructures!
        if component_type <: Composed
            temp_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
            mod_data_dict[:component_obj] = temp_obj
        end

        # Call get_datastructures!
        create_component = get_datastructures!(component_type, M, mod_data_dict)

        if !create_component
            continue
        end

        # Instantiate the final component object
        component_obj = resolve_technical_primitive(mod_data_dict, M, M[:hyperpriors], M[:prior_scheme])
        mod_data_dict[:component_obj] = component_obj

        # Call get_precomputes
        M_nt = NamedTuple(M)
        precomputes = get_precomputes(component_obj, M_nt, mod_data_dict)

        # Assemble the final spec
        spec = (
            key=Symbol(key), 
            structure=mod_data_dict[:type], 
            var=join(string.(mod_data_dict[:variables]), "_"), 
            component_obj=component_obj, 
            params=mod_data_dict[:params], 
            hyper=precomputes
        )
        push!(M[:components], spec)
    end

    all_fixed_effects = copy(decomposed_formula.fixed_effects)
    if haskey(M, :fixed_effects_from_modules)
        append!(all_fixed_effects, M[:fixed_effects_from_modules])
    end
    _process_fixed_effects!(M, unique(all_fixed_effects))
    _process_fixed_effects_priors!(M)
    _finalize_config!(M)
    
    return NamedTuple(M)
end


# Version 1.0.0 (2026-08-06)
# Purpose: Generates a NamedTuple of full variable names for a given component.
# Rationale: This utility function centralizes the naming convention for all parameters
#            associated with a model component, ensuring consistency across the framework.
#            It handles suffixes for multivariate models and prefixes for different parameter types.
# Assumptions: The component specification `spec` contains a unique `key`.
function generate_full_variable_names(spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}; prefix::String="")
    base_key = string(spec.key)
    full_key = isempty(prefix) ? base_key : "$(prefix)_$(base_key)"

    is_multivariate = arch == "multivariate"
    is_shared = get(spec.params, :shared, false)

    suffix = (is_multivariate && !is_shared) ? "_$(outcome_idx)" : ""

    names = Dict{Symbol, Symbol}()
    
    all_prefixes = [
        :sigma, :rho, :rho1, :rho2, :kappa, :ls, :range, :period, :amplitude, :phase,
        :velocity, :diffusion, :pca_sd, :pdef_sd, :L_corr, :sigma_effects,
        :r, :K, :q, :M_nat, :alpha, :beta, :gamma, :delta,
        :raw, :innov, :latent, :struct, :iid, :beta_cos, :beta_sin, :rho_field,
        :W, :b, :v_raw, :factors_flat, :thresh_raw, :W1, :b1, :W2, :amplitude_raw,
        :innov_predator
    ]

    for p in all_prefixes
        names[p] = Symbol("$(p)_$(full_key)$(suffix)")
    end

    return NamedTuple(names)
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



# Version 1.0.2 (2026-08-06)
# Purpose: Pre-computes a matrix factorization for static components.
# Rationale: This version reverts to using `cholesky` factorization instead of `lu`.
#            The `CanonicalIndexError` that previously prompted the switch to `lu` is now
#            addressed in the code generator by explicitly creating a `SparseMatrixCSC`
#            from the Cholesky factor before solving. This change restores the use of the
#            more efficient and appropriate Cholesky decomposition for symmetric
#            positive-definite precision matrices.
# Assumptions: The input `Q_template` is a sparse matrix representing a valid precision structure.
function _precompute_static_components!(M::Dict)
    noise = M[:noise]
    new_components = []
    # Define component types that do not have dynamic structure parameters like `rho`.
    static_component_types = [IID, ICAR, Besag, RW1, RW2, Cyclic, PSpline, TPS, BSpline, Eigen, Moran, Spherical, Barycentric, TensorProductSmooth]

    for spec_in in M[:components]
        current_spec = spec_in
        m_obj = current_spec.component_obj

        if m_obj isa MixedComponent || m_obj isa SVC
            inner_model = m_obj.model
            is_inner_static = any(T -> inner_model isa T, static_component_types)
            if is_inner_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
                try
                    Q_concrete = sparse(current_spec.Q_template)
                    # Revert to cholesky
                    F = cholesky(Symmetric(Q_concrete + noise * I))
                    final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                    push!(new_components, final_spec)
                    continue
                catch e
                    @warn "Cholesky factorization failed for static inner model in $(current_spec.key). Reverting to dynamic computation. Error: $e"
                end
            end
        end

        if m_obj isa Composed && m_obj.operator == :pipe
            state_spec = get(current_spec.hyper, :state_spec, nothing)
            if !isnothing(state_spec)
                state_m_obj = m_obj.components[2]
                is_state_static = any(T -> state_m_obj isa T, static_component_types)

                if is_state_static && !isnothing(state_spec.Q_template) && size(state_spec.Q_template, 1) > 0
                    try
                        Q_concrete = sparse(state_spec.Q_template)
                        # Revert to cholesky
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

        is_main_static = !(current_spec.component_obj isa Composed) && any(T -> current_spec.component_obj isa T, static_component_types)

        if is_main_static && !isnothing(current_spec.Q_template) && size(current_spec.Q_template, 1) > 0
            try
                Q_concrete = sparse(current_spec.Q_template)
                # Revert to cholesky
                F = cholesky(Symmetric(Q_concrete + noise * I))
                final_spec = merge(current_spec, (is_static=true, cholesky_factor=F))
                push!(new_components, final_spec)
            catch e
                @warn "Cholesky factorization failed for static component $(current_spec.key). Reverting to dynamic computation. Error: $e"
                final_spec = merge(current_spec, (is_static=false,))
                push!(new_components, final_spec)
            end
        else
            final_spec = merge(current_spec, (is_static=get(current_spec, :is_static, false),))
            push!(new_components, final_spec)
        end
    end
    M[:components] = new_components
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
        println("\n--- Generated Model Code ---")
        println(new_config.generated_model_code)
        println("----------------------------------------\n")

        # Call the new detailed parameter printing function
        _print_finalized_parameters(new_config)

        println("\n--- Sample from priors as a check ---")
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


function bstm(formula::String, data::DataFrame; kwargs...)
    # Convenience overload defaulting to the Main execution scope
    # The semicolon below is the critical fix.
    return bstm(formula, data, Main; kwargs...)
end


"""
    bstm_text_assembler(config::NamedTuple, model_func_name::Symbol)

Assembles the full Turing model code as a string and a Julia `Expr` by iterating
through the components defined in the configuration and calling their `get_priors`
and `get_updates` methods.

# Rationale for Update
This function is the core of the refactored code generation engine. It replaces the
legacy `_generate_component_code_fragments` function with direct calls to the explicit
`get_priors` and `get_updates` interface methods. This change makes the assembler
simpler, more transparent, and directly aligned with the new component contract. It
iterates through the component specifications, collects the prior and update code
fragments for each, and then interpolates them into a complete `@model` block.
"""
function bstm_text_assembler(config::NamedTuple, model_func_name::Symbol)
    priors_acc = String[]
    updates_acc = String[]
    
    is_multivariate = config.model_arch == "multivariate"
    arch_str = config.model_arch

    # --- 1. Likelihood-specific Priors ---
    # These are global parameters for the observation model.
    push!(priors_acc, "# --- Likelihood Priors ---")
    for (i, spec) in enumerate(config.likelihood_specs)
        family = get(spec, :family, "gaussian")
        suffix = is_multivariate ? "_$i" : ""
        if family == "negbin"
            push!(priors_acc, "r_nb$(suffix) ~ NamedDist(Exponential(1.0), :r_nb$(suffix))")
        end
        if family in ["gaussian", "lognormal", "student_t", "laplace", "half_normal", "half_student_t"] && !get(config, :volatility, false)
            push!(priors_acc, "y_sigma$(suffix) ~ NamedDist(Exponential(1.0), :y_sigma$(suffix))")
        end
        if get(config, :use_zi, false)
            push!(priors_acc, "phi_zi$(suffix) ~ NamedDist(Beta(1, 1), :phi_zi$(suffix))")
        end
    end

    # --- 2. Intercept and Fixed Effects Priors ---
    if config.add_intercept
        push!(priors_acc, "# --- Intercept Prior ---")
        intercept_prior_str = _distribution_to_string(config.intercept_prior)
        if is_multivariate
            push!(priors_acc, "intercept ~ NamedDist(MvNormal(zeros(T, $(config.outcomes_N)), $(config.intercept_prior.σ) * I), :intercept)")
        else
            push!(priors_acc, "intercept ~ NamedDist($(intercept_prior_str), :intercept)")
        end
    end

    if get(config, :Xfixed_N, 0) > 0
        push!(priors_acc, "# --- Fixed Effects Priors ---")
        if is_multivariate
            # For multivariate, create a block of priors for each outcome
            for k in 1:config.outcomes_N
                for i in 1:config.Xfixed_N
                    prior_str = _distribution_to_string(config.Xfixed_priors_vec[i])
                    push!(priors_acc, "Xfixed_beta_$(k)_$(i) ~ NamedDist($(prior_str), :Xfixed_beta_$(k)_$(i))")
                end
            end
        else
            priors_str = join(["NamedDist($(_distribution_to_string(p)), :Xfixed_beta_$(i))" for (i, p) in enumerate(config.Xfixed_priors_vec)], ", ")
            push!(priors_acc, "Xfixed_beta ~ NamedDist(Product([$(priors_str)]), :Xfixed_beta)")
        end
    end

    # --- 3. Multivariate Correlation Prior ---
    if is_multivariate
        push!(priors_acc, "# --- Multivariate Correlation Prior ---")
        push!(priors_acc, "L_corr ~ NamedDist(LKJCholesky($(config.outcomes_N), T(1.0)), :L_corr)")
    end

    # --- 4. Main Component Loop ---
    # This loop iterates through all defined components (spatial, temporal, smooth, etc.)
    # and calls their specific `get_priors` and `get_updates` methods.
    for spec in config.components
        m_obj = spec.component_obj
        
        # Skip components that are handled globally or have no code to generate
        if m_obj isa NoneComponent; continue; end

        push!(priors_acc, "\n# --- Priors for component: $(spec.key) ---")
        push!(updates_acc, "\n# --- Updates for component: $(spec.key) ---")

        if is_multivariate
            for k in 1:config.outcomes_N
                push!(priors_acc, get_priors(m_obj, spec, arch_str, k, config))
                push!(updates_acc, get_updates(m_obj, spec, arch_str, k, config))
            end
        else # Univariate
            push!(priors_acc, get_priors(m_obj, spec, arch_str, nothing, config))
            push!(updates_acc, get_updates(m_obj, spec, arch_str, nothing, config))
        end
    end

    # --- 5. Assemble Code Fragments ---
    priors_code = join(filter(!isempty, priors_acc), "\n    ")
    updates_code = join(filter(!isempty, updates_acc), "\n")

    # --- 6. Construct Linear Predictor and Likelihood ---
    eta_assembly_code = ""
    likelihood_code = ""

    if is_multivariate
        # Assemble beta matrix for fixed effects
        beta_matrix_assembly = if get(config, :Xfixed_N, 0) > 0
            "local Xfixed_beta = hcat($([ "vcat($(["Xfixed_beta_$(k)_$(i)" for i in 1:config.Xfixed_N]))" for k in 1:config.outcomes_N ]...))"
        else
            "local Xfixed_beta = zeros(T, 0, $(config.outcomes_N))"
        end

        eta_assembly_code = """
        # --- Linear Predictor Assembly (Multivariate) ---
        local eta_latent = zeros(T, M.y_N, M.outcomes_N)
        if M.add_intercept; eta_latent .+= intercept'; end
        if M.Xfixed_N > 0
            $(beta_matrix_assembly)
            eta_latent .+= M.Xfixed * Xfixed_beta
        end
        
        $(updates_code)
        
        local eta = eta_latent * L_corr.L'
        """
        
        # Construct likelihood block for multivariate models
        lik_lines = ["for i in 1:M.y_N"]
        for k in 1:config.outcomes_N
            lik_spec = config.likelihood_specs[k]
            fam = get(lik_spec, :family, "gaussian")
            lik_params = "family=\"$fam\", trial=M.trials[i,$k], phi_zi=phi_zi$k, r_nb=r_nb$k, sigma_y=y_sigma$k, censor_lower=M.censor_lower[i,$k], censor_upper=M.censor_upper[i,$k], hurdle=M.hurdle[i,$k]"
            push!(lik_lines, "    M.y_obs[i,$k] ~ bstm_Likelihood(eta[i,$k]; $(lik_params))")
        end
        push!(lik_lines, "end")
        likelihood_code = join(lik_lines, "\n")

    else # Univariate
        eta_assembly_code = """
        # --- Linear Predictor Assembly (Univariate) ---
        local eta = zeros(T, M.y_N)
        if M.add_intercept; eta .+= intercept; end
        if M.Xfixed_N > 0; eta .+= M.Xfixed * Xfixed_beta; end
        
        $(updates_code)
        """
        
        lik_spec = config.likelihood_specs[1]
        fam = get(lik_spec, :family, "gaussian")
        lik_params = "family=\"$fam\", trial=M.trials, phi_zi=phi_zi, r_nb=r_nb, sigma_y=y_sigma, censor_lower=M.censor_lower, censor_upper=M.censor_upper, hurdle=M.hurdle"
        likelihood_code = "M.y_obs ~ bstm_Likelihood(eta; $(lik_params))"
    end

    # --- 7. Final Model Assembly ---
    model_string = """
    @model function $(model_func_name)(M, spec_registry; T=Float64)
        # --- Priors ---
        $(priors_code)

        # --- Model Definition ---
        $(eta_assembly_code)
        
        # --- Likelihood ---
        if !get(M, :likelihood_handled, false)
            $(likelihood_code)
        end
    end
    """

    # --- 8. Create Spec Registry and Expression ---
    spec_registry = NamedTuple(spec.key => spec for spec in config.components)
    expr = Meta.parse(model_string)

    return model_string, expr, spec_registry
end


function bstm_codegen(config::NamedTuple)
    # Purpose: Generates the necessary components to define and instantiate a Turing model.
    # Rationale: Decouples code generation from evaluation to better handle Julia's world-age issues.
    # v1.0.0 (2026-07-18)
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


"""
    resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)

Instantiates a `ComponentModel` object from its parsed formula representation. This function
acts as a factory, resolving hyperpriors and calling the appropriate constructor from the
`COMPONENT_CONSTRUCTORS` registry.
"""
function resolve_technical_primitive(module_metadata::Dict{Symbol, Any}, M, priors_dict, scheme::Symbol)
    m_type = module_metadata[:type]
    m_params = module_metadata[:params]
    calling_mod = get(M, :calling_module, Main)

    # Handle composed components recursively
    if m_type == :interact
        op = m_params[:operator]
        components_data = m_params[:components]
        components_metadata = map(c_node -> Dict(:key=>"temp", :type => c_node.module_type, :params => c_node.args, :variables => get(c_node.args, :positional_args, [])), components_data)
        resolved_components = [resolve_technical_primitive(comp_meta, M, priors_dict, scheme) for comp_meta in components_metadata]
        return Composed(resolved_components, op)
    end

    # Handle standard components
    model_name = get(m_params, :model, :iid)
    if m_type in keys(LEGACY_MODULES); model_name = m_type; end

    model_name_str = string(model_name)
    resolved_priors = resolve_hyperpriors(model_name_str, priors_dict, m_params, scheme, calling_mod)
    
    if !haskey(COMPONENT_CONSTRUCTORS, model_name)
        error("Component model ':$model_name' is not a recognized model type.")
    end
    
    constructor_func = COMPONENT_CONSTRUCTORS[model_name]
    return constructor_func(resolved_priors, m_params)
end



# Version 1.5.3 (2026-08-06)
# Purpose: Creates a precision matrix template and its spectral decomposition for a GMRF model.
# Rationale: This version is updated to pre-compute and return the eigendecomposition (eigenvectors `U`
#            and eigenvalues `L`) of the precision matrix template. This enables the use of AD-friendly
#            spectral sampling methods for latent Gaussian fields, as recommended for improving HMC
#            performance. Instead of sampling from `MvNormalCanon(0, Q)` where `Q` might depend on
#            sampled parameters (requiring a non-differentiable Cholesky decomposition), the model can
#            sample `z ~ N(0,I)` and construct the latent field as `latent = U * D * z`, where `D` is a
#            diagonal matrix constructed from the eigenvalues `L` and other hyperparameters. This avoids
#            all matrix decompositions within the model's execution path.
# Assumptions: The input `model_type` is a recognized GMRF structure.
function build_structure_template(model_type::Symbol, n::Int; W::Union{AbstractMatrix, Nothing}=nothing)
    # This function creates a standardized precision matrix template (Q) and its
    # eigendecomposition for various Gaussian Markov Random Field (GMRF) models.

    Q_template = spzeros(Float64, n, n)
    rank_deficiency = 0

    if n == 0
        return (matrix=Q_template, scaling_factor=1.0, U=spzeros(Float64, 0, 0), L=Float64[])
    end

    if model_type in [:icar, :besag, :bym2, :leroux, :localadaptive]
        if isnothing(W); error("Spatial model '$model_type' requires an adjacency matrix `W`."); end
        if size(W, 1) != n || size(W, 2) != n; error("Adjacency matrix `W` dimensions ($(size(W))) do not match `n` ($n)."); end
        W_sym = sparse((W + W') .> 0)
        D = spdiagm(0 => vec(sum(W_sym, dims=2)))
        Q_template = D - W_sym
        rank_deficiency = 1
    elseif model_type == :rw1
        if n > 1
            Q_template = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
            Q_template[1, 1] = 1.0
            Q_template[n, n] = 1.0
        elseif n == 1
            Q_template[1,1] = 1.0
        end
        rank_deficiency = 1
    elseif model_type == :rw2
        # Implementation for RW2 precision matrix
        if n > 1
            Q_template = spdiagm(0 => fill(6.0, n), -1 => fill(-4.0, n-1), 1 => fill(-4.0, n-1), -2 => fill(1.0, n-2), 2 => fill(1.0, n-2))
            Q_template[1, 1] = 1.0; Q_template[2, 2] = 5.0
            Q_template[1, 2] = -2.0; Q_template[2, 1] = -2.0
            Q_template[n-1, n-1] = 5.0; Q_template[n, n] = 1.0
            Q_template[n-1, n] = -2.0; Q_template[n, n-1] = -2.0
        elseif n == 1
            Q_template[1,1] = 1.0
        end
        rank_deficiency = 2
    elseif model_type == :cyclic
        if n > 0
            Q_template = spdiagm(0 => fill(2.0, n), 1 => fill(-1.0, n-1), -1 => fill(-1.0, n-1))
            Q_template[1, n] = -1.0
            Q_template[n, 1] = -1.0
        end
        rank_deficiency = 1
    elseif model_type == :ar1
        if n > 1
            Q_template = spdiagm(0 => zeros(n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
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

    local U, L, scaling_factor
    # Always compute eigendecomposition for potential use by spectral methods.
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    if rank_deficiency > 0
        scaling_factor = _compute_scaling_factor(L, rank_deficiency)
        Q_template = Q_template ./ scaling_factor
        # Rescale eigenvalues as well for consistency.
        L = L ./ scaling_factor
    else
        scaling_factor = 1.0
    end

    return (matrix=Q_template, scaling_factor=scaling_factor, U=U, L=L)
end



# Version 1.0.0 (2026-07-21)
# Purpose: Computes a robust scaling factor for a precision matrix from its eigenvalues.
# Rationale: The scaling factor is the geometric mean of the non-zero eigenvalues.
#            This method avoids using a fixed tolerance to identify zero eigenvalues,
#            which can be sensitive to floating-point noise. Instead, it uses the known
#            rank deficiency of the GMRF model to correctly identify the structural zero
#            eigenvalues.
# Assumptions: `evals` is a vector of eigenvalues from a symmetric matrix.
function _compute_scaling_factor(evals::Vector{Float64}, rank_deficiency::Int)
    # Sort eigenvalues in ascending order to easily discard the smallest ones, which correspond to the null space.
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
    # This is a standard method for ensuring the determinant of the scaled precision matrix is 1.
    return exp(mean(log.(positive_evals)))
end





# Version 1.5.4 (2026-08-06)
# Purpose: Computes the cross-covariance kernel matrix between two sets of coordinates.
# Rationale: This version is updated for improved type stability and AD-friendliness,
#            mirroring the changes in `evaluate_kernel_matrix`. It avoids explicit
#            casts by using `convert` and relying on Julia's promotion system, which is
#            safer for automatic differentiation.
# Assumptions: `coords1` and `coords2` are matrices of data points.
function evaluate_cross_kernel_matrix(coords1::AbstractMatrix, coords2::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol)
    T = promote_type(eltype(coords1), eltype(coords2), typeof(param_val), eltype(ls))
    coords1_T = convert(AbstractMatrix{T}, coords1)
    coords2_T = convert(AbstractMatrix{T}, coords2)
    ls_T = convert(typeof(ls) <: Real ? T : AbstractVector{T}, ls)

    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords1_T, 2) != length(ls_T) || size(coords2_T, 2) != length(ls_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions does not match number of lengthscales.")
        end
        # Calculate weighted squared Euclidean distance
        dist_sq = pairwise(SqEuclidean(), coords1_T ./ ls_T', coords2_T ./ ls_T', dims=1)
    else # Isotropic case
        dist_sq = pairwise(SqEuclidean(), coords1_T, coords2_T, dims=1) ./ ls_T^2
    end

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d)
    
    # Matern 3/2
    elseif kernel_type == :matern32
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val)
    
    # Matern 5/2
    elseif kernel_type == :matern52
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val)

    # Fallback Dispatch
    else
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq)
    end
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


# Version 1.2.0 (2026-08-06)
# Purpose: Automatically constructs an efficient composite Gibbs sampler for a `bstm` model.
# Rationale: This version implements a sophisticated block-Gibbs strategy. It groups component-specific
#            parameters (e.g., a spatial field and its hyperparameters) into a single block for joint
#            sampling with NUTS, which is crucial for handling the "funnel" geometry in hierarchical models.
#            For remaining parameters, it assigns specialized samplers based on their prior's support:
#            ESS for Gaussian, Slice for bounded, and PG for discrete parameters. This hybrid approach
#            significantly improves sampling efficiency and convergence over a single, general-purpose sampler.
#            It also adds an `adtype` keyword to allow user control over the AD backend for NUTS.
function get_optimal_sampler(
    model_obj::DynamicPPL.Model;
    sampler_choice=:auto,
    sampler_map::Dict{Symbol, <:AbstractMCMC.AbstractSampler}=Dict{Symbol, AbstractMCMC.AbstractSampler}(),
    adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff(),
    target_acceptance=0.8,
    adaptation_steps=1000,
    group_components::Bool=true,
    n_particles=20,
    hmc_leapfrog_steps=10
)
    # This function constructs a composite Gibbs sampler by assigning optimal MCMC algorithms
    # to different blocks of parameters based on their characteristics. The process follows a
    # clear hierarchy to provide both flexibility and efficiency.
    #
    # Sampler Selection Workflow:
    # 1. Manual Override: If a specific sampler is provided via `sampler_choice`, it is used directly.
    #    If a `sampler_map` is provided, it assigns specific samplers to designated parameters,
    #    taking the highest precedence.
    #
    # 2. Component Grouping: If `group_components=true`, the function identifies all parameters
    #    belonging to the same model component (e.g., a spatial field and its hyperparameters).
    #    These are grouped into a single block and assigned a NUTS sampler. This is crucial for
    #    efficiently exploring the correlated posterior geometry of hierarchical components.
    #
    # 3. Default Categorization: Any remaining parameters are categorized by their prior's support:
    #    - `:discrete` (e.g., from Categorical priors) are assigned Particle Gibbs (PG).
    #    - `:gaussian` (from Normal or MvNormal priors) are assigned Elliptical Slice Sampling (ESS).
    #    - `:bounded` (from priors like Beta, Exponential) are assigned Slice sampling.
    #    - `:other_continuous` (unbounded, non-Gaussian) are assigned NUTS.
    #
    # This strategy balances the use of efficient gradient-based samplers (NUTS) for complex,
    # high-dimensional blocks with robust gradient-free samplers for simpler or constrained parameters.

    if sampler_choice isa AbstractMCMC.AbstractSampler
        @info "Using user-specified sampler: $(typeof(sampler_choice))"
        return sampler_choice
    end

    vi = DynamicPPL.VarInfo(model_obj)
    vns = DynamicPPL.keys(vi)

    sampler_assignments = []
    all_processed_vns = Set{VarName}()

    # --- Stage 1: Handle user-provided sampler map (highest precedence) ---
    for (param_sym, sampler) in sampler_map
        sym_vns = filter(vn -> DynamicPPL.getsym(vn) == param_sym, vns)
        if !isempty(sym_vns)
            push!(sampler_assignments, Tuple(sym_vns) => sampler)
            union!(all_processed_vns, sym_vns)
            @info "Applying user-defined sampler $(typeof(sampler)) for parameter group: $(param_sym)"
        else
            @warn "Parameter :$(param_sym) in sampler_map not found in model."
        end
    end

    # --- Stage 2: Group parameters by model component if enabled ---
    if group_components
        @info "Component grouping enabled. Grouping hyperparameters and latent fields for joint sampling."
        component_groups = Dict{String, Set{VarName}}()
        
        # This list contains the standard prefixes for parameters generated by `bstm`.
        # It is used to parse variable names like `sigma_spatial_main` into a prefix (`sigma`)
        # and a component key (`spatial_main`).
        all_prefixes = [
            "sigma", "rho", "rho1", "rho2", "kappa", "ls", "range", "period",
            "amplitude", "phase", "velocity", "diffusion", "pca_sd", "pdef_sd",
            "L_corr", "sigma_effects", "r", "K", "q", "M_nat", "alpha", "beta", "gamma", "delta",
            "raw", "innov", "latent", "struct", "iid", "beta_cos", "beta_sin", "rho_field",
            "W", "b", "v_raw", "factors_flat", "thresh_raw", "W1", "b1", "W2", "amplitude_raw",
            "innov_predator"
        ]
        prefix_regex = Regex("^(" * join(all_prefixes, "|") * ")_(.+)\$")

        for vn in vns
            if vn in all_processed_vns; continue; end

            vn_str = string(DynamicPPL.getsym(vn))
            m = match(prefix_regex, vn_str)
            
            if !isnothing(m)
                component_key = m.captures[2]
                if !haskey(component_groups, component_key)
                    component_groups[component_key] = Set{VarName}()
                end
                push!(component_groups[component_key], vn)
            end
        end

        # Assign a NUTS sampler to each identified component group.
        for (key, params_vns) in component_groups
            if !isempty(params_vns)
                sampler = NUTS(adaptation_steps, target_acceptance; adtype=adtype)
                push!(sampler_assignments, Tuple(params_vns) => sampler)
                union!(all_processed_vns, params_vns)
                param_syms = Set(DynamicPPL.getsym.(params_vns))
                @info "Created NUTS block for component '$(key)' with parameters: $(param_syms)"
            end
        end
    end

    # --- Stage 3: Assign samplers to remaining parameters based on their prior's support ---
    remaining_vns = filter(vn -> !(vn in all_processed_vns), vns)
    if !isempty(remaining_vns)
        param_groups = Dict(:discrete => Set{VarName}(), :gaussian => Set{VarName}(), :bounded => Set{VarName}(), :other_continuous => Set{VarName}())

        for vn in remaining_vns
            try
                dist = DynamicPPL.getdist(vi, vn)
                support = Distributions.value_support(typeof(dist))
                if support isa Distributions.Discrete
                    push!(param_groups[:discrete], vn)
                elseif support isa Distributions.Continuous
                    if dist isa Union{Normal, MvNormal, Truncated{<:Normal}}
                        push!(param_groups[:gaussian], vn)
                    elseif isfinite(minimum(dist)) || isfinite(maximum(dist))
                        push!(param_groups[:bounded], vn)
                    else
                        push!(param_groups[:other_continuous], vn)
                    end
                end
            catch e
                # If we can't determine the distribution, default to the 'other' category.
                push!(param_groups[:other_continuous], vn)
            end
        end

        # Assign appropriate samplers to each category.
        if !isempty(param_groups[:discrete]); params = Tuple(param_groups[:discrete]); push!(sampler_assignments, params => PG(n_particles)); @info "Using Particle Gibbs (PG) for: $(DynamicPPL.getsym.(params))"; end
        if !isempty(param_groups[:gaussian]); params = Tuple(param_groups[:gaussian]); push!(sampler_assignments, params => ESS()); @info "Using Elliptical Slice Sampling (ESS) for: $(DynamicPPL.getsym.(params))"; end
        if !isempty(param_groups[:bounded]); params = Tuple(param_groups[:bounded]); push!(sampler_assignments, params => Slice()); @info "Using Slice sampling for: $(DynamicPPL.getsym.(params))"; end
        if !isempty(param_groups[:other_continuous]); params = Tuple(param_groups[:other_continuous]); push!(sampler_assignments, params => NUTS(adaptation_steps, target_acceptance; adtype=adtype)); @info "Using NUTS for remaining continuous parameters: $(DynamicPPL.getsym.(params))"; end
    end

    # --- Stage 4: Construct and return the final composite sampler ---
    if isempty(sampler_assignments)
        @warn "Could not identify any parameters to sample. Defaulting to a single NUTS sampler for all parameters."
        return NUTS(adaptation_steps, target_acceptance; adtype=adtype)
    elseif length(sampler_assignments) == 1
        # If only one group was formed, return that sampler directly, not wrapped in Gibbs.
        return sampler_assignments[1][2]
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
        println(model_pseudocode(m))
        println("\n--- End Reconstructed Model Source ---")
    end
    println("\n--- End Model Summary ---")
    return nothing
end


function model_pseudocode(m::DynamicPPL.Model)
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
        ) # Pass custom_knots explicitly
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
    # Qualify call with the dynamically loaded module.
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
        # Qualify call with the dynamically loaded module.
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

    # Qualify call with the dynamically loaded module.
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


# Version 1.5.4 (2026-08-06)
# Purpose: Computes the covariance kernel matrix for a given set of coordinates.
# Rationale: This version is updated for improved type stability and AD-friendliness.
#            It replaces explicit type casts like `T(val)` with `convert(T, val)` or
#            relies on Julia's promotion system. For example, `T(noise) * I` is replaced
#            with `noise * I`, as `UniformScaling` correctly promotes types during addition.
#            Numeric literals are also handled via `convert` or promotion to avoid issues
#            with AD libraries.
# Assumptions: `coords` contains the data points, and other arguments define the kernel.
function evaluate_kernel_matrix(coords::AbstractMatrix, param_val::Real, ls::Union{Real, AbstractVector}, kernel_type::Symbol, noise::Real; wavelet_levels=3)
    T = promote_type(eltype(coords), typeof(param_val), eltype(ls), typeof(noise))
    coords_T = convert(AbstractMatrix{T}, coords)
    ls_T = convert(typeof(ls) <: Real ? T : AbstractVector{T}, ls)

    local dist_sq
    if ls isa AbstractVector # ARD case
        if size(coords_T, 2) != length(ls_T)
            error("Dimension mismatch for ARD kernel: Number of coordinate dimensions ($(size(coords_T, 2))) does not match number of lengthscales ($(length(ls_T))).")
        end
        # Calculate weighted squared Euclidean distance
        dist_sq = pairwise(SqEuclidean(), coords_T ./ ls_T', dims=1)
    else # Isotropic case
        dist_sq = pairwise(SqEuclidean(), coords_T, dims=1) ./ ls_T^2
    end

    # Gaussian / Squared Exponential
    if kernel_type == :gaussian || kernel_type == :se
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    
    # Exponential / Matern 1/2
    elseif kernel_type == :exponential || kernel_type == :matern12 
        d = sqrt.(dist_sq)
        return param_val^2 .* exp.(-d) .+ (noise * I)
    
    # Matern 3/2
    elseif kernel_type == :matern32 
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 3.0)) .* d
        return param_val^2 .* (one(T) .+ val) .* exp.(-val) .+ (noise * I)
    
    # Matern 5/2
    elseif kernel_type == :matern52 
        d = sqrt.(dist_sq)
        val = sqrt(convert(T, 5.0)) .* d
        return param_val^2 .* (one(T) .+ val .+ (val.^2 ./ convert(T, 3.0))) .* exp.(-val) .+ (noise * I)

    # Constant Kernel (Identity innovation)
    elseif kernel_type == :constant
        return fill(convert(T, param_val^2), size(dist_sq))

    # Linear Kernel
    elseif kernel_type == :linear
        return param_val^2 .* dist_sq

    # Wavelet Multiscale Kernel
    elseif kernel_type == :wavelet
        K_accum = zeros(T, size(dist_sq))
        for wv_scale in 1:wavelet_levels
            ls_scale_sq = (ls isa Real ? ls_T^2 : one(T)) / (convert(T, 4.0)^(wv_scale-1))
            weight_scale = param_val^2 * exp(convert(T, -wv_scale) / ls_T)
            K_accum .+= weight_scale .* exp.(-one(T)/2 .* dist_sq ./ ls_scale_sq)
        end
        return K_accum + (noise * I)

    # Fallback Dispatch
    else
        return param_val^2 .* exp.(-one(T)/2 .* dist_sq) .+ (noise * I)
    end
end




# Version 1.5.4 (2026-08-06)
# Purpose: Recomposes a precision matrix from a template and hyperparameters.
# Rationale: This version is updated for improved type stability and AD-friendliness.
#            It replaces explicit type casts like `T_num(1.0)` with `one(T_num)` and
#            `T_num(0.0)` with `zero(T_num)`. It also ensures that identity matrices
#            are constructed with the correct element type, preventing potential
#            type mismatches when using AD libraries like ForwardDiff.jl.
#            The use of `convert(T_num, 0.5)` is preferred over `T_num(0.5)` for
#            consistency with Julia's promotion system.
# Assumptions: The input `template_s` is a parameter-free structure matrix.
function recompose_precision(m_type::Symbol, template_s::AbstractMatrix, param_val::Real; extra_param=nothing, noise=1e-4, kwargs...)
    n_s = size(template_s, 1)
    T_num = promote_type(typeof(param_val), typeof(noise), eltype(template_s), typeof(extra_param))

    if m_type == :SPDE
        kappa = isnothing(extra_param) ? one(T_num) : extra_param
        local Q_kappa
        if kappa isa Real
            Q_kappa = kappa^2 * I(n_s) # UniformScaling promotes correctly
        else
            if length(kappa) != n_s; error("Anisotropic kappa vector length must match number of spatial units."); end
            Q_kappa = Diagonal(kappa.^2) # Let promotion handle eltype
        end
        L_spde = Q_kappa + template_s
        return Symmetric(L_spde' * L_spde)
    end

    if m_type == :None || m_type == :FIXED
        return Symmetric(sparse(I, n_s, n_s))
    end

    if m_type == :Besag || m_type == :ICAR || m_type == :Cyclic
        return Symmetric(template_s)
    end

    if m_type == :AR1
        rho = isnothing(extra_param) ? zero(T_num) : extra_param
        
        Q = spdiagm(0 => fill(one(T_num) + rho^2, n_s))
        if n_s > 0
            Q[1, 1] = one(T_num)
            Q[n_s, n_s] = one(T_num)
        end
        
        Q .+= rho .* template_s
        return Symmetric(Q)
    end

    if m_type == :RW1 || m_type == :RW2
        error("recompose_precision should not be called for $(m_type) models. Use the state-space implementation.")
    end

    if m_type == :Leroux || m_type == :LocalAdaptive
        lambda_val = isnothing(extra_param) ? convert(T_num, 0.5) : extra_param
        I_prom = spdiagm(0 => ones(T_num, n_s))
        return Symmetric(lambda_val .* template_s + (one(T_num) - lambda_val) .* I_prom)
    end

    if m_type == :ST_I || m_type == :ST_II || m_type == :ST_III || m_type == :ST_IV
        error("recompose_precision should not be called for ST_I/II/III/IV models directly.")
    end

    if m_type == :NetworkFlow
        rho_net = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
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
        rho_p = isnothing(extra_param) ? convert(T_num, 0.8) : extra_param
        L_op = I(n_s) - rho_p .* template_s
        return Symmetric(L_op' * L_op)
    end

    if m_type == :GP
        ls = isnothing(extra_param) ? one(T_num) : extra_param
        K = param_val^2 .* exp.(-(Matrix(template_s).^2) ./ (convert(T_num, 2.0) * ls^2 + convert(T_num, noise)))
        return inv(Symmetric(K))
    end

    if m_type == :RFF || m_type == :FFT || m_type == :BSpline || m_type == :PSpline || m_type == :TPS
        return Symmetric(template_s)
    end

    return Symmetric(template_s)
end




# Version 1.5.3 (2026-08-06)
# Purpose: Generates the code block for calculating exploitation in dynamics models.
# Rationale: This version is updated to use `zeros(T_num_dyn, M.s_N)` for the default exploitation value.
#            This ensures that the `exploitation` variable is always initialized as a vector of the
#            correct generic type `T_num_dyn` (which becomes `ForwardDiff.Dual` during automatic
#            differentiation), resolving a type instability that caused a `MethodError` when no
#            effort or removal data was provided.
function generate_exploitation_block(spec, time_var)
    effort_keys = get(spec.hyper, :effort_keys, [])
    removal_keys = get(spec.hyper, :removal_keys, [])
    
    if isempty(effort_keys) && isempty(removal_keys)
        # Always initialize as a vector of the correct type and size.
        return "local exploitation = zeros(T_num_dyn, M.s_N)"
    end
    
    lines = ["local exploitation = zeros(T_num_dyn, M.s_N)"]
    for key in effort_keys
        push!(lines, "exploitation .+= q_$(key) .* spec_registry[\"$(spec.key)\"].hyper.processed_params[:$(key)][:, $(time_var)] .* N_prev")
    end
    for key in removal_keys
        push!(lines, "exploitation .+= spec_registry[\"$(spec.key)\"].hyper.processed_params[:$(key)][:, $(time_var)]")
    end
    return join(lines, "\n    ")
end



 
# Version 1.0.1 (2026-08-06)
# Purpose: Converts a `Distribution` object into a type-stable string for code generation.
# Rationale: This version adds explicit handling for `filldist` and `Product` distributions.
#            It recursively calls `_distribution_to_string` on the inner distributions,
#            ensuring that complex, nested distribution types are correctly and robustly
#            represented in the generated Turing model code. This prevents errors and
#            improves the clarity of the generated model.
function _distribution_to_string(d::Distribution)
    dist_name = string(typeof(d).name.name)
    if d isa Exponential
        return "$(dist_name){T}($(rate(d)))"
    elseif d isa Normal
        return "$(dist_name){T}($(mean(d)), $(std(d)))"
    elseif d isa LogNormal
        return "$(dist_name){T}($(params(d)[1]), $(params(d)[2]))"
    elseif d isa Beta
        return "$(dist_name){T}($(params(d)[1]), $(params(d)[2]))"
    elseif d isa InverseGamma
        return "$(dist_name){T}($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Gamma
        return "$(dist_name){T}($(Distributions.shape(d)), $(Distributions.scale(d)))"
    elseif d isa Uniform
        return "$(dist_name){T}($(minimum(d)), $(maximum(d)))"
    elseif d isa Truncated
        inner_dist_str = _distribution_to_string(d.untruncated)
        return "truncated($(inner_dist_str), $(d.lower), $(d.upper))"
    elseif d isa Product
        if hasproperty(d, :v) && d.v isa Fill
            inner_dist_str = _distribution_to_string(d.v.value)
            n = length(d.v)
            return "filldist($(inner_dist_str), $(n))"
        elseif hasproperty(d, :v) && all(dist -> dist == d.v[1], d.v)
             inner_dist_str = _distribution_to_string(d.v[1])
             n = length(d.v)
             return "filldist($(inner_dist_str), $(n))"
        else
            inner_strs = [_distribution_to_string(dist) for dist in d.v]
            return "Product([$(join(inner_strs, ", "))])"
        end
    elseif d isa Fill
        inner_dist_str = _distribution_to_string(d.value)
        n = length(d)
        return "filldist($(inner_dist_str), $(n))"
    else
        @warn "String conversion for distribution type `$(typeof(d))` is not explicitly handled. The generated code may not be type-stable."
        return string(d)
    end
end



 
function bstm_dynamic_model(config::NamedTuple)
    # Purpose: A unified entry point for compiling and instantiating any dynamically generated model.
    # Rationale: Decouples model generation from execution.
    # v1.0.0 (2026-07-16)
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





# Code Generators for Advanced Components
 

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


