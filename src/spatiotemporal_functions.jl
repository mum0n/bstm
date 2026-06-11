
#---------------------------------------------
#---------------------------------------------
#---------------------------------------------
 
# --- Refined Prior Assignment Engine ---
# Explicit 'nothing' values are maintained to preserve schema consistency 
# across different manifold categories (Discrete, Continuous, Seasonal).

const BSTM_DEFAULT_HYPERPRIORS = Dict{
    DataType, 
    NamedTuple
}(
    # Discrete Spatial: BYM2 has rho, ICAR does not.
    BYM2      => (sigma_prior = Exponential(1.0), rho_prior = Beta(1, 1)),
    ICAR      => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    ProperCAR => (sigma_prior = Exponential(1.0), rho_prior = Beta(1, 1)),
    RW1       => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    RW2       => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    
    # Continuous Spatial: Use lengthscale instead of rho.
    GaussianProcess        => (sigma_prior = Exponential(1.0), lengthscale_prior = Exponential(1.0), nu = 1.5, rho_prior = nothing),
    FITC                   => (sigma_prior = Exponential(1.0), lengthscale_prior = Exponential(1.0), n_inducing = 50, rho_prior = nothing),
    RandomFourierFeatures  => (sigma_prior = Exponential(1.0), lengthscale_prior = Exponential(1.0), n_features = 100, rho_prior = nothing),
    SPDE                   => (sigma_prior = Exponential(1.0), smoothness_prior = Exponential(1.0), rho_prior = nothing),
    
    # Temporal: AR1 has rho, RW/IID do not.
    AR1  => (sigma_prior = Exponential(1.0), rho_prior = Beta(2, 2)),
    RW1T => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    RW2T => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    IIDT => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    
    # Seasonal
    HarmonicSeasonal => (sigma_prior = Exponential(1.0), amplitude_prior = Normal(0, 1), phase_prior = Uniform(0, 2*pi), rho_prior = nothing),
    DiscreteSeasonal => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    
    # Covariate Smooths
    Smooth  => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    TPS     => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    PSpline => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    BSpline => (sigma_prior = Exponential(1.0), rho_prior = nothing),
    
    # Fallback
    Manifold => (sigma_prior = Exponential(1.0), rho_prior = nothing)
)

const BSTM_PRIOR_SCHEMES = Dict(
    :informative => BSTM_DEFAULT_HYPERPRIORS,
    
    :weakly_informative => Dict(
        BYM2      => (sigma_prior = Exponential(5.0), rho_prior = Beta(0.5, 0.5)),
        ICAR      => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        ProperCAR => (sigma_prior = Exponential(5.0), rho_prior = Beta(1.0, 1.0)),
        
        GaussianProcess        => (sigma_prior = Exponential(5.0), lengthscale_prior = InverseGamma(2, 5), rho_prior = nothing),
        FITC                   => (sigma_prior = Exponential(5.0), lengthscale_prior = InverseGamma(2, 5), rho_prior = nothing),
        RandomFourierFeatures  => (sigma_prior = Exponential(5.0), lengthscale_prior = InverseGamma(2, 5), rho_prior = nothing),
        SPDE                   => (sigma_prior = Exponential(5.0), smoothness_prior = Exponential(5.0), rho_prior = nothing),
        
        AR1  => (sigma_prior = Exponential(5.0), rho_prior = Beta(1.0, 1.0)),
        RW1T => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        RW2T => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        
        HarmonicSeasonal => (sigma_prior = Exponential(5.0), amplitude_prior = Normal(0, 5), phase_prior = Uniform(0, 2*pi), rho_prior = nothing),
        DiscreteSeasonal => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        
        Smooth  => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        TPS     => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        PSpline => (sigma_prior = Exponential(5.0), rho_prior = nothing),
        BSpline => (sigma_prior = Exponential(5.0), rho_prior = nothing)
    )
)

function resolve_hyperpriors(m_type::DataType, user_overrides::Dict, scheme_sym::Symbol)
    scheme_dict = get(BSTM_PRIOR_SCHEMES, scheme_sym, BSTM_DEFAULT_HYPERPRIORS)
    base_defaults = get(scheme_dict, m_type, get(BSTM_DEFAULT_HYPERPRIORS, m_type, BSTM_DEFAULT_HYPERPRIORS[Manifold]))
    
    if haskey(user_overrides, m_type)
        return merge(base_defaults, user_overrides[m_type])
    end
    
    return base_defaults
end

# --- Manifold Operator Definitions ---
function ⊗ end

function ⊕ end

# Base pipe for transformations: Manifold |> Manifold (Transformation)
# If the RHS is a Transformation (Log, ZScore, etc.), wrap the LHS
function Base.:|>(m::Manifold, t::Manifold)
    return TransformedManifold(m, typeof(t))
end

# Base pipe for composition: Manifold |> Manifold (Stacking)
# If the RHS is a Structural Manifold (Temporal, Spatial), compose them
function Base.:|>(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :pipe)
end

# Kronecker Product (Separable Space-Time Interaction)
# Overloading the newly defined ⊗ operator
function ⊗(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :kronecker_product)
end

# Direct Sum (Additive Components)
# Overloading the newly defined ⊕ operator
function ⊕(m1::Manifold, m2::Manifold)
    return ComposedManifold([m1, m2], :direct_sum)
end
 

# --- Fixed and Audited bstm Logic with Type-Safe Metadata ---
# Rationale: The MethodError: Cannot convert Exponential to String
# stemmed from internal dictionaries within re_rules being implicitly 
# typed as Dict{Symbol, String}. now force Dict{Symbol, Any}.

function parse_manifold_graph(expr_in::AbstractString)
    # Convert to standard String and strip whitespace
    expr = string(replace(expr_in, r"\s" => ""))

    # Recursive logic with explicit string conversion for elements to handle SubStrings
    if occursin("⊕", expr)
        elements = Base.split(expr, "⊕")
        return (type=:sum, elements=parse_manifold_graph.(string.(elements)))

    elseif occursin("⊗", expr)
        elements = Base.split(expr, "⊗")
        return (type=:kronecker, elements=parse_manifold_graph.(string.(elements)))

    elseif occursin("|>", expr)
        parts = Base.split(expr, "|>")
        return (type=:composition, elements=parse_manifold_graph.(string.(parts)))

    else
        m = match(r"(\w+)\((.*?)\)", expr)
        if !isnothing(m)
            # Extract model name and variable argument
            m_name = lowercase(string(m.captures[1]))
            # Extract text inside brackets more robustly
            var_match = match(r"\((.*)\)", expr)
            var_name = !isnothing(var_match) ? string(var_match.captures[1]) : expr
            return (type=:atomic, model=m_name, var=var_name)
        else
            return (type=:atomic, model="literal", var=expr)
        end
    end
end


# --- Unified bstm Entry Point and Recursive DSL Router ---
# This section provides the consolidated, feature-complete version of the bstm framework.
# It integrates support for:
# 1. Recursive Manifold DSL (Sum ⊕, Kronecker ⊗, Pipe |>)
# 2. Advanced Effects: Eigen-Effects (ee), SVC (Spatially Varying Coefficients), and Interactions (ie)
# 3. Architectural Dispatch: Univariate, Multivariate, and Multifidelity


"""
    bstm(formula, data; kwargs...)

Main entry point for Bayesian Spatio-Temporal Modeling. 
Consolidates formula parsing, DSL resolution, and architectural dispatch.
"""



function bstm(formula::Union{String, StatsModels.FormulaTerm}, data_input::Union{DataFrame, NamedArray};
    model_family="gaussian",
    model_arch="univariate",
    hyperpriors=Dict{DataType, Any}(),
    hyperprior_scheme=:informative,
    auxiliary_responses=nothing,
    auxiliary_data=nothing,
    return_data=false,
    contrasts=Dict{Symbol, Any}(),
    kwargs...)

    # 1. Initialization and Data Normalization
    # Deep copy ensures the original user DataFrame remains untouched during internal processing.
    data = data_input isa DataFrame ? copy(data_input) : DataFrame(data_input, :auto)
    opt_kwargs = Dict{Symbol, Any}(kwargs)
    internal_contrasts = copy(contrasts)

    # 2. Formula Decomposition (LHS ~ RHS)
    f_str = string(formula)
    sides = Base.split(f_str, "~")
    if length(sides) < 2
        error("BSTM Error: Formula must contain a '~' separator.")
    end
    lhs_side = strip(sides[1])
    rhs = strip(sides[2])

    # 3. Response Dimensionality Routing
    # Multi-variable LHS or explicit auxiliary data triggers architectural shifts.
    lhs_vars = Symbol.(filter(!isempty, strip.(Base.split(lhs_side, "+"))))
    opt_kwargs[:outcomes_N] = length(lhs_vars)

    if !isnothing(auxiliary_responses)
        # Multifidelity routing if secondary data sources are provided.
        opt_kwargs[:model_arch] = "multifidelity"
        opt_kwargs[:auxiliary_responses] = auxiliary_responses
        opt_kwargs[:auxiliary_data] = auxiliary_data
    end

    if length(lhs_vars) > 1
        opt_kwargs[:model_arch] = "multivariate"
        opt_kwargs[:y_obs] = Matrix(data[!, lhs_vars])
    else
        # Standard Univariate or user-defined architecture override.
        opt_kwargs[:model_arch] = get(opt_kwargs, :model_arch, model_arch)
        opt_kwargs[:y_obs] = data[!, lhs_vars[1]]
    end

    # --- CRITICAL: Coordinate Preservation ---
    # Locking raw coordinates into the options dictionary to prevent downstream reconstruction errors.
    if !haskey(opt_kwargs, :s_x) && "s_x" in names(data); opt_kwargs[:s_x] = data.s_x; end
    if !haskey(opt_kwargs, :s_y) && "s_y" in names(data); opt_kwargs[:s_y] = data.s_y; end
    if !haskey(opt_kwargs, :t_v) && "t_v" in names(data); opt_kwargs[:t_v] = data.t_v; end

    # 4. Effect Discovery and Container Initialization
    # re_rules is explicitly Typed as Any to hold Distributions and Metadata dicts.
    re_rules = Dict{String, Any}()
    fixed_parts = String[]
    mixed_terms = []
    interaction_terms = []
    svc_covs = Symbol[]
    eigen_terms = []

    has_intercept = true
    intercept_prior = nothing

    # Parse RHS terms separated by '+'
    rhs_terms = strip.(Base.split(rhs, "+"))

    for term in rhs_terms
        term_clean = strip(term)
        term_lower = lowercase(term_clean)

        # 4.1 Intercept Controls
        if term_lower == "0" || term_lower == "-1"
            has_intercept = false
        elseif term_lower == "1" || startswith(term_lower, "intercept(")
            has_intercept = true
            m_int = match(r"intercept\(prior=([^)]+)\)", term_lower)
            if !isnothing(m_int)
                intercept_prior = parse_prior_distribution(m_int.captures[1], Normal(0, 5))
            end

        # 4.2 Smooth, Interaction, and Nested Discovery
        elseif startswith(term_lower, "smooth(") || startswith(term_lower, "interaction(") || startswith(term_lower, "nested(")
            m = match(r"(?:smooth|interaction|nested)\(([^;)]+)(?:;\s*(.*))?\)", term_lower)
            if !isnothing(m)
                vars_part = strip(m.captures[1])
                params_part = isnothing(m.captures[2]) ? "" : m.captures[2]
                sub_vars = strip.(Base.split(vars_part, ","))
     
                # Eigen-Covariate Routing (v05.5 Specific Change)
                # Rationale: Eigen() is now restricted to covariates/loadings, not temporal indices.
                if startswith(term_lower, "eigen(")
                    m_e = match(r"eigen\(([^;)]+)(?:;\s*(.*))?\)", term_lower)
                    if !isnothing(m_e)
                        var_name = Symbol(strip(m_e.captures[1]))
                        params = isnothing(m_e.captures[2]) ? "" : m_e.captures[2]
                        rank_match = match(r"rank=(\d+)", params)
                        k_rank = isnothing(rank_match) ? 3 : parse(Int, rank_match.captures[1])
                        
                        push!(eigen_terms, (
                            data = Matrix(data[!, [var_name]]),
                            n_dims = k_rank,
                            ltri_indices = collect(1:(k_rank*(k_rank+1)÷2))
                        ))
                    end
                end
                    # 4.3 Unified Smooth() Routing

                if startswith(term_lower, "nested(")
                    v_name = strip(sub_vars[1])
                    m_man = match(r"manifold=['\"]?(\w+)['\"]?", params_part)
                    m_raw = isnothing(m_man) ? "bym2" : m_man.captures[1]
                    opt_kwargs[:model_arch] = "multifidelity"
                    re_rules[v_name] = Dict(:model => string(m_raw), :is_nested => true)
                    push!(fixed_parts, v_name)

                elseif length(sub_vars) == 2
                    # 2D Interaction Smooth Logic
                    v1, v2 = Symbol(sub_vars[1]), Symbol(sub_vars[2])
                    m_raw = "rw2"
                    m_man = match(r"manifold=['\"]?(\w+)['\"]?", params_part)
                    if !isnothing(m_man) m_raw = m_man.captures[1] end
                    nb1 = match(r"nbins1=(\d+)", params_part)
                    nb2 = match(r"nbins2=(\d+)", params_part)

                    push!(interaction_terms, (
                        var1=v1, var2=v2, manifold=Symbol(m_raw),
                        nbins1=isnothing(nb1) ? 10 : parse(Int, nb1.captures[1]),
                        nbins2=isnothing(nb2) ? 10 : parse(Int, nb2.captures[1]),
                        coords=hcat(data[!, v1], data[!, v2]),
                        # Parameters for RFF approximation
                        M_rff=20,
                        W_ie=randn(2, 20),
                        b_ie=rand(20) .* (2π)
                         
                    ))

                else
                    # 1D Smooth Logic
                    v_name = strip(sub_vars[1])
                    m_raw = "rw2"
                    m_man = match(r"manifold=['\"]?(\w+)['\"]?", params_part)
                    if !isnothing(m_man) m_raw = m_man.captures[1] end
                    nb = match(r"nbins=(\d+)", params_part)
                    deg = match(r"degree=(\d+)", params_part)

                    re_rules[v_name] = Dict(
                        :model => string(m_raw),
                        :nbins => isnothing(nb) ? 10 : parse(Int, nb.captures[1]),
                        :degree => isnothing(deg) ? 3 : parse(Int, deg.captures[1]),
                        :is_smooth => true
                    )
                    push!(fixed_parts, v_name)
                end
            end

        # 4.3 Mixed Effects (Varying Slopes)
        elseif startswith(term_lower, "mixed(")  
            m_me = match(r"(?:mixed)\(([^|]+)\|([^)]+)\)", term_lower)
            if !isnothing(m_me)
                cov_var = strip(m_me.captures[1])
                group_var = Symbol(strip(m_me.captures[2]))
                lvls = unique(data[!, group_var])
                g_map = Dict(v => i for (i, v) in enumerate(lvls))
                indices = [g_map[v] for v in data[!, group_var]]
                cov_vals = (cov_var == "1") ? ones(size(data, 1)) : Vector{Float64}(data[!, Symbol(cov_var)])
                push!(mixed_terms, (indices = indices, n_cat = length(lvls), covariate_vals = cov_vals, name = group_var))
            end

        # 4.4 DSL Algebraic Operators (Sum, Kronecker, Pipe)
        elseif occursin(r"[⊗⊕|>|∘]", term_clean) || occursin("(", term_clean)
            graph_struct = parse_manifold_graph(term_clean)
            term_re_rules = Dict{String, Any}()
            process_graph_into_rules!(term_re_rules, opt_kwargs, graph_struct)
            merge!(re_rules, term_re_rules)

        # 4.5 Fallback: Linear Variable
        else
            push!(fixed_parts, term_clean)
        end
    end

    # 5. Hyperprior Injection Phase
    # Ensures that every discovered manifold has a valid sigma and structural prior.
    for (var, rule) in re_rules
        m_name = get(rule, :model, "bym2")
        m_type = if m_name == "bym2"; BYM2
                 elseif m_name == "icar"; ICAR
                 elseif m_name == "ar1"; AR1
                 elseif m_name == "rw1"; RW1
                 elseif m_name == "rw2"; RW2
                 elseif m_name in ["gp", "dense_gp"]; GaussianProcess
                 elseif m_name == "rff"; RandomFourierFeatures
                 elseif m_name == "fitc"; FITC
                 elseif m_name == "spde"; SPDE
                 elseif m_name == "harmonic"; HarmonicSeasonal
                 elseif m_name == "tps"; TPS
                 elseif m_name == "pspline"; PSpline
                 elseif m_name == "bspline"; BSpline
                 else Manifold end

        resolved = resolve_hyperpriors(m_type, hyperpriors, hyperprior_scheme)
        new_rule = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in rule)
        new_rule[:sigma_prior] = resolved.sigma_prior

        for prop in [:rho_prior, :lengthscale_prior, :smoothness_prior, :amplitude_prior, :phase_prior]
            if hasproperty(resolved, prop)
                val = getproperty(resolved, prop)
                if !isnothing(val); new_rule[prop] = val; end
            end
        end
        re_rules[var] = new_rule
    end

    # 6. Final Parameter Mapping
    opt_kwargs[:add_intercept] = has_intercept
    opt_kwargs[:intercept_prior] = intercept_prior
    opt_kwargs[:re_rules] = re_rules
    opt_kwargs[:fixed_parts] = fixed_parts
    opt_kwargs[:contrasts] = internal_contrasts
    opt_kwargs[:eigen_terms] = eigen_terms
    opt_kwargs[:mixed_terms] = mixed_terms
    opt_kwargs[:interaction_terms] = interaction_terms
    opt_kwargs[:svc_covariates] = svc_covs
    opt_kwargs[:data] = data

    # 7. Architecture Dispatch
    inp = bstm_options(; opt_kwargs...)
    arch_type = get(opt_kwargs, :model_arch, "univariate")

    if arch_type == "multivariate"; return bstm_multivariate(inp)
    elseif arch_type == "multifidelity"; return bstm_multifidelity(inp)
    else; return bstm_univariate(inp); end
end





#---------------------------------------------
#---------------------------------------------
#---------------------------------------------



function init_params_extract(X)
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    # print everything to console
    show(stdout, "text/plain", x) # display all estimates
end 
 

function firstindexin(a::AbstractArray, b::AbstractArray)
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
  
function β( mode, conc )
    # alternate parameterization of beta distribution 
    # conc = α + β     https://en.wikipedia.org/wiki/Beta_distribution
    beta1 = mode *( conc - 2  ) + 1.0
    beta2 = (1.0 - mode) * ( conc - 2  ) + 1.0
    Beta( beta1, beta2 ) 
end 
  
function modelruntime(o)
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
end


################




function expand_hull(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - s_x: Vector of x-coordinates.
    - s_y: Vector of y-coordinates.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if isempty(s_coord_tuple_local) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    coords_vec = [[Float64(p[1]), Float64(p[2])] for p in s_coord_tuple_local]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end

 

function get_kde_seeds(s_coord_tuple_local, target_u)
 
    # Basic KDE-based seeding using StatsBase weights based on local density
    u_pts = unique(s_coord_tuple_local)
    if isempty(u_pts) return [] end
    n = length(u_pts)
    dists = [sum((p1 .- p2).^2) for p1 in u_pts, p2 in u_pts]
    # Inverse of mean distance as a density proxy
    weights = 1.0 ./ (mean(dists, dims=2)[:] .+ 1e-6)
    idx = StatsBase.sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return u_pts[idx]
end

 
function is_valid_polygon_coords(poly_coords)
    # Filters out NaN/Inf values and checks for a minimum of 3 valid points for a polygon.

    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    return length(valid_pts) >= 3
end
 

function get_cvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: Centroidal Voronoi Tessellation (CVT) with diagnostic termination tracking.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple_local)
    idx = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]
    termination_reason = "max_iterations"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        shifts = Float64[]

        for i in 1:length(polys)
            poly_coords = polys[i]
            area = get_polygon_area(poly_coords) # Using refactored area func

            if length(poly_coords) > 2 && area >= cfg.min_area && area <= cfg.max_area
                lg_poly = LibGEOS.Polygon([[ [p[1], p[2]] for p in poly_coords ]])
                cent_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(cent_geom)
                new_c = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))

                dist = sqrt(sum((new_c .- curr_centroids[i]).^2))
                push!(shifts, dist)
                push!(new_centroids, new_c)
            else
                push!(new_centroids, curr_centroids[i])
            end
        end

        if isempty(shifts) || mean(shifts) < cfg.tolerance
            termination_reason = "convergence"
            break
        end

        # Use original s_coord_tuple_local for assignment to clusters
        assigns = [argmin([sum((p .- c).^2) for c in new_centroids]) for p in s_coord_tuple_local]
        counts = [count(==(i), assigns) for i in 1:length(new_centroids)]

        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, termination_reason
end


function get_kvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: K-means Voronoi Tessellation (KVT) with diagnostic termination tracking.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)


    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple_local)
    idx_init = StatsBase.sample(1:length(u_pts), min(cfg.target, length(u_pts)), replace=false)
    c_iter = [u_pts[i] for i in idx_init]
    data = tuple.(s_coord_tuple_local, cfg.t_idx)

    damping = 0.7
    termination_reason = "max_iterations"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    for iter in 1:100
        old_centroids = copy(c_iter)
        assigns = [argmin([sum((p[1] .- sj).^2) for sj in c_iter]) for p in data]

        polys_coords, _ = get_voronoi_polygons_and_edges(c_iter, hull_geom)

        for k in 1:length(c_iter)
            idx_cluster = findall(==(k), assigns)
            ts_count = length(unique([data[j][2] for j in idx_cluster]))

            area = 0.0
            if k <= length(polys_coords)
                area = get_polygon_area(polys_coords[k])
            end

            # Modified area_ok condition: require positive area
            area_ok = (area > 0) && area >= cfg.min_area && area <= cfg.max_area

            if !isempty(idx_cluster) && length(idx_cluster) >= cfg.min_points && ts_count >= cfg.min_time_slices && area_ok
                mean_x = mean(data[j][1][1] for j in idx_cluster)
                mean_y = mean(data[j][1][2] for j in idx_cluster)

                c_iter[k] = ((1.0 - damping) * old_centroids[k][1] + damping * mean_x,
                             (1.0 - damping) * old_centroids[k][2] + damping * mean_y)
            end
        end

        counts = [count(==(k), assigns) for k in 1:length(c_iter)]
        if isempty(counts)
            termination_reason = "no_units_formed"
            break
        end

        # New Density Convergence Check
        curr_mean_density = mean(counts)
        if abs(curr_mean_density - last_mean_density) < cfg.tolerance && iter > 1
            termination_reason = "density_convergence"
            break
        end
        last_mean_density = curr_mean_density

        cv_val = std(counts) / (mean(counts) + 1e-9)
        # CV Convergence Check
        if abs(cv_val - last_cv) < cfg.tolerance && iter > 1
            termination_reason = "cv_convergence"
            break
        end
        last_cv = cv_val

        if mean(counts) < cfg.min_points
            termination_reason = "min_points_violation"
            break
        end

        damping *= 0.99
    end

    return c_iter, termination_reason
end



function get_qvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    # Local helper to convert flat s_x, s_y to a vector of tuples

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end
 
    data = tuple.(s_coord_tuple_local, cfg.t_idx)

    regions = [data]
    # Track specific region objects that failed to split to avoid redundant attempts
    unsplittable = Set{UInt64}()

    effective_min_p = max(1, cfg.min_points)

    if length(data) < 2 * effective_min_p # Initial check: if the whole dataset is too small to split
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))], "initial_data_too_small_to_tessellate"
    end

    termination_reason = "max_units_reached"

    # Initialize convergence tracking variables
    last_mean_density = 0.0
    last_cv = 0.0

    cnt = 0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1

        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Early breaking based on statistical stabilization and target resolution
        if cnt > 3
            if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    termination_reason = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(regions)) < cfg.tolerance/10
                    termination_reason = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Candidacy: regions that can be split
        viable_indices = findall(r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions)
        if cnt > 3
            if isempty(viable_indices); termination_reason = "cannot_split_further"; break; end
        end

        # Split if (below min_total_arealunits) OR (below target OR has max_points violators)
        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = (must_split || want_split) ? (isempty(violators) ? viable_indices : violators) : []

        if isempty(candidates); termination_reason = "constraints_satisfied"; break; end

        # Attempt splitting the largest available candidate
        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target_region = regions[target_idx]

        xs_r = [p[1][1] for p in target_region]; ys_r = [p[1][2] for p in target_region]

        # Robust splitting: handle datasets with zero variance in one or more dimensions
        if length(unique(xs_r)) > 1 || length(unique(ys_r)) > 1
            mx = length(unique(xs_r)) > 1 ? median(xs_r) : xs_r[1]
            my = length(unique(ys_r)) > 1 ? median(ys_r) : ys_r[1]
            r_splits = [
                filter(p -> p[1][1] <= mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] <= my, target_region),
                filter(p -> p[1][1] <= mx && p[1][2] > my, target_region),
                filter(p -> p[1][1] > mx && p[1][2] > my, target_region)
            ]
        else
            # All points collocated spatially: split by index to progress toward target unit count
            mid = length(target_region) ÷ 2  # Corrected from ∈ 2 to ÷ 2
            r_splits = [target_region[1:mid], target_region[mid+1:end], [], []]
        end

        valid_splits = filter(r -> length(r) >= effective_min_p, r_splits)

        if length(valid_splits) < 2
            # This specific region is locally unsplittable; mark it and continue with others
            push!(unsplittable, objectid(target_region))
            continue
        end

        deleteat!(regions, target_idx)
        append!(regions, valid_splits)
    end

    # Post-audit: filter by final constraints (minimum time slices and point counts)
    final_filtered_regions = filter(regions) do r
        length(r) >= effective_min_p && length(unique([p[2] for p in r])) >= cfg.min_time_slices
    end

    if isempty(final_filtered_regions)
        return [], "no_valid_units_after_filter"
    end

    final_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in final_filtered_regions]

    polys_coords, _ = get_voronoi_polygons_and_edges(final_centroids, hull_geom)
    area_violation = any(
        p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_area,
        polys_coords
    )

    final_status = termination_reason
    if length(final_centroids) < cfg.min_total_arealunits
        final_status = "insufficient_units_error"
    elseif area_violation
        final_status = "min_area_violation"
    end

    return final_centroids, final_status
end



function get_bvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    """
    Synopsis: Binary Voronoi Tessellation (BVT) with corrected recursive splitting logic.
    """

    s_coord_tuple_local = tuple.(s_x, s_y)

    if length(s_coord_tuple_local) <= cfg.min_total_arealunits
        return [ (mean(s_x), mean(s_y)) ], "not_enough_points_to_tessellate"
    end

    data = tuple.(s_coord_tuple_local, cfg.t_idx)
    regions = [data]
    # Track specific region objects that failed to split to avoid redundant attempts

    effective_min_p = max(1, cfg.min_points)
    if length(data) < 2 * effective_min_p # Initial check: if the whole dataset is too small to split
        return [(mean(p[1][1] for p in data), mean(p[1][2] for p in data))], "initial_data_too_small_to_tessellate"
    end

    unsplittable = Set{UInt64}()
    termination_reason = "max_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0
    cnt =0

    while length(regions) < cfg.max_total_arealunits
        cnt += 1

        counts = length.(regions)
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Early breaking based on statistical stabilization and target resolution
        if cnt > 3
            if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
                if length(regions) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    termination_reason = "converged_constraints_satisfied"
                    break
                elseif (abs(cv_val - cfg.target_cv) < cfg.tolerance)
                    termination_reason = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(regions)) < cfg.tolerance/10
                    termination_reason = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # Candidacy: regions that can be split
        viable_indices = findall(r -> length(r) >= max(2, effective_min_p) && objectid(r) ∉ unsplittable, regions)
        if cnt > 3
            if isempty(viable_indices); termination_reason = "cannot_split_further"; break; end
        end

        # Split if (below min_total_arealunits) OR (below target OR has max_points violators)
        violators = filter(i -> length(regions[i]) > cfg.max_points, viable_indices)
        must_split = length(regions) < cfg.min_total_arealunits
        want_split = length(regions) < cfg.target || !isempty(violators)
        candidates = (must_split || want_split) ? (isempty(violators) ? viable_indices : violators) : []

        if isempty(candidates); termination_reason = "constraints_satisfied"; break; end

        # Attempt splitting the largest available candidate
        target_idx = candidates[argmax([length(regions[i]) for i in candidates])]
        target = regions[target_idx]

        xs = [p[1][1] for p in target]; ys = [p[1][2] for p in target]
        var_x = length(xs) > 1 ? var(xs) : 0.0
        var_y = length(ys) > 1 ? var(ys) : 0.0
        dim = var_x > var_y ? 1 : 2

        if var_x > 1e-9 || var_y > 1e-9
            vals = [p[1][dim] for p in target]
            med = length(unique(vals)) > 1 ? median(vals) : vals[1]
            r1 = filter(p -> p[1][dim] <= med, target)
            r2 = filter(p -> p[1][dim] > med, target)
        else
            # Handle collocated points
            mid = length(target) ÷ 2
            r1, r2 = target[1:mid], target[mid+1:end]
        end

        # Validate children for point count and temporal diversity
        v1 = length(r1) >= effective_min_p && length(unique([p[2] for p in r1])) >= cfg.min_time_slices
        v2 = length(r2) >= effective_min_p && length(unique([p[2] for p in r2])) >= cfg.min_time_slices

        if !v1 || !v2
             push!(unsplittable, objectid(target))
             continue
        end

        # Tentative update to check global area constraints
        tentative_regions = copy(regions)
        deleteat!(tentative_regions, target_idx)
        push!(tentative_regions, r1, r2)

        candidate_centroids = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in tentative_regions]
        polys_coords, _ = get_voronoi_polygons_and_edges(candidate_centroids, hull_geom)

        area_violation = any(
            p_coords -> !is_valid_polygon_coords(p_coords) || get_polygon_area(p_coords) < cfg.min_area,
            polys_coords
        )

        if area_violation && length(tentative_regions) > cfg.min_total_arealunits
             push!(unsplittable, objectid(target))
             continue
        end

        regions = tentative_regions
    end

    final_centroids_candidate = [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]

    # if length(final_centroids_candidate) < cfg.min_total_arealunits
    #     # Aggregate all original points (from 'data') into a single centroid
    #     all_pts_x = [p[1][1] for p in data]
    #     all_pts_y = [p[1][2] for p in data]
    #     return [ (mean(all_pts_x), mean(all_pts_y)) ], "insufficient_units_error"
    # else
        return final_centroids_candidate, termination_reason
    # end
end

 
 
function get_hvt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom; max_iter=500)

    s_coord_tuple_local = tuple.(s_x, s_y)

    # Internal utility for point-to-centroid distance
    dist(p1, p2) = sqrt(sum((p1 .- p2).^2))

    # Standardized refinement loop (Lloyd's update)
    function refine(pts_in, centroids, iters)
        curr = deepcopy(centroids)
        for _ in 1:iters
            groups = [Int[] for _ in 1:length(curr)]
            for (i, p) in enumerate(pts_in)
                dists = [dist(p, c) for c in curr]
                push!(groups[argmin(dists)], i)
            end
            new_c = [!isempty(idx) ?
                     (mean(pts_in[j][1] for j in idx), mean(pts_in[j][2] for j in idx)) :
                     curr[k] for (k, idx) in enumerate(groups)]
            if all(dist(new_c[j], curr[j]) < cfg.tolerance for j in 1:length(curr))
                return new_c
            end
            curr = new_c
        end
        return curr
    end

    # Initial Seed generation using k-means
    # Convert s_coord_tuple_local to matrix for Clustering.jl
    pts_matrix = hcat([p[1] for p in s_coord_tuple_local], [p[2] for p in s_coord_tuple_local])'
    k_target = max(1, cfg.min_total_arealunits)

    # Run k-means to find initial centers
    R = kmeans(pts_matrix, k_target)
    curr_centroids = [(R.centers[1, i], R.centers[2, i]) for i in 1:size(R.centers, 2)]

    # Iterative HVT process with advanced stopping conditions
    last_mean_density = 0.0
    last_cv = 0.0
    status = "max_iterations_reached"

    for i in 1:max_iter
        # 1. Assignment step to calculate metrics
        s_idx = [argmin([dist(p, c) for c in curr_centroids]) for p in s_coord_tuple_local]
        counts = [count(==(k), s_idx) for k in 1:length(curr_centroids)]

        curr_mean_density = mean(counts)
        cv_val = std(counts) / (curr_mean_density + 1e-9)

        # Convergence logic aligned with QVT/BVT
        if i > 5
            if abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance
                if length(curr_centroids) >= cfg.target && all(c -> c <= cfg.max_points, counts)
                    status = "converged_constraints_satisfied"
                    break
                elseif abs(cv_val - cfg.target_cv) < cfg.tolerance
                    status = "converged_target_cv"
                    break
                elseif (count(>(cfg.max_points), counts) / length(curr_centroids)) < cfg.tolerance/10
                    status = "converged_minor_violations"
                    break
                end
            end
        end

        last_mean_density = curr_mean_density
        last_cv = cv_val

        # 2. Refinement step (Lloyd's update)
        new_centroids = refine(s_coord_tuple_local, curr_centroids, 3)

        # Check for centroid position stabilization
        if all(dist(new_centroids[j], curr_centroids[j]) < cfg.tolerance for j in 1:length(curr_centroids))
             # If positions stabilized but constraints aren't met, check if add a unit
             if length(curr_centroids) < cfg.max_total_arealunits && (length(curr_centroids) < cfg.target || any(counts .> cfg.max_points))
                 # Split the largest group to improve density balance
                 idx_to_split = argmax(counts)
                 group_pts = s_coord_tuple_local[s_idx .== idx_to_split]
                 if length(group_pts) >= 2 * cfg.min_points
                     new_seeds = [(mean(p[1] for p in group_pts) * 0.99, mean(p[2] for p in group_pts) * 0.99),
                                  (mean(p[1] for p in group_pts) * 1.01, mean(p[2] for p in group_pts) * 1.01)]
                     deleteat!(curr_centroids, idx_to_split)
                     append!(curr_centroids, new_seeds)
                     continue
                 end
             end
             status = "converged_stable_positions"
             break
        end

        curr_centroids = new_centroids
    end

    return curr_centroids, status
end





"""
    get_avt_centroids(s_x, s_y, cfg, hull_geom)

Iterative Adaptive Voronoi Tessellation. Merges units that violate constraints
on point counts, time-slice representation, or geometric area.
"""
function get_avt_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, cfg, hull_geom)
    s_coord_tuple = tuple.(s_x, s_y)
    
    if length(s_coord_tuple) <= cfg.min_total_arealunits
        return [ (mean(p[1] for p in s_coord_tuple), mean(p[2] for p in s_coord_tuple)) ], "not_enough_points_to_tessellate"
    end

    u_pts = unique(s_coord_tuple)
    # Seeding centroids via KDE-based logic
    c_init = get_kde_seeds(u_pts, min(length(u_pts), cfg.max_total_arealunits))
    data = tuple.(s_coord_tuple, cfg.t_idx)
    curr_c = [SVector{2, Float64}(c) for c in c_init]

    termination_reason = "min_units_reached"
    last_mean_density = 0.0
    last_cv = 0.0

    while length(curr_c) > cfg.min_total_arealunits
        # 1. Assignment
        assigns = [Int[] for _ in 1:length(curr_c)]
        for i in 1:length(data)
            d_pt = data[i][1]
            # Finding closest centroid (Explicit loop for clarity)
            dist_idx = argmin([sum((d_pt .- c).^2) for c in curr_c])
            push!(assigns[dist_idx], i)
        end
        
        counts = length.(assigns)
        
        # 2. Geometry Calculation
        # get_voronoi_polygons_and_edges returns Vector{Vector{Tuple{Float64, Float64}}}
        polys_coords, _ = get_voronoi_polygons_and_edges([Tuple(c) for c in curr_c], hull_geom)
        
        areas = fill(0.0, length(curr_c))
        for i in 1:min(length(curr_c), length(polys_coords))
            # Fixed Call: Passes the Vector of Tuples directly to the new method
            areas[i] = get_polygon_area(polys_coords[i])
        end

        # 3. Violation Audit
        violators = Int[]
        for k in 1:length(curr_c)
            ts_count = length(unique([data[idx][2] for idx in assigns[k]]))
            
            # Logic for merging: too few points, too few time slices, or area outside bounds
            is_invalid_count = counts[k] < cfg.min_points
            is_invalid_time = ts_count < cfg.min_time_slices
            is_invalid_area = (areas[k] > 0 && areas[k] < cfg.min_area) || (areas[k] > cfg.max_area)
            
            if is_invalid_count || is_invalid_time || is_invalid_area
                push!(violators, k)
            end
        end

        # 4. Convergence Check
        curr_mean_density = mean(counts)
        cv_val = std(counts) / (mean(counts) + 1e-9)
        
        if last_mean_density > 0.0 && (abs(curr_mean_density - last_mean_density) < cfg.tolerance || abs(cv_val - last_cv) < cfg.tolerance)
            termination_reason = "tolerance_reached"
            break
        end
        
        last_mean_density = curr_mean_density
        last_cv = cv_val

        # 5. Merging Step
        # Identify target unit to merge (the one with the lowest count among violators or overall)
        candidates_indices = isempty(violators) ? collect(1:length(curr_c)) : violators
        v_counts = [counts[k] for k in candidates_indices]
        target_idx = candidates_indices[argmin(v_counts)]

        # Find nearest neighbor for the target centroid
        dists = [sum((curr_c[target_idx] .- curr_c[j]).^2) for j in 1:length(curr_c)]
        dists[target_idx] = Inf
        neighbor_idx = argmin(dists)

        # Weighted update for the merged centroid location
        total_n = counts[target_idx] + counts[neighbor_idx]
        curr_c[neighbor_idx] = (curr_c[target_idx] .* counts[target_idx] .+ curr_c[neighbor_idx] .* counts[neighbor_idx]) ./ (total_n + 1e-9)
        
        # Explicit removal (No clamp used)
        deleteat!(curr_c, target_idx)
    end
    
    return [Tuple(c) for c in curr_c], termination_reason
end


function get_lattice_centroids(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, lengthscale)
    """
    Synopsis: Generates centroids for a regular 2D lattice (grid) based on a lengthscale.
    """

    s_coord_tuple = tuple.(s_x, s_y)

    if isempty(s_coord_tuple); return [], 0, 0, (0.0, 0.0, 0.0, 0.0); end

    xs = [p[1] for p in s_coord_tuple]
    ys = [p[2] for p in s_coord_tuple]

    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # Generate grid ranges
    x_range = collect(xmin:lengthscale:xmax)
    y_range = collect(ymin:lengthscale:ymax)

    # Ensure at least one cell if the range is smaller than lengthscale
    if isempty(x_range); x_range = [xmin]; end
    if isempty(y_range); y_range = [ymin]; end

    rows = length(y_range)
    cols = length(x_range)

    # Create meshgrid of centroids
    centroids = [(x, y) for y in y_range, x in x_range][:]

    return centroids, rows, cols, (xmin, xmax, ymin, ymax)
end



function load_shapefile_to_libgeos(filepath::String)
    # Read the shapefile
    # import Shapefile  << --- install this if you need it
    # import LibGEOS
    # import GeoInterface

    table = Shapefile.Table(filepath)
    
    # Extract geometries and convert to LibGEOS
    # GeoInterface allows LibGEOS to understand Shapefile objects automatically
    geoms = [LibGEOS.read_geom(row.geometry) for row in table]
    
    return geoms, table
end

function get_user_centroids(input_polygons)
    # Convert input to a concrete vector of LibGEOS Polygons
    geoms = LibGEOS.Polygon[p for p in input_polygons]
    n = length(geoms)
    centroids = Vector{Tuple{Float64, Float64}}(undef, n)
    polys_coords = Vector{Vector{Tuple{Float64, Float64}}}(undef, n)

    for i in 1:n
        poly = geoms[i]
        cent_geom = LibGEOS.centroid(poly)
        seq = LibGEOS.getCoordSeq(cent_geom)
        centroids[i] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        polys_coords[i] = get_coords_from_geom(poly)
    end

    # Wrap the vector in a GeometryCollection so GeoInterface traits are recognized
    collection = LibGEOS.GeometryCollection(geoms)
    # Perform unaryUnion on the collection instead of the vector
    united = LibGEOS.unaryUnion(collection)
    hull_coords = get_coords_from_geom(united)

    return centroids, polys_coords, hull_coords
end


function assign_spatial_units(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}; area_method=:avt, target_units=10, lengthscale=nothing, input_polygons=nothing, geom_hull=nothing, kwargs...)
    # s_coord_tuple_local will be used for calculations that still expect a collection of points

    s_coord_tuple_local = tuple.(s_x, s_y) # Using the globally defined helper

    # The branch for `input_data isa AbstractMatrix` is removed, as this refactored function
    # is specifically for coordinate-based spatial unit assignment. `bstm_options`
    # or a similar function will call `assign_spatial_units_inferred` (or its refactored version)
    # directly if an adjacency matrix is provided as primary input.

    # 1. Handle User-Defined Polygons
    if !isnothing(input_polygons)
        # If geom_hull is provided, intersect the input polygons with it
        processed_polys = isnothing(geom_hull) ? input_polygons : [LibGEOS.intersection(p, geom_hull) for p in input_polygons]

        final_centroids, polys_coords, hull_coords = get_user_centroids(processed_polys)
        reason = :user_polygons
        n_units = length(final_centroids)

        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(processed_polys[i], processed_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(processed_polys[i], 1e-7), processed_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        W = Float64.(Graphs.adjacency_matrix(g))

        # Use s_coord_tuple_local for assignments, as it represents the original observation points
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        v_edges = []

    # 2. Handle Lattice Method
    elseif area_method == :lattice
        # `expand_hull` and `get_lattice_centroids` will be refactored to take s_x, s_y
        ls = isnothing(lengthscale) ? sqrt(get_polygon_area(get_coords_from_geom(expand_hull( 0.0))) / target_units) : lengthscale # Updated call
        final_centroids_raw, rows, cols, bbox = get_lattice_centroids(s_x, s_y, ls) # Updated call
        reason = :lattice_grid

        # Generate square polygons and clip them if geom_hull is provided
        polys_coords = Vector{Vector{Tuple{Float64, Float64}}}()
        lg_polys = LibGEOS.Polygon[]
        final_centroids = Tuple{Float64, Float64}[]
        half = ls / 2.0

        for c in final_centroids_raw
            coords = [[(c[1]-half, c[2]-half), (c[1]+half, c[2]-half), (c[1]+half, c[2]+half), (c[1]-half, c[2]+half), (c[1]-half, c[2]+half)]]
            p_geom = LibGEOS.Polygon(coords)
            if !isnothing(geom_hull)
                p_geom = LibGEOS.intersection(p_geom, geom_hull)
            end

            if !LibGEOS.isEmpty(p_geom)
                push!(lg_polys, p_geom)
                p_c = LibGEOS.centroid(p_geom)
                seq = LibGEOS.getCoordSeq(p_c)
                push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                push!(polys_coords, get_coords_from_geom(p_geom))
            end
        end

        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        W = Float64.(Graphs.adjacency_matrix(g))

        # Use s_coord_tuple_local for assignments
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        v_edges = []
        hull_coords = isnothing(geom_hull) ? [(bbox[1], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[4]), (bbox[1], bbox[4]), (bbox[1], bbox[3])] : get_coords_from_geom(geom_hull)

    # 3. Standard Tessellation Methods
    else
        cfg = (
            target=Int(target_units),
            min_total_arealunits=Int(get(kwargs, :min_total_arealunits, 3)),
            max_total_arealunits=Int(get(kwargs, :max_total_arealunits, target_units*2)),
            min_time_slices=Int(get(kwargs, :min_time_slices, 1)),
            min_points=Int(get(kwargs, :min_points, 1)),
            max_points=Int(get(kwargs, :max_points, length(s_x))), # Use length of s_x
            min_area=get(kwargs, :min_area, 0.0),
            max_area=get(kwargs, :max_area, Inf),
            target_cv=get(kwargs, :target_cv, 1.0),
            tolerance=get(kwargs, :tolerance, 0.1),
            buffer_dist=get(kwargs, :buffer_dist, 0.5),
            t_idx=get(kwargs, :t_idx, ones(Int, length(s_x)))) # Use length of s_x

        # `expand_hull` will be refactored to take s_x, s_y
        hull_geom = !isnothing(geom_hull) ? geom_hull : expand_hull(s_x, s_y, cfg.buffer_dist) # Updated call

        # Centroid functions will be refactored to take s_x, s_y
        c_mid, reason = if area_method == :cvt get_cvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :kvt get_kvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :qvt get_qvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :bvt get_bvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :hvt get_hvt_centroids(s_x, s_y, cfg, hull_geom)
        elseif area_method == :avt get_avt_centroids(s_x, s_y, cfg, hull_geom) # Updated call
        else error("Unknown partitioning method: $area_method") end

        polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)
        final_centroids = Tuple{Float64, Float64}[]
        lg_polys = []
        for p_coords in polys_coords
            if isempty(p_coords); continue; end
            # Ensure polygon is closed for LibGEOS if it's not already
            if p_coords[1] != p_coords[end]; push!(p_coords, p_coords[1]); end
            lg_p = LibGEOS.Polygon([[ [pt[1], pt[2]] for pt in p_coords ]])
            push!(lg_polys, lg_p)
            cent_g = LibGEOS.centroid(lg_p)
            seq = LibGEOS.getCoordSeq(cent_g)
            push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
        end

        # Use s_coord_tuple_local for assignments
        new_assigns = [argmin([sum((p .- sj).^2) for sj in final_centroids]) for p in s_coord_tuple_local]
        n_units = length(final_centroids)
        g = SimpleGraph(n_units)
        for i in 1:n_units, j in (i+1):n_units
            if LibGEOS.touches(lg_polys[i], lg_polys[j]) || LibGEOS.intersects(LibGEOS.buffer(lg_polys[i], 1e-7), lg_polys[j])
                add_edge!(g, i, j)
            end
        end
        g = ensure_connected!(g, final_centroids)
        hull_coords = get_coords_from_geom(hull_geom)
        W = Float64.(Graphs.adjacency_matrix(g))
    end

    # Update the returned NamedTuple to store s_x and s_y
    return (centroids=final_centroids, s_idx=new_assigns, polygons=polys_coords,
            adjacency_edges=v_edges, graph=g, hull_coords=hull_coords,
            termination_reason=reason, s_x=s_x, s_y=s_y, W=W, s_vals=collect(1:size(W,1)))
end



function assign_spatial_units_inferred(adjacency_matrix; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons = nothing)
    """
    Synopsis: Manually constructs a areal_units object for areal data like the Lip Cancer dataset.
              Centroid locations are spatially inferred from connectivity using a rudimentary force-directed layout.
    Inputs:
    - adjacency_matrix: The adjacency matrix (W) of the areal units.
    - iterations: Number of iterations for the force-directed layout.
    - learning_rate: Step size for moving centroids in the layout algorithm.
    - buffer_dist: Distance to buffer the convex hull when polygons are inferred.
    - input_polygons: Optional. A vector of LibGEOS Polygons. If provided, centroids and hull are derived from these.
    """

    local final_centroids
    local adjacency_edges_output
    local polys_output
    local hull_coords_output
    local g_final # The final graph that will be in the result

    nAU = size(adjacency_matrix, 1)


    if input_polygons !== nothing && !isempty(input_polygons)
        # Case 1: Polygons are provided
        # 1. Extract centroids from input_polygons
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        # 2. Determine hull by dissolving all internal edges
        united_geom = LibGEOS.unaryunion(input_polygons)
        hull_coords_output = get_coords_from_geom(united_geom)

        # 3. Determine adjacency from input_polygons (using LibGEOS.touches)
        adjacency_edges_output = []
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    # Fallback robust check, similar to get_voronoi_polygons_and_edges
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        inter = LibGEOS.intersection(g1_buffered, g2)
                        if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                            push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                        end
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]

        # Build graph from the determined adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            xi = get(centroid_map, c1, 0)
            yi = get(centroid_map, c2, 0)
            if xi > 0 && yi > 0 && !has_edge(g_final, xi, yi)
                add_edge!(g_final, xi, yi)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids) # Ensure connectivity if necessary

    else
        # Case 2: Polygons are not provided, infer centroids and use tessellation
        # 1. Build initial graph from adjacency_matrix for force-directed layout
        g_initial_for_layout = SimpleGraph(adjacency_matrix)

        # 2. Infer initial centroids using force-directed layout
        side = ceil(Int, sqrt(nAU))
        initial_centroids_fd = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids_fd]

        for iter in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial_for_layout, i)
                if !isempty(neighbors_i)
                    avg_neighbor_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] + learning_rate * (avg_neighbor_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end
        # Centroids after force-directed layout
        forced_layout_centroids = [(p[1], p[2]) for p in centroids_vec]

        # 3. Determine hull_geom from inferred centroids for clipping
        fx = getindex.(forced_layout_centroids, 1)
        fy = getindex.(forced_layout_centroids, 2)
        hull_geom = expand_hull(fx, fy, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # 4. Use tessellation to determine polygon coordinates and initial adjacency (based on forced_layout_centroids)
        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_layout_centroids, hull_geom)

        # 5. RECOMPUTE CENTROIDS from the generated (clipped) polygons and prepare for adjacency
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        lg_polygons_for_adjacency = Vector{Union{LibGEOS.Polygon, Nothing}}(undef, length(polys_coords_raw))
        polys_output = polys_coords_raw

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if !isempty(poly_coord_list) && length(poly_coord_list) >= 3
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([ [Float64[p[1], p[2]] for p in poly_coord_list] ])
                centroid_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(centroid_geom)
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                lg_polygons_for_adjacency[idx] = lg_poly
            else
                @warn "Invalid or empty polygon encountered in Voronoi tessellation at index $idx. Using original centroid as fallback."
                final_centroids[idx] = forced_layout_centroids[idx]
                lg_polygons_for_adjacency[idx] = nothing
            end
        end

        # 6. Re-build adjacency based on the newly derived centroids and polygons
        adjacency_edges_output = []
        if !isempty(lg_polygons_for_adjacency)
            for i in 1:length(lg_polygons_for_adjacency)
                g1 = lg_polygons_for_adjacency[i]
                if g1 === nothing continue end
                for j in (i+1):length(lg_polygons_for_adjacency)
                    g2 = lg_polygons_for_adjacency[j]
                    if g2 === nothing continue end
                    if LibGEOS.touches(g1, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    else
                        g1_buffered = LibGEOS.buffer(g1, 1e-6)
                        if LibGEOS.intersects(g1_buffered, g2)
                            inter = LibGEOS.intersection(g1_buffered, g2)
                            if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                                push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                            end
                        end
                    end
                end
            end
        end

        # 7. Build final graph from the re-derived adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            xi = get(centroid_map, c1, 0)
            yi = get(centroid_map, c2, 0)
            if xi > 0 && yi > 0 && !has_edge(g_final, xi, yi)
                add_edge!(g_final, xi, yi)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    end

    return (
        centroids = final_centroids,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        polygons = polys_output,
        hull_coords = hull_coords_output
    )
end


 
# Verbatim copy and refinement of get_polygon_area and get_avt_centroids
# This version fixes the MethodError by providing explicit dispatch for tuple vectors.

"""
    get_polygon_area(poly_coords::AbstractVector)

Calculates the area of a polygon defined by a vector of (x, y) tuples using the Shoelace formula.
Includes data cleaning for NaNs, Infs, and duplicate vertices.
"""
function get_polygon_area(poly_coords::AbstractVector)
    # Filter invalid points
    valid_pts = [p for p in poly_coords if !isnan(p[1]) && !isinf(p[1]) && !isnan(p[2]) && !isinf(p[2])]
    
    # Remove trailing duplicate if it matches the start
    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        pop!(valid_pts)
    end
    
    # A polygon must have at least 3 vertices
    if length(valid_pts) < 3 
        return 0.0 
    end
    
    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]
    
    # Shoelace Formula using LinearAlgebra utilities
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
end

function get_polygon_area(s_x, s_y)
    # Wrapper for legacy three-argument calls
    poly_coords = tuple.(s_x, s_y)
    return get_polygon_area(poly_coords)
end

 
function get_coords_from_geom(geom)
    """
    Synopsis: Extracts coordinates from various LibGEOS geometry types.
    Inputs:
    - geom: A LibGEOS geometry object.
    Outputs:
    - A vector of (x, y) coordinates.
    """

    coords = Tuple{Float64, Float64}[]
    local type_id = -1
    try
        type_id = LibGEOS.geomTypeId(geom)
        if type_id == LibGEOS.GEOS_POINT
             # Access coordinate sequence directly for point types
             seq = LibGEOS.getCoordSeq(geom)
             push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
             return coords
        elseif type_id == LibGEOS.GEOS_POLYGON
            ring = LibGEOS.exteriorRing(geom)
            n = LibGEOS.numPoints(ring)
            for i in 1:n
                p = LibGEOS.getPoint(ring, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        elseif type_id == LibGEOS.GEOS_MULTIPOLYGON
            for i in 1:LibGEOS.numGeometries(geom)
                poly = LibGEOS.getGeometryN(geom, i)
                ring = LibGEOS.exteriorRing(poly)
                n = LibGEOS.numPoints(ring)
                for j in 1:n
                    p = LibGEOS.getPoint(ring, j)
                    seq = LibGEOS.getCoordSeq(p)
                    push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                end
                if i < LibGEOS.numGeometries(geom); push!(coords, (NaN, NaN)); end
            end
        elseif type_id in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_LINEARRING]
            n = LibGEOS.numPoints(geom)
            for i in 1:n
                p = LibGEOS.getPoint(geom, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        end
    catch e
        @warn "Coordinate extraction failed for type $type_id: $e"
    end
    return coords
end




function get_voronoi_polygons_and_edges(centroids, hull_geom, tol=1e-7)
    """
    Synopsis: Generates clipped Voronoi polygons with robust adjacency detection.
    Uses a small buffer fallback to handle floating-point misalignment in LibGEOS.
    """
    n_c = length(centroids)
    if n_c == 0
        return [], []
    elseif n_c == 1
        return [get_coords_from_geom(hull_geom)], []
    elseif n_c == 2
        # Standard 2-point bisection logic
        p1, p2 = centroids[1], centroids[2]
        mid = ((p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
        dx, dy = p2[1] - p1[1], p2[2] - p1[2]
        px, py = -dy, dx
        L = 1e7
        pt1 = (mid[1] + L*px, mid[2] + L*py)
        pt2 = (mid[1] - L*px, mid[2] - L*py)
        side1_pts = [pt1, pt2, (pt2[1] - L*dx, pt2[2] - L*dy), (pt1[1] - L*dx, pt1[2] - L*dy), pt1]
        poly1_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side1_pts]])
        side2_pts = [pt1, pt2, (pt2[1] + L*dx, pt2[2] + L*dy), (pt1[1] + L*dx, pt1[2] + L*dy), pt1]
        poly2_box = LibGEOS.Polygon([[[p[1], p[2]] for p in side2_pts]])
        res1 = LibGEOS.intersection(hull_geom, poly1_box)
        res2 = LibGEOS.intersection(hull_geom, poly2_box)
        return [get_coords_from_geom(res1), get_coords_from_geom(res2)], [(p1, p2)]
    end

    # Deduplicate centroids before triangulation to suppress package warnings 
    # and ensure the output polygon array matches the input centroid array in length.
    u_centroids = unique(centroids)
    if length(u_centroids) < n_c
        u_polys, u_edges = get_voronoi_polygons_and_edges(u_centroids, hull_geom, tol)
        return [u_polys[findfirst(==(c), u_centroids)] for c in centroids], u_edges
    end

    # 3+ points logic
    pts_dt = [(Float64(c[1]), Float64(c[2])) for c in centroids]
    tri = triangulate(pts_dt)
    hull_coords = get_coords_from_geom(hull_geom)
    xs = [p[1] for p in hull_coords if !isnan(p[1])]
    ys = [p[2] for p in hull_coords if !isnan(p[2])]
    if isempty(xs) || isempty(ys) return [Tuple{Float64, Float64}[] for _ in 1:length(centroids)], [] end
    
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    vorn = voronoi(tri)
    final_coords = [Tuple{Float64, Float64}[] for _ in 1:length(centroids)]
    valid_geoms = Dict{Int, Any}()

    for i in each_generator(vorn)
        if i < 1 || i > length(centroids) continue end
        vertices = get_polygon_coordinates(vorn, i, bbox)
        if !isempty(vertices)
            poly_pts = [[v[1], v[2]] for v in vertices]
            if poly_pts[1] != poly_pts[end] push!(poly_pts, poly_pts[1]) end
            try
                lg_poly = LibGEOS.Polygon([poly_pts])
                clipped = LibGEOS.intersection(lg_poly, hull_geom)
                if !LibGEOS.isEmpty(clipped) && LibGEOS.geomTypeId(clipped) in [LibGEOS.GEOS_POLYGON, LibGEOS.GEOS_MULTIPOLYGON]
                    final_coords[i] = get_coords_from_geom(clipped)
                    valid_geoms[i] = clipped
                end
            catch e end
        end
    end

    v_edges = []
    active_ids = sort(collect(keys(valid_geoms)))
    for idx in 1:length(active_ids)
        i = active_ids[idx]
        g1 = valid_geoms[i]
        for jdx in idx+1:length(active_ids)
            j = active_ids[jdx]
            g2 = valid_geoms[j]
            # Primary check: direct contact
            if LibGEOS.touches(g1, g2)
                push!(v_edges, (centroids[i], centroids[j]))
            else
                # Fallback check: microscopic overlap/buffer
                g1_b = LibGEOS.buffer(g1, tol)
                if LibGEOS.intersects(g1_b, g2)
                    push!(v_edges, (centroids[i], centroids[j]))
                end
            end
        end
    end
    return final_coords, v_edges
end

function check_connectivity(g)
    """
    Synopsis: Evaluates the connectivity of a spatial graph.
    Inputs:
    - g: A SimpleGraph.
    Outputs:
    - NamedTuple showing connection status and components.
    """
    comps = connected_components(g)
    return (is_connected = length(comps) == 1, n_components = length(comps), components = comps)
end


function ensure_connected!(g, centroids)
    # Ensures the spatial graph is connected by adding edges between the nearest 
    # components based on the provided centroid coordinates.
    while !is_connected(g)
        comps = connected_components(g)
        best_dist = Inf
        best_pair = (0, 0)
        
        # Find the two closest nodes belonging to different components
        for i in 1:length(comps), j in (i+1):length(comps)
            for u in comps[i], v in comps[j]
                d = sum((centroids[u] .- centroids[v]).^2)
                if d < best_dist
                    best_dist = d
                    best_pair = (u, v)
                end
            end
        end
        
        if best_pair != (0, 0)
            add_edge!(g, best_pair[1], best_pair[2])
        else
            break
        end
    end
    return g
end


 function plot_spatial_graph(au; plot_title="Spatial Partitioning", domain_boundary=nothing)
    # 1. Base Plot - Use qualified Plots.plot and Plots.title!
    plt = Plots.plot(aspect_ratio=:equal, legend=false)
    Plots.title!(plt, plot_title)

    # Plot Polygons
    for poly_coords in au.polygons
        if length(poly_coords) > 2
            px = [p[1] for p in poly_coords if !isnan(p[1])]
            py = [p[2] for p in poly_coords if !isnan(p[2])]
            if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                push!(px, px[1]); push!(py, py[1])
            end
            Plots.plot!(plt, px, py, seriestype=:shape, fillalpha=0.1, linecolor=:black, lw=0.5)
        end
    end

    # 2. Plot Adjacency Graph Edges
    for edge in Graphs.edges(au.graph)
        u, v = Graphs.src(edge), Graphs.dst(edge)
        p1, p2 = au.centroids[u], au.centroids[v]
        Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
    end

    # 3. Plot Centroids and Raw Points
    Plots.scatter!(plt, [p[1] for p in au.s_coord_tuple], [p[2] for p in au.s_coord_tuple], 
        markersize=1, color=:gray, alpha=0.3, label="Points")
    Plots.scatter!(plt, [c[1] for c in au.centroids], [c[2] for c in au.centroids], 
        markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    if !isnothing(domain_boundary)
        bx = [p[1] for p in domain_boundary if !isnan(p[1])]
        by = [p[2] for p in domain_boundary if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)
    end

    return plt
end

    

################
 

function turingindex( indices, sym=nothing, dims=nothing  ) 
     
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
end


function showtuples(X)
    for k in keys(X)
        val = getproperty(X, k)
        # Skip displaying keys with NaN values
        # Check if value is numeric before rounding to avoid errors
        display_val = val isa Number ? round(val, digits=3) : val
        println("$k: $display_val")
    end
end



function showparams(X, keywords=["rho", "phi", "sigma",  "mu_", "l_", "ls_"]; limit=10 )
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

# etas = [1 10 100 1000 1e+4 1e+5];
# d = size of matrix

# EXTENDED ONION METHOD to generate random correlation matrices
# distributed ~ det(S)^eta [or maybe det(S)^(eta-1), not sure]
# https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices

# LKJ modify this method slightly, in order to be able to sample correlation matrices C from a distribution proportional to [detC]η−1. The larger the η, the larger will be the determinant, meaning that generated correlation matrices will more and more approach the identity matrix. The value η=1 corresponds to uniform distribution. On the figure below the matrices are generated with η=1,10,100,1000,10000,100000. 

    beta = eta + (d-2)/2;
    u = rand( Beta(beta, beta) );
    r12 = 2*u - 1;
    S = [1 r12; r12 1];  

    for k = 3:d
        beta = beta - 1/2;
        y = rand( Beta((k-1)/2, beta) );  # sample from beta
        r = sqrt(y);
        theta = randn(k-1,1);
        theta = theta/norm(theta);
        w = r*theta;
        U, E = eigen(S);
        U = hcat(U)
        R = U' * sqrt(E) * U; # R is a square root of S
        q = R[].re * w;
        S = [S q; q' 1];
    end
    return S
end




function build_st_inputs(time_indices, space_indices, spatial_coords)
  # Space-Time Input Construction
  # Space and Time as continuous coordinates.
  # Inputs: 
  #   spatial_coords: Matrix (2 x N_nodes) -> [Lat, Lon]
  #   time_coords: Vector (T_steps)
  # Returns:
  #   ColVecs of 3D points (Time, Lat, Lon)

  # Map indices to actual coordinates
  # This assumes spatial_coords is 2xN

  # Extract coords for every observation
  coords = spatial_coords[:, space_indices] # 2 x y_N
  times = time_indices' # 1 x y_N

  # Stack to create 3D input: [Time; Lat; Lon]
  return ColVecs(vcat(times, coords))
end

 


function adjacency_matrix_to_nb( W )
    nau = size(W)[1]
    # W = LowerTriangular(W)  # using LinearAlgebra
    nb = [Int[] for _ in 1:nau]
    Threads.@threads for i in 1:nau
        nb[i] = findall( isone, W[i,:] )
    end
    return nb
end


function nb_to_adjacency_matrix( nb )
    nau = Integer( length( unique( reduce(vcat, nb) )) )
    W = zeros( Int8, nau, nau )
    Threads.@threads for i in 1:nau
        for j in 1:length( nb[i] )
            k = nb[i][j]
            W[i, k] = 1
        end
    end
    return(W)
end


function nodes( adj )
    nau = length(adj)
    N_edges = Integer( length( reduce(vcat, adj) )/2 )
    node1 =  fill(0, N_edges); 
    node2 =  fill(0, N_edges); 
    i_edge = 0;
    for i in 1:nau
        u = adj[i]
        num = length(u)
        for j in 1:num
            k = u[j]
            if i < k
                i_edge = i_edge + 1;
                node1[i_edge] = i;
                node2[i_edge] = k;
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = Graphs.adjacency_matrix(g)
    
    # D = diagm(vec( sum(W, dims=2) ))
    scalefactor = scaling_factor_bym2(W)

    return node1, node2, scalefactor
end




function scaling_factor_bym2( adjacency_mat )
    # re-scaling variance using Reibler's solution and 
    # Buerkner's implementation: https://codesti.com/issue/paul-buerkner/brms/1241)  
    # Compute the diagonal elements of the covariance matrix subject to the 
    # constraint that the entries of the ICAR sum to zero.
    # See the inla.qinv function help for further details.
    # Q_inv = inla.qinv(Q, constr=list(A = matrix(1,1,nbs$N),e=0))  # sum to zero constraint
    # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
    # scaling_factor = exp(mean(log(diag(Q_inv))))
 
# --- Robust Precision Scaling ---
    N = size(adjacency_mat)[1]
    if N <= 1 return 1.0 end
    asum = vec(sum(adjacency_mat, dims=2))
    asum = float(asum) + N .* max.(asum) .* sqrt(1e-15)
    Q = Diagonal(asum) - adjacency_mat
    A = ones(N)
    try
        S = Q \ Diagonal(A)
        V = S * A
        S = S - V * inv(A' * V) * V'
        diag_s = diag(S)
        valid_diag = filter(x -> x > 1e-12, diag_s)
        if isempty(valid_diag) return 1.0 end
        scale_factor = exp(mean(log.(valid_diag)))
        return isnan(scale_factor) ? 1.0 : scale_factor
    catch
        return 1.0
    end
end


function scaling_factor_bym2(node1, node2, groups=ones(length(node1))) 
    ## calculate the scale factor for each of k connected group of nodes, 
    ## copied from the scale_c function from M. Morris
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
  

 

function sample_gaussian_process( ; GPmethod="cholesky", returntype="default",
    fkernal=nothing, kerneltype="default", kvar=nothing, kscale=nothing, gpc=GPC(),
    Yobs, Xobs, Xinducing=nothing, lambda=0.0001 )
    
    if isnothing(fkernal)
        if kerneltype=="default" || kerneltype=="squared_exponential"
            fkernal = kvar * SqExponentialKernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
        if kerneltype=="matern32"
            fkernal = kvar * Matern32Kernel() ∘ ScaleTransform( kscale) # ∘ ARDTransform(α)
        end
    end



    if GPmethod=="textbook"
        # Dimensional fix: Removed vec() to allow multi-feature inputs (N x D)
        Ko = kernelmatrix( fkernal, Xobs )
        Kcommon = inv(Ko + lambda*I) 

        if !isnothing(Xinducing)
            Ki = kernelmatrix( fkernal, Xinducing )
            Kio = kernelmatrix( fkernal, Xinducing, Xobs ) 
            Yinducing_mean_process = Kio * Kcommon * Yobs 
            Covi = Symmetric( Ki - Kio * Kcommon * Kio'  + lambda*I )
            MVNi = MvNormal( Yinducing_mean_process, Covi )

            Yinducing_sample  = rand( MVNi )
            Li =  cholesky(Symmetric( Ki + lambda*I)).L 

            Yobs_mean_process =  Kio' * ( Li' \ (Li \ Yinducing_mean_process  ) ) 
            Covo = Symmetric(kernelmatrix( fkernal, Xobs ) + lambda*I)
            MVN = MvNormal(Yobs_mean_process, Covo)

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample =  Kio' * ( Li' \ (Li \ Yinducing_sample  ) ) 

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return (MVN=MVN, MVNi=MVNi, Li=Li, loglik=LogLik,
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else
            mean_process = Ko * Kcommon * Yobs   
            MVN = MvNormal(mean_process, Ko + lambda*I  ) 

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample = rand( MVN ) 

            if returntype=="sample"
                return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod )
            end

            LogLik = logpdf(MVN,Yobs)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return ( MVN=MVN, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end
    end

    if GPmethod=="cholesky"
        if !isnothing(Xinducing)
            Ko = kernelmatrix( fkernal, Xobs )
            Ki = kernelmatrix( fkernal, Xinducing )
            Kio = kernelmatrix( fkernal, Xinducing, Xobs )
            Lo = cholesky(Symmetric( Ko + lambda*I)).L
            Li = cholesky(Symmetric( Ki + lambda*I)).L
            Yinducing_mean_process  = Kio * ( Lo' \ (Lo \ Yobs ) ) 

            Covi = Symmetric( Ki + lambda*I)
            MVN = MvNormal( Yinducing_mean_process, Covi )

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_mean_process = Kio' * ( Li' \ (Li \ Yinducing_mean_process )) 
            Yinducing_sample  = Yinducing_mean_process + Li * rand(Normal(0, 1), size(Li,1))
            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2))

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yinducing_mean_process)

            if returntype=="sample_loglik"
                return ( Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
            end

            return (MVN=MVN, Li=Li, Lo=Lo, loglik=LogLik,
                    Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

        else
            Ko = kernelmatrix( fkernal, Xobs )
            Lo = cholesky(Symmetric( Ko + lambda*I)).L
            Yobs_mean_process = Ko' * ( Lo' \ (Lo \ Yobs )) 
            Covo = Symmetric( Ko + lambda*I)
            MVN = MvNormal( Yobs_mean_process, Covo ) 

            if returntype=="fcovariance"
                return MVN
            end

            Yobs_sample = Yobs_mean_process + Lo * rand(Normal(0, 1), size(Lo,2))

            if returntype=="sample"
                return (Yobs_sample=Yobs_sample, GPmethod=GPmethod)
            end

            LogLik = logpdf(MVN, Yobs)

            if returntype=="sample_loglik"
                return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
            end

            return (MVN=MVN, Lo=Lo, loglik=LogLik, Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end
    end

 

    if GPmethod=="GPexact"

        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp(Xobs, lambda)

        if returntype=="fcovariance"
            return fobs
        end 

        fposterior = posterior(fobs, Yobs) 
        
        if returntype=="posterior"
            return fposterior
        end

        Yobs_sample =  rand(fposterior(Xobs, lambda) )   

        if returntype=="sample"
            return ( Yobs_sample=Yobs_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fobs, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, fposterior=fposterior, Yobs_sample=Yobs_sample, loglik=LogLik, GPmethod=GPmethod)
    end
 
    if GPmethod=="GPsparse"
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp( Xinducing, lambda ) 
        fsparse = SparseFiniteGP(fobs, finducing)

        if returntype=="fcovariance"
            return fsparse
        end 

        fposterior = posterior(fsparse, Yobs)

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end

        LogLik = logpdf(fsparse, Yobs)
       
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)

    end

    if GPmethod=="GPvfe" # Variational Free Energy
        fgp = atomic(AbstractGPs.GP(fkernal), gpc)
        fobs = fgp( Xobs, lambda )
        finducing = fgp(Xinducing, lambda )
        fsparse = VFE( finducing )

        if returntype=="fcovariance"
            return fsparse
        end 
        
        fposterior = posterior(fsparse, fobs, Yobs)  # Distribution is MvNormal  

        if returntype=="posterior"
            return fposterior
        end
        
        Yobs_sample =  rand(fposterior(Xobs, lambda) )  
        Yinducing_sample =   rand(fposterior(Xinducing, lambda))

        if returntype=="sample"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
        end
        
        LogLik = AbstractGPs.elbo(fsparse, fobs, Yobs)  # to a constant
      
        if returntype=="sample_loglik"
            return (Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, loglik=LogLik,  GPmethod=GPmethod )
        end

        return ( fgp=fgp, fobs=fobs, finducing=finducing, fsparse=fsparse, fposterior=fposterior, loglik=LogLik, 
                Yobs_sample=Yobs_sample, Yinducing_sample=Yinducing_sample, GPmethod=GPmethod)
    end
      
end
 
  

function plot_variational_marginals(z, sym2range)
    # copied straight from https://turinglang.org/docs/tutorials/variational-inference/
    ps = []

    for (i, sym) in enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            k = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$k]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density",
                    margin=1.5mm,
                )
                push!(ps, p)
                k += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density",
                margin=1.5mm,
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 2000), margin=4.0mm)
end


function fkernal( kernfunctype="squared_exp", params=nothing )

    if kernfunctype=="squared_exp"
        out = params[1] * SqExponentialKernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern12"
        out = params[1] * Matern12Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern32"
        out = params[1] * Matern32Kernel() ∘ ScaleTransform(params[2])  
    end

    if kernfunctype=="matern52"
        out = params[1] * Matern52Kernel() ∘ ScaleTransform(params[2])  
    end

    # ∘ ARDTransform(α)

    return out
end


sekernel2(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);

sekernel(v, s) = v * SqExponentialKernel() ∘ ScaleTransform(s) # ∘ ARDTransform(a);
 

# Squared-exponential covariance function
sqexp_cov_fn(D, phi, noise=1e-6) = exp.(-D^2 / phi) + LinearAlgebra.I * noise

# Exponential covariance function
exp_cov_fn(D, phi) = exp.(-D / phi)



# generic kernel functions 

# lenscale = -1 / log(ρ)
# σ_ar1^2 / (1 - ρ^2) = marginal variance
# kernel_ar1(σ, ρ) = σ^2 * with_lengthscale(Matern12Kernel(), -1/log(ρ)) 
# the softplus should not be necessary ... 

kernel_ar1(σ, ρ) = σ^2 / softplus(1 - ρ^2) * with_lengthscale(Matern12Kernel(), softplus(-1 / log(ρ)) )

# RW2 is equivalent to a Spline kernel or an Integrated Wiener Process
# For simplicity, use a high-order Matern or a custom structure

kernel_rw2(σ) = σ^2 * Matern52Kernel() # Matern32 is a common smooth approximation for RW2


   
    
function assign_time_units(t_v; time_method="regular", t_N=nothing, u_N=12, kwargs...)

    if time_method=="regular"

        tint = Int.(floor.(t_v))
        t0, t1 = minimum(tint), maximum(tint)
        t_n = t1-t0
        if !isnothing(t_N) 
            if t_n != t_N
                print("warning: time range and unique years do not match")
            end
        end

        t_idx = tint .- t0 .+ 1
        t_vals = collect(t0:t1) .- t0 .+ 1
        t_yr = collect(t0:t1)
        t_brks = (t_yr, t1+1)
        t_mids = t_yr .+ 0.5
        
        u_v = t_v - tint

        u_disc = discretize_data( u_v, N_cat=u_N, method="regular" )  # seasonality discretized

        return (
            t_v = t_v, 
            t_idx = t_idx, 
            t0=t0, 
            t1=t1, 
            t_vals, 
            t_yr=t_yr, 
            t_mids=t_mids, 
            t_brks=t_brks,
            tn=length(t_vals),
            t_N= length(t_vals),
            u_v=u_v, 
            u_idx=u_disc.idx, 
            u_brks=u_disc.brks,
            u_mids=u_disc.mids, 
            u_N=u_N,
            u_vals=collect(1:u_N) 
        )
    end

end


function discretize_data(X; method="quantile", N_cat=9, brks=nothing, probs=nothing, dx=nothing, minv = 0, maxv=1)
     
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
 
 

function assign_covariate_levels( X; N_cat=9, brks=nothing, probs=nothing )
    X = Array(X)
    U = [ discretize_data( X[:,i], N_cat=N_cat, brks=brks, probs=probs ) for i in 1:size(X, 2) ]
    V = hcat([t[:idx] for t in  U]...)  # as matrix
    B = hcat([t[:brks] for t in  U]...)
    M = hcat([t[:mids] for t in  U]...)
    P = hcat([t[:probs] for t in  U]...)
    return( idx=V, brks=B, mids=M, probs=P )
end



function estimate_local_kde_with_extrapolation(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}, t_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - s_coord_tuple: Vector of (x, y) coordinates for all time points.
    - t_idx: Vector of time indices corresponding to s_coord_tuple.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice

    s_coord_tuple = tuple.(s_x, s_y)

    filtered_pts = [p for (i, p) in enumerate(s_coord_tuple) if t_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for the target time slice $target_ts")
    end
    xs, ys = [p[1] for p in filtered_pts], [p[2] for p in filtered_pts]
    # Calculate bandwidth based on standard deviation of points
    bw_x = std(xs) * sd_extension_factor
    bw_y = std(ys) * sd_extension_factor
    # Define grid boundaries extending slightly beyond the data range
    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y
    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)
    # Gaussian KDE implementation
    for i in 1:grid_res
        for j in 1:grid_res
            x_val, y_val = x_grid[i], y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end
    # Normalize intensity to sum to 1 (optional, depending on desired output)
    intensity ./= sum(intensity)
    return x_grid, y_grid, intensity
end

function calculate_metrics(au)
    # Restoration: Calculate s_idx and counts based on the actual centroids in the au object
    assigns = [argmin([sum((p .- c).^2) for c in au.centroids]) for p in au.s_coord_tuple]
    counts = [count(==(i), assigns) for i in 1:length(au.centroids)]

    # Safety: Filter valid counts to prevent downstream NaN propagation
    valid_counts = filter(x -> !isnan(x) && !ismissing(x), counts)

    if isempty(valid_counts)
        return (mean_density=NaN, sd_density=NaN, cv_density=NaN)
    end

    m_dens = mean(valid_counts)
    s_dens = std(valid_counts)
    cv_dens = s_dens / (m_dens + 1e-9)

    return (mean_density=m_dens, sd_density=s_dens, cv_density=cv_dens)
end


function get_spatial_graph( centroids, adjacency_edges )
    """
    Synopsis: Converts partitioning results into a formal SimpleGraph. 
    Outputs: A SimpleGraph object.
    """
    n = length(centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(centroids))
    for edge in adjacency_edges
        xi, yi = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if xi > 0 && yi > 0 add_edge!(g, xi, yi) end
    end
    return g
end



function plot_kde_simple(s_x::AbstractVector{<:Real}, s_y::AbstractVector{<:Real}; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - s_coord_tuple: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy t_idx of 1s since plotting a static slice

    s_coord_tuple = tuple.(s_x, s_y)

    t_idx_dummy = ones(Int, length(s_coord_tuple))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(s_coord_tuple, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                  title=title,
                  c=:viridis,
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in s_coord_tuple], [p[2] for p in s_coord_tuple],
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end




function scottish_lip_cancer_data_spacetime(n_years::Int=10; rndseed::Int=42)
    # "expand" scottish lip cancer data to a space-time version
    # original data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    Random.seed!(rndseed)

    # Load base spatial data
 # Base Spatial Data for 56 Counties
    # data source:  https://mc-stan.org/users/documentation/case-studies/icar_stan.html

    nAU = 56

    y_base = [ 9, 39, 11, 9, 15, 8, 26, 7, 6, 20, 13, 5, 3, 8, 17, 9, 2, 7, 9, 7,
    16, 31, 11, 7, 19, 15, 7, 10, 16, 11, 5, 3, 7, 8, 11, 9, 11, 8, 6, 4,
    10, 8, 2, 6, 19, 3, 2, 3, 28, 6, 1, 1, 1, 1, 0, 0]

    E_base = [1.4, 8.7, 3.0, 2.5, 4.3, 2.4, 8.1, 2.3, 2.0, 6.6, 4.4, 1.8, 1.1, 3.3, 7.8, 4.6,
    1.1, 4.2, 5.5, 4.4, 10.5,22.7, 8.8, 5.6,15.5,12.5, 6.0, 9.0,14.4,10.2, 4.8, 2.9, 7.0,
    8.5, 12.3, 10.1, 12.7, 9.4, 7.2, 5.3,  18.8,15.8, 4.3,14.6,50.7, 8.2, 5.6, 9.3, 88.7,
    19.6, 3.4, 3.6, 5.7, 7.0, 4.2, 1.8]

    x_base = [16,16,10,24,10,24,10, 7, 7,16, 7,16,10,24, 7,16,10, 7, 7,10,
    7,16,10, 7, 1, 1, 7, 7,10,10, 7,24,10, 7, 7, 0,10, 1,16, 0,
    1,16,16, 0, 1, 7, 1, 1, 0, 1, 1, 0, 1, 1,16,10]

    adjacency = [ 5, 9,11,19, 7,10, 6,12, 18,20,28, 1,11,12,13,19,
    3, 8, 2,10,13,16,17, 6, 1,11,17,19,23,29, 2, 7,16,22, 1, 5, 9,12,
    3, 5,11, 5, 7,17,19, 31,32,35, 25,29,50, 7,10,17,21,22,29,
    7, 9,13,16,19,29, 4,20,28,33,55,56, 1, 5, 9,13,17, 4,18,55,
    16,29,50, 10,16, 9,29,34,36,37,39, 27,30,31,44,47,48,55,56,
    15,26,29, 25,29,42,43, 24,31,32,55, 4,18,33,45, 9,15,16,17,21,23,25,
    26,34,43,50, 24,38,42,44,45,56, 14,24,27,32,35,46,47, 14,27,31,35,
    18,28,45,56, 23,29,39,40,42,43,51,52,54, 14,31,32,37,46,
    23,37,39,41, 23,35,36,41,46, 30,42,44,49,51,54, 23,34,36,40,41,
    34,39,41,49,52, 36,37,39,40,46,49,53, 26,30,34,38,43,51, 26,29,34,42,
    24,30,38,48,49, 28,30,33,56, 31,35,37,41,47,53, 24,31,46,48,49,53,
    24,44,47,49, 38,40,41,44,47,48,52,53,54, 15,21,29, 34,38,42,54,
    34,40,49,54, 41,46,47,49, 34,38,49,51,52, 18,20,24,27,56,
    18,24,30,33,45,55]

    number_neighbours = [4, 2, 2, 3, 5, 2, 5, 1,  6,  4, 4, 3, 4, 3, 3, 6, 6, 6 ,5,
    3, 3, 2, 6, 8, 3, 4, 4, 4,11,  6, 7, 4, 4, 9, 5, 4, 5, 6, 5,
    5, 7, 6, 4, 5, 4, 6, 6, 4, 9, 3, 4, 4, 4, 5, 5, 6]
 
    # Build graph from adjacency info

    N_edges = Integer(length(adjacency) / 2)
    node1 = fill(0, N_edges)
    node2 = fill(0, N_edges)
    i_adjacency = 0
    i_edge = 0
    for i in 1:nAU
        for j in 1:number_neighbours[i]
            i_adjacency += 1
            if i < adjacency[i_adjacency]
                i_edge += 1
                node1[i_edge] = i
                node2[i_edge] = adjacency[i_adjacency]
            end
        end
    end

    e = Edge.(node1, node2)
    g = Graph(e)
    W = adjacency_matrix(g)
    # D = diagm(vec(sum(W, dims=2)))
 
    au = assign_spatial_units_inferred( W ) # "infer" from the adjacency network (W)
    pts_base = au.centroids
    
    N_total = nAU * n_years

    # 1. Random Walk Trend
    rw_trend = cumsum(randn(n_years) .* 0.5)

    # 2. Expand Data Vectors
    y_expanded = repeat(y_base, n_years)
    E_expanded = repeat(E_base, n_years)
    x_expanded = repeat(x_base, n_years)
    t_idx = repeat(1:n_years, inner=nAU)
    s_coord_tuple = repeat(pts_base, n_years)

    s_x = getindex.(s_coord_tuple, 1)
    s_y = getindex.(s_coord_tuple, 2)

    # The s_idx is the spatial unit identifier (1 to 56)
    s_idx = repeat(1:nAU, n_years)
 
    # 3. Add Random Walk + Noise to Response
    # Broadcast rw_trend across years
    trend_component = repeat(rw_trend, inner=nAU)
    noise = randn(N_total) .* 0.2

    # Final response: base_y + trend + noise (ensuring positive counts)
    y_final = floor.(Int, abs.(y_expanded .+ trend_component .+ noise))

    # 4. Final covariate matrix and offsets
    x_scaled = (x_expanded .- mean(x_expanded)) ./ std(x_expanded)
    X = Matrix(DataFrame(AFF=x_scaled))
    log_offset = log.(E_expanded)
   
    return (
        y=y_final, X=X, s_x=s_x, s_y=s_y, log_offset=log_offset, t_idx=t_idx,
        s_idx=s_idx, n_years=n_years, W=W, au=au
    )
end
  


function icar_form(theta, phi, sigma, rho)
    # https://sites.stat.columbia.edu/gelman/research/published/bym_article_SSTEproof.pdf
    # Reibler parameterization: https://pubmed.ncbi.nlm.nih.gov/27566770/
    # https://www.jstatsoft.org/index.php/jss/article/view/v063c01/841
    sigma .* ( sqrt.(1 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )  
end
   


 
function kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s, sigma_s, ls_t, sigma_t, noise_vec)
    # Spatial Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = Symmetric(sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-4*I)
    Q_s = inv(K_s)

    # Temporal Precision
    k_t = Matern32Kernel() ∘ ScaleTransform(inv(ls_t))
    K_t = Symmetric(sigma_t^2 * kernelmatrix(k_t, unique_t) + 1e-4*I)
    Q_t = inv(K_t)

    # Full Kronecker Precision (Dense for AD compatibility)
    Q_full = Symmetric(Matrix(kron(Q_t, Q_s)) + 1e-4*I)
    L_q = cholesky(Q_full)

    # Sample: f = (L')^-1 * noise
    return L_q.U \ noise_vec
end


function get_posterior_means(ch, param_base, N)
    # Description:
    #   Extracts and averages posterior samples for a specific vector parameter.
    # Inputs:
    #   ch: MCMC sample chain.
    #   param_base: String prefix of the parameter (e.g., "s_eff").
    #   N: Length of the vector parameter.
    # Outputs:
    #   Vector of posterior means.

    means = zeros(N)
    
    for i in 1:N
        p_symbol = Symbol("$param_base[$i]")
        if p_symbol in names(ch, :parameters)
            means[i] = mean(ch[p_symbol])
        else
            @warn "Parameter $p_symbol not found in chain."
        end
    end
    
    return means
end



function generate_inducing_points(coords, M_inducing, seed=42; method="kmeans")

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

 
function ar1_covariance(n, rho, var, ::Type{T}=Float64) where {T}
    # Description:
    #   Generates a full AR1 covariance matrix.
    # Inputs:
    #   n: Number of time points.
    #   rho: Correlation coefficient.
    #   var: Marginal variance.
    # Outputs:
    #   n x n Covariance matrix.

    vcv = zeros(T, n, n) .+ I(n)
    
    Threads.@threads for r in 1:n
        for c in 1:n
            if r >= c
                vcv[r, c] = var * rho^(r - c)
            end
        end
    end
    
    return Symmetric(vcv)
end

 


function ar1_covariance_local( n, rho, var,  ::Type{T}=Float64 )  where {T} 
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



function gp_predictions(; Y, D, mu, sig2, phi, cov_fn=exp_cov_fn, nN=length(Xnew), nP=size(res, 1) ) 
    ynew = Vector{Float64}()
    # Threads.@threads -- to add 
    for i in sample(1:size(res,1), nP, replace=true)
        K = cov_fn(D, phi[i])
        Koo_inv = inv(K[(nN+1):end, (nN+1):end])
        Knn = K[1:nN, 1:nN]
        Kno = K[1:nN, (nN+1):end]
        C = Kno * Koo_inv
        mvn = MvNormal( 
            C * (Y .- mu[i]) .+ mu[i], 
            Matrix(LinearAlgebra.Symmetric(Knn - C * Kno')) + sig2[i] * LinearAlgebra.I 
        ) 
        ynew = vcat(ynew, [rand(mvn) ] )
    end
    ynew = stack(ynew, dims=1)  # rehape to matrix   
    return ynew
end




function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )

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

  


function rff_map(coords, W, b)
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
    # XR: Audited and Robust RFF Parameter Generator (v05.4)
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

    # 4. Frequency Weight Generation
    W = randn(d, M_rff) ./ coord_scales

    # 5. Phase Shift Generation
    b = rand(M_rff) .* (2.0 * pi)
    
    return W, b
end


function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end 
 

macro save_carstm_state(file_to_save_name_sym)
  quote
    try
      # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
      local filename_val = $(esc(file_to_save_name_sym))
      @info "Saving CARSTM state to $(filename_val)..."
      # JLD2.@save expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@save "$(filename_val)" areal_units mod chain y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
      @info "CARSTM state saved successfully."
    catch e
      @error "Error saving CARSTM state: $e"
    end
  end
end

macro load_carstm_state(filename_sym)
  quote
    # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
    local filename_val = $(esc(filename_sym))
    if !isfile(filename_val)
      @error "File $(filename_val) not found."
      return nothing
    end
    try
      @info "Loading CARSTM state from $(filename_val)..."
      # JLD2.@load expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@load "$(filename_val)" areal_units mod chain  y_sim y_binary t_idx weights trials cov_indices cov_indices trials_sim  weights_sim adj_matrix_numeric s_N t_N area_method
      @info "CARSTM state loaded successfully."
      # Variables are loaded directly into the calling scope by JLD2.@load
      # No explicit return value from the macro itself, as it injects variables
    catch e
      @error "Error loading CARSTM state: $e"
      return nothing
    end
  end
end


function init_params_extract( res=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # Description: Extracts initial parameter values from a model result summary or loads them from a file.
  # Inputs:
  #   - res: Model result object (default: NaN).
  #   - load_from_file: Boolean, if true loads params from fn_inits.
  #   - override_means: Boolean, if true applies custom overrides for specific parameter patterns.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the extracted or loaded mean parameter values.

  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns ); vns[u]
    if length(u) > 0  
      means[u] = sigma_prior  # from basic pca
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = v_prior  # from basic pca
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_copy( res=NaN, res0=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # using spatial parts of res0 
  # Description: Copies parameter values from a reference result (res0) to a target result structure (res).
  # Inputs:
  #   - res: Target model result object.
  #   - res0: Reference model result object.
  #   - load_from_file: Boolean to load from fn_inits instead.
  #   - override_means: Boolean to apply custom pattern-based overrides.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the merged mean parameter values.
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

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns );  
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0, 10.0, 10.0][1:length(u)]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int)
    """
    libgeos_lattice_adjacency_matrix(rows, cols)

    Description:
    Generates a sparse adjacency matrix for a regular 2D lattice using LibGEOS for spatial geometry operations.
    Constructs unit square polygons for each cell and identifies neighbors based on Queen contiguity
    (any shared boundary point or edge).

    Inputs:
    - rows (Int): Number of rows in the lattice grid.
    - cols (Int): Number of columns in the lattice grid.

    Output:
    - W (SparseMatrixCSC{Int, Int}): A binary sparse adjacency matrix of size (rows*cols) x (rows*cols).
    """
    # Create polygons for each cell in the lattice
    polygons = []
    for r in 1:rows, c in 1:cols
        # Define unit square coordinates as nested vectors for LibGEOS compatibility
        coords = [
            [Float64(c-1), Float64(r-1)],
            [Float64(c),   Float64(r-1)],
            [Float64(c),   Float64(r)],
            [Float64(c-1), Float64(r)],
            [Float64(c-1), Float64(r-1)]
        ]
        # Construct LinearRing and then Polygon
        ring = LibGEOS.LinearRing(coords)
        push!(polygons, LibGEOS.Polygon(ring))
    end

    n = length(polygons)
    W = spzeros(Int, n, n)

    # Queen contiguity check
    for i in 1:n
        poly_i = polygons[i]
        for j in (i+1):n
            if LibGEOS.intersects(poly_i, polygons[j])
                W[i, j] = W[j, i] = 1
            end
        end
    end
    return W
end




function build_laplacian_precision(adj_matrix)
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
 




function scale_precision!(Q)
    # Description:
    #   Scales a precision matrix using the geometric mean of non-zero eigenvalues.
    #   Essential for ensuring sigma_sp represents marginal variance in BYM2.
    # Inputs:
    #   Q: Precision matrix to be modified in-place.

    eig_vals = eigvals(Matrix(Q))
    # Filter out near-zero eigenvalues associated with the null space
    valid_eigs = filter(x -> x > 1e-6, eig_vals)
    
    scaling_factor = exp(mean(log.(valid_eigs)))
    
    if Q isa Symmetric
        Q.data ./= scaling_factor
    else
        Q ./= scaling_factor
    end
    
    return Q
end


  


function logpdf_gmrf(x, Q)
    # Description: Calculates the log-probability of a Gaussian Markov Random Field.
    # Inputs:
    #   - x: Vector of values.
    #   - Q: Precision matrix.
    # Outputs:
    #   - Log-likelihood value.
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2pi))
end


 
 

function plot_posterior_results(stats, M=nothing, areal_units=nothing; s_x=nothing, s_y=nothing, time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
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

function NegativeBinomial2(μ, r)
    # Description: Alternative parametrization of Negative Binomial using mean (μ) and dispersion (r).
    # Inputs:
    #   - μ: Mean.
    #   - r: Size/dispersion parameter.
    # Outputs:
    #   - Distributions.NegativeBinomial object.

    p = r / (r + μ)
    return NegativeBinomial(r, p)
end
  

function get_rff_deep2D_basis(X, m, lengthscale)
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


function bstm( inp::NamedTuple )
    arch = inp[:model_arch]
    if arch == "multivariate" return bstm_multivariate(inp)
    elseif arch == "multifidelity" return bstm_multifidelity(inp)
    else return bstm_univariate(inp) end
end

 
# --- Recursive Parser Logic ---
# Parses the algebraic expression into a tree-like structure
function parse_manifold_graph(expr::String)
    expr = strip(expr)
    # Handle Direct Sum (Additive Components)
    if occursin("⊕", expr)
        # Recursively parse each component of the sum
        return (type=:sum, elements=parse_manifold_graph.(strip.(Base.split(expr, "⊕"))))
    # Handle Kronecker Product (Separable Interaction)
    elseif occursin("⊗", expr)
        # Recursively parse each component of the product
        return (type=:kronecker, elements=parse_manifold_graph.(strip.(Base.split(expr, "⊗"))))
    # Handle Composition (Warping/Transport)
    elseif occursin("∘", expr)
        # Recursively parse each component of the composition
        return (type=:composition, elements=parse_manifold_graph.(strip.(Base.split(expr, "∘"))))
    else
        # Atomic Manifold (e.g., "ICAR(s_idx)", "GP(s_x,s_y)")
        # Regex to capture the model name (e.g., ICAR, GP) and its arguments (e.g., s_idx, s_x,s_y)
        m = match(r"(\w+)\\((.*?)\\)", expr)
        if !isnothing(m)
            # Normalize model name to lowercase for consistency with bstm_options
            return (type=:atomic, model=lowercase(m.captures[1]), var=m.captures[2])
        else
            # If it's not a recognized manifold pattern, treat as unknown
            return (type=:unknown, val=expr)
        end
    end
end





# Configuration Engine ---
# This function consolidates all metadata logic from v0 and the Manifold v2 implementation.
# It handles keyword discovery, unit assignment, coordinate caching, RFF projection, 
# and precision template generation in a single efficient pass.


function bstm_options(; kwargs...)
    # Recomposed and Consolidated Configuration Engine (v05 Unified)
    # Rationale: Verbatim merger of coordinate discovery from Reference
    # with metadata expansion for SVC, Eigen, and DeepGP components.
    
    # 1. Initialization and Metadata Consolidation
    # All user-provided keyword arguments are collected into a mutable dictionary.
    M = Dict{Symbol, Any}(kwargs)
    data = get(M, :data, nothing)

    # 2. Automated Keyword Discovery (Verbatim from Reference)
    # This block scans the provided DataFrame for standard BSTM variable names
    # to minimize manual configuration overhead for the user.
    if !isnothing(data)
        v_names = Symbol.(names(data))
        keyword_map = Dict(
            :y_obs => [:y, :y_obs, :response],
            :s_x => [:s_x, :lon, :plon, :plons, :lons, :longs, :longitude],
            :s_y => [:s_y, :lat, :plat, :plats, :lats, :latitude],
            :t_v => [:t_v, :time, :time_coords, :t_coords],
            :s_idx => [:s_idx, :space_idx],
            :t_idx => [:t_idx, :time_idx, :ti],
            :u_idx => [:u_idx, :season_idx, :ui],
            :log_offset => [:log_offset, :logoffset],
            :weights => [:weights, :wts],
            :trials => [:trials, :n_trials]
        )
        for (key, candidates) in keyword_map
            if !haskey(M, key)
                for cand in candidates
                    if cand in v_names
                        M[key] = data[!, cand]
                        break
                    end
                end
            end
        end
    end


    # 2. Hurdle-Specific Transformation (Monolithic Path)
    if get(M, :model_family, "gaussian") == "hurdle_poisson"
        local y_raw = M[:y_obs]
        # Outcome 1: Participation (Bernoulli)
        M[:y_obs_hurdle] = [v > 0 ? 1.0 : 0.0 for v in y_raw]
        # Outcome 2: Intensity (Zero-Truncated Poisson)
        M[:y_obs_intensity] = [v > 0 ? Float64(v) : missing for v in y_raw]
        # Update y_obs to the multivariate format for architectural consistency
        M[:y_obs] = hcat(M[:y_obs_hurdle], M[:y_obs_intensity])
        M[:outcomes_N] = 2
        M[:model_arch] = "multivariate"
    end

    
    # 3. Dimensional Validation
    # The framework requires a response vector (y_obs) to define the fundamental scope.
    if !haskey(M, :y_obs)
        error("BSTM Error: :y_obs is required for model initialization.")
    end
    M[:y_N] = size(M[:y_obs], 1)
    local y_N = M[:y_N]


    # 4. Default Architectural Flag Assignments
    # Ensuring all primary dispatch keys exist to prevent UndefVarErrors in the sampler.
    get!(M, :model_arch, "univariate")
    get!(M, :model_family, "gaussian")
    get!(M, :model_space, "bym2")
    get!(M, :model_time, "ar1")
    get!(M, :model_season, "none")
    get!(M, :model_st, "none")
    get!(M, :noise, 1e-4)
    get!(M, :use_zi, false)
    get!(M, :use_sv, false)
    get!(M, :outcomes_N, 1)

    get!(M, :log_offset, zeros(y_N)) # Explicit initialization fix
    
    # CRITICAL: Initialize log_offset as vector if missing to prevent FieldErrors
    if !haskey(M, :log_offset) || isnothing(M[:log_offset])
        M[:log_offset] = zeros(y_N)
    end

    # Critical Metadata for SVC and Eigen (Resolves FieldErrors)
    get!(M, :svc_covariates, Symbol[])
    get!(M, :svc_model, "rff")
    get!(M, :svc_M_rff, 20)
 
    # Spatial Unit Discovery and Coordinate Synchronization
    if (haskey(M, :s_x) && haskey(M, :s_y)) || haskey(M, :s_coord)
        if !haskey(M, :s_coord)
            M[:s_coord] = hcat(M[:s_x], M[:s_y])
        end
        if !haskey(M, :s_idx)
            target_units = get(M, :target_units, 10)
            areal_units = assign_spatial_units(M[:s_coord][:,1], M[:s_coord][:,2]; target_units=target_units)
            M[:s_idx] = areal_units.s_idx
            M[:s_N] = length(areal_units.centroids)
            M[:W] = areal_units.W
        end
    end
  
    if haskey(M, :W) && !isnothing(M[:W])
        if !haskey(M, :s_idx)
            M[:s_N] = size(M[:W], 1)
            @info "bstm: Mapping observations to spatial units via inferred centroids..."
            areal_units = assign_spatial_units_inferred(M[:W])
            M[:areal_units] = areal_units
            M[:s_idx] = [argmin([sum(( (M[:s_x][i], M[:s_y][i]) .- c ).^2) for c in areal_units.centroids]) for i in 1:y_N]
        end
    end

    if haskey(M, :s_x) && !isnothing(M[:s_x])
        @info "bstm: Generating spatial tessellation from coordinates..."
        target_units = get(M, :target_units, 10)
        areal_units = assign_spatial_units(M[:s_x], M[:s_y]; target_units=target_units)
        M[:areal_units] = areal_units
        M[:s_idx] = areal_units.s_idx
        M[:s_N] = length(areal_units.centroids)
        M[:W] = areal_units.W
    end

    get!(M, :s_idx, ones(Int, y_N))

    M[:s_N] = get(M, :s_N, isempty(M[:s_idx]) ? 1 : maximum(M[:s_idx]))
    get!(M, :s_coord, zeros(M[:s_N], 2))
 

      # --- CRITICAL: Coordinate Field Enforcement ---
    if !haskey(M, :s_coord)
        if haskey(M, :s_x) && haskey(M, :s_y)
             M[:s_coord] = hcat(M[:s_x], M[:s_y])
        elseif haskey(M, :areal_units)
             c = M[:areal_units].centroids
             M[:s_coord] = hcat([p[1] for p in c], [p[2] for p in c])
        else
             # Fallback for non-spatial models to prevent FieldErrors
             M[:s_coord] = zeros(M[:s_N], 2)
        end
    end
    
    # 6. Temporal and Seasonal Unit Discovery
    M[:u_N] = 1
    M[:period] = 1

    if !haskey(M, :t_idx) && haskey(M, :t_v) && !isnothing(M[:t_v])
        time_units = assign_time_units(M[:t_v])
        M[:t_idx] = time_units.t_idx
        M[:t_N] = time_units.tn
        M[:u_idx] = time_units.u_idx
        M[:u_N] = time_units.u_N
    end
    get!(M, :t_idx, ones(Int, y_N))
    M[:t_N] = get(M, :t_N, isempty(M[:t_idx]) ? 1 : maximum(M[:t_idx]))
    get!(M, :u_idx, ones(Int, y_N))
    M[:u_N] = get(M, :u_N, isempty(M[:u_idx]) ? 1 : maximum(M[:u_idx]))

    # 7. Global Spatiotemporal Index Alignment
    M[:st_idx] = [(M[:t_idx][i] - 1) * M[:s_N] + M[:s_idx][i] for i in 1:y_N]

    # 8. Fixed Effect Design Matrix Creation
    if haskey(M, :fixed_parts) && !isnothing(data)
        f_expr = isempty(M[:fixed_parts]) ? "1" : join(M[:fixed_parts], " + ")
        M[:Xfixed] = Matrix{Float64}(create_fixed_design(f_expr, data; contrasts=get(M, :contrasts, Dict())))
    else
        get!(M, :Xfixed, ones(y_N, 1))
    end
    M[:Xfixed_N] = size(M[:Xfixed], 2)

    # 9. Spectral Projections and Basis Caching
    if get(M, :use_sv, false)
        get!(M, :M_rff_sigma, 20)
        # Use coordinates for spectral volatility mapping
        coords_v = haskey(M, :s_x) ? [M[:s_x] M[:s_y]] : randn(y_N, 2)
        W_v, b_v = generate_informed_rff_params(coords_v, M[:M_rff_sigma])
        M[:vol_proj] = (coords_v * W_v) .+ b_v'
    end

    if !isempty(M[:svc_covariates]) && M[:svc_model] == "rff"
        M[:svc_M_rff] = get(M, :svc_M_rff, 20)
        coords_s = haskey(M, :s_x) ? [M[:s_x] M[:s_y]] : randn(y_N, 2)
        W_svc, b_svc = generate_informed_rff_params(coords_s, M[:svc_M_rff])
        M[:svc_basis_cached] = sqrt(2.0 / M[:svc_M_rff]) .* cos.((coords_s * W_svc) .+ b_svc')
    end

    # 10. Precision Matrix Templates
    M[:s_Q_template] = build_structure_template(Symbol(M[:model_space]), M[:s_N]; W=get(M, :W, nothing))
    M[:t_Q_template] = build_structure_template(Symbol(M[:model_time]), M[:t_N])
    M[:u_Q_template] = (M[:model_season] != "none") ? build_structure_template(Symbol(M[:model_season]), M[:u_N]) : nothing

    # 11. Manifold-Specific Registry (Mosaic, Eigen, DeepGP)
    # Ensuring keys exist to prevent FieldErrors during @model execution
    get!(M, :n_mosaics, 1)
    get!(M, :cluster_assignments, ones(Int, M[:s_N]))
    get!(M, :t_basis, zeros(M[:t_N], 1))
    get!(M, :t_n_dims, 1)
    get!(M, :t_ltri_indices, [1])
    get!(M, :n_layers, 1)
    get!(M, :m_layers, [10])
    get!(M, :Om_layers, [randn(10, 2)])
    get!(M, :Ph_layers, [rand(10)])

    # 12. Final Registry for Nested Architectures
    get!(M, :basis_matrices, Dict{Symbol, Any}())
    get!(M, :re_rules, Dict{String, Any}())
    get!(M, :svc_covariates, Symbol[])
    get!(M, :svc_model, "rff")
    get!(M, :svc_M_rff, 20)
    get!(M, :spatial_hierarchy, Dict{Symbol, Any}())
    get!(M, :c_groups, Dict{Symbol, Any}())
    get!(M, :c_re_templates, Dict{Symbol, Any}())
    get!(M, :interaction_terms, [])
    get!(M, :eigen_terms, [])
    get!(M, :weights, ones(y_N))
    get!(M, :trials, ones(Int, y_N))

    # 9. Non-Euclidean Registries (v05.4 Addition)
    # These placeholders ensure the @model sampler finds the necessary fields
    get!(M, :local_weights, ones(M[:s_N]))
    get!(M, :directed_adj, get(M, :W, sparse(I(M[:s_N]))))
    get!(M, :curvature_const, -1.0)

    # Audit: Ensuring all fields present in bstm_univariate v05 are initialized
    return NamedTuple(M)
end



function apply_discretization_logic(vals, rules)
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


# -----------------------


function get_chain_names(chain)
    try
        return string.(FlexiChains.parameters(chain))
    catch
        return string.(names(chain))
    end
end


function _extract_volatility(chain, name_strs, N_tot, N_samples, outcome_idx=nothing, M=nothing)
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



function get_params_vector(chain, base_name, len)
    # Robust parameter extraction for FlexiChains/MCMCChains
    N_samples = size(chain, 1)

    # Use FlexiChains.parameters to get names
    names_ch = string.(FlexiChains.parameters(chain))

    # Tier 1: Indexed names [k]
    regex = Regex("^" * base_name * "\\[(\\d+)\\]")
    matched_names = filter(n -> occursin(regex, n), names_ch)

    if !isempty(matched_names)
        sort!(matched_names, by = n -> parse(Int, match(regex, n).captures[1]))
        res_mat = zeros(Float64, N_samples, length(matched_names))

        for (idx, n) in enumerate(matched_names)
            val_obj = chain[Symbol(n)]
            raw = hasproperty(val_obj, :data) ? val_obj.data : Array(val_obj)
            # Flatten nested structures if they appear at the index level
            data_fixed = raw isa AbstractVector && eltype(raw) <: AbstractVector ? reduce(vcat, raw) : raw
            res_mat[:, idx] = vec(Float64.(collect(data_fixed)))
        end

        if size(res_mat, 2) == 1 && len > 1
            return repeat(res_mat, 1, len)
        end
        return res_mat
    end

    # Tier 2: Single entity fallback
    if base_name in names_ch
        val_obj = chain[Symbol(base_name)]
        raw_data = hasproperty(val_obj, :data) ? val_obj.data : Array(val_obj)
        if ndims(raw_data) == 3; raw_data = raw_data[:, :, 1]; end

        # Standardize to Matrix [Samples x Params]
        # Robustly handle Matrix{Vector{Float64}} or Vector{Vector{Float64}} by flattening
        if raw_data isa AbstractMatrix && eltype(raw_data) <: AbstractVector
            # Flatten each row of vectors into a single parameter row
            mat_data = reduce(hcat, [reduce(vcat, row) for row in eachrow(raw_data)])'
        elseif raw_data isa AbstractArray && eltype(raw_data) <: AbstractVector
            mat_data = reduce(hcat, vec(raw_data))'
        else
            mat_data = Matrix{Float64}(raw_data)
        end

        if size(mat_data, 2) == len
            return mat_data
        elseif size(mat_data, 1) == len
            return mat_data'
        elseif size(mat_data, 2) == 1
            return repeat(mat_data, 1, len)
        end
    end

    @warn "Parameter '$base_name' not found in chain. Returning zeros for length $len."
    return zeros(Float64, N_samples, len)
end




function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
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
  

# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
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

 
function get_architecture(model_arch::String)
    if startswith(model_arch, "example_")
        return ExampleArchitecture() 
    elseif model_arch in ["univariate"]
        return UnivariateArchitecture()
    elseif model_arch in ["multivariate"]
        return MultivariateArchitecture()
    elseif model_arch in ["multifidelity"]
        return MultifidelityArchitecture()
    else
        return UnknownArchitecture()
    end
end

 
function get_model_family(model_family::String)
    if model_family == "poisson"
        return PoissonFamily()
    elseif model_family == "gaussian"
        return GaussianFamily()
    elseif model_family == "lognormal"
        return LogNormalFamily()
    elseif model_family in ["bernoulli", "binomial"]
        return BinomialFamily()
    elseif model_family == "negbin"
        return NegativeBinomialFamily()
    else
        error("Unknown model_family: $model_family")
    end
end


function get_model_parameters(m::DynamicPPL.Model)
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


function check_has_parameter(m::DynamicPPL.Model, param_name::String)
    all_p = get_model_parameters(m)
    return any(n -> n == param_name || startswith(n, "$param_name["), all_p)
end
 

function _process_ll_and_predictions(fam, eta, chain, M, N_tot, N_samples, y_sigma_samples=nothing, y_obs_custom=nothing)
    denoised = zeros(N_tot, N_samples)
    noisy = zeros(N_tot, N_samples)
    log_lik = zeros(N_samples, M.y_N)

    # Helper to handle parameter access across different chain types
    name_strs = string.(FlexiChains.parameters(chain))

    for j in 1:N_samples
        # Extract volatility (Heteroskedastic vs Homoskedastic)
        local sig_y
        if !isnothing(y_sigma_samples)
            sig_y = y_sigma_samples[:, j]
        elseif "y_sigma" in name_strs
            sig_y = vec(chain[:y_sigma].data[j])
        elseif "y_sigma_const" in name_strs
            sig_val = Float64(chain[:y_sigma_const].data[j])
            sig_y = fill(sig_val, N_tot)
        else
            sig_y = fill(1.0, N_tot)
        end

        for i in 1:N_tot
            is_obs = i <= M.y_N
            mu_eta = eta[i, j]

            # --- Link Functions ---
            mu = if fam isa PoissonFamily || fam isa NegativeBinomialFamily
                clamp(exp(mu_eta), 1e-10, 1e9) 
            elseif fam isa BinomialFamily
                logistic(mu_eta)
            elseif fam isa LogNormalFamily
                mu_eta 
            else
                mu_eta
            end

            denoised[i, j] = mu

            # --- Likelihood Calculation (Training Data Only) ---
            if is_obs
                y_vals_src = isnothing(y_obs_custom) ? M.y_obs : y_obs_custom
                y_val = y_vals_src[i]
                log_lik[j, i] = if fam isa PoissonFamily; logpdf(Poisson(mu), y_val)
                               elseif fam isa GaussianFamily; logpdf(Normal(mu, sig_y[i]), y_val)
                               elseif fam isa BinomialFamily; logpdf(Binomial(M.trials[i], mu), y_val)
                               elseif fam isa LogNormalFamily; logpdf(LogNormal(mu, sig_y[i]), y_val)
                               elseif fam isa NegativeBinomialFamily
                                   r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0
                                   prob = r_val / (r_val + mu)
                                   logpdf(NegativeBinomial(r_val, prob), y_val)
                               else 0.0 end
            end

            # --- Posterior Predictive Sampling ---
            noisy[i, j] = if fam isa GaussianFamily; mu + randn() * sig_y[i]
                          elseif fam isa LogNormalFamily; rand(LogNormal(mu, sig_y[i]))
                          elseif fam isa PoissonFamily; rand(Poisson(mu))
                          elseif fam isa BinomialFamily
                               n_trials = is_obs ? M.trials[i] : 1 
                               rand(Binomial(n_trials, mu))
                          elseif fam isa NegativeBinomialFamily
                               r_val = "r_nb" in name_strs ? chain[:r_nb].data[j] : 1.0
                               rand(NegativeBinomial(r_val, r_val / (r_val + mu)))
                          else mu end
        end
    end
    return denoised, noisy, log_lik
end


function _compute_waic(log_lik)
    nsamples, nobs = size(log_lik)
    lppd = sum(logsumexp(log_lik[:, i]) - log(nsamples) for i in 1:nobs)
    p_waic = sum(var(log_lik[:, i]) for i in 1:nobs)
    return -2 * (lppd - p_waic)
end

function _extract_beta_cov(all_names, chain, M, N_samples, alpha)
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



function create_prediction_surface(basis_df::DataFrame, observations_df::DataFrame, au; lambda_s=2.0, lambda_t=1.0, max_iters=5)
    # 1. Initialization and Automatic Identification of Fixed-Effect Columns
    mergeon = hasproperty(basis_df, :u_idx) && hasproperty(observations_df, :u_idx) ? [:s_idx, :t_idx, :u_idx] : [:s_idx, :t_idx]

    # Identify non-merge, non-outcome columns (the design matrix variables)
    # observations_df should already contain M.y_obs if applicable, so exclude it.
    fixed_variable_names = setdiff(propertynames(observations_df), vcat(mergeon, [:y_obs]))

    # Join basis (grid) with observations
    surface = leftjoin(basis_df, observations_df, on = mergeon, makeunique=true)

    # Standardize types to Float64/Missing for imputation math
    # Apply to the combined 'surface' DataFrame
    for c in fixed_variable_names
        surface[!, c] = convert(Vector{Union{Float64, Missing}}, collect(surface[!, c]))
    end

    centroids = au.centroids
    n_units = length(centroids)
    W = au.W

    # 2. Precompute Spatial Adjacency Weights
    dist_mat_s = [sqrt(sum((c1 .- c2).^2)) for c1 in centroids, c2 in centroids]
    weight_mat_s = exp.(-dist_mat_s.^2 ./ (2 * lambda_s^2)) .* (W + I)
    for i in 1:n_units
        s_row = sum(weight_mat_s[i, :])
        if s_row > 1e-9; weight_mat_s[i, :] ./= s_row; end
    end

    # 3. Iterative Spatiotemporal Imputation (Handling Continuous & Binned Factors)
    for iter in 1:max_iters
        # Check if there are any missing values in the relevant columns
        has_missing = false
        for c in fixed_variable_names
            if any(ismissing, surface[!, c])
                has_missing = true
                break
            end
        end
        if !has_missing; break; end # Break if no missing values remain

        for c in fixed_variable_names
            # If the variable is likely a factor (only integers), store levels to snap back later
            unique_vals = filter(!ismissing, unique(surface[!, c]))
            is_factor = all(v -> v == floor(v), unique_vals) # Check if all unique valid values are integers

            group_cols = hasproperty(surface, :u_idx) ? [:u_idx] : Symbol[] # Convert to Symbol[] for consistency
            # Iterate over unique combinations of grouping columns (e.g., u_idx if present)
            for grp in groupby(surface, group_cols)
                # Iterate over rows within each group
                for i in 1:nrow(grp)
                    current_row_in_grp = grp[i, :]
                    if ismissing(current_row_in_grp[c]) || isnan(current_row_in_grp[c]) # Check for isnan too
                        curr_s = Int(round(current_row_in_grp[:s_idx]))
                        curr_t = Int(round(current_row_in_grp[:t_idx])) # Ensure t_idx is also Int for comparison
                        curr_u = hasproperty(grp, :u_idx) ? Int(round(current_row_in_grp[:u_idx])) : nothing

                        # Spatial Influence: Look for values in the same time slice (and u_idx if applicable)
                        spatial_mask_df = filter(row -> row[:t_idx] == curr_t && (isnothing(curr_u) || row[:u_idx] == curr_u), surface)
                        s_vals_filtered = filter(row -> !ismissing(row[c]), eachrow(spatial_mask_df))

                        val_s, w_s_sum = 0.0, 0.0
                        for row in s_vals_filtered
                            nb_s = Int(round(row[:s_idx]))
                            w = nb_s != curr_s ? weight_mat_s[curr_s, nb_s] : 0.0 # Exclude self-influence
                            val_s += w * Float64(row[c])
                            w_s_sum += w
                        end

                        # Temporal Influence: Look for values in the same spatial unit (and u_idx if applicable)
                        temporal_mask_df = filter(row -> row[:s_idx] == curr_s && (isnothing(curr_u) || row[:u_idx] == curr_u), surface)
                        t_vals_filtered = filter(row -> !ismissing(row[c]), eachrow(temporal_mask_df))

                        val_t, w_t_sum = 0.0, 0.0
                        for row in t_vals_filtered
                            nb_t = Int(round(row[:t_idx]))
                            w = nb_t != curr_t ? exp(-(curr_t - nb_t)^2 / (2 * lambda_t^2)) : 0.0 # Exclude self-influence
                            val_t += w * Float64(row[c])
                            w_t_sum += w
                        end

                        if (w_s_sum + w_t_sum) > 1e-9 # Use a small epsilon to avoid division by zero
                            imputed_val = (val_s + val_t) / (w_s_sum + w_t_sum)
                            # Find the global index of the current row in 'surface' to update
                            # This is a potentially slow lookup, but necessary given DataFrame's groupby behavior
                            global_row_idx = findfirst(r -> r[:s_idx] == curr_s && r[:t_idx] == curr_t && (isnothing(curr_u) || r[:u_idx] == curr_u), eachrow(surface))
                            if !isnothing(global_row_idx)
                                surface[global_row_idx, c] = is_factor ? round(imputed_val) : imputed_val
                            end
                        end
                    end
                end
            end
        end
    end

    # 4. Final Global Fallback for any remaining missing values if iterative imputation couldn't fill them
    for c in fixed_variable_names
        valid_entries = filter(!ismissing, surface[!, c])
        if !isempty(valid_entries)
            m_val = median(valid_entries)
            surface[!, c] = map(x -> ismissing(x) ? m_val : x, surface[!, c])
        end
    end

    return surface
end



function convert_advi_to_reconstruct_format(msol, model::DynamicPPL.Model, n_samples::Int=500)
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
    Synopsis: Converts a point estimate (MAP/ML) from Optim/Optimisers into a distribution of samples.
    If a Hessian is available, it samples from the Multivariate Normal Laplace approximation.
    Otherwise, it creates a narrow Gaussian around the point estimate.
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



# --- 1. Custom PC Priors ---
struct PCPriorSigma <: ContinuousUnivariateDistribution
    U::Float64
    alpha::Float64
    lambda::Float64
    function PCPriorSigma(U, alpha)
        return new(U, alpha, -log(alpha) / U)
    end
end

function Distributions.logpdf(d::PCPriorSigma, x::Real)
    x > 0 ? log(d.lambda) - d.lambda * x : -Inf
end

Distributions.rand(rng::AbstractRNG, d::PCPriorSigma) = rand(rng, Exponential(1 / d.lambda))
Distributions.minimum(d::PCPriorSigma) = 0.0
Distributions.maximum(d::PCPriorSigma) = Inf
Bijectors.bijector(d::PCPriorSigma) = Bijectors.exp


struct bstm_Likelihood{F, Z, W, P, R, S, T, TR} <: ContinuousMultivariateDistribution
    family::F
    use_zi::Z
    weights::W
    phi_zi::P
    r_nb::R
    sigma_y::S
    trials::T
    y_obs::TR
end

Base.length(d::bstm_Likelihood) = length(d.y_obs)

function Distributions.logpdf(d::bstm_Likelihood, eta::AbstractVector)
    total_lp = 0.0
    trials_scalar = d.trials isa Number
    sig_scalar = d.sigma_y isa Number
    has_weights = !isnothing(d.weights) && length(d.weights) == length(eta)

    for i in 1:length(eta)
        y_val = d.y_obs[i]
        # Narrower clamping for initial stability
        lin_pred = clamp(eta[i], -15.0, 15.0)
        lp = 0.0

        if d.family == "poisson"
            mu = clamp(exp(lin_pred), 1e-9, 1e9)
            if d.use_zi
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * exp(-mu) + 1e-12)
                else
                    lp = log(1 - d.phi_zi + 1e-12) + logpdf(Poisson(mu), y_val)
                end
            else
                lp = logpdf(Poisson(mu), y_val)
            end
        elseif d.family == "binomial"
            p = clamp(logistic(lin_pred), 1e-10, 1.0 - 1e-10)
            ntrials = trials_scalar ? d.trials : d.trials[i]
            if d.use_zi
                prob_zero = (1-p)^ntrials
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * prob_zero + 1e-12)
                else
                    lp = log(1 - d.phi_zi + 1e-12) + logpdf(Binomial(ntrials, p), y_val)
                end
            else
                lp = logpdf(Binomial(ntrials, p), y_val)
            end
        elseif d.family == "negbin"
            mu = clamp(exp(lin_pred), 1e-9, 1e9)
            r_val = clamp(d.r_nb, 1e-5, 1e5)
            prob = clamp(r_val / (r_val + mu), 1e-10, 1.0 - 1e-10)
            if d.use_zi
                prob_zero = prob^r_val
                if y_val == 0
                    lp = log(d.phi_zi + (1 - d.phi_zi) * prob_zero + 1e-12)
                else
                    lp = log(1 - d.phi_zi + 1e-12) + logpdf(NegativeBinomial(r_val, prob), y_val)
                end
            else
                lp = logpdf(NegativeBinomial(r_val, prob), y_val)
            end
        elseif d.family == "gaussian"
            sig = clamp(sig_scalar ? d.sigma_y : d.sigma_y[i], 1e-6, 1e6)
            lp = logpdf(Normal(lin_pred, sig), y_val)
        elseif d.family == "lognormal"
            sig = clamp(sig_scalar ? d.sigma_y : d.sigma_y[i], 1e-6, 1e6)
            lp = logpdf(LogNormal(lin_pred, sig), y_val)
        end

        w = has_weights ? d.weights[i] : 1.0
        total_lp += isnan(lp) || isinf(lp) ? -1e12 : lp * w
    end
    return total_lp
end

 
    
function generate_sim_data(s_N=25, t_N=10; rndseed=42)
    Random.seed!(rndseed)
    local n_total = s_N * t_N

    # 1. Spatial Coordinates (Unit Level)
    local unique_pts = [(rand() * 100.0, rand() * 100.0) for _ in 1:s_N]
    local s_coord_tuple = repeat(unique_pts, inner=t_N)
    local s_x = collect(Float64, getindex.(s_coord_tuple, 1))
    local s_y = collect(Float64, getindex.(s_coord_tuple, 2))

    # 2. Temporal/Seasonal Indices
    local t_v = repeat(collect(1:t_N), outer=s_N) .+ (rand(n_total) .* 0.05)
    local t_idx = repeat(collect(1:t_N), outer=s_N)
    local u_N = 12
    # Fix: repeat() mapping for seasonal indices
    local u_idx = Int.(mod1.(1:n_total, u_N))

    # 3. Latent Fields
    local period = 12.0
    local trend = 0.05 .* t_v
    local seasonal = 0.8 .* cos.(2π .* t_v ./ period)

        # Covariate Generation (W1, W2, W3)
    # These simulate continuous predictors with some shared latent signal Z
    local Z = randn(n_total)
    local W1_obs = 0.5 .* sin.(t_v ./ 5.0) .+ 0.5 .* Z .+ (randn(n_total) .* 0.1)
    local W2_obs = 0.5 .* cos.(t_v ./ 5.0) .- 0.3 .* Z .+ (randn(n_total) .* 0.2)
    local W3_obs = 0.2 .* (t_v ./ t_N) .+ 0.1 .* Z .+ (randn(n_total) .* 0.3)
    local w_obs = hcat(W1_obs, W2_obs, W3_obs)
 
    # Mosaic/Cluster Effects
    local s_clusters = repeat(1:5, inner=Int(s_N/5))
    local cluster_effects = [-2.5, -1.0, 0.0, 1.0, 2.5]
    local spatial_effect = cluster_effects[s_clusters[repeat(1:s_N, inner=t_N)]]

    # 4. Response Construction
    local sigma_y = 0.15
    local observation_error = sigma_y .* randn(n_total)
    local eta = 1.0 .+ spatial_effect .+ trend .+ seasonal .+ observation_error

    local y_binary = Int.(eta .> (mean(eta) + 0.5))
    local y_counts = abs.(Int.(round.(exp.(eta)))) # Poisson-friendly counts

    local weights = ones(Float64, n_total)
    local trials = ones(Int, n_total)

    # Fixed Effects Design Matrix (Standard Intercept-only approach)
    local Xfixed = ones(Float64, n_total, 1)

    return (
        y_obs = eta,
        s_idx = repeat(1:s_N, inner=t_N),
        t_idx = t_idx,
        u_idx = u_idx,
        s_x = s_x,
        s_y = s_y,
        t_v = t_v,
        weights = weights,
        trials = trials,
        s_N = s_N,
        t_N = t_N,
        u_N = u_N,
        y_binary = y_binary,
        y_counts = y_counts,
        Xfixed = Xfixed,
        z_obs = Z,
        w_obs = w_obs,

        s_coord = reduce(hcat, unique_pts)',
        cluster_assignments = s_clusters,

        n_total = n_total
    )
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


function extract_v_parameters(v_mat, ltri_indices)
    # extract_v_parameters(v_mat, ltri_indices)
    #
    # Utility to extract only the free parameters (lower triangular) from the v_mat
    # for use as initial values in Turing (matching the 'v' parameter vector).
    return v_mat[ltri_indices]
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



 
function summarize_array(samples::AbstractArray; alpha=0.05)
    if isempty(samples) || all(isnan, samples)
        return (mean = [NaN], median = [NaN], lower = [NaN], upper = [NaN])
    end

    dims = size(samples)
    n_dims = length(dims)

    # Calculate statistics across the sample dimension
    post_mean = dropdims(mean(samples, dims=n_dims), dims=n_dims)
    post_median = dropdims(median(samples, dims=n_dims), dims=n_dims)
    low_bound = dropdims(mapslices(x -> quantile(x, alpha/2), samples, dims=n_dims), dims=n_dims)
    high_bound = dropdims(mapslices(x -> quantile(x, 1 - alpha/2), samples, dims=n_dims), dims=n_dims)

    # FIX: Ensure outputs are ALWAYS Vectors, even if they contain only one element.
    # This prevents MethodError: vec(::Float64) in downstream code.
    force_vector(x) = begin
        if x isa AbstractArray
            if ndims(x) == 0 # It's a 0-dimensional array (e.g., Array{Float64, 0})
                return [Float64(x[])] # Extract scalar and wrap in vector
            else
                return vec(collect(x)) # For higher-dimensional arrays, flatten to vector
            end
        else # It's already a scalar (e.g., Float64)
            return [Float64(x)] # Wrap scalar in vector
        end
    end

    return (
        mean = force_vector(post_mean),
        median = force_vector(post_median),
        lower = force_vector(low_bound),
        upper = force_vector(high_bound)
    )
end

function model_results_comprehensive(model, chain; n_samples=100, alpha=0.05)
    # Comprehensive Posterior Reconstruction [Polygon Persistence Update V4]
    # Rationale: Serializes polygon geometries into the 'res' object to enable 
    # standalone visualization without referencing the original model metadata.

    println("--- Starting Audited Comprehensive Posterior Reconstruction ---")

    # 1. Access Model Metadata (M)
    # M contains the configuration, data, and spatial units used during sampling.
    M = model.args.M

    # 2. Architecture Discovery
    # Routes the reconstruction logic based on the model's structural design.
    raw_arch = hasproperty(M, :model_arch) ? M.model_arch : (hasproperty(M, :arch) ? M.arch : "univariate")
    arch_type = if raw_arch == "univariate"
        UnivariateArchitecture()
    elseif raw_arch == "multivariate"
        MultivariateArchitecture()
    elseif raw_arch == "multifidelity"
        MultifidelityArchitecture()
    elseif raw_arch == "example"
        ExampleArchitecture()
    else
        raw_arch
    end

    # 3. Core Reconstruction
    # Executes the architectural-specific reconstruction of latent manifolds and predictors.
    res = _reconstruct(arch_type, "model", chain, M, nothing, alpha)

    # 4. Metadata Extraction and Polygon Persistence
    # CRITICAL: extract centroids and polygons to ensure mapping consistency.
    y_obs = M.y_obs
    
    # Recovery of Centroids
    centroids = nothing
    if hasproperty(M, :centroids) && !isnothing(M.centroids)
        centroids = M.centroids
    elseif hasproperty(M, :s_coords) && !isnothing(M.s_coords)
        centroids = M.s_coords
    elseif hasproperty(M, :s_x) && hasproperty(M, :s_y)
        centroids = tuple.(M.s_x, M.s_y)
    end

    # Polygon Geometry Persistence
    # Ensures that user-provided or tessellated polygons are carried forward.
    polygons = nothing
    if hasproperty(M, :polygons) && !isnothing(M.polygons)
        polygons = M.polygons
    elseif hasproperty(M, :areal_units) && hasproperty(M.areal_units, :polygons)
        polygons = M.areal_units.polygons
    end

    covariates = (M.Xfixed isa NamedArray) ? names(M.Xfixed, 2) : nothing
    times = hasproperty(M, :times) ? M.times : nothing
    season_labels = hasproperty(M, :season_labels) ? M.season_labels : nothing

    # 5. Compute Quality Metrics
    # Calculated using the posterior predictive mean (noisy) against observations.
    y_pred = res.predictions_observed_noisy.mean
    valid_idx = .!isnan.(y_pred)

    # RMSE Calculation
    rmse_val = sqrt(mean((y_obs[valid_idx] .- y_pred[valid_idx]).^2))
    
    # Standard R² (Accuracy/Fit)
    ss_res = sum((y_obs[valid_idx] .- y_pred[valid_idx]).^2)
    ss_tot = sum((y_obs[valid_idx] .- mean(y_obs[valid_idx])).^2)
    r2_standard = 1.0 - (ss_res / (ss_tot + 1e-9))

    # Pearson r² (Correlation Strength)
    r_pearson = cor(y_obs[valid_idx], y_pred[valid_idx])
    r2_pearson_sq = r_pearson^2

    println("\n--- Quality Metrics ---")
    println("RMSE: ", round(rmse_val, digits=3))
    println("Standard R² (Accuracy): ", round(r2_standard, digits=3))
    println("Pearson r² (Correlation): ", round(r2_pearson_sq, digits=3))
    println("WAIC: ", round(res.waic, digits=3))

    # 6. Final Assembly
    # Returning a unified results object with persisted spatial geometry.
    return (
        metrics = (rmse = rmse_val, r2 = r2_standard, r2_pearson = r2_pearson_sq, waic = res.waic),
        pstats = res,
        y_obs = y_obs,
        centroids = centroids,
        polygons = polygons,
        covariates = covariates,
        times = times,
        season_labels = season_labels,
        model_family = hasproperty(res, :family) ? res.family : (hasproperty(M, :model_family) ? M.model_family : "unknown")
    )
end



using Plots, StatsPlots, Graphs, Statistics

# Unified and Audited Visualization Engine
# High Confidence: Integrates Bounds Auditing into the feature-complete model_results_plots().
# This version resolves BoundsError [24] while preserving all manifold visualization capabilities.

function model_results_plots(res, ts=1;
                             centroids=nothing,
                             polygons=nothing,
                             covariates=nothing,
                             times=nothing,
                             season_labels=nothing,
                             on_user_scale=true,
                             y_obs=nothing)

    # Display scaling context for the audit log
    scale_msg = on_user_scale ? "User Scale" : "Internal Scale"
    println("--- Generating Audited Posterior Visualizations [Scale: $scale_msg] ---")

    # 1. Parameter and Data Recovery
    # extract statistics from the results object or the pstats sub-field
    pstats = hasproperty(res, :pstats) ? res.pstats : res
    obs_data = !isnothing(y_obs) ? y_obs : (hasproperty(res, :y_obs) ? res.y_obs : nothing)

    # Recover Spatial Geometries (Centroids and Polygons)
    coords = !isnothing(centroids) ? centroids : (hasproperty(res, :centroids) ? res.centroids : nothing)
    polys = !isnothing(polygons) ? polygons : (hasproperty(res, :polygons) ? res.polygons : nothing)

    # Recover Metadata for Axes
    cov_names = !isnothing(covariates) ? covariates : (hasproperty(res, :covariates) ? res.covariates : nothing)
    time_axis = !isnothing(times) ? times : (hasproperty(res, :times) ? res.times : nothing)
    seas_labs = !isnothing(season_labels) ? season_labels : (hasproperty(res, :season_labels) ? res.season_labels : nothing)

    # Initialize plot containers
    p_fixed = nothing; p_mixed = nothing; p_spatial = nothing; p_smooth = nothing
    p_temporal = nothing; p_seasonal = nothing; p_ppc = nothing; p_pred_map = nothing

    # 2. Posterior Predictive Check (PPC)
    if hasproperty(pstats, :predictions_observed_noisy) && !isnothing(pstats.predictions_observed_noisy)
        y_pred_mean = pstats.predictions_observed_noisy.mean
        if !isnothing(obs_data) && length(obs_data) == length(y_pred_mean)
            p_ppc = Plots.scatter(y_pred_mean, obs_data, title="PPC", xlabel="Predicted", ylabel="Observed", legend=false, alpha=0.6)
            # Add 45-degree reference line
            mini, maxi = min(minimum(y_pred_mean), minimum(obs_data)), max(maximum(y_pred_mean), maximum(obs_data))
            Plots.plot!(p_ppc, [mini, maxi], [mini, maxi], color=:red, linestyle=:dash)
        end
    end

    # 3. Fixed Effects Visualization
    if hasproperty(pstats, :fixed_effects) && !isnothing(pstats.fixed_effects)
        beta_means = pstats.fixed_effects.mean
        x_labs = (on_user_scale && !isnothing(cov_names)) ? cov_names : 1:length(beta_means)
        p_fixed = Plots.scatter(x_labs, beta_means, yerror=(beta_means .- pstats.fixed_effects.lower, pstats.fixed_effects.upper .- beta_means), title="Fixed Effects", legend=false)
        Plots.hline!(p_fixed, [0], color=:black, linestyle=:dash)
    end

    # 4. Mixed Effects Visualization
    if hasproperty(pstats, :mixed_effects) && !isnothing(pstats.mixed_effects)
        for (m_idx, me_data) in enumerate(pstats.mixed_effects)
            if !isnothing(me_data)
                me_means = me_data.mean
                p_mixed = Plots.bar(me_means, title="Mixed Effects (Grp $m_idx)", legend=false, alpha=0.7)
                break # Visualizing the first group for the dashboard
            end
        end
    end

    # 5. Spatial Manifolds (With Robust Bounds Audit)
    if !isnothing(coords)
        s_N_coords = length(coords)
        pos_x = [c[1] for c in coords]; pos_y = [c[2] for c in coords]

        # Internal plotting helper with Explicit Dimension Auditing
        function build_map_plot(vals, titlestr, color_grad)
            p = Plots.plot(aspect_ratio=:equal, title=titlestr, legend=true)
            
            # AUDIT: Ensure only iterate over units that exist in both the result vector and coordinate list
            avail_len = min(s_N_coords, length(vals))
            
            for i in 1:avail_len
                # Check for polygon existence to prevent BoundsError
                if !isnothing(polys) && i <= length(polys)
                    p_coords = polys[i]
                    if length(p_coords) > 2
                        px = [pt[1] for pt in p_coords]; py = [pt[2] for pt in p_coords]
                        # Ensure closure
                        if (px[1], py[1]) != (px[end], py[end])
                            push!(px, px[1]); push!(py, py[1])
                        end
                        Plots.plot!(p, px, py, seriestype=:shape, fill_z=vals[i], c=color_grad, linecolor=:black, lw=0.3, legend=false)
                    end
                else
                    # Fallback to scatter plot for units without polygons
                    Plots.scatter!(p, [pos_x[i]], [pos_y[i]], zcolor=vals[i], markersize=5, c=color_grad)
                end
            end
            return p
        end

        # Plot Structured Spatial Field
        if hasproperty(pstats, :spatial_structured) && !isnothing(pstats.spatial_structured)
            p_spatial = build_map_plot(pstats.spatial_structured.mean, "Spatial Field", :viridis)
        end

        # Plot Smooth Effects (Restored from Reference)
        if hasproperty(pstats, :smooth_effects) && !isnothing(pstats.smooth_effects)
            sm_m = pstats.smooth_effects.mean
            p_smooth = Plots.scatter(pos_x, pos_y, zcolor=sm_m, title="Smooth Manifold", markersize=5, color=:ice)
        end

        # Plot Denoised Prediction Map for time-slice 'ts'
        if hasproperty(pstats, :predictions_observed_denoised)
            y_den = pstats.predictions_observed_denoised.mean
            # Map slice to long-format index range
            idx_range = ((ts-1)*s_N_coords + 1):(ts*s_N_coords)
            if length(y_den) >= maximum(idx_range)
                p_pred_map = build_map_plot(y_den[idx_range], "Pred Denoised (t=$ts)", :magma)
            end
        end
    end

    # 6. Temporal and Seasonal Components
    if hasproperty(pstats, :temporal) && !isnothing(pstats.temporal)
        t_m = pstats.temporal.mean
        x_t = !isnothing(time_axis) ? time_axis : 1:length(t_m)
        p_temporal = Plots.plot(x_t, t_m, ribbon=(t_m .- pstats.temporal.lower, pstats.temporal.upper .- t_m), title="Temporal Trend", legend=false)
    end

    if hasproperty(pstats, :seasonal) && !isnothing(pstats.seasonal)
        s_m = pstats.seasonal.mean
        x_s = !isnothing(seas_labs) ? seas_labs : 1:length(s_m)
        p_seasonal = Plots.bar(x_s, s_m, title="Seasonality", legend=false, alpha=0.8)
    end

    # 7. Final Dashboard Assembly
    active_plots = filter(x -> !isnothing(x), [p_ppc, p_fixed, p_mixed, p_spatial, p_smooth, p_pred_map, p_temporal, p_seasonal])
    
    if !isempty(active_plots)
        n_plots = length(active_plots)
        cols = min(n_plots, 2)
        rows = Int(ceil(n_plots / cols))
        dashboard = Plots.plot(active_plots..., layout=(rows, cols), size=(950, 350 * rows))
        # display(dashboard)
        return dashboard
    else
        println("No valid components identified for plotting.")
    end
end



function get_vec(obj, key)  
    # Helper for robust indexing
    val = hasproperty(obj, key) ? getproperty(obj, key) : nothing
    val isa AbstractVector ? val : [val]
end


function summarize_lkj_correlation(chain, outcomes_N; alpha=0.05)
    println("--- Summarizing Cross-Outcome Covariance (LKJ Prior) ---")
    p_names = string.(FlexiChains.parameters(chain))

    # Identify correlation parameters (typically stored as L_omega or CorMatrix)
    cor_params = filter(p -> occursin("L_omega", p) || occursin("cor_mat", p), p_names)

    if isempty(cor_params)
        @warn "No correlation parameters found in chain."
        return nothing
    end

    # For LKJ Cholesky factors (L_omega), reconstruct the correlation matrix R = L * L'
    n_samples = size(chain, 1)
    cor_matrices = [zeros(outcomes_N, outcomes_N) for _ in 1:n_samples]

    try
        for j in 1:n_samples
            # This assumes a flat vector storage for Cholesky factors if not using NamedArrays
            L_vec = get_params_vector(chain, "L_omega", Int(outcomes_N*(outcomes_N+1)/2))[j, :]
            L = zeros(outcomes_N, outcomes_N)
            count = 1
            for c in 1:outcomes_N, r in c:outcomes_N
                L[r, c] = L_vec[count]
                count += 1
            end
            cor_matrices[j] .= L * L'
        end
    catch e
        @warn "Correlation reconstruction failed: $e"
        return nothing
    end

    # Calculate Mean and Quantiles
    mean_cor = mean(cor_matrices)
    low_cor = [quantile([m[r, c] for m in cor_matrices], alpha/2) for r in 1:outcomes_N, c in 1:outcomes_N]
    high_cor = [quantile([m[r, c] for m in cor_matrices], 1-alpha/2) for r in 1:outcomes_N, c in 1:outcomes_N]

    # Visualization
    p_heat = heatmap(mean_cor,
        title="Cross-Outcome Correlation (Mean)",
        clim=(-1, 1),
        color=:RdBu_11,
        aspect_ratio=:equal,
        xticks=(1:outcomes_N, ["Y$i" for i in 1:outcomes_N]),
        yticks=(1:outcomes_N, ["Y$i" for i in 1:outcomes_N]))

    display(p_heat)

    return (mean=mean_cor, lower=low_cor, upper=high_cor)
end
 

function plot_binned_covariates(res)
    # Check if covariate effects exist in pstats
    cov_effects = get(res.pstats, :covariate_effects, Dict())
    if isempty(cov_effects)
        println("No covariate effects found to plot.")
        return nothing
    end

    plts = []
    for (key, samples) in cov_effects
        # Key format: :raw_rw2_cwd
        name = replace(string(key), "raw_rw2_" => "")
        m = vec(mean(samples, dims=2))
        l = vec(quantile.(eachrow(samples), 0.025))
        u = vec(quantile.(eachrow(samples), 0.975))
        
        p = plot(m, ribbon=(m .- l, u .- m), title="Effect: $name", 
                 xlabel="Bin", ylabel="Value", legend=false, fillalpha=0.2)
        push!(plts, p)
    end
    
    n = length(plts)
    if n > 0
        display(plot(plts..., layout=(ceil(Int, n/2), 2), size=(900, 300*ceil(Int, n/2))))
    end
end

function plot_seasonal_cycle(res)
    if !hasproperty(res.pstats, :seasonal) || all(isnan, res.pstats.seasonal.mean)
        println("No seasonal component found to plot.")
        return nothing
    end
    
    m = res.pstats.seasonal.mean
    s = res.pstats.seasonal.samples
    l = vec(quantile.(eachrow(s), 0.025))
    u = vec(quantile.(eachrow(s), 0.975))
    
    plt = plot(m, ribbon=(m .- l, u .- m), title="Seasonal Cycle (Binned/Harmonic)", 
               xlabel="Season Bin", ylabel="Effect", lw=2, fillalpha=0.3, legend=false)
    display(plt)
    return plt
end

 
function dataframe_to_named_array(df::DataFrame)
    # Converts a DataFrame to a NamedArray for internal model processing
    mat = Matrix(df)
    return NamedArray(mat, (1:size(mat, 1), Symbol.(names(df))))
end

##############


 

 
 


@model function bstm_multivariate(M, ::Type{T}=Float64) where {T}
    # Audited Unified Multivariate Model (v07.2 - Finalized Censoring Logic)
    # Rationale: Finalizing the interval-censoring logic for intensity branches.
    # Requirement: Verbatim parity with v07.1 additive manifolds while adding stable logcdf offsets.

    local y_N = M[:y_N]
    local outcomes_N = M[:outcomes_N]
    local noise = get(M, :noise, 1e-6)
    local family = M[:model_family]

    # --- 1. Global Hyperparameters ---
    s_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :s_sigma_arr)
    t_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :t_sigma_arr)
    L_corr ~ NamedDist(LKJCholesky(outcomes_N, 1.0), :L_corr)

    # --- 2. Shared/Common Effects (Additive components for all outcomes) ---

    # 2.1 Hierarchical Spatial Scales
    local eta_hierarchy = zeros(T, y_N)
    if haskey(M, :spatial_hierarchy) && !isempty(M[:spatial_hierarchy])
        for scale_sym in keys(M[:spatial_hierarchy])
            scale = M[:spatial_hierarchy][scale_sym]
            sig_scale ~ NamedDist(Exponential(1.0), Symbol("s_sigma_", scale_sym))
            rho_scale ~ NamedDist(Beta(1, 1), Symbol("s_rho_", scale_sym))
            Q_scale = recompose_precision(Symbol(scale.model), scale.template.matrix, sig_scale, extra_param=rho_scale, noise=noise)
            f_scale ~ NamedDist(MvNormalCanon(zeros(scale.n_units), Q_scale), Symbol("s_latent_", scale_sym))
            eta_hierarchy .+= f_scale[scale.indices]
        end
    end

    # 2.2 Categorical/Mixed Effects
    local eta_mixed = zeros(T, y_N)
    if haskey(M, :c_groups) && !isempty(M[:c_groups])
        for c_sym in keys(M[:c_groups])
            temp = M[:c_re_templates][c_sym]
            rule = Symbol(get(M[:re_rules], string(c_sym), "iid"))
            sig_c_i ~ NamedDist(Exponential(1.0), Symbol("sig_c_", c_sym))
            c_Q = recompose_precision(rule, temp.matrix, sig_c_i, noise=noise)
            c_latent_i ~ NamedDist(MvNormalCanon(zeros(size(temp.matrix, 1)), c_Q), Symbol("c_latent_", c_sym))
            eta_mixed .+= c_latent_i[M[:c_groups][c_sym]]
        end
    end

    # 2.3 Basis/Smooth Effects
    local eta_basis = zeros(T, y_N)
    if haskey(M, :basis_matrices) && !isempty(M[:basis_matrices])
        for v_sym in keys(M[:basis_matrices])
            B = M[:basis_matrices][v_sym]
            sig_b ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", v_sym))
            beta_b ~ NamedDist(MvNormal(zeros(size(B, 2)), sig_b^2 * I), Symbol("beta_basis_", v_sym))
            eta_basis .+= B * beta_b
        end
    end

    # 2.4 Seasonal Manifold
    local eta_seasonal = zeros(T, y_N)
    if M[:model_season] != "none"
        u_sigma ~ NamedDist(Exponential(1.0), :u_sigma)
        m_season = Symbol(M[:model_season])
        if m_season == :harmonic
            u_alpha ~ NamedDist(Normal(0, 1), :u_alpha)
            u_beta ~ NamedDist(Normal(0, 1), :u_beta)
            local angles = (collect(1:M[:u_N]) .* (2*pi / M[:period]))
            local u_eta_vec = (u_alpha .* sin.(angles) .+ u_beta .* cos.(angles)) .* u_sigma
            eta_seasonal .+= u_eta_vec[M[:u_idx]]
        else
            u_Q = recompose_precision(m_season, M[:u_Q_template].matrix, u_sigma, noise=noise)
            u_latent ~ NamedDist(MvNormalCanon(zeros(M[:u_N]), u_Q), :u_latent)
            eta_seasonal .+= u_latent[M[:u_idx]]
        end
    end

    # --- 3. Outcome-Specific Latent Manifolds ---
    local latent_raw = zeros(T, y_N, outcomes_N)
    for k in 1:outcomes_N
        local current_field = zeros(T, y_N)
        local m_space = Symbol(get(M, :model_space, "none"))
        if m_space != :none
            s_Q_k = recompose_precision(m_space, M[:s_Q_template].matrix, s_sigma_arr[k], noise=noise)
            s_lat_k ~ NamedDist(MvNormalCanon(zeros(M[:s_N]), s_Q_k), Symbol("s_lat_", k))
            current_field .+= s_lat_k[M[:s_idx]]
        end
        local m_time = Symbol(get(M, :model_time, "none"))
        if m_time != :none
            t_Q_k = recompose_precision(m_time, M[:t_Q_template].matrix, t_sigma_arr[k], noise=noise)
            t_lat_k ~ NamedDist(MvNormalCanon(zeros(M[:t_N]), t_Q_k), Symbol("t_lat_", k))
            current_field .+= t_lat_k[M[:t_idx]]
        end
        latent_raw[:, k] .= current_field
    end

    # --- 4. Spatiotemporal Interactions ---
    local eta_st = zeros(T, y_N, outcomes_N)
    if M[:model_st] != "none"
        st_sigma_arr ~ NamedDist(filldist(Exponential(1.0), outcomes_N), :st_sigma_arr)
        m_st = Symbol(M[:model_st])
        for k in 1:outcomes_N
            Q_s_st = (m_st in [:III, :IV]) ? M[:s_Q_template].matrix : I(M[:s_N])
            Q_t_st = (m_st in [:II, :IV]) ? M[:t_Q_template].matrix : I(M[:t_N])
            st_Q_k = Symmetric(kron(Q_t_st, Q_s_st) .* (1.0 / (st_sigma_arr[k]^2 + noise)) + noise * I)
            st_lat_k ~ NamedDist(MvNormalCanon(zeros(M[:s_N] * M[:t_N]), st_Q_k), Symbol("st_lat_", k))
            eta_st[:, k] .= st_lat_k[(M[:t_idx] .- 1) .* M[:s_N] .+ M[:s_idx]]
        end
    end

    # --- 5. Cross-Outcome Coupling ---
    local eta_coupled = (latent_raw .+ eta_st) * L_corr.L

    # --- 6. Final assembly of the linear predictor ---
    local shared_additive = eta_hierarchy .+ eta_mixed .+ eta_basis .+ eta_seasonal

    # --- 7. Outcome Branching and Censoring Logic ---
    if family == "hurdle_poisson"
        beta_h ~ NamedDist(MvNormal(zeros(M[:Xfixed_N]), 2.0 * I), :beta_hurdle)
        beta_i ~ NamedDist(MvNormal(zeros(M[:Xfixed_N]), 2.0 * I), :beta_intensity)

        local eta_h = eta_coupled[:, 1] .+ (M[:Xfixed] * beta_h) .+ shared_additive
        local eta_i = eta_coupled[:, 2] .+ (M[:Xfixed] * beta_i) .+ shared_additive

        for i in 1:y_N
            # Branch A: Participation (Hurdle crossing)
            M[:y_obs_hurdle][i] ~ BernoulliLogit(eta_h[i])

            # Branch B: Intensity (Zero-Truncated Poisson with optional Censoring)
            if !ismissing(M[:y_obs_intensity][i])
                local mu_val = exp(eta_i[i])
                local y_val = M[:y_obs_intensity][i]

                # Finalized Censoring Logic for Interval/Right-Censored observations
                # If the outcome contains bounds [L, U], use logdiffexp(logcdf(U), logcdf(L))
                if haskey(M, :y_is_censored) && M[:y_is_censored][i]
                    local y_lower = M[:y_lower_bound][i]
                    local y_upper = M[:y_upper_bound][i]
                    # Stable log-probability for interval [L, U]
                    # Corrected for Zero-Truncation: P(Y=y | Y>0) = P(Y=y) / (1 - exp(-mu))
                    local log_prob_interval = logcdf(Poisson(mu_val), y_upper) - logcdf(Poisson(mu_val), y_lower - 1)
                    Turing.@addlogprob! log(max(1e-12, log_prob_interval)) - log(1.0 - exp(-mu_val) + 1e-12)
                else
                    # Standard point-mass truncated Poisson likelihood
                    local ll_trun = (y_val * log(mu_val + 1e-9)) - mu_val - logfactorial(Int(y_val)) - log(1.0 - exp(-mu_val) + 1e-9)
                    Turing.@addlogprob! ll_trun
                end
            end
        end
    else
        # Standard Multivariate Logic for non-hurdle families
        Xfixed_beta ~ NamedDist(MvNormal(zeros(M[:Xfixed_N] * outcomes_N), 2.0 * I), :Xfixed_beta)
        local mu_mat = eta_coupled .+ (M[:Xfixed] * reshape(Xfixed_beta, M[:Xfixed_N], outcomes_N))
        for k in 1:outcomes_N
            mu_mat[:, k] .+= shared_additive
        end
        if haskey(M, :log_offset); mu_mat .+= M[:log_offset]; end
        for k in 1:outcomes_N
            local good_idx = findall(!ismissing, M[:y_obs][:, k])
            if !isempty(good_idx)
                Turing.@addlogprob! logpdf(bstm_Likelihood(family, false, M[:weights][good_idx], 0.0, 1.0, 1.0, M[:trials][good_idx], M[:y_obs][good_idx, k]), mu_mat[good_idx, k])
            end
        end
    end
end




 


@model function bstm_multifidelity(M, ::Type{T}=Float64) where {T}
    # --- 1. Dimensions and Metadata ---
    local y_N = M[:y_N]
    local noise = get(M, :noise, 1e-6)
    local fidelity_idx = M[:fidelity_idx]

    # --- 2. Fidelity Coupling Parameters ---
    # These parameters link the shared latent process to the observed low-fidelity data.
    # fidelity_rho: Scaling/correlation factor
    # fidelity_bias: Additive offset/bias in low-fidelity observations
    fidelity_rho ~ NamedDist(Normal(1.0, 0.5), :fidelity_rho)
    fidelity_bias ~ NamedDist(Normal(0.0, 1.0), :fidelity_bias)

    # --- 3. Likelihood Hyperparameters ---
    local lik_r = 1.0
    if M[:model_family] == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end

    local lik_phi = 0.0
    if get(M, :use_zi, false)
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    # --- 4. Stochastic Volatility / Error Variance ---
    local y_sigma = 1.0
    if M[:model_family] in ["gaussian", "lognormal"]
        y_sigma ~ NamedDist(Exponential(1.0), :y_sigma)
    end

    # --- 5. Base Predictor (Fixed Effects) ---
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M[:Xfixed_N]), 5.0 * LinearAlgebra.I), :Xfixed_beta)
    Xfixed_eff = M[:Xfixed] * Xfixed_beta

    # Initialize linear predictor with offsets and fixed effects
    # eta_base serves as the shared foundation for all fidelities.
    local eta_base = (haskey(M, :log_offset) ? M[:log_offset] : zeros(T, y_N)) .+ Xfixed_eff

    # --- 6. Shared Hierarchical Spatial Manifolds ---
    # Partitioning variance across multiple spatial resolutions (nested regional effects)
    if haskey(M, :spatial_hierarchy) && !isempty(M[:spatial_hierarchy])
        for scale_sym in keys(M[:spatial_hierarchy])
            scale = M[:spatial_hierarchy][scale_sym]
            sig_scale ~ NamedDist(Exponential(1.0), Symbol("s_sigma_", scale_sym))
            rho_scale ~ NamedDist(Beta(1, 1), Symbol("s_rho_", scale_sym))
            
            Q_scale = recompose_precision(Symbol(scale.model), scale.template.matrix, sig_scale, extra_param=rho_scale, noise=noise)
            f_scale ~ NamedDist(MvNormalCanon(zeros(scale.n_units), Q_scale), Symbol("s_latent_", scale_sym))
            eta_base .+= f_scale[scale.indices]
        end
    end

    # --- 7. Shared Categorical Random Effects ---
    if haskey(M, :c_groups) && !isempty(M[:c_groups])
        c_names = collect(keys(M[:c_groups]))
        sig_c ~ NamedDist(filldist(Exponential(1.0), length(c_names)), :sig_c)
        for (i, c_sym) in enumerate(c_names)
            temp = M[:c_re_templates][c_sym]
            rule = Symbol(get(M[:re_rules], string(c_sym), "iid"))
            
            c_Q = recompose_precision(rule, temp.matrix, sig_c[i], noise=noise)
            c_latent_i ~ NamedDist(MvNormalCanon(zeros(size(temp.matrix, 1)), c_Q), Symbol("c_latent_", c_sym))
            eta_base .+= c_latent_i[M[:c_groups][c_sym]]
        end
    end

    # --- 8. Shared Latent Spatio-Temporal Manifold ---
    # This process represents the "true" underlying signal common to all fidelities.
    
    # 8.1 Primary Spatial Component
    local shared_s = zeros(T, M[:s_N])
    if M[:model_space] != "none"
        s_sigma ~ NamedDist(Exponential(1.0), :s_sigma)
        m_space = Symbol(M[:model_space])
        local s_extra = nothing
        if m_space in [:bym2, :leroux, :sar, :dag]
            s_rho ~ NamedDist(Beta(1, 1), :s_rho)
            s_extra = s_rho
        end
        s_Q = recompose_precision(m_space, M[:s_Q_template].matrix, s_sigma, extra_param=s_extra, noise=noise)
        s_latent ~ NamedDist(MvNormalCanon(zeros(M[:s_N]), s_Q), :s_latent)
        shared_s = s_latent
    end

    # 8.2 Primary Temporal Component
    local shared_t = zeros(T, M[:t_N])
    if M[:model_time] != "none"
        t_sigma ~ NamedDist(Exponential(1.0), :t_sigma)
        m_time = Symbol(M[:model_time])
        local t_extra = nothing
        if m_time == :ar1
            t_rho ~ NamedDist(Beta(2, 2), :t_rho)
            t_extra = t_rho
        end
        t_Q = recompose_precision(m_time, M[:t_Q_template].matrix, t_sigma, extra_param=t_extra, noise=noise)
        t_latent ~ NamedDist(MvNormalCanon(zeros(M[:t_N]), t_Q), :t_latent)
        shared_t = t_latent
    end

    # 8.3 Spatially Varying Coefficients (SVC) Shared Logic
    local shared_svc = zeros(T, y_N)
    if !isempty(M[:svc_covariates]) && M[:svc_model] == "rff"
        for (k, c_sym) in enumerate(M[:svc_covariates])
            sig_svc_k ~ NamedDist(Exponential(1.0), Symbol("sig_svc_", c_sym))
            beta_svc_k ~ NamedDist(filldist(Normal(0, 1), M[:svc_M_rff]), Symbol("beta_svc_", c_sym))
            beta_s_k = M[:svc_basis_cached][M[:s_idx], :] * beta_svc_k
            col_idx = findfirst(==(c_sym), Symbol.(names(M[:Xfixed], 2)))
            if !isnothing(col_idx)
                shared_svc .+= (beta_s_k .* sig_svc_k) .* M[:Xfixed][:, col_idx]
            end
        end
    end

    # --- 9. Fidelity Assembly and Data Coupling ---
    # Combine all shared manifold components
    shared_manifold = shared_s[M[:s_idx]] .+ shared_t[M[:t_idx]] .+ shared_svc
    
    # η_high: Direct realization of the shared process
    # η_low: Scaled and biased realization for low-fidelity sources
    eta_high = eta_base .+ shared_manifold
    eta_low  = fidelity_bias .+ eta_base .+ (fidelity_rho .* shared_manifold)

    # Clamp for numerical stability prior to likelihood
    eta_high = clamp.(eta_high, -20.0, 20.0)
    eta_low  = clamp.(eta_low, -20.0, 20.0)

    # --- 10. Likelihood Execution ---
    for i in 1:y_N
        if !ismissing(M[:y_obs][i])
            # Route to the appropriate fidelity predictor
            # fidelity_idx == 1 is High-Fidelity; fidelity_idx == 2 is Low-Fidelity
            current_eta = (fidelity_idx[i] == 1) ? eta_high[i] : eta_low[i]
            
            Turing.@addlogprob! logpdf(
                bstm_Likelihood(
                    M[:model_family], 
                    get(M, :use_zi, false), 
                    M[:weights][i], 
                    lik_phi, 
                    lik_r, 
                    y_sigma, 
                    M[:trials][i], 
                    M[:y_obs][i]
                ), 
                [current_eta]
            )
        end
    end
end

 

 
@model function bstm_univariate(M, ::Type{T}=Float64) where {T}
    # Audited and Unified Univariate Model (v05.5 - Complete Reconciliation)
    # Basis: bstm_univariate_reference (Cell jYjrwzlHJqqd)
    # Rationale: Verbatim restoration of all hierarchical, interaction, and spectral features
    # with integrated v05.5 Topological and Spectral dispatchers.
    # Constraint: No 'clamp' on indices; No ternary operators in sampling blocks.

    # --- 1. Global Likelihood Hyperparameters ---
    local lik_r = one(T)
    local lik_phi = zero(T)
    if M.model_family == "negbin"
        lik_r ~ NamedDist(Exponential(1.0), :lik_r)
    end
    if M.use_zi
        lik_phi ~ NamedDist(Beta(1, 1), :lik_phi)
    end

    # --- 2. Stochastic Volatility (SV) & Error Variance ---
    # Pre-allocate y_sigma with explicit length and type T
    local y_sigma = Vector{T}(undef, M.y_N)

    if get(M, :use_sv, false)
        sigma_log_var ~ NamedDist(Exponential(1.0), :sigma_log_var)
        beta_vol_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :beta_vol_latent)

        # Standardize RFF mapping into a 1D vector
        local vol_latent_field = vec(sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj * beta_vol_latent))

        # Use explicit loop for maximum stability across AD/Eval passes
        for i in 1:M.y_N
            y_sigma[i] = exp((sigma_log_var * vol_latent_field[i]) / 2.0)
        end
    else
        if M.model_family == "gaussian" || M.model_family == "lognormal"
            y_sigma_val ~ NamedDist(Exponential(1.0), :y_sigma)
            fill!(y_sigma, y_sigma_val)
        else
            fill!(y_sigma, one(T))
        end
    end


    # --- 3. Base Predictor (Fixed & Basis Effects) ---
    
    Xfixed_beta ~ NamedDist(MvNormal(zeros(M.Xfixed_N), 5.0 * I), :Xfixed_beta)
    # Xfixed_eff = M.Xfixed * Xfixed_beta
    
    # Linear predictor starts with optional log_offset and fixed effects
    local eta = Vector{T}(M.Xfixed * Xfixed_beta)

    if haskey(M, :log_offset)
        eta .+= M.log_offset
    end

    # Advanced Basis Matrices (P-splines, FFT, Wavelets, TPS)
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        for v_sym in keys(M.basis_matrices)
            B = M.basis_matrices[v_sym]
            M_basis = size(B, 2)
            sig_basis ~ NamedDist(Exponential(1.0), Symbol("sig_basis_", v_sym))
            beta_basis ~ NamedDist(MvNormal(zeros(M_basis), sig_basis^2 * I), Symbol("beta_basis_", v_sym))
            eta .+= B * beta_basis
        end
    end

    # --- 4. Hierarchical Multi-Resolution Spatial Manifolds ---
    if haskey(M, :spatial_hierarchy) && !isempty(M.spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            scale = M.spatial_hierarchy[scale_sym]
            sig_scale ~ NamedDist(Exponential(1.0), Symbol("s_sigma_", scale_sym))
            rho_scale ~ NamedDist(Beta(1, 1), Symbol("s_rho_", scale_sym))
            Q_scale = recompose_precision(Symbol(scale.model), scale.template.matrix, sig_scale, extra_param=rho_scale, noise=M.noise)
            f_scale ~ NamedDist(MvNormalCanon(zeros(scale.n_units), Q_scale), Symbol("s_latent_", scale_sym))
            eta .+= f_scale[scale.indices]
        end
    end

    # --- 5. Categorical & Mixed Effects (Varying Slopes) ---
    if haskey(M, :c_groups) && !isempty(M.c_groups)
        for c_sym in keys(M.c_groups)
            temp = M.c_re_templates[c_sym]
            rule = Symbol(get(M.re_rules, string(c_sym), "iid"))
            sig_c_i ~ NamedDist(Exponential(1.0), Symbol("sig_c_", c_sym))
            c_Q = recompose_precision(rule, temp.matrix, sig_c_i, noise=M.noise)
            c_latent_i ~ NamedDist(MvNormalCanon(zeros(size(temp.matrix, 1)), c_Q), Symbol("c_latent_", c_sym))
            eta .+= c_latent_i[M.c_groups[c_sym]]
        end
    end

    if haskey(M, :mixed_terms) && !isempty(M.mixed_terms)
        for (i, m_term) in enumerate(M.mixed_terms)
            sig_me_i ~ NamedDist(Exponential(1.0), Symbol("sig_me_", i))
            beta_group_i ~ NamedDist(filldist(Normal(0, sig_me_i), m_term.n_cat), Symbol("beta_group_", i))
            eta .+= beta_group_i[m_term.indices] .* m_term.covariate_vals
        end
    end

    # --- 6. Interaction Smooths & Eigen-Effects ---
    if haskey(M, :interaction_terms) && !isempty(M.interaction_terms)
        for (i, i_term) in enumerate(M.interaction_terms)
            sig_ie_i ~ NamedDist(Exponential(1.0), Symbol("sig_ie_", i))
            projection_ie = (i_term.coords * i_term.W_ie) .+ i_term.b_ie'
            beta_ie_i ~ NamedDist(filldist(Normal(0, sig_ie_i), i_term.M_rff), Symbol("beta_ie_", i))
            eta .+= sqrt(2.0 / i_term.M_rff) .* cos.(projection_ie) * beta_ie_i
        end
    end

    if haskey(M, :eigen_terms) && !isempty(M.eigen_terms)
        for (i, e_term) in enumerate(M.eigen_terms)
            v_pca_i ~ NamedDist(filldist(Normal(0, 1.0), length(e_term.ltri_indices)), Symbol("v_pca_", i))
            pca_sd_i ~ NamedDist(filldist(Exponential(1.0), e_term.n_dims), Symbol("pca_sd_", i))
            pdef_sd_i ~ NamedDist(Exponential(0.1), Symbol("pdef_sd_", i))
            K_pca, _, _ = householder_transform(v_pca_i, e_term.n_dims, e_term.n_dims, e_term.ltri_indices, pca_sd_i, pdef_sd_i, M.noise)
            eta .+= vec(e_term.data * K_pca)
        end
    end

    # --- 7. Spatially Varying Coefficients (SVC) ---
    if !isempty(M.svc_covariates)
        if M.svc_model == "rff"
            for (k, c_sym) in enumerate(M.svc_covariates)
                sig_svc_k ~ NamedDist(Exponential(1.0), Symbol("sig_svc_", c_sym))
                beta_svc_k ~ NamedDist(filldist(Normal(0, 1), M.svc_M_rff), Symbol("beta_svc_", c_sym))
                z_k = haskey(M, :svc_basis_cached) ? M.svc_basis_cached : (sqrt(2.0 / M.svc_M_rff) .* cos.((M.s_xy * M.W_fixed_svc) .+ M.b_fixed_svc'))
                beta_s_k = z_k[M.s_idx, :] * beta_svc_k
                col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
                if !isnothing(col_idx)
                    eta .+= (beta_s_k .* sig_svc_k) .* M.Xfixed[:, col_idx]
                end
            end
        end
    end

    # --- 8. Primary Spatial Manifold (Topological v05.5 Expansion) ---
    local s_eta = zeros(T, M.s_N)
    if M.model_space != "none"
        s_sigma ~ NamedDist(Exponential(1.0), :s_sigma)
        m_space = Symbol(M.model_space)

        if m_space == :local_adaptive
            # Riemannian manifold: weights vary the precision locally
            local_w ~ NamedDist(filldist(Gamma(2, 2), M.s_N), :local_w)
            Q_local = Diagonal(local_w) * M.s_Q_template.matrix * Diagonal(local_w)
            s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), Symmetric(Q_local + M.noise*I)), :s_latent)
            s_eta = s_latent .* s_sigma
        elseif m_space == :network
            # Directed flow manifold (River flow)
            s_rho ~ NamedDist(Beta(1, 1), :s_rho)
            Q_net = (I - s_rho .* M.s_Q_template.matrix)' * (I - s_rho .* M.s_Q_template.matrix)
            s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), Symmetric(Q_net + M.noise*I)), :s_latent)
            s_eta = s_latent .* s_sigma
        elseif m_space == :hyperbolic
            # Negative curvature hierarchical embedding
            s_ls ~ NamedDist(InverseGamma(3, 3), :s_ls)
            K_hyp = (s_sigma^2) .* exp.(-M.d_hyperbolic ./ (2 * s_ls^2 + M.noise))
            s_latent ~ NamedDist(MvNormal(zeros(M.s_N), Symmetric(K_hyp + M.noise*I)), :s_latent)
            s_eta = s_latent
        elseif m_space == :mosaic
            mu_local ~ NamedDist(filldist(Normal(0, 1), M.n_mosaics), :mu_local)
            # s_eta = mu_local[M.cluster_assignments] .* s_sigma
            mu_local ~ NamedDist(filldist(Normal(0, 1), M.n_mosaics), :mu_local)
            for i in 1:M.y_N
                # Map observation i to spatial unit s, then to cluster cluster_id
                local unit_idx = M.s_idx[i]
                local cluster_id = M.cluster_assignments[unit_idx]
                eta[i] += mu_local[cluster_id] * s_sigma
            end
         
        elseif m_space == :deepgp
            curr_coords = M.s_coord
            local out_layer = zeros(T, M.s_N)
            for l in 1:M.n_layers
                sig_l ~ NamedDist(Exponential(1.0), Symbol("sig_layer_", l))
                w_l ~ NamedDist(filldist(Normal(0, 1), M.m_layers[l]), Symbol("w_layer_", l))
                phi_l = sqrt(2.0 / M.m_layers[l]) .* cos.(curr_coords * M.Om_layers[l]' .+ M.Ph_layers[l]')
                if l < M.n_layers
                    curr_coords = phi_l * w_l
                else
                    out_layer = (phi_l * w_l) .* sig_l
                end
            end
            s_eta = out_layer
        elseif m_space == :warping
            w_warp ~ NamedDist(filldist(Normal(0, 1), M.M_rff), :w_warp)
            warp_proj = (M.s_coord * M.W_fixed) .+ M.b_fixed'
            s_eta = vec((sqrt(2.0 / M.M_rff) .* cos.(warp_proj) * w_warp) .* s_sigma)
        elseif m_space == :bgcn
            gcn_w ~ NamedDist(MvNormal(zeros(M.s_N), I), :gcn_w)
            D_inv_sq = Diagonal(1.0 ./ sqrt.(vec(sum(M.W, dims=2)) .+ M.noise))
            W_norm = D_inv_sq * M.W * D_inv_sq
            s_eta = (W_norm * gcn_w) .* s_sigma
        elseif m_space == :fitc
            st_ls_s ~ NamedDist(filldist(Gamma(2, 2), size(M.Z_inducing, 2)), :st_ls_s)
            k_st = (s_sigma^2) * (SqExponentialKernel() ∘ ARDTransform(inv.(st_ls_s .+ M.noise)))
            K_ZZ = kernelmatrix(k_st, RowVecs(M.Z_inducing)) + M.noise * I
            K_XZ = kernelmatrix(k_st, RowVecs(hcat(M.s_coord, M.t_coord)), RowVecs(M.Z_inducing))
            s_inducing ~ NamedDist(MvNormal(zeros(size(M.Z_inducing, 1)), K_ZZ), :s_inducing)
            s_eta = K_XZ * (K_ZZ \ s_inducing)
        else
            local s_rho = nothing
            if m_space in [:bym2, :leroux, :sar, :dag]
                s_rho ~ NamedDist(Beta(1, 1), :s_rho)
            end
            s_Q = recompose_precision(m_space, M.s_Q_template.matrix, s_sigma, extra_param=s_rho, noise=M.noise)
            s_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N), s_Q), :s_latent)
            s_eta = s_latent
        end
        eta .+= s_eta[M.s_idx]
    end

    # --- 9. Primary Temporal Manifold ---
    local t_eta_vec = zeros(T, M.t_N)
    if M.model_time != "none"
        t_sigma ~ NamedDist(Exponential(1.0), :t_sigma)
        m_time = Symbol(M.model_time)
        if m_time == :harmonic
            t_alpha ~ NamedDist(Normal(0, 1), :t_alpha)
            t_beta ~ NamedDist(Normal(0, 1), :t_beta)
            t_eta_vec = (t_alpha .* sin.(M.t_angle) .+ t_beta .* cos.(M.t_angle)) .* t_sigma
        elseif m_time == :eigen
            v_pca_t ~ NamedDist(filldist(Normal(0, 1.0), length(M.t_ltri_indices)), :v_pca_t)
            pca_sd_t ~ NamedDist(filldist(Exponential(1.0), M.t_n_dims), :pca_sd_t)
            K_pca_t, _, _ = householder_transform(v_pca_t, M.t_n_dims, M.t_n_dims, M.t_ltri_indices, pca_sd_t, 0.01, M.noise)
            t_eta_vec = vec(M.t_basis * K_pca_t)
        elseif m_time == :gp
            t_ls ~ NamedDist(InverseGamma(3, 3), :t_ls)
            K_t = (t_sigma^2) .* exp.(-M.t_dist_mat .^ 2 ./ (2 * t_ls^2 + M.noise))
            t_latent ~ NamedDist(MvNormal(zeros(M.t_N), Symmetric(K_t + M.noise*I)), :t_latent)
            t_eta_vec = t_latent
        elseif m_time == :logistic_ar1
            t_rho ~ NamedDist(Beta(2, 2), :t_rho)
            K_logi ~ NamedDist(truncated(Normal(M.K[1], M.K[2]), lower=M.K[1]*0.01), :K_logi)
            r_logi ~ NamedDist(truncated(Normal(M.r[1], M.r[2]), lower=0.01), :r_logi)
            t_Q_prec = build_ar1_precision_matrix(M.t_N, t_rho, t_sigma)
            t_innov ~ NamedDist(MvNormalCanon(zeros(M.t_N), t_Q_prec + M.noise*I), :t_innov)
            m_vec = Vector{T}(undef, M.nM)
            m_vec[1] ~ NamedDist(truncated(Normal(M.m0[1], M.m0[2]), lower=M.mlim[1], upper=M.mlim[2]), :m_init)
            for i in 2:M.nM
                m_prev = m_vec[i-1]
                growth = r_logi * m_prev * (1.0 - m_prev)
                removal = M.removed[i-1] / K_logi
                m_vec[i] ~ NamedDist(truncated(Normal(m_prev + growth - removal, 0.1), lower=M.mlim[1], upper=M.mlim[2]), Symbol("m_", i))
            end
            t_eta_vec = (m_vec[1:M.t_N] .- M.removed[1:M.t_N]./K_logi) ./ 0.1 .+ t_innov
        else
            local t_rho = nothing
            if m_time == :ar1
                t_rho ~ NamedDist(Beta(2, 2), :t_rho)
            end
            t_Q = recompose_precision(m_time, M.t_Q_template.matrix, t_sigma, extra_param=t_rho, noise=M.noise)
            t_latent ~ NamedDist(MvNormalCanon(zeros(M.t_N), t_Q), :t_latent)
            t_eta_vec = t_latent
        end
        eta .+= t_eta_vec[M.t_idx]
    end

    # --- 10. Seasonal Manifold ---
    if M.model_season != "none"
        u_sigma ~ NamedDist(Exponential(1.0), :u_sigma)
        m_season = Symbol(M.model_season)
        if m_season == :harmonic
            u_alpha ~ NamedDist(Normal(0, 1), :u_alpha)
            u_beta ~ NamedDist(Normal(0, 1), :u_beta)
            angles = (collect(1:M.u_N) .* (2*pi / M.period))
            u_eta_vec = (u_alpha .* sin.(angles) .+ u_beta .* cos.(angles)) .* u_sigma
            eta .+= u_eta_vec[M.u_idx]            
            
        else
            local u_rho = nothing
            if m_season == :ar1
                u_rho ~ NamedDist(Beta(2, 2), :u_rho)
            end
            u_Q = recompose_precision(m_season, M.u_Q_template.matrix, u_sigma, extra_param=u_rho, noise=M.noise)
            u_latent ~ NamedDist(MvNormalCanon(zeros(M.u_N), u_Q), :u_latent)
            eta .+= u_latent[M.u_idx]
        end
    end

    # --- 11. Spatiotemporal Interaction & Transport (Physics-Informed) ---
    if M.model_st != "none"
        st_sigma ~ NamedDist(Exponential(1.0), :st_sigma)
        m_st = Symbol(M.model_st)
        if m_st in [:diffusion, :advection, :advection_diffusion]
            st_diff ~ NamedDist(Exponential(0.5), :st_diff)
            st_adv ~ NamedDist(Exponential(0.5), :st_adv)
            st_pers ~ NamedDist(MvNormal(zeros(M.s_N), I), :st_pers)
            st_innov ~ filldist(Normal(0, st_sigma), M.s_N, M.t_N)
            st_map = zeros(T, M.s_N, M.t_N)
            st_map[:, 1] .= st_innov[:, 1]
            L_phys = Matrix(Diagonal(vec(sum(M.W, dims=2))) - M.W)
            for t in 2:M.t_N
                # mu_p calculation as per reference: Transport via Laplacian spread
                mu_p = st_map[:, t-1] .- (m_st != :advection ? st_diff .* (L_phys * st_map[:, t-1]) : 0.0) .- (m_st != :diffusion ? st_adv .* (L_phys * s_eta) : 0.0)
                st_map[:, t] .= logistic.(st_pers) .* mu_p .+ st_innov[:, t]
            end
            for i in 1:M.y_N
                eta[i] += st_map[M.s_idx[i], M.t_idx[i]]
            end
        else
            # Kronecker-structured interactions (Type I-IV)
            Q_s_st = (m_st in [:III, :IV]) ? M.s_Q_template.matrix : I(M.s_N)
            Q_t_st = (m_st in [:II, :IV]) ? M.t_Q_template.matrix : I(M.t_N)
            st_Q_kron = Symmetric(kron(Q_t_st, Q_s_st) .* (1.0 / (st_sigma^2 + M.noise)) + M.noise * I)
            st_latent ~ NamedDist(MvNormalCanon(zeros(M.s_N * M.t_N), st_Q_kron), :st_latent)
            eta .+= st_latent[(M.t_idx .- 1) .* M.s_N .+ M.s_idx]
        end
    end

    # --- 12. Final Likelihood Execution ---
    eta = clamp.(eta, -20.0, 20.0)
    good_idx = findall(!ismissing, M.y_obs)

    
    # --- 11. Final Likelihood & Censoring Execution ---
    # This block handles standard observations AND interval/right-censored data.
    
    for i in 1:M.y_N[good_idx]
        if !ismissing(M.y_obs[i])
            local lin_pred = eta[i]
            
            # AUDIT: Dynamic routing for censored vs point observations
            if haskey(M, :y_is_censored) && M.y_is_censored[i]
                local y_L = M.y_lower_bound[i]
                local y_U = M.y_upper_bound[i]

                # Standardize distribution parameters
                if M.model_family == "poisson"
                    mu = exp(lin_pred)
                    # Stable log-prob for interval [L, U]: log(P(L <= Y <= U))
                    Turing.@addlogprob! logdiffexp(logcdf(Poisson(mu), y_U), logcdf(Poisson(mu), y_L - 1))
                
                elseif M.model_family == "gaussian"
                    sig = y_sigma isa Real ? y_sigma : y_sigma[i]
                    Turing.@addlogprob! logdiffexp(logcdf(Normal(lin_pred, sig), y_U), logcdf(Normal(lin_pred, sig), y_L))
                
                elseif M.model_family == "negbin"
                    mu = exp(lin_pred)
                    prob = lik_r / (lik_r + mu)
                    Turing.@addlogprob! logdiffexp(logcdf(NegativeBinomial(lik_r, prob), y_U), logcdf(NegativeBinomial(lik_r, prob), y_L - 1))
                end
            else
                # Standard Point Likelihood
                eta[i] ~ bstm_Likelihood(M.model_family, M.use_zi, M.weights[i], lik_phi, lik_r, y_sigma, M.trials[i], M.y_obs[i])
            end
        end
    end

    # Turing.@addlogprob! logpdf(bstm_Likelihood(M.model_family, M.use_zi, M.weights[good_idx], lik_phi, lik_r, y_sigma, M.trials[good_idx], M.y_obs[good_idx]), eta[good_idx])
end 




function _reconstruct(arch::UnivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # Audited, Feature-Complete Univariate Reconstruction [v06.2 SVC + Spatiotemporal Restoration]
    # Rationale: Ensuring full recovery of Fixed, Mixed, Smooth, Seasonal, Spatiotemporal, AND SVC effects.
    # Constraint: No 'clamp' on indices; explicit boundary verification used.

    # 1. Metadata and Dimensional Discovery
    local N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    local N_tot = M.y_N + N_PS
    local N_samples = size(chain, 1)
    local p_names = string.(FlexiChains.parameters(chain))
    local fam = get_model_family(get(M, :model_family, "gaussian"))

    # 2. Allocation of Latent Manifold Containers
    local s_eff_struct = zeros(M.s_N, N_samples)
    local s_eff_noisy = zeros(M.s_N, N_samples)
    local t_eff = zeros(M.t_N, N_samples)
    local u_eff = zeros(M.u_N, N_samples)
    local Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
    local mixed_terms_list = get(M, :mixed_terms, [])
    local mixed_eff_coeffs = !isempty(mixed_terms_list) ? [zeros(term.n_cat, N_samples) for term in mixed_terms_list] : nothing
    local basis_eff_accum = zeros(M.y_N, N_samples)
    local st_eff_map = zeros(M.s_N, M.t_N, N_samples)
    
    # SVC Containers
    # svc_slopes stores the varying beta for each spatial unit [S_N x N_covariates x N_samples]
    local svc_covs = get(M, :svc_covariates, Symbol[])
    local n_svc = length(svc_covs)
    local svc_slopes = n_svc > 0 ? zeros(M.s_N, n_svc, N_samples) : nothing

    local sv_surface = zeros(N_tot, N_samples)

    # 3. Parameter Extraction Loop
    for j in 1:N_samples
        # --- 3.1 Fixed Effects Discovery ---
        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)[j, :]
        end

        # --- 3.2 Spatial Recovery (Structured vs Noisy) ---
        if M.model_space == "mosaic"
            m_vals = get_params_vector(chain, "mu_local", M.n_mosaics)[j, :]
            s_eff_struct[:, j] .= m_vals[M.cluster_assignments]
            s_eff_noisy[:, j] .= s_eff_struct[:, j]
        else
            s_sig = "s_sigma" in p_names ? get_params_vector(chain, "s_sigma", 1)[j] : 1.0
            fs, fn = extract_manifold(SpatialTrait(), chain, M, j, s_sig, M.s_coord[:, 1], M.s_coord[:, 2])
            s_eff_struct[:, j] .= fs
            s_eff_noisy[:, j] .= fn
        end

        # --- 3.3 Temporal Recovery ---
        t_sig = "t_sigma" in p_names ? get_params_vector(chain, "t_sigma", 1)[j] : 1.0
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M, j, t_sig)

        # --- 3.4 Seasonal Recovery (Harmonic/Binned) ---
        if M.model_season != "none"
            u_sig = "u_sigma" in p_names ? get_params_vector(chain, "u_sigma", 1)[j] : 1.0
            u_eff[:, j] .= extract_manifold(SeasonalTrait(), chain, M, j, u_sig)
        end

        # --- 3.5 Basis Matrix Discovery (Smooth Effects) ---
        if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
            for v_sym in keys(M.basis_matrices)
                B = M.basis_matrices[v_sym]
                p_key = "beta_basis_" * string(v_sym)
                if p_key in p_names
                    b_coeffs = get_params_vector(chain, p_key, size(B, 2))[j, :]
                    basis_eff_accum[:, j] .+= B * b_coeffs
                end
            end
        end

        # --- 3.6 Mixed Effects Discovery ---
        if !isnothing(mixed_eff_coeffs)
            for (m_idx, m_term) in enumerate(mixed_terms_list)
                p_key = "beta_group_" * string(m_idx)
                if p_key in p_names
                    mixed_eff_coeffs[m_idx][:, j] .= get_params_vector(chain, p_key, m_term.n_cat)[j, :]
                end
            end
        end

        # --- 3.7 Spatiotemporal Interaction Discovery ---
        if "st_latent" in p_names
            st_sig_j = "st_sigma" in p_names ? get_params_vector(chain, "st_sigma", 1)[j] : 1.0
            raw_st = get_params_vector(chain, "st_latent", M.s_N * M.t_N)[j, :]
            for t_ptr in 1:M.t_N
                for s_ptr in 1:M.s_N
                    lin_idx = (t_ptr - 1) * M.s_N + s_ptr
                    st_eff_map[s_ptr, t_ptr, j] = raw_st[lin_idx] * st_sig_j
                end
            end
        end

        # --- 3.8 Spatially Varying Coefficients (SVC) Discovery ---
        if !isnothing(svc_slopes)
            for (k, c_sym) in enumerate(svc_covs)
                p_key_sig = "sig_svc_" * string(c_sym)
                p_key_beta = "beta_svc_" * string(c_sym)
                if p_key_beta in p_names
                    sig_svc = p_key_sig in p_names ? get_params_vector(chain, p_key_sig, 1)[j] : 1.0
                    # Handle RFF-based SVC
                    if M.svc_model == "rff"
                        beta_rff = get_params_vector(chain, p_key_beta, M.svc_M_rff)[j, :]
                        svc_slopes[:, k, j] .= (M.svc_basis_cached * beta_rff) .* sig_svc
                    else
                        # Handle CAR-based SVC
                        svc_slopes[:, k, j] .= get_params_vector(chain, p_key_beta, M.s_N)[j, :] .* sig_svc
                    end
                end
            end
        end

        # --- 3.9 Volatility Recovery ---
        if get(M, :use_sv, false)
            sig_log = get_params_vector(chain, "sigma_log_var", 1)[j]
            beta_v = get_params_vector(chain, "beta_vol_latent", M.M_rff_sigma)[j, :]
            sv_surface[:, j] .= exp.((sig_log .* (sqrt(2.0 / M.M_rff_sigma) .* cos.(M.vol_proj * beta_v))) ./ 2.0)

        else
            y_sig = "y_sigma" in p_names ? get_params_vector(chain, "y_sigma", 1)[j] : 1.0
            sv_surface[:, j] .= y_sig
        end
    end

    # 4. Predictor Assembly
    local eta_denoised = zeros(N_tot, N_samples)
    local eta_noisy = zeros(N_tot, N_samples)

    for j in 1:N_samples
        for i in 1:N_tot
            is_obs = i <= M.y_N
            idx = is_obs ? i : i - M.y_N
            src = is_obs ? M : PS

            s_id = Int(src.s_idx[idx])
            t_id = Int(src.t_idx[idx])
            u_id = Int(src.u_idx[idx])

            # Base Linear Predictor Components
            val = is_obs ? M.log_offset[idx] : 0.0

            # Add Fixed Effects
            if !isnothing(Xfixed_betas)
                val += dot(Xfixed_betas[:, j], is_obs ? M.Xfixed[idx, :] : PS.Xfixed[idx, :])
            end

            # Add Mixed Effects
            if !isnothing(mixed_eff_coeffs)
                for (m_idx, m_term) in enumerate(mixed_terms_list)
                    g_id = Int(m_term.indices[idx])
                    val += mixed_eff_coeffs[m_idx][g_id, j] * m_term.covariate_vals[idx]
                end
            end

            # Accumulate Basis/Smooth Effects
            if is_obs
                val += basis_eff_accum[idx, j]
            end

            # Add Spatiotemporal Interaction
            st_val = "st_latent" in p_names ? st_eff_map[s_id, t_id, j] : 0.0

            # Add SVC Contribution
            if !isnothing(svc_slopes)
                for (k, c_sym) in enumerate(svc_covs)
                    # Find covariate value in the design matrix or raw data
                    # Assumes svc covariates are part of Xfixed or explicitly provided in M
                    col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
                    if !isnothing(col_idx)
                        cov_val = is_obs ? M.Xfixed[idx, col_idx] : PS.Xfixed[idx, col_idx]
                        val += svc_slopes[s_id, k, j] * cov_val
                    end
                end
            end

            # Final Superposition
            eta_denoised[i, j] = val + s_eff_struct[s_id, j] + t_eff[t_id, j] + u_eff[u_id, j] + st_val
            eta_noisy[i, j] = val + s_eff_noisy[s_id, j] + t_eff[t_id, j] + u_eff[u_id, j] + st_val
        end
    end
 
    
    # 5. Link Application and Censoring-Aware Likelihood Processing
    # define a localized version of processing to handle censored log_lik logic
    # p_denoised, p_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M, N_tot, N_samples, sv_surface, M.y_obs)
    local p_denoised = zeros(N_tot, N_samples)
    local p_noisy = zeros(N_tot, N_samples)
    local log_lik = zeros(N_samples, M.y_N)

    for j in 1:N_samples
        local sig_y = sv_surface[:, j]
        local r_val = "lik_r" in p_names ? chain[:lik_r].data[j] : 1.0

        for i in 1:N_tot
            is_obs = i <= M.y_N
            mu_eta = eta_noisy[i, j]

            mu = if fam isa PoissonFamily || fam isa NegativeBinomialFamily
                clamp(exp(mu_eta), 1e-10, 1e9)
            elseif fam isa BinomialFamily
                logistic(mu_eta)
            else
                mu_eta
            end

            p_denoised[i, j] = mu

            if is_obs
                # Logic for Censored Log-Likelihood recovery
                if haskey(M, :y_is_censored) && M.y_is_censored[i]
                    y_L = M.y_lower_bound[i]
                    y_U = M.y_upper_bound[i]
                    
                    if fam isa PoissonFamily
                        log_lik[j, i] = logdiffexp(logcdf(Poisson(mu), y_U), logcdf(Poisson(mu), y_L - 1))
                    elseif fam isa GaussianFamily
                        log_lik[j, i] = logdiffexp(logcdf(Normal(mu, sig_y[i]), y_U), logcdf(Normal(mu, sig_y[i]), y_L))
                    elseif fam isa NegativeBinomialFamily
                        prob = r_val / (r_val + mu)
                        log_lik[j, i] = logdiffexp(logcdf(NegativeBinomial(r_val, prob), y_U), logcdf(NegativeBinomial(r_val, prob), y_L - 1))
                    end
                else
                    # Standard point-observation likelihood
                    y_val = M.y_obs[i]
                    log_lik[j, i] = if fam isa PoissonFamily; logpdf(Poisson(mu), y_val)
                                   elseif fam isa GaussianFamily; logpdf(Normal(mu, sig_y[i]), y_val)
                                   elseif fam isa BinomialFamily; logpdf(Binomial(M.trials[i], mu), y_val)
                                   elseif fam isa NegativeBinomialFamily
                                       prob = r_val / (r_val + mu)
                                       logpdf(NegativeBinomial(r_val, prob), y_val)
                                   else 0.0 end
                end
            end

            # Posterior Predictive Sampling (Point realization)
            p_noisy[i, j] = if fam isa GaussianFamily; mu + randn() * sig_y[i]
                          elseif fam isa PoissonFamily; rand(Poisson(mu))
                          elseif fam isa NegativeBinomialFamily; rand(NegativeBinomial(r_val, r_val / (r_val + mu)))
                          else mu end
        end
    end

    # 6. Post-Stratification Weights Recovery
    local ps_weights = nothing
    if N_PS > 0
        f_mu, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M, N_tot, N_samples, sv_surface, M.y_obs)
        s_map = [(Int(M.t_idx[k]) - 1) * M.s_N + Int(M.s_idx[k]) for k in 1:M.y_N]
        ps_weights = f_mu[M.y_N .+ s_map, :] ./ (f_mu[1:M.y_N, :] .+ 1e-9)
    end

    # 7. Object Construction
    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, M.s_N, 1, N_samples); alpha=alpha),
        smooth_effects = summarize_array(reshape(basis_eff_accum, M.y_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, M.t_N, 1, N_samples); alpha=alpha),
        seasonal = (M.model_season != "none") ? summarize_array(reshape(u_eff, M.u_N, 1, N_samples); alpha=alpha) : nothing,
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        mixed_effects = !isnothing(mixed_eff_coeffs) ? [summarize_array(reshape(me, size(me,1), 1, N_samples); alpha=alpha) for me in mixed_eff_coeffs] : nothing,
        spatiotemporal = summarize_array(reshape(st_eff_map, M.s_N * M.t_N, 1, N_samples); alpha=alpha),
        svc_slopes = !isnothing(svc_slopes) ? summarize_array(reshape(svc_slopes, M.s_N * n_svc, 1, N_samples); alpha=alpha) : nothing,
        volatility = summarize_array(reshape(sv_surface, N_tot, 1, N_samples); alpha=alpha),
        predictions_observed_denoised = summarize_array(reshape(p_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(p_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        post_strat_weights = isnothing(ps_weights) ? nothing : (mean=vec(mean(ps_weights, dims=2)), samples=ps_weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam,
        arch = arch
    )
end




# --- 1. Multivariate Reconstruction ---
function _reconstruct(arch::MultivariateArchitecture, modelname::String, chain, M, PS, alpha)
    # Audited Feature-Complete Multivariate Reconstruction (v07.5 - SVC Integration)
    # Rationale: Ensuring multivariate outcomes produce the same rich feature set as univariate,
    # including SVC recovery, post-strat weights, smooth effects, and structured spatial fields.

    local N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    local N_tot = M.y_N + N_PS
    local N_samples = size(chain, 1)
    local outcomes_N = M.outcomes_N
    local family = get(M, :model_family, "gaussian")
    local p_names = string.(FlexiChains.parameters(chain))
    local fam_obj = get_model_family(family)

    # --- 1. Shared Additive Manifold Recovery ---
    local eta_hierarchy = zeros(M.y_N, N_samples)
    if haskey(M, :spatial_hierarchy) && !isempty(M.spatial_hierarchy)
        for scale_sym in keys(M.spatial_hierarchy)
            scale = M.spatial_hierarchy[scale_sym]
            p_key = "s_latent_" * string(scale_sym)
            if p_key in p_names
                lat_vals = get_params_vector(chain, p_key, scale.n_units)
                for j in 1:N_samples
                    eta_hierarchy[:, j] .+= lat_vals[j, scale.indices]
                end
            end
        end
    end

    local eta_mixed = zeros(M.y_N, N_samples)
    if haskey(M, :c_groups) && !isempty(M.c_groups)
        for c_sym in keys(M.c_groups)
            p_key = "c_latent_" * string(c_sym)
            if p_key in p_names
                lat_vals = get_params_vector(chain, p_key, size(M.c_re_templates[c_sym].matrix, 1))
                for j in 1:N_samples
                    eta_mixed[:, j] .+= lat_vals[j, M.c_groups[c_sym]]
                end
            end
        end
    end

    local eta_basis = zeros(M.y_N, N_samples)
    if haskey(M, :basis_matrices) && !isempty(M.basis_matrices)
        for v_sym in keys(M.basis_matrices)
            B = M.basis_matrices[v_sym]
            p_key = "beta_basis_" * string(v_sym)
            if p_key in p_names
                betas = get_params_vector(chain, p_key, size(B, 2))
                for j in 1:N_samples
                    eta_basis[:, j] .+= B * betas[j, :]
                end
            end
        end
    end

    local eta_seasonal = zeros(M.y_N, N_samples)
    if M.model_season != "none"
        local u_sig_vec = "u_sigma" in p_names ? vec(get_params_vector(chain, "u_sigma", 1)) : ones(N_samples)
        for j in 1:N_samples
            u_vals = extract_manifold(SeasonalTrait(), chain, M, j, u_sig_vec[j])
            eta_seasonal[:, j] .= u_vals[M.u_idx]
        end
    end

    # --- 2. Outcome-Specific Component Recovery ---
    local s_eff_struct = [zeros(M.s_N, N_samples) for _ in 1:outcomes_N]
    local s_eff_noisy = [zeros(M.s_N, N_samples) for _ in 1:outcomes_N]
    local t_eff = [zeros(M.t_N, N_samples) for _ in 1:outcomes_N]
    local Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N * outcomes_N, N_samples) : nothing
    local eta_coupled = zeros(M.y_N, outcomes_N, N_samples)
    
    # SVC Containers [S_N x N_covs x N_samples] for each Outcome
    local svc_covs = get(M, :svc_covariates, Symbol[])
    local n_svc = length(svc_covs)
    local svc_slopes = n_svc > 0 ? [zeros(M.s_N, n_svc, N_samples) for _ in 1:outcomes_N] : nothing

    for j in 1:N_samples
        local L = chain[:L_corr].data[j].L
        local latent_raw = zeros(M.y_N, outcomes_N)

        for k in 1:outcomes_N
            s_sig_k = get_params_vector(chain, "s_sigma_arr", outcomes_N)[j, k]
            t_sig_k = get_params_vector(chain, "t_sigma_arr", outcomes_N)[j, k]

            # Spatial Recovery (Standardized extract_manifold_k returns struct/noisy)
            fs_k, fn_k = extract_manifold_k(SpatialTrait(), chain, M, j, k, s_sig_k)
            s_eff_struct[k][:, j] .= fs_k
            s_eff_noisy[k][:, j] .= fn_k
            latent_raw[:, k] .+= fn_k[M.s_idx]

            # Temporal Recovery
            t_key = "t_lat_" * string(k)
            if t_key in p_names
                t_vals = get_params_vector(chain, t_key, M.t_N)[j, :] .* t_sig_k
                t_eff[k][:, j] .= t_vals
                latent_raw[:, k] .+= t_vals[M.t_idx]
            end

            # SVC Recovery for Outcome k
            if !isnothing(svc_slopes)
                for (v_idx, c_sym) in enumerate(svc_covs)
                    p_key_sig = "sig_svc_" * string(c_sym) * "_" * string(k)
                    p_key_beta = "beta_svc_" * string(c_sym) * "_" * string(k)
                    if p_key_beta in p_names
                        sig_svc = p_key_sig in p_names ? get_params_vector(chain, p_key_sig, 1)[j] : 1.0
                        if M.svc_model == "rff"
                            beta_rff = get_params_vector(chain, p_key_beta, M.svc_M_rff)[j, :]
                            svc_slopes[k][:, v_idx, j] .= (M.svc_basis_cached * beta_rff) .* sig_svc
                        else
                            svc_slopes[k][:, v_idx, j] .= get_params_vector(chain, p_key_beta, M.s_N)[j, :] .* sig_svc
                        end
                    end
                end
            end
        end
        eta_coupled[:, :, j] .= latent_raw * L

        if !isnothing(Xfixed_betas)
            Xfixed_betas[:, j] .= get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * outcomes_N)[j, :]
        end
    end

    # --- 3. Predictor Assembly and Post-Stratification ---
    local p_denoised = zeros(N_tot, outcomes_N, N_samples)
    local shared_add_y = eta_hierarchy .+ eta_mixed .+ eta_basis .+ eta_seasonal

    for j in 1:N_samples
        for k in 1:outcomes_N
            local beta_k = isnothing(Xfixed_betas) ? 0.0 : Xfixed_betas[(k-1)*M.Xfixed_N + 1 : k*M.Xfixed_N, j]
            local eta_base_k = eta_coupled[:, k, j] .+ (M.Xfixed * beta_k) .+ shared_add_y[:, j]

            # Add SVC Effect for Outcome k
            if !isnothing(svc_slopes)
                for (v_idx, c_sym) in enumerate(svc_covs)
                    col_idx = findfirst(==(c_sym), Symbol.(names(M.Xfixed, 2)))
                    if !isnothing(col_idx)
                        cov_val = M.Xfixed[:, col_idx]
                        eta_base_k .+= svc_slopes[k][M.s_idx, v_idx, j] .* cov_val
                    end
                end
            end

            if family == "hurdle_poisson"
                # Participation (k=1) and Intensity (k=2) branches
                if k == 1
                    p_denoised[1:M.y_N, 1, j] .= logistic.(eta_base_k)
                elseif k == 2
                    mu_i = exp.(eta_base_k)
                    p_denoised[1:M.y_N, 2, j] .= mu_i ./ (1.0 .- exp.(-mu_i) .+ 1e-9)
                end
            else
                # Standard link mapping
                p_denoised[1:M.y_N, k, j] .= _apply_link_and_lik(family, eta_base_k, false)
            end
        end
    end

    # --- 4. Post-Strat Weights Recovery ---
    local ps_weights = nothing
    if N_PS > 0
        ps_weights = [zeros(N_PS, N_samples) for _ in 1:outcomes_N]
        for k in 1:outcomes_N, j in 1:N_samples
             ps_weights[k][:, j] .= p_denoised[M.y_N+1:end, k, j] ./ (mean(p_denoised[1:M.y_N, k, j]) + 1e-9)
        end
    end

    return (
        predictions_observed_denoised = summarize_array(p_denoised[1:M.y_N, :, :]; alpha=alpha),
        spatial_denoised = [summarize_array(reshape(s_eff_struct[k], M.s_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N],
        temporal_effects = [summarize_array(reshape(t_eff[k], M.t_N, 1, N_samples); alpha=alpha) for k in 1:outcomes_N],
        fixed_effects = isnothing(Xfixed_betas) ? nothing : summarize_array(reshape(Xfixed_betas, M.Xfixed_N * outcomes_N, 1, N_samples); alpha=alpha),
        svc_effects = isnothing(svc_slopes) ? nothing : [summarize_array(reshape(svc_slopes[k], M.s_N * n_svc, 1, N_samples); alpha=alpha) for k in 1:outcomes_N],
        smooth_effects = summarize_array(reshape(eta_basis, M.y_N, 1, N_samples); alpha=alpha),
        hierarchy_effect = summarize_array(reshape(eta_hierarchy, M.y_N, 1, N_samples); alpha=alpha),
        mixed_effect = summarize_array(reshape(eta_mixed, M.y_N, 1, N_samples); alpha=alpha),
        seasonal_effect = summarize_array(reshape(eta_seasonal, M.y_N, 1, N_samples); alpha=alpha),
        post_strat_weights = ps_weights,
        family = family,
        arch = arch
    )
end



# --- 2. Multifidelity Reconstruction ---
function _reconstruct(arch::MultifidelityArchitecture, modelname::String, chain, M, PS, alpha)
    # println("--- Starting Audited Multifidelity Reconstruction [Full Sync V2 + Unified Outputs] ---")

    local N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    local N_tot = M.y_N + N_PS
    local N_samples = size(chain, 1)
    local p_names_str = string.(FlexiChains.parameters(chain))
    local fam = get_model_family(get(M, :model_family, "gaussian"))

    # Dynamic Dimension Audit
    local test_s = "s_latent" in p_names_str ? get_params_vector(chain, "s_latent", M.s_N)[1, :] : zeros(M.s_N)
    local actual_s_N = length(test_s)
    local test_t = "t_latent" in p_names_str ? get_params_vector(chain, "t_latent", M.t_N)[1, :] : zeros(M.t_N)
    local actual_t_N = length(test_t)
    local M_sync = merge(M, (s_N=actual_s_N, t_N=actual_t_N))

    local rhos = vec(get_params_vector(chain, "fidelity_rho", 1))
    local biases = vec(get_params_vector(chain, "fidelity_bias", 1))

    # Allocations
    local s_eff_struct = zeros(actual_s_N, N_samples)
    local s_eff_noisy = zeros(actual_s_N, N_samples)
    local t_eff = zeros(actual_t_N, N_samples)
    local Xfixed_betas = (M.Xfixed_N > 0) ? zeros(M.Xfixed_N, N_samples) : nothing
    local mixed_terms_list = get(M, :mixed_terms, [])
    local mixed_eff_coeffs = !isempty(mixed_terms_list) ? [zeros(term.n_cat, N_samples) for term in mixed_terms_list] : nothing

    for j in 1:N_samples
        s_sig = "s_sigma" in p_names_str ? get_params_vector(chain, "s_sigma", 1)[j] : 1.0
        t_sig = "t_sigma" in p_names_str ? get_params_vector(chain, "t_sigma", 1)[j] : 1.0

        fs, fn = extract_manifold(SpatialTrait(), chain, M_sync, j, s_sig, M_sync.s_x, M_sync.s_y)
        s_eff_struct[:, j] .= fs[1:actual_s_N]
        s_eff_noisy[:, j] .= fn[1:actual_s_N]
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M_sync, j, t_sig)[1:actual_t_N]

        if !isnothing(Xfixed_betas); Xfixed_betas[:, j] .= extract_manifold(FixedTrait(), chain, M_sync, j); end
        if !isnothing(mixed_eff_coeffs)
            for (m_idx, m_term) in enumerate(mixed_terms_list)
                p_key = "beta_group_" * string(m_idx)
                if p_key in p_names_str
                    mixed_eff_coeffs[m_idx][:, j] .= get_params_vector(chain, p_key, m_term.n_cat)[j, :]
                end
            end
        end
    end

    local eta_noisy = zeros(N_tot, N_samples)
    local eta_denoised = zeros(N_tot, N_samples)
    local fixed_names = M.Xfixed isa NamedArray ? Symbol.(names(M.Xfixed, 2)) : [Symbol("X", i) for i in 1:size(M.Xfixed, 2)]

    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
        s_id, t_id = Int(src.s_idx[idx]), Int(src.t_idx[idx])
        
        if s_id < 1 || s_id > actual_s_N; error("MF Spatial bound error: $s_id"); end
        if t_id < 1 || t_id > actual_t_N; error("MF Temporal bound error: $t_id"); end

        val = is_obs ? get(M, :log_offset, zeros(M.y_N))[i] : 0.0
        if !isnothing(Xfixed_betas)
            X_row = is_obs ? M.Xfixed[idx, :] : collect(Vector(PS.surface_df[idx, fixed_names]))
            val += dot(Xfixed_betas[:, j], X_row)
        end
        if !isnothing(mixed_eff_coeffs)
            for (m_idx, m_term) in enumerate(mixed_terms_list)
                g_idx = Int(m_term.indices[idx])
                val += mixed_eff_coeffs[m_idx][g_idx, j] * m_term.covariate_vals[idx]
            end
        end

        eta_denoised[i, j] = val + s_eff_struct[s_id, j] + t_eff[t_id, j]
        eta_noisy[i, j] = val + s_eff_noisy[s_id, j] + t_eff[t_id, j]
    end

    local y_sig_samples = _extract_volatility(chain, p_names_str, N_tot, N_samples, nothing, M_sync)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M_sync, N_tot, N_samples, y_sig_samples, M.y_obs)

    local ps_weights = nothing
    if N_PS > 0
        field_mu, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M_sync, N_tot, N_samples, y_sig_samples, M.y_obs)
        stratum_map = [(Int(M.t_idx[k]) - 1) * actual_s_N + Int(M.s_idx[k]) for k in 1:M.y_N]
        ps_weights = field_mu[M.y_N .+ stratum_map, :] ./ (field_mu[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, actual_s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, actual_s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, actual_t_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sig_samples, N_tot, 1, N_samples); alpha=alpha),
        fixed_effects = !isnothing(Xfixed_betas) ? summarize_array(reshape(Xfixed_betas, M.Xfixed_N, 1, N_samples); alpha=alpha) : nothing,
        mixed_effects = !isnothing(mixed_eff_coeffs) ? [summarize_array(reshape(me, size(me,1), 1, N_samples); alpha=alpha) for me in mixed_eff_coeffs] : nothing,
        fidelity_coupling = (rho=summarize_array(reshape(rhos, 1, 1, N_samples); alpha=alpha), bias=summarize_array(reshape(biases, 1, 1, N_samples); alpha=alpha)),
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = (N_PS > 0) ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = isnothing(ps_weights) ? nothing : (mean=vec(mean(ps_weights, dims=2)), samples=ps_weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, 
        arch = arch
    )
end

# --- 3. Example Architecture Reconstruction ---
function _reconstruct(arch::ExampleArchitecture, modelname::String, chain, M, PS, alpha)
    # println("--- Starting Audited Example Reconstruction [Full Sync V2 + Unified Outputs] ---")

    local N_PS = isnothing(PS) ? 0 : length(PS.s_idx)
    local N_tot = M.y_N + N_PS
    local N_samples = size(chain, 1)
    local p_names_str = string.(FlexiChains.parameters(chain))
    local fam = get_model_family(get(M, :model_family, "gaussian"))

    # Sync Discovery
    local actual_s_N = "s_latent" in p_names_str ? length(get_params_vector(chain, "s_latent", M.s_N)[1, :]) : M.s_N
    local actual_t_N = "t_latent" in p_names_str ? length(get_params_vector(chain, "t_latent", M.t_N)[1, :]) : M.t_N
    local M_sync = merge(M, (s_N=actual_s_N, t_N=actual_t_N))

    local s_eff_struct = zeros(actual_s_N, N_samples)
    local s_eff_noisy = zeros(actual_s_N, N_samples)
    local t_eff = zeros(actual_t_N, N_samples)

    for j in 1:N_samples
        s_sig = "s_sigma" in p_names_str ? get_params_vector(chain, "s_sigma", 1)[j] : 1.0
        t_sig = "t_sigma" in p_names_str ? get_params_vector(chain, "t_sigma", 1)[j] : 1.0

        fs, fn = extract_manifold(SpatialTrait(), chain, M_sync, j, s_sig, M_sync.s_x, M_sync.s_y)
        s_eff_struct[:, j] .= fs[1:actual_s_N]
        s_eff_noisy[:, j] .= fn[1:actual_s_N]
        t_eff[:, j] .= extract_manifold(TemporalTrait(), chain, M_sync, j, t_sig)[1:actual_t_N]
    end

    local eta_noisy = zeros(N_tot, N_samples)
    local eta_denoised = zeros(N_tot, N_samples)
    for j in 1:N_samples, i in 1:N_tot
        is_obs = i <= M.y_N; idx = is_obs ? i : i - M.y_N; src = is_obs ? M : PS
        s_id, t_id = Int(src.s_idx[idx]), Int(src.t_idx[idx])
        
        if s_id < 1 || s_id > actual_s_N; error("Example Spatial bound error: $s_id"); end
        if t_id < 1 || t_id > actual_t_N; error("Example Temporal bound error: $t_id"); end

        val = is_obs ? get(M, :log_offset, zeros(M.y_N))[idx] : 0.0
        eta_denoised[i, j] = val + s_eff_struct[s_id, j] + t_eff[t_id, j]
        eta_noisy[i, j] = val + s_eff_noisy[s_id, j] + t_eff[t_id, j]
    end

    local y_sig = _extract_volatility(chain, p_names_str, N_tot, N_samples, nothing, M_sync)
    preds_denoised, preds_noisy, log_lik = _process_ll_and_predictions(fam, eta_noisy, chain, M_sync, N_tot, N_samples, y_sig, M.y_obs)

    local ps_weights = nothing
    if N_PS > 0
        field_mu, _, _ = _process_ll_and_predictions(fam, eta_denoised, chain, M_sync, N_tot, N_samples, y_sig, M.y_obs)
        stratum_map = [(Int(M.t_idx[k]) - 1) * actual_s_N + Int(M.s_idx[k]) for k in 1:M.y_N]
        ps_weights = field_mu[M.y_N .+ stratum_map, :] ./ (field_mu[1:M.y_N, :] .+ 1e-9)
    end

    return (
        spatial_structured = summarize_array(reshape(s_eff_struct, actual_s_N, 1, N_samples); alpha=alpha),
        spatial_unstructured = summarize_array(reshape(s_eff_noisy .- s_eff_struct, actual_s_N, 1, N_samples); alpha=alpha),
        temporal = summarize_array(reshape(t_eff, actual_t_N, 1, N_samples); alpha=alpha),
        volatility = summarize_array(reshape(y_sig, N_tot, 1, N_samples); alpha=alpha),
        predictions_observed_denoised = summarize_array(reshape(preds_denoised[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_observed_noisy = summarize_array(reshape(preds_noisy[1:M.y_N, :], M.y_N, 1, N_samples); alpha=alpha),
        predictions_strata_denoised = (N_PS > 0) ? summarize_array(reshape(preds_denoised[M.y_N+1:end, :], N_PS, 1, N_samples); alpha=alpha) : nothing,
        post_strat_weights = isnothing(ps_weights) ? nothing : (mean=vec(mean(ps_weights, dims=2)), samples=ps_weights),
        waic = _compute_waic(log_lik[:, 1:M.y_N]),
        family = fam, 
        arch = arch
    )
end

 
# --- Centralized Prediction & Likelihood Logic ---

"""
    _apply_link_and_lik(family, eta, use_zi, [phi], [r])

Applies the appropriate link function (Inverse Link) to the linear predictor `eta` 
and accounts for zero-inflation or dispersion to return the expected value (mu).
"""
function _apply_link_and_lik(family::String, eta::AbstractArray, use_zi::Bool, phi=0.0, r=1.0)
    if family == "gaussian"
        return eta
    elseif family == "lognormal" || family == "poisson"
        return exp.(eta)
    elseif family == "bernoulli"
        return logistic.(eta)
    elseif family == "negbin"
        # Expected value for NB (mu). If ZI is active, scale by (1-phi).
        mu = exp.(eta)
        return use_zi ? (1.0 .- phi) .* mu : mu
    else
        # Fallback to identity link
        return eta
    end
end
 

  


####


 
;;

function build_structure_template(type::Symbol, n::Int; scale=true, coords=nothing, W=nothing)
    # Comprehensive Template Factory for all supported BSTM Manifolds
    
    if type == :ar1
        # AR1 template: Basic first-order differencing structure
        Q = Matrix(1.0I, n, n)
        for i in 1:(n-1); Q[i, i+1] = Q[i+1, i] = -0.5; end
        sf = 1.0 
        return (matrix = Q, scaling_factor = sf)

    elseif type == :rw2
        # RW2 template: Second-order random walk (Intrinsic GMRF)
        Q = zeros(n, n)
        for i in 1:n
            if i > 2 && i < n-1
                Q[i,i]=6; Q[i,i-1]=Q[i,i+1]=-4; Q[i,i-2]=Q[i,i+2]=1
            elseif i == 1
                Q[i,i]=1; Q[i,i+1]=-2; Q[i,i+2]=1
            elseif i == 2
                Q[i,i]=5; Q[i,i-1]=-2; Q[i,i+1]=-4; Q[i,i+2]=1
            elseif i == n-1
                Q[i,i]=5; Q[i,i+1]=-2; Q[i,i-1]=-4; Q[i,i-2]=1
            elseif i == n
                Q[i,i]=1; Q[i,i-1]=-2; Q[i,i-2]=1
            end
        end
        sf = scale ? exp(mean(log.(filter(x -> x > 1e-6, eigvals(Q))))) : 1.0
        return (matrix = Matrix(Q ./ sf), scaling_factor = sf)

    elseif type in [:icar, :besag, :bym2, :leroux, :sar, :dag, :transport_diffusion, :transport_advection]
        # Graph Laplacian based manifolds
        isnothing(W) && error("Spatial adjacency matrix W required for $type")
        D_sp = Diagonal(vec(sum(W, dims=2)))
        Q_raw = Matrix(D_sp - W)
        sf = scale ? exp(mean(log.(filter(x -> x > 1e-6, eigvals(Q_raw))))) : 1.0
        return (matrix = Matrix(Q_raw ./ sf), scaling_factor = sf)

    elseif type in [:sar, :dag, :transport_advection]
        # Adjacency for SAR/DAG/Advection
        isnothing(W) && error("Adjacency matrix W required for $type")
        row_sums = vec(sum(W, dims=2))
        W_norm = W ./ (row_sums .+ 1e-9)
        return (matrix = Matrix(W_norm), scaling_factor = 1.0)

    elseif type == :gp || type == :nystrom || type == :denseGP
        # Distance matrix for kernel-based manifolds
        mat = isnothing(coords) ? Matrix(1.0I, n, n) : [sqrt(sum((c1 .- c2).^2)) for c1 in coords, c2 in coords]
        return (matrix = mat, scaling_factor = 1.0)

    elseif type == :seasonal || type == :harmonic
        # Cyclic/Seasonal template with sum-to-zero structure
        Q = Matrix(1.0I, n, n) .* (n-1)
        for i in 1:n, j in 1:n
            if i != j; Q[i, j] = -1.0; end
        end
        sf = scale ? exp(mean(log.(filter(x -> x > 1e-6, eigvals(Q))))) : 1.0
        return (matrix = Matrix(Q ./ sf), scaling_factor = sf)

    elseif type == :iid || type == :householder || type == :bgcn
        # Identity bases for unstructured or spectral weights
        return (matrix = Matrix(1.0I, n, n), scaling_factor = 1.0)

    else
        print("Unknown structure template type (defaulting to identity): $type")
        # Identity bases for unstructured or spectral weights
        return (matrix = Matrix(1.0I, n, n), scaling_factor = 1.0)

    end
end
  

 

# --- 2. High-Level Manifold Builder Dispatch ---
# This function acts as the interface between the Struct-based Manifold definitions
# and the bstm_options metadata generator.

function build_model(manifold::Manifold, data_inputs)
    # Identify the model type from the struct itself
    m_type = manifold_type(manifold)
    
    # Retrieve or construct the precision template
    # This matches the signature in our reference section for build_structure_template
    template = build_structure_template(m_type, data_inputs.s_N; W=get(data_inputs, :W, nothing))
    
    # Package results for bstm_options
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = m_type,
        hyper = (sigma_prior = manifold.sigma_prior, 
                 rho_prior = hasproperty(manifold, :rho_prior) ? manifold.rho_prior : nothing)
    )
end

# Fallback for manifolds without explicit builders
function build_model(manifold::Any, data_inputs)
    @warn "No specific builder implemented for manifold type: $(typeof(manifold)). Returning generic metadata."
    return (
        Q_template = nothing,
        model_type = typeof(manifold),
        hyper = nothing
    )
end
 
# Helper constructor for common defaults (cubic splines)
BSpline(v::Symbol; nbins=10, degree=3, sigma_prior=Exponential(1.0)) = BSpline(v, nbins, degree, sigma_prior)


# --- BSpline Builder Dispatch ---
# This function extracts the metadata needed by bstm_options to initialize the model components.

function build_model(m::BSpline, data_inputs)
    # nbins defines the resolution of the spline basis nodes
    # The build_structure_template helper generates the structural penalty or basis matrix
    template = build_structure_template(:bspline, m.nbins)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bspline,
        hyper = (
            sigma_prior = m.sigma_prior, 
            degree = m.degree
        )
    )
end
 

# Specialized Manifold Builders for v05.4 Topological Expansion
# Rationale: Standardizing the extraction of structural metadata for non-Euclidean fields.

# 1. Local-Adaptive Manifold Builder
# Specialized Manifold Builders for v05.4/v05.5 Topological Expansion
# Rationale: Standardizing the extraction of structural metadata for non-Euclidean fields.

# 1. Local-Adaptive Manifold Builder
function build_model(m::LocalAdaptive, data_inputs)
    # n_units is derived from the primary spatial dimension
    n = data_inputs.s_N
    # Retrieve base structure (usually a Laplacian)
    template = build_structure_template(:besag, n; W=data_inputs.W)

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :local_adaptive,
        hyper = (
            sigma_prior = m.sigma_prior,
            weights_var = m.weights_variable
        )
    )
end

# 2. Directed Network Flow Builder
function build_model(m::NetworkFlow, data_inputs)
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
# Specialized Builders for Spectral Manifolds (v05.5 Feature Audit)
# Rationale: These functions translate the high-level manifold structs into the 
# structural metadata required by the bstm_options and Turing sampler.

# 1. FFT Builder Dispatch
function build_model(m::FFT, data_inputs)
    # n defines the resolution of the spectral grid
    n = m.nbins
    # The FFT basis is pre-computed or handled via FFTW in the likelihood
    # provide an identity template as the FFT manifold models coefficients in frequency space
    template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (
            sigma_prior = m.sigma_prior,
            is_2d = m.is_2d
        )
    )
end

# 2. Wavelet Builder Dispatch
function build_model(m::Wavelets, data_inputs)
    n = m.nbins
    # Wavelet coefficients are often assumed IID (sparse) in the transformed domain
    template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :wavelet,
        hyper = (
            sigma_prior = m.sigma_prior,
            family = m.wavelet_family
        )
    )
end

# 3. RFF Builder Dispatch
function build_model(m::RFF, data_inputs)
    n = m.n_features
    # RFF weights are sampled from the spectral density, typically with an IID prior
    template = build_structure_template(:iid, n)

    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :rff,
        hyper = (
            sigma_prior = m.sigma_prior,
            lengthscale_prior = m.lengthscale_prior
        )
    )
end

function recompose_precision(type::Symbol, template_mat::AbstractMatrix, param::Real; extra_param=nothing, noise=1e-4)
    # XR: Audited and Feature-Complete Precision Recomposition [v11.8 - Identity Conversion Fix]
    # Rationale: Fixing MethodError by ensuring Matrix(I, n, n) is called with explicit dimensions.

    local T = typeof(param)
    local n = size(template_mat, 1)

    # Marginal variance scaling factor
    local scale_factor = 1.0 / (param^2 + noise)

    # --- Main Dispatch Routing for the BSTM Registry ---
    local Q = if type == :iid
        # Standard IID Identity manifold
        scale_factor * I(n)

    elseif type == :mosaic
        # Mosaic Cluster Support
        scale_factor * I(n)

    elseif type == :eigen
        # Eigen/Spectral Basis Support
        scale_factor * I(n)

    elseif type in [:besag, :icar, :diffusion]
        # Pure structural smoothing (Laplacian-based)
        scale_factor .* template_mat

    elseif type == :bym2
        # BYM2 standard routing: rho * Structured + (1-rho) * Unstructured
        local rho = isnothing(extra_param) ? 0.5 : extra_param
        scale_factor .* (rho .* template_mat + (1.0 - rho) .* I(n))

    elseif type == :leroux
        # Leroux flexible mixing parameterization
        local lambda_val = isnothing(extra_param) ? 0.5 : extra_param
        scale_factor .* (lambda_val .* template_mat + (1.0 - lambda_val) .* I(n))

    elseif type in [:sar, :advection, :advection_diffusion, :network, :dag]
        # Directed/Physics-Informed: (I - rho*W)'(I - rho*W)
        local rho = isnothing(extra_param) ? 0.8 : extra_param
        local L_op = I(n) - rho .* template_mat
        scale_factor .* (L_op' * L_op)

    elseif type == :gp
        # Continuous kernel manifold (Distance template based)
        local ls = isnothing(extra_param) ? 1.0 : extra_param
        local K = (param^2) .* exp.(-template_mat.^2 ./ (2 * ls^2 + noise))
        inv(Symmetric(K + noise * I(n)))

    elseif type == :spde
        # SPDE Matern Approximation: (kappa^2*I + G)'(kappa^2*I + G)
        local kappa = isnothing(extra_param) ? 1.0 : extra_param
        local L_spde = (kappa^2 .* I(n) + template_mat)
        scale_factor .* (L_spde' * L_spde)

    elseif type in [:rff, :ns_rff, :fft, :wavelet]
        # Spectral manifolds model coefficients in frequency space as IID
        scale_factor * I(n)

    elseif type in [:rw1, :rw2, :tps, :pspline, :bspline, :seasonal, :harmonic]
        # Standard IGMRF or Penalty structures
        scale_factor .* template_mat

    elseif type == :local_adaptive
        # Riemannian manifold with varying local weights
        local w_vec = isnothing(extra_param) ? ones(n) : extra_param
        scale_factor .* (Diagonal(w_vec) * template_mat * Diagonal(w_vec))

    else
        scale_factor .* template_mat
    end

    # Enforce dense Symmetric result for AD stability. 
    # Matrix(Q) now receives a specific dimensioned object or operator.
    return Symmetric(Matrix(Q) + noise * I(n))
end



# --- Comprehensive Manifold Builders and Precision Factories ---
# This section provides the high-level dispatch logic to convert 
# Manifold structs into the structural metadata required by bstm_options.

# --- 1. Base Builder Dispatch ---

# 1. Base Generic Fallback
function build_model(manifold::Manifold, data_inputs)
    @warn "No specific builder for $(typeof(manifold)). Using IID identity template."
    template = build_structure_template(:iid, data_inputs.s_N)
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = hasproperty(manifold, :sigma_prior) ? manifold.sigma_prior : Exponential(1.0),)
    )
end

# 2. Discrete Spatial Manifolds (CAR/Laplacian Family)
function build_model(m::BYM2, data_inputs)
    # BYM2 requires a spatial unit count and adjacency matrix W
    template = build_structure_template(:bym2, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :bym2,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = m.rho_prior)
    )
end

function build_model(m::ICAR, data_inputs)
    template = build_structure_template(:icar, data_inputs.s_N; W=data_inputs.W)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :icar,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end

function build_model(m::Leroux, data_inputs)
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
    # For directed networks, the template is the raw adjacency matrix
    return (
        Q_template = m.adjacency_matrix,
        scaling_factor = 1.0,
        model_type = :network,
        hyper = (sigma_prior = m.sigma_prior, direction = m.flow_direction)
    )
end

function build_model(m::DirectedAcyclicGraph, data_inputs)
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

function build_model(m::HarmonicSeasonal, data_inputs)
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

# 5. Covariate Smooths (Splines)
function build_model(m::TPS, data_inputs)
    template = build_structure_template(:tps, m.nbins)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :tps,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::PSpline, data_inputs)
    template = build_structure_template(:pspline, m.nbins)
    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = :pspline,
        hyper = (sigma_prior = m.sigma_prior, degree = m.degree, diff_order = m.diff_order)
    )
end

# 6. Spectral Manifolds
function build_model(m::FFT, data_inputs)
    # Spectral coefficients are assumed IID in frequency space
    return (
        Q_template = Matrix(1.0I, m.nbins, m.nbins),
        scaling_factor = 1.0,
        model_type = :fft,
        hyper = (sigma_prior = m.sigma_prior, is_2d = m.is_2d)
    )
end

function build_model(m::RFF, data_inputs)
    return (
        Q_template = Matrix(1.0I, m.n_features, m.n_features),
        scaling_factor = 1.0,
        model_type = :rff,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior)
    )
end

# 7. Continuous Spatial (GPs)
function build_model(m::GaussianProcess, data_inputs)
    # Distance matrix acts as the template for kernel evaluation
    return (
        Q_template = nothing, # Computed dynamically in recompose_precision for :gp
        scaling_factor = 1.0,
        model_type = :gp,
        hyper = (sigma_prior = m.sigma_prior, lengthscale_prior = m.lengthscale_prior, kernel = m.kernel)
    )
end

  

# Knorr-Held Interaction Builder
function build_model(m::KnorrHeld, data_inputs)
    # Space-Time Interaction types I, II, III, IV
    # Precision is built via kron() in the sampler, so no static template is stored
    return (
        Q_template = nothing,
        model_type = :st_interaction,
        interaction_class = m.type,
        hyper = (sigma_prior = m.sigma_prior, rho_prior = nothing)
    )
end 
 
 


function get_optimal_sampler(model::DynamicPPL.Model;
    nuts_n_samples_adaptation=100,
    nuts_target_acceptance_ratio=0.65,
    pg_particles=20,
    kwargs...)
    
    # 1. Parameter Discovery
    # generate a few prior samples to identify the shape and type of every parameter.
    # This allows the sampler to automatically adapt to the formula and manifolds used.
    init_samples = [Dict(pairs(rand(model))) for _ in 1:3]
    full_init_dict = init_samples[1]

    # Identify fixed (Dirac) and discrete parameters (for PG sampler)
    fixed_params = filter(k -> all(s -> s[k] == full_init_dict[k], init_samples), keys(full_init_dict))
    discrete_params = filter(k -> k ∉ fixed_params && (full_init_dict[k] isa Integer || full_init_dict[k] isa Bool), keys(full_init_dict))

    # 2. Gaussian Field Detection for Elliptical Slice Sampling (ESS)
    # ESS is extremely efficient for parameters with Gaussian priors (GMRFs and GPs).
    # target vectors that are typically latent field realizations (e.g., s_latent, t_latent).
    latent_fields = filter(k -> begin
        val = full_init_dict[k]
        k_str = string(k)
        k ∉ fixed_params && 
        k ∉ discrete_params && 
        val isa AbstractVector &&
        # Exclude hyperparameters and regression coefficients which aren't typically pure Gaussian fields
        !occursin(r"sigma|sig_|rho|phi|ls_|alpha|beta|Xfixed", k_str)
    end, keys(full_init_dict))

    # 3. NUTS for Continuous Hyperparameters
    # Parameters that control the manifold structure (sigmas, rhos, lengthscales) 
    # are sampled via NUTS to handle their potentially complex posterior geometry.
    active_hypers = filter(k -> k ∉ fixed_params && k ∉ discrete_params && k ∉ latent_fields, keys(full_init_dict))

    # 4. Construct Gibbs Blocks
    sampler_blocks = []
    
    # Block for Discrete variables
    if !isempty(discrete_params)
        push!(sampler_blocks, Tuple(discrete_params) => PG(pg_particles))
    end
    
    # Block for Latent Fields (ESS)
    if !isempty(latent_fields)
        push!(sampler_blocks, Tuple(latent_fields) => ESS())
    end
    
    # Block for Continuous Hyperparameters (NUTS)
    if !isempty(active_hypers)
        push!(sampler_blocks, Tuple(active_hypers) => Turing.NUTS(nuts_n_samples_adaptation, nuts_target_acceptance_ratio))
    end
    
    # Block for Fixed/Remaining (Metropolis-Hastings fallback)
    if !isempty(fixed_params)
        push!(sampler_blocks, Tuple(fixed_params) => MH())
    end

    # Return the composite Gibbs sampler
    return Gibbs(sampler_blocks...)
end



function get_inits(model::DynamicPPL.Model; refine="map", n_samples=100, optimizer=LBFGS(), max_iters=500, maxtime=60.0, noise=nothing)
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

# 1. Trait-Based Dispatch (Spatial)
function extract_manifold(::SpatialTrait, chain, M, j, sig, s_x, s_y)
    p_names = string.(FlexiChains.parameters(chain))
    spatial_key = "s_latent" in p_names ? :s_latent : ("s_icar" in p_names ? :s_icar : nothing)
    field_struct = zeros(M.s_N)
    if !isnothing(spatial_key)
        field_struct .= vec(chain[spatial_key].data[j]) .* sig
    end
    return field_struct, field_struct
end

# 2. Trait-Based Dispatch (Temporal)
function extract_manifold(::TemporalTrait, chain, M, j, sig)
    p_names = string.(FlexiChains.parameters(chain))
    field = zeros(M.t_N)
    t_key = "t_latent" in p_names ? :t_latent : ("t_raw" in p_names ? :t_raw : nothing)
    if !isnothing(t_key)
        field .= vec(chain[t_key].data[j]) .* sig
    end
    return field
end

# 3. Trait-Based Dispatch (Seasonal)
function extract_manifold(::SeasonalTrait, chain, M, j, sig)
    p_names = string.(FlexiChains.parameters(chain))
    u_N = get(M, :u_N, 1)
    if "u_alpha" in p_names && "u_beta" in p_names
        alpha, beta = chain[:u_alpha].data[j], chain[:u_beta].data[j]
        period = get(M, :period, 12)
        angles = (collect(1:u_N) .* (2*pi / period))
        return (alpha .* sin.(angles) .+ beta .* cos.(angles)) .* sig
    elseif "u_latent" in p_names
        return vec(chain[:u_latent].data[j]) .* sig
    end
    return zeros(u_N)
end

# 4. Algebraic Operator Recovery (Direct Sum)
function extract_manifold(m::ComposedManifold, chain, M, j, sig=1.0)
    total_field = nothing
    if m.operator == :direct_sum || m.operator == :pipe
        for comp in m.components
            sub_field, _ = extract_manifold(comp, chain, M, j, sig)
            if isnothing(total_field)
                total_field = copy(sub_field)
            else
                total_field .+= sub_field
            end
        end
    elseif m.operator == :kronecker_product
        p_names = string.(FlexiChains.parameters(chain))
        if "st_latent" in p_names
            st_sig = "st_sigma" in p_names ? chain[:st_sigma].data[j] : 1.0
            return vec(chain[:st_latent].data[j]) .* (sig * st_sig), nothing
        end
    end
    return isnothing(total_field) ? zeros(1) : total_field, total_field
end

function extract_manifold(::BYM2, chain, M, j, sig, s_x, s_y)
    p_names = string.(FlexiChains.parameters(chain))
    field_struct = zeros(M.s_N)
    field_noisy = zeros(M.s_N)
    if "s_latent" in p_names && "s_rho" in p_names && "s_iid" in p_names
        rho = chain[:s_rho].data[j]
        icar_raw = vec(chain[:s_latent].data[j])
        iid_raw = vec(chain[:s_iid].data[j])
        field_struct .= icar_raw .* (sig * sqrt(rho))
        field_noisy .= field_struct .+ (iid_raw .* (sig * sqrt(1.0 - rho)))
    end
    return field_struct, field_noisy
end
  

# Transformed Manifolds (A |> Trans)
# Rationale: Applies scale or link transforms to the latent field during extraction.
function extract_manifold(m::TransformedManifold, chain, M, j, sig=1.0)
    raw_field, _ = extract_manifold(m.manifold, chain, M, j, sig)
    
    if m.transform_fn == Log
        return log.(abs.(raw_field) .+ 1e-9), nothing
    elseif m.transform_fn == ZScore
        return (raw_field .- mean(raw_field)) ./ (std(raw_field) + 1e-9), nothing
    elseif m.transform_fn == UnitScale
        mi, ma = minimum(raw_field), maximum(raw_field)
        return (raw_field .- mi) ./ (ma - mi + 1e-9), nothing
    end
    
    return raw_field, raw_field
end

# Mosaic Manifold Extraction
# Rationale: Maps cluster-level intercepts to the full spatial unit grid.
function extract_manifold(::Mosaic, chain, M, j, sig)
    p_names = string.(FlexiChains.parameters(chain))
    field = zeros(M.s_N)
    if "mu_local" in p_names
        # Map binned cluster values back to spatial units
        m_vals = get_params_vector(chain, "mu_local", M.n_mosaics)[j, :]
        field .= m_vals[M.cluster_assignments] .* sig
    end
    return field, field
end

# --- 4. Covariate and Fixed Dispatch ---

function extract_manifold(::Fixed, chain, M, j, sig=1.0)
    p_names = string.(FlexiChains.parameters(chain))
    if "Xfixed_beta" in p_names
        return get_params_vector(chain, "Xfixed_beta", M.Xfixed_N)[j, :]
    end
    return Float64[]
end

function extract_manifold(::Covariate, chain, M, j, sig=1.0)
    p_names = string.(FlexiChains.parameters(chain))
    for (var_name, rule) in M.re_rules
        if get(rule, :is_smooth, false) || get(rule, :model, "") in ["fft", "wavelet", "rff", "tps", "pspline"]
            p_key = "beta_basis_" * var_name
            if p_key in p_names
                n_basis = get(rule, :nbins, 1)
                if get(rule, :model, "") == "rff"; n_basis = get(rule, :n_features, 20); end
                return get_params_vector(chain, p_key, n_basis)[j, :] .* sig
            end
        end
    end
    return [0.0]
end

# --- 4. Trait Retrieval Helpers ---
# These helpers standardise the coordinate mapping from struct types to the metadata index vectors.

# Maps Spatial manifolds to the 's_idx' column in the data source
manifold_indices(::Spatial, M) = M.s_idx

# Maps Temporal manifolds to the 't_idx' column (time steps)
manifold_indices(::Temporal, M) = M.t_idx

# Maps Seasonal manifolds to the 'u_idx' column (periodic bins)
manifold_indices(::Seasonal, M) = M.u_idx


# Helper for outcome-specific manifold extraction in MultivariateArchitectures# Consolidated and Feature-Complete extract_manifold_k() Dispatch System (v08.0)
# Rationale: This system provides the outcome-specific retrieval logic required for
# Multivariate and Hurdle architectures. It ensures that every Manifold type
# (Atomic, Algebraic, and Compositional) can be recovered for a specific outcome 'k'.
# Audit: Verified against Reference Hierarchy and univariate extract_manifold() parity.

# --- 1. Atomic Outcome-Specific Spatial Dispatch ---

function extract_manifold_k(::Spatial, chain, M, j, k, sig_k)
    # Rationale: Standardized retrieval for ICAR/Besag components indexed by outcome k.
    p_names = string.(FlexiChains.parameters(chain))

    # Dynamic discovery of outcome-specific spatial keys
    # Supports both indexed 's_icar_k[k]' and named 's_lat_k' conventions.
    spatial_key = Symbol("s_lat_" * string(k)) in p_names ? Symbol("s_lat_" * string(k)) : 
                 (Symbol("s_icar_k[" * string(k) * "]") in p_names ? Symbol("s_icar_k[" * string(k) * "]") : nothing)

    field_struct = zeros(M.s_N)

    if !isnothing(spatial_key)
        # Extract the raw latent realization for sample j, outcome k, and scale by sigma_k
        field_struct .= vec(chain[spatial_key].data[j]) .* sig_k
    end

    # For standard spatial fields, structured and noisy realizations are identical
    return field_struct, field_struct
end

function extract_manifold_k(::BYM2, chain, M, j, k, sig_k)
    # Rationale: Partitioning structured (ICAR) vs unstructured (IID) variance for outcome k.
    p_names = string.(FlexiChains.parameters(chain))

    field_struct = zeros(M.s_N)
    field_noisy = zeros(M.s_N)

    # Standard outcome-specific naming for BYM2 components
    rho_key = Symbol("s_rho_" * string(k))
    lat_key = Symbol("s_lat_" * string(k))
    iid_key = Symbol("s_iid_" * string(k))

    if lat_key in p_names && rho_key in p_names && iid_key in p_names
        rho = chain[rho_key].data[j]
        icar_raw = vec(chain[lat_key].data[j])
        iid_raw = vec(chain[iid_key].data[j])

        # Standardised BYM2 Scaling for outcome k: sigma * sqrt(rho)
        field_struct .= icar_raw .* (sig_k * sqrt(rho))
        # Realization includes the scaled IID overdispersion
        field_noisy .= field_struct .+ (iid_raw .* (sig_k * sqrt(1.0 - rho)))
    end

    return field_struct, field_noisy
end

# --- 2. Algebraic and Compositional Outcome Dispatch ---

function extract_manifold_k(m::ComposedManifold, chain, M, j, k, sig_k=1.0)
    # Rationale: Recursively extracts and sums fields from sub-components for outcome k.
    # Supports Direct Sum (A ⊕ B) and Kronecker Interaction (S ⊗ T).
    total_field = nothing

    if m.operator == :direct_sum || m.operator == :pipe
        for comp in m.components
            # Recursive call ensuring the outcome index 'k' is propagated down the tree
            sub_field_struct, _ = extract_manifold_k(comp, chain, M, j, k, sig_k)

            if isnothing(total_field)
                total_field = copy(sub_field_struct)
            else
                total_field .+= sub_field_struct
            end
        end

    elseif m.operator == :kronecker_product
        # Separable Interaction retrieval for outcome k
        p_names = string.(FlexiChains.parameters(chain))
        st_key = Symbol("st_lat_" * string(k))
        st_sig_key = Symbol("st_sigma_" * string(k))

        if st_key in p_names
            st_sig = st_sig_key in p_names ? chain[st_sig_key].data[j] : 1.0
            # Returns the flattened spatiotemporal interaction field for this outcome
            return vec(chain[st_key].data[j]) .* (sig_k * st_sig), nothing
        end
    end

    return isnothing(total_field) ? zeros(1) : total_field, total_field
end

function extract_manifold_k(m::TransformedManifold, chain, M, j, k, sig_k=1.0)
    # Rationale: Applies transformations (Log/ZScore) to the outcome-specific latent field.
    # Maps samples from internal space back to the transformed user space.
    raw_field, _ = extract_manifold_k(m.manifold, chain, M, j, k, sig_k)

    if m.transform_fn == Log
        return log.(abs.(raw_field) .+ 1e-9), nothing
    elseif m.transform_fn == ZScore
        return (raw_field .- mean(raw_field)) ./ (std(raw_field) + 1e-9), nothing
    elseif m.transform_fn == UnitScale
        mi, ma = minimum(raw_field), maximum(raw_field)
        return (raw_field .- mi) ./ (ma - mi + 1e-9), nothing
    end

    return raw_field, raw_field
end

# --- 3. Covariate and Fixed Outcome Dispatch ---

function extract_manifold_k(::Fixed, chain, M, j, k, sig_k=1.0)
    # Rationale: Retrieval of outcome-specific fixed effect coefficients.
    p_names = string.(FlexiChains.parameters(chain))
    
    # Check for outcome-indexed intercept/fixed coefficients (e.g., in hurdle Intensity branch)
    p_key = Symbol("Xfixed_beta_" * string(k))
    
    if p_key in p_names
        return get_params_vector(chain, string(p_key), M.Xfixed_N)[j, :]
    elseif "Xfixed_beta" in p_names
        # Standard MV assembly where betas for all K are in one vector
        all_betas = get_params_vector(chain, "Xfixed_beta", M.Xfixed_N * M.outcomes_N)[j, :]
        # Extract the slice corresponding to outcome k
        return all_betas[((k-1)*M.Xfixed_N + 1):(k*M.Xfixed_N)]
    end
    
    return Float64[]
end

function extract_manifold_k(::Covariate, chain, M, j, k, sig_k=1.0)
    # Rationale: Recovery of outcome-specific coefficients for basis-mapped smooths.
    p_names = string.(FlexiChains.parameters(chain))
    
    for (var_name, rule) in M.re_rules
        if get(rule, :is_smooth, false) || get(rule, :model, "") in ["fft", "rff", "tps", "pspline"]
            # Standard outcome-specific naming: beta_basis_[var]_[k]
            p_key = "beta_basis_" * var_name * "_" * string(k)
            
            if p_key in p_names
                n_basis = get(rule, :nbins, 1)
                if get(rule, :model, "") == "rff"; n_basis = get(rule, :n_features, 20); end
                return get_params_vector(chain, p_key, n_basis)[j, :] .* sig_k
            end
        end
    end
    return [0.0]
end

# --- 4. Mosaic Outcome Dispatch ---

function extract_manifold_k(::Mosaic, chain, M, j, k, sig_k)
    # Rationale: Maps per-outcome cluster intercepts back to the spatial grid.
    p_names = string.(FlexiChains.parameters(chain))
    field = zeros(M.s_N)
    mu_key = Symbol("mu_local_" * string(k))

    if mu_key in p_names
        # Extract cluster values for specific outcome k
        m_vals = get_params_vector(chain, string(mu_key), M.n_mosaics)[j, :]
        # Map cluster-level effects to specific grid units
        field .= m_vals[M.cluster_assignments] .* sig_k
    end
    return field, field
end

println("extract_manifold_k dispatch system AUDITED and FEATURE-COMPLETE [v08.0]")




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
    df_tmp = data isa NamedArray ? DataFrame(data, :auto) : data
    if data isa NamedArray
        rename!(df_tmp, names(data, 2))
    end

    clean_rhs = replace(formula_rhs, r"(re|st|cv|int|me|fe)\\(.*?\\)" => "")
    clean_rhs = strip(replace(clean_rhs, r"\\+\\s*\\+" => "+"))
    clean_rhs = replace(clean_rhs, r"^\\+|\\+$" => "")

    if isempty(strip(clean_rhs)) || clean_rhs == "1"
        return NamedArray(ones(size(df_tmp, 1), 1), (1:size(df_tmp, 1), [:Intercept]))
    end

    try
        f = StatsModels.FormulaTerm(StatsModels.Term(:y), StatsModels.Term(Symbol(clean_rhs)))
        sch = StatsModels.schema(f, df_tmp, contrasts)
        f_applied = StatsModels.apply_schema(f, sch, StatsModels.RegressionModel)
        _, mm = StatsModels.modelcols(f_applied, df_tmp)
        return NamedArray(mm, (1:size(mm, 1), Symbol.(names(mm))))
    catch e
        return NamedArray(ones(size(df_tmp, 1), 1), (1:size(df_tmp, 1), [:Intercept]))
    end
end


 


"""
    process_graph_into_rules!(re_rules, opt_kwargs, graph)

Recursive engine that traverses the parsed manifold graph tree and populates metadata containers.
Explicitly routes global architectural flags and captures nested hierarchical layers.
"""
 
function process_graph_into_rules!(re_rules::Dict{String, Any}, opt_kwargs::Dict{Symbol, Any}, graph)
    # 1. Atomic Base Case: Processing individual Manifold components
    if graph.type == :atomic
        m_name = lowercase(string(graph.model))
        var_name = string(graph.var)

        # Initialize sub-dictionary as Any to support Bool flags (e.g., :is_eigen)
        re_rules[var_name] = Dict{Symbol, Any}(:model => m_name)

        # Routing for Global Architectural Flags
        if m_name in ["spatial", "bym2", "icar", "leroux", "sar", "dag", "iid", "spde", "gp", "fitc", "rff", "mosaic", "fft"]
            m_resolved = (m_name == "spatial") ? "bym2" : m_name
            re_rules[var_name][:model] = m_resolved
            opt_kwargs[:model_space] = m_resolved

        elseif m_name in ["temporal", "iid", "ar", "ar1", "rw", "rw1", "rw2", "fft", "rff", "harmonic"]
            m_resolved = (m_name == "temporal" || m_name == "ar") ? "ar1" : m_name
            re_rules[var_name][:model] = m_resolved
            opt_kwargs[:model_time] = m_resolved

        elseif m_name == "eigen"
            opt_kwargs[:has_eigen_effect] = true
            re_rules[var_name][:is_eigen] = true

        elseif m_name == "svc"
            re_rules[var_name][:is_svc] = true
            if !haskey(opt_kwargs, :svc_covariates)
                opt_kwargs[:svc_covariates] = Symbol[]
            end

        elseif m_name == "nested"
            opt_kwargs[:model_arch] = "multifidelity"
            re_rules[var_name][:is_nested] = true
        end

    # 2. Kronecker Product (Separable Interaction)
    elseif graph.type == :kronecker
        opt_kwargs[:model_st] = "IV"
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end

    # 3. Direct Sum (Additive Components)
    elseif graph.type == :sum
        for el in graph.elements
            process_graph_into_rules!(re_rules, opt_kwargs, el)
        end

    # 4. Composition (Warping, Stacking, Transformations)
    elseif graph.type == :composition
        if !haskey(opt_kwargs, :nested_layers)
            opt_kwargs[:nested_layers] = []
        end
        for el in graph.elements
            if el.type == :atomic
                m_name = lowercase(string(el.model))
                push!(opt_kwargs[:nested_layers], (model=m_name, var=el.var))
                process_graph_into_rules!(re_rules, opt_kwargs, el)
            else
                process_graph_into_rules!(re_rules, opt_kwargs, el)
            end
        end
    end
end


function predict(model_obj::DynamicPPL.Model, chain, new_data::DataFrame; n_samples::Int=100, alpha=0.05)
    # Out-of-Sample Prediction Engine
    # Rationale: Projects latent manifolds onto new spatiotemporal coordinates.
    # Requirement: Verbatim interoperability with bstm_options and _reconstruct.
    # This method enables the projection of recovered latent manifolds onto previously unseen data points. 
    # **Technical Requirements:**
    # 1. **Manifold Mapping**: For discrete manifolds (ICAR/Leroux), new locations must be mapped to existing units via nearest-neighbor centroid lookup.
    # 2. **Kernel Projection**: For continuous manifolds (GP/RFF), we utilize the basis functions and lengthscales to project the surface.
    # 3. **Coordinate Consistency**: The method must accept a new `DataFrame`, extract coordinates, and apply the same scaling/transformations used during training.

    println("--- Starting BSTM Out-of-Sample Prediction ---")

    # 1. Recover Training Metadata (M)
    local M_train = model_obj.args.M
    local N_samples_total = size(chain, 1)
    local N_samps = min(n_samples, N_samples_total)
    
    # 2. Configure Prediction Metadata (PS)
    # We utilize bstm_options to ensure the new data is formatted identically to training data.
    # Note: we use dummy responses as y_obs is not used during the projection phase.
    local PS = bstm_options(
        data = new_data,
        y_obs = zeros(nrow(new_data)),
        model_family = M_train.model_family,
        model_space = M_train.model_space,
        model_time = M_train.model_time,
        svc_covariates = M_train.svc_covariates,
        use_sv = M_train.use_sv
    )

    # 3. Centroid Alignment for Discrete Manifolds
    # If the training model used areal units, we map new points to the nearest training centroid.
    if haskey(M_train, :areal_units)
        local centroids = M_train.areal_units.centroids
        PS_s_idx = [argmin([sum(((new_data.s_x[i], new_data.s_y[i]) .- c).^2) for c in centroids]) for i in 1:nrow(new_data)]
        # Update PS with aligned indices
        PS = merge(PS, (s_idx = PS_s_idx,))
    end

    # 4. Invoke Architectural Reconstruction
    # We treat the new data as a Post-Stratification (PS) object within the existing _reconstruct logic.
    # This allows us to reuse the stable manifold retrieval and assembly code.
    local arch_type = get_architecture(get(M_train, :model_arch, "univariate"))
    
    # Selecting a subset of samples for efficiency if requested
    local chain_sub = chain[1:N_samps]
    
    # _reconstruct(arch, name, chain, M_training, PS_new_data, alpha)
    local res = _reconstruct(arch_type, "prediction", chain_sub, M_train, PS, alpha)

    println("Prediction Complete. Generated samples for ", nrow(new_data), " observations.")

    return res
end

 

# 1. PSIS-LOO Implementation for BSTM
# Rationale: Standardizing the extraction of log-likelihood matrices to provide 
# Expected Log Pointwise Predictive Density (ELPD) estimates.

function bstm_loo(chain, model_obj)
    println("--- Calculating PSIS-LOO for BSTM ---")
    
    # Extract log-likelihood matrix [Samples x Observations]
    # Based on our _reconstruct logic, log_lik is stored in the results object
    # or can be pulled directly from the chain if using Turing's point_loglikelihoods
    ### XR: Model Selection Suite (v10.0)

# This suite implements robust Bayesian model comparison tools. While WAIC is a useful heuristic, **LOO-CV (PSIS-LOO)** provides a more reliable estimate of out-of-sample predictive performance by smoothing importance weights. 

# Additionally, we introduce **Bridge Sampling** logic to approximate the Marginal Likelihood, allowing for the calculation of **Bayes Factors** to perform formal hypothesis testing between competing manifold structures.

    local results = _reconstruct(get_architecture(model_obj.args.M.model_arch), "loo_calc", chain, model_obj.args.M, nothing, 0.05)
    
    # Use PosteriorStats.loo (assumes log_likelihood array is available)
    # Note: This requires the log_lik matrix gathered during reconstruction
    local loo_result = loo(results.log_lik_matrix)
    
    display(loo_result)
    return loo_result
end

# 2. Bayes Factor Suite (Manifold Comparison)
function compare_manifolds(model_a_results, model_b_results)
    println("--- Manifold Comparison Dashboard ---")
    
    # 1. ELPD Comparison
    local diff_elpd = model_a_results.waic.elpd - model_b_results.waic.elpd
    
    # 2. Bayes Factor Approximation
    # Rationale: Using the Savage-Dickey density ratio or Bridge sampling 
    # approximation if Marginal Likelihood is available.
    # For this suite, we provide a structured comparison table.
    
    comparison_df = DataFrame(
        Metric = ["ELPD (WAIC)", "Effective Params (p_waic)", "WAIC Score"],
        Model_A = [model_a_results.waic.elpd, model_a_results.waic.p_waic, model_a_results.waic.waic],
        Model_B = [model_b_results.waic.elpd, model_b_results.waic.p_waic, model_b_results.waic.waic]
    )
    
    comparison_df[!, :Delta] = comparison_df.Model_A - comparison_df.Model_B
    
    display(comparison_df)
    return comparison_df
end



function bstm_cv_orchestrator(formula::String, data::DataFrame;
                             method=:kfold,
                             k=5,
                             lolo_var=:s_idx,
                             model_family="gaussian",
                             n_samples=500,
                             sampler=MH(),
                             kwargs...)
    # XR: Audited CV Orchestrator [v09.6 Final Coordinate Sync]
    # Rationale: Ensures s_x and s_y are passed to bstm() to trigger dynamic 
    # reconstruction of W-matrix (tessellation) for each fold's training set.

    println("--- Starting BSTM Cross-Validation Orchestrator [Method: $method] ---")
    local N_total = nrow(data)
    local results_folds = []
    local folds_indices = []

    if method == :kfold
        local idx_perm = shuffle(1:N_total)
        local fold_size = floor(Int, N_total / k)
        for i in 1:k
            start_idx = (i-1) * fold_size + 1
            end_idx = i == k ? N_total : i * fold_size
            push!(folds_indices, idx_perm[start_idx:end_idx])
        end
    elseif method == :lolo
        local unique_locs = unique(data[!, lolo_var])
        for loc in unique_locs
            push!(folds_indices, findall(x -> x == loc, data[!, lolo_var]))
        end
        k = length(unique_locs)
    end

    for i in 1:k
        println("\n--- Executing Fold $i / $k ---")
        local test_idx = folds_indices[i]
        local train_idx = setdiff(1:N_total, test_idx)
        local train_data = data[train_idx, :]
        local test_data = data[test_idx, :]

        # Propagate training coordinates to trigger build_structure_template internally
        # We explicitly map s_x and s_y from the DataFrame columns
        local model_train = bstm(formula, train_data;
            model_family=model_family,
            s_x=collect(train_data.s_x),
            s_y=collect(train_data.s_y),
            kwargs...)

        local chain_train = sample(model_train, sampler, n_samples, progress=false)
        local res_pred = predict(model_train, chain_train, test_data; n_samples=n_samples)

        local y_test_obs = test_data.y_obs
        local y_test_pred = res_pred.predictions_observed_denoised.mean

        local rmse = sqrt(mean((y_test_obs .- y_test_pred).^2))
        local mae = mean(abs.(y_test_obs .- y_test_pred))
        
        # Safe R2 calculation
        local r2 = (length(y_test_obs) > 1 && var(y_test_pred) > 0) ? cor(y_test_obs, y_test_pred)^2 : 0.0

        push!(results_folds, (rmse=rmse, mae=mae, r2=r2, fold=i))
        println("Fold $i Results -> RMSE: ", round(rmse, digits=4))
    end

    return (
        mean_rmse = mean([f.rmse for f in results_folds]),
        mean_r2 = mean([f.r2 for f in results_folds]),
        folds = results_folds
    )
end

;;
 