"""
    Barycentric <: ComponentModel

A component for non-linear smoothing using barycentric interpolation on a 2D Delaunay
triangulation of knot points. This method is particularly well-suited for modeling
smooth spatial effects on irregular domains.

# Version
v1.0.1 (2026-08-08)

# Mathematical Summary
The component models a function f(s) where s is a 2D coordinate.
1.  **Knot Triangulation**: A set of N_knots knot points are defined over the
    spatial domain. A Delaunay triangulation is performed on these knots to create a
    mesh of non-overlapping triangles.
2.  **Barycentric Coordinates**: For any observation point s_obs, the model finds
    the triangle in the mesh that encloses it. It then computes the barycentric
    coordinates (λ₁, λ₂, λ₃) of s_obs with respect to the triangle's vertices
    (v₁, v₂, v₃). These coordinates are non-negative weights that sum to 1.
3.  **Basis Construction**: The barycentric coordinates form the basis functions. For
    an observation i falling in a triangle with vertices (j, k, l), the
    corresponding row in the basis matrix B will have non-zero values only at
    columns j, k, l, where B[i,j] = λ₁, B[i,k] = λ₂, and B[i,l] = λ₃.
4.  **Final Effect**: The final smooth effect is a linear combination of the basis
    functions, with coefficients β (representing the latent field values at the
    knots) scaled by a standard deviation σ:
    f(s) = (B ⋅ (βσ))(s)
    where β ∼ N(0, I).

# Assumptions
- The spatial effect is smooth and can be reasonably approximated by a piecewise
  linear surface over the triangulated domain.
- The provided coordinates are 2-dimensional.

# Best Use Case
Modeling smooth spatial effects, particularly when the domain is complex or
irregularly shaped. It is an alternative to Thin Plate Splines or Gaussian Processes
that is computationally efficient and easy to interpret, as the coefficients
directly correspond to the value of the field at the knot locations.

# Key References
- Wikipedia: Barycentric coordinate system
- Amid, E., & Warmuth, M. K. (2019). TriMap: Large-scale Dimensionality Reduction
  Using Triplets. *arXiv preprint arXiv:1910.00204*. (For applications of
  triangulation in machine learning).

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the basis
  function coefficients.
"""
struct Barycentric <: ComponentModel
    sigma::UnivariateDistribution
end

COMPONENT_TYPE_REGISTRY[:barycentric] = Barycentric
COMPONENT_CONSTRUCTORS[:barycentric] = (p, params) -> Barycentric(p.sigma)
MODEL_TO_STRUCTURE_MAP[:barycentric] = :smooth


# --- Helper functions for Delaunay triangulation, restored from archive/model.jl ---

function _get_barycentric_coords(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    area_total = abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))
    if area_total < 1e-9 return nothing end
    area1 = abs((p2.x - p.x) * (p3.y - p.y) - (p3.x - p.x) * (p2.y - p.y))
    area2 = abs((p3.x - p.x) * (p1.y - p.y) - (p1.x - p.x) * (p3.y - p.y))
    area3 = abs((p1.x - p.x) * (p2.y - p.y) - (p2.x - p.x) * (p1.y - p.y))
    w_sum = area1 + area2 + area3
    if abs(w_sum - area_total) > 1e-6 return nothing end # Point is outside
    return (area1 / area_total, area2 / area_total, area3 / area_total)
end

function _is_inside_triangle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    coords = _get_barycentric_coords(p, p1, p2, p3)
    return !isnothing(coords) && all(c -> c >= -1e-9, coords)
end

function _get_circumcircle(p1::Point2D, p2::Point2D, p3::Point2D)
    D = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if abs(D) < 1e-9 return nothing, nothing end
    p1_sq, p2_sq, p3_sq = p1.x^2 + p1.y^2, p2.x^2 + p2.y^2, p3.x^2 + p3.y^2
    center_x = (p1_sq * (p2.y - p3.y) + p2_sq * (p3.y - p1.y) + p3_sq * (p1.y - p2.y)) / D
    center_y = (p1_sq * (p3.x - p2.x) + p2_sq * (p1.x - p3.x) + p3_sq * (p2.x - p1.x)) / D
    center = Point2D(center_x, center_y)
    radius_sq = (p1.x - center.x)^2 + (p1.y - center.y)^2
    return center, radius_sq
end

function _is_in_circumcircle(p::Point2D, p1::Point2D, p2::Point2D, p3::Point2D)
    center, radius_sq = _get_circumcircle(p1, p2, p3)
    if isnothing(center) return false end
    dist_sq = (p.x - center.x)^2 + (p.y - center.y)^2
    return dist_sq < radius_sq
end

function _delaunay_triangulation(points::Vector{Point2D})
    n = length(points)
    if n < 3 return [] end
    min_x, max_x = extrema(p.x for p in points); min_y, max_y = extrema(p.y for p in points)
    dx, dy = max_x - min_x, max_y - min_y; delta_max = max(dx, dy)
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    p_super1 = Point2D(mid_x - 20 * delta_max, mid_y - delta_max)
    p_super2 = Point2D(mid_x + 20 * delta_max, mid_y - delta_max)
    p_super3 = Point2D(mid_x, mid_y + 20 * delta_max)
    super_triangle = Triangle(n + 1, n + 2, n + 3)
    all_points = [points; p_super1; p_super2; p_super3]
    triangulation = [super_triangle]
    for i in 1:n
        point = points[i]; bad_triangles = []
        for tri in triangulation
            p1, p2, p3 = all_points[tri.v1], all_points[tri.v2], all_points[tri.v3]
            if _is_in_circumcircle(point, p1, p2, p3); push!(bad_triangles, tri); end
        end
        polygon = []
        for tri in bad_triangles
            edges = [(tri.v1, tri.v2), (tri.v2, tri.v3), (tri.v3, tri.v1)]
            for edge in edges
                is_shared = false
                for other_tri in bad_triangles
                    if tri === other_tri continue end
                    other_edges = [(other_tri.v1, other_tri.v2),
                                   (other_tri.v2, other_tri.v3),
                                   (other_tri.v3, other_tri.v1)]
                    if (edge in other_edges) || ((edge[2], edge[1]) in other_edges)
                        is_shared = true; break;
                    end
                end
                if !is_shared; push!(polygon, edge); end
            end
        end
        filter!(t -> !(t in bad_triangles), triangulation)
        for edge in polygon; push!(triangulation, Triangle(edge[1], edge[2], i)); end
    end
    filter!(t -> !(t.v1 > n || t.v2 > n || t.v3 > n), triangulation)
    return triangulation
end

function bstm_barycentric_basis_2D(coords::AbstractMatrix, knots::Vector{Point2D})
    n_obs, n_knots = size(coords, 1), length(knots)
    B = spzeros(Float64, n_obs, n_knots)
    triangles = _delaunay_triangulation(knots)
    if isempty(triangles)
        @warn "Delaunay triangulation failed. Returning empty basis."
        return B
    end
    for i in 1:n_obs
        obs_point = Point2D(coords[i, 1], coords[i, 2])
        for tri in triangles
            v1_idx, v2_idx, v3_idx = tri.v1, tri.v2, tri.v3
            p1, p2, p3 = knots[v1_idx], knots[v2_idx], knots[v3_idx]
            if _is_inside_triangle(obs_point, p1, p2, p3)
                bary_coords = _get_barycentric_coords(obs_point, p1, p2, p3)
                if !isnothing(bary_coords)
                    w1, w2, w3 = bary_coords
                    B[i, v1_idx], B[i, v2_idx], B[i, v3_idx] = w1, w2, w3
                end
                break
            end
        end
    end
    return B
end

# --- Component Interface Methods ---

"""
    get_datastructures!(m_type::Type{<:Barycentric}, M::Dict, mod_data::Dict)

Extracts the 2D coordinate variables from the formula and stores them.

# Assumptions
- The `random()` call provides exactly two variables for the 2D coordinates.
"""
function get_datastructures!(
    m_type::Type{<:Barycentric}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]
    if length(variables) != 2
        error(
            "The Barycentric model requires exactly two coordinate variables, e.g., " *
            "`random(x, y, model=:barycentric)`."
        )
    end
    coords_matrix = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords_matrix
    return true
end

"""
    get_precomputes(m::Barycentric, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes the barycentric basis matrix by generating knots, performing Delaunay
triangulation, and calculating barycentric coordinates for each observation.
"""
function get_precomputes(
    m::Barycentric, M::NamedTuple, mod_data::Dict
)::NamedTuple
    coords = mod_data[:params][:coords]
    nbins = get(mod_data[:params], :nbins, 25)
    knot_method = get(mod_data[:params], :knot_method, :quantile)
    n_marginal = Int(floor(sqrt(nbins)))
    
    local kx, ky
    if knot_method == :range
        kx = collect(range(extrema(coords[:, 1])..., length=n_marginal))
        ky = collect(range(extrema(coords[:, 2])..., length=n_marginal))
    else # Default to :quantile
        kx = quantile(coords[:, 1], range(0, 1, length=n_marginal))
        ky = quantile(coords[:, 2], range(0, 1, length=n_marginal))
    end
    
    knot_points = [Point2D(x, y) for x in kx for y in ky]
    basis_matrix = bstm_barycentric_basis_2D(coords, knot_points)
    n_knots = length(knot_points)

    return (B=basis_matrix, n_knots=n_knots)
end

"""
    get_priors(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for the basis coefficients (`innov`) and overall scale (`sigma`).
"""
function get_priors(
    m::Barycentric, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_knots = spec.hyper.n_knots
    return """
    $(p_names.innov) ~ MvNormal(zeros(T, $(n_knots)), I)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    """
end

"""
    get_updates(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx, M)

Generates code to compute the smooth effect as a product of the basis matrix and
the scaled coefficients, and adds it to the linear predictor `eta`.
"""
function get_updates(
    m::Barycentric, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    return """
        # --- Barycentric Component: $(key) ---
        let
            local B = spec_registry[:$(key)].hyper.B
            local scaled_coeffs = $(p_names.innov) .* $(p_names.sigma)
            local barycentric_effect = B * scaled_coeffs
            $(eta_target) .+= barycentric_effect
        end
    """
end

"""
    get_effects(m::Barycentric, chain, M::NamedTuple, ...)

Reconstructs the barycentric smooth effect from posterior samples.
"""
function get_effects(
    m::Barycentric, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    # Re-create basis matrix for prediction set if it exists
    B_train = spec.hyper.B
    B_full = if !isnothing(PS)
        coord_vars = get(spec.params, :positional_args, [])
        coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        
        # Re-use the same knots from the training phase
        nbins = get(spec.params, :nbins, 25)
        n_marginal = Int(floor(sqrt(nbins)))
        kx = quantile(spec.hyper.coords[:, 1], range(0, 1, length=n_marginal))
        ky = quantile(spec.hyper.coords[:, 2], range(0, 1, length=n_marginal))
        knot_points = [Point2D(x, y) for x in kx for y in ky]
        
        B_pred = bstm_barycentric_basis_2D(coords_pred, knot_points)
        vcat(B_train, B_pred)
    else
        B_train
    end

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        innov_samples = get_params_vector(chain, string(p_names.innov), spec.hyper.n_knots)
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)
        
        scaled_coeffs = innov_samples .* sigma_samples
        effect_k = B_full * scaled_coeffs'
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
