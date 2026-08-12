"""
    Barycentric <: ComponentModel

A component for non-linear smoothing using barycentric interpolation on a 2D Delaunay
triangulation of knot points. This method is particularly well-suited for modeling
smooth spatial effects on irregular domains.

# Version
v1.1.0 (2026-08-11)

# Mathematical Summary
The component models a function \$f(s)\$ where \$s\$ is a 2D coordinate.
1.  **Knot Triangulation**: A set of \$N_{\\text{knots}}\$ knot points are defined over the
    spatial domain. A Delaunay triangulation is performed on these knots to create a
    mesh of non-overlapping triangles.
2.  **Barycentric Coordinates**: For any observation point \$s_{\\text{obs}}\$, the model finds
    the triangle in the mesh that encloses it. It then computes the barycentric
    coordinates \$(\\lambda_1, \\lambda_2, \\lambda_3)\$ of \$s_{\\text{obs}}\$ with respect to the triangle's vertices
    \$ (v_1, v_2, v_3) \$. These coordinates are non-negative weights that sum to 1.
3.  **Basis Construction**: The barycentric coordinates form the basis functions. For
    an observation \$i\$ falling in a triangle with vertices \$ (j, k, l) \$, the
    corresponding row in the basis matrix \$B\$ will have non-zero values only at
    columns \$j, k, l\$, where \$B[i,j] = \\lambda_1\$, \$B[i,k] = \\lambda_2\$, and \$B[i,l] = \\lambda_3\$.
4.  **Final Effect**: The final smooth effect is a linear combination of the basis
    functions, with coefficients \$\\beta\$ (representing the latent field values at the
    knots) scaled by a standard deviation \$\\sigma\$:
    \$f(s) = (B \\cdot \\beta)(s)\$, where the prior on \$\\beta\$ depends on the chosen method.

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where coefficients are
  sampled from a standard normal and scaled by `sigma`. Recommended for AD.
- `:centered`: A centered parameterization where coefficients are sampled directly
  from `N(0, sigma^2)`. Didactic, can be less efficient and not AD-friendly.
- `:gmrfsmooth`: Imposes a spatial ICAR prior on the knot coefficients, encouraging
  a smoother interpolation surface. AD-safe via spectral decomposition.

# Fields
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the basis
  function coefficients.
- `method::Symbol`: The computational method, one of `:noncentered`, `:centered`,
  or `:gmrfsmooth`.

# Inputs
- **Required**:
  - Exactly two coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `nbins`: `Int`, the approximate number of knot points to use. Default: `25`.
  - `knot_method`: `Symbol`, method for placing knots (`:quantile` or `:range`). Default: `:quantile`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the basis
    function coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered`, `:centered`, `:gmrfsmooth`).
    Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the basis coefficients.
- `innov_<key>`: The latent standard normal innovations for basis coefficients (for `:noncentered` and `:gmrfsmooth`).
- `latent_<key>`: The latent basis coefficients (for `:centered`).
"""
struct Barycentric <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:barycentric] = Barycentric
COMPONENT_CONSTRUCTORS[:barycentric] = (p, params) -> Barycentric(
    p.sigma, get(params, :method, :noncentered)
)
MODEL_TO_STRUCTURE_MAP[:barycentric] = :smooth


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

Pre-computes the barycentric basis matrix. For the `:gmrfsmooth` method, it also
computes the precision matrix template and spectral decomposition for the knot grid.
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

    precomputes = Dict{Symbol, Any}(
        :B => basis_matrix,
        :n_knots => n_knots,
        :knots => knot_points
    )

    if m.method == :gmrfsmooth
        W_knots = spzeros(Int, n_knots, n_knots)
        for i in 1:n_knots, j in (i+1):n_knots
            dist_sq = (knot_points[i].x - knot_points[j].x)^2 + (knot_points[i].y - knot_points[j].y)^2
            if dist_sq <= ((kx[2]-kx[1])^2 + (ky[2]-ky[1])^2) * 1.1
                W_knots[i, j] = W_knots[j, i] = 1
            end
        end
        
        template = build_structure_template(:besag, n_knots; W=W_knots)
        precomputes[:Q_template] = template.matrix
        precomputes[:U] = template.U
        precomputes[:L] = template.L
    end

    return NamedTuple(precomputes)
end

"""
    get_priors(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for the basis coefficients and scale, dispatching on the method.
"""
function get_priors(
    m::Barycentric, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_knots = spec.hyper.n_knots
    
    priors = ["$(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))"] # Prior for the standard deviation

    if m.method in [:noncentered, :gmrfsmooth]
        push!(priors, "$(p_names.innov) ~ MvNormal(zeros(T, $(n_knots)), I)")
    elseif m.method == :centered
        push!(priors, "$(p_names.latent) ~ MvNormal(zeros(T, $(n_knots)), I)")
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Barycentric, spec::NamedTuple, arch::String, outcome_idx, M)

Generates code to compute the smooth effect, dispatching on the chosen method.
"""
function get_updates(
    m::Barycentric, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    noncentered_code = """
        # --- Barycentric Component (Non-Centered): $(key) ---
        let # Access pre-computed data from spec_registry
            hyper = spec_registry[:$(key)].hyper
            B = hyper.B
            innov_coeffs = $(p_names.innov)
            scaled_coeffs = innov_coeffs .* $(p_names.sigma)
            barycentric_effect = B * scaled_coeffs
            $(eta_target) .+= barycentric_effect
        end
    """

    centered_code = """
        # --- Barycentric Component (Centered): $(key) ---
        let # Access pre-computed data from spec_registry
            hyper = spec_registry[:$(key)].hyper
            B = hyper.B
            # Sample the latent coefficients directly from a centered prior.
            $(p_names.latent) ~ MvNormal(zeros(T, hyper.n_knots), $(p_names.sigma)^2 * I)
            coeffs = $(p_names.latent)
            barycentric_effect = B * coeffs
            $(eta_target) .+= barycentric_effect
        end
    """

    gmrfsmooth_code = """
        # --- Barycentric Component (GMRF Smooth): $(key) ---
        let # Access pre-computed data from spec_registry
            hyper = spec_registry[:$(key)].hyper
            B = hyper.B
            U = hyper.U
            L = hyper.L
            innov_coeffs = $(p_names.innov)
            diag_D = $(p_names.sigma) ./ sqrt.(L .+ M.noise) # Scale by sigma
            diag_D[1] = 0.0 # Sum-to-zero constraint for ICAR on knots
            coeffs = U * (diag_D .* innov_coeffs)
            barycentric_effect = B * coeffs
            $(eta_target) .+= barycentric_effect
        end
    """

    if m.method == :noncentered; return noncentered_code;
    elseif m.method == :centered; return centered_code;
    elseif m.method == :gmrfsmooth; return gmrfsmooth_code;
    else; error("Unsupported method '$(m.method)' for Barycentric component."); end
end


"""
    get_effects(m::Barycentric, chain, M::NamedTuple, ...)

Reconstructs the barycentric smooth effect from posterior samples, dispatching
on the method used during sampling.
"""
function get_effects(
    m::Barycentric, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    B_train = spec.hyper.B
    is_multivariate_model = M.model_arch == "multivariate"
    B_full = if !isnothing(PS)
        coord_vars = get(spec.params, :positional_args, [])
        coords_pred = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        B_pred = bstm_barycentric_basis_2D(coords_pred, spec.hyper.knots)
        vcat(B_train, B_pred)
    else
        B_train
    end

    n_knots = spec.hyper.n_knots
    p_names_vec = string.(FlexiChains.parameters(chain))

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        
        if isempty(sigma_name)
            @warn "Parameters for Barycentric component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, size(B_full, 1), n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        
        local coeffs_samples_matrix
        if m.method == :centered
            latent_name = _find_parameter(p_names_vec, string(spec.key), "latent", k, is_multivariate_model)
            coeffs_samples_matrix = get_params_vector(chain, latent_name, n_knots)
        else # :noncentered or :gmrfsmooth
            innov_name = _find_parameter(p_names_vec, string(spec.key), "innov", k, is_multivariate_model)
            innov_samples_matrix = get_params_vector(chain, innov_name, n_knots)
            coeffs_samples_matrix = zeros(n_knots, n_samples)
            if m.method == :noncentered
                coeffs_samples_matrix = innov_samples_matrix .* sigma_samples'
            else # :gmrfsmooth
                U, L = spec.hyper.U, spec.hyper.L
                noise_val = get(M, :noise, 1e-6)
                for i in 1:n_samples
                    diag_D = sigma_samples[i] ./ sqrt.(L .+ noise_val); diag_D[1] = 0.0; coeffs_samples_matrix[:, i] = U * (diag_D .* innov_samples_matrix[i, :]); end
            end
        end
        
        effect_k = B_full * coeffs_samples_matrix
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end



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

 