"""
    Movement <: ComponentModel

A component for simulating population dynamics using an Advection-Diffusion-Reaction
(ADR) process on a discrete spatial graph. This component models the change in a
latent field over time due to three primary processes: advection (directional
movement), diffusion (random movement), and reaction (local population growth/decay).
It can also integrate mark-recapture telemetry data to inform movement parameters.

# Version
v1.0.0

# Mathematical Summary
The component approximates the solution to the Advection-Diffusion-Reaction PDE:

\$\\frac{\\partial C}{\\partial t} = \\nabla \\cdot (D \\nabla C) - \\nabla \\cdot
  (\\mathbf{v} C) + f(C)\$

where:
- `C` is the concentration or density of the population.
- `D` is the diffusion coefficient, which can be spatially varying.
- `v` is the velocity field for advection.
- `f(C)` is the reaction term, modeled here as logistic growth: `r*C*(1 - C/K)`.

This implementation uses a discrete state-space representation on a graph, where
the spatial operators are derived from the graph's adjacency matrix `W`. The
temporal evolution is modeled using either an explicit or implicit Euler scheme.
This approach is common in hierarchical Bayesian models for ecological processes.

# Telemetry Data Integration
If `mark_recapture_data` is provided, the model includes a likelihood component for
these observations. The transition probability over `k` time steps is calculated
from the one-step transition matrix `Gamma` (the inverse of the propagator) as `Gamma^k`.

The `mark_recapture_data` can be provided in two formats:
1.  A `DataFrame` in "long" format with columns: `tagid`, `s_idx`, `time`, `tag`, and an
  optional `individual_covariate`. The component will automatically process this into
  transitions.
2.  A pre-processed `Matrix` in "wide" format with columns: `[release_unit, recapture_unit,
  time_steps, individual_covariate]`.

# Computational Methods
- **`:explicit` (Default, AD-friendly)**: Uses an explicit Euler time-stepping
  scheme that includes the reaction term. This method is fully compatible with
  automatic differentiation (AD) but is only conditionally stable.
- **`:implicit` (Didactic, Not AD-friendly)**: Uses an implicit Euler time-stepping
  scheme for the advection-diffusion part (no reaction term). This method is
  unconditionally stable but not AD-compatible.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - A temporal index variable (e.g., `year`).
  - An adjacency matrix `W`.
- **Optional**:
  - `habitat`: A covariate influencing diffusion.
  - `mark_recapture_data`: A `DataFrame` or `Matrix` with telemetry data.
  - `method`: `categorical`, `:explicit`,  `:implicit`.
  - `velocity`: Prior for the advection velocity.
  - `diffusion`: Prior for the diffusion rate.
  - `sigma`: Prior for the process noise standard deviation.
  - `r`: Prior for the intrinsic growth rate (for reaction term).
  - `K`: Prior for the carrying capacity (for reaction term).
  - `beta_het`: Prior for the individual heterogeneity effect in telemetry data.

# Outputs (Parameter Names)
- `velocity_<key>`, `diffusion_<key>`, `sigma_<key>`
- `r_<key>`, `K_<key>` (if reaction is modeled)
- `beta_het_<key>` (if telemetry is modeled)
- `beta_habitat_diffusion_<key>` (if habitat covariate is used)
- `innovations_<key>`
"""
struct Movement <: ComponentModel
    velocity::UnivariateDistribution
    diffusion::UnivariateDistribution
    sigma::UnivariateDistribution   
    gamma::UnivariateDistribution
    r::Union{UnivariateDistribution, Nothing}
    K::Union{UnivariateDistribution, Nothing}
    beta_het::Union{UnivariateDistribution, Nothing}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:movement] = Movement
COMPONENT_CONSTRUCTORS[:movement] = (p, params) -> Movement(
    get(p, :velocity, get(params, :velocity, truncated(Normal(0.2, 0.2), 0.0, 0.95))),
    get(p, :diffusion, get(params, :diffusion, truncated(Normal(0.1, 0.2), 0.0, Inf))),
    get(p, :sigma, get(params, :sigma, Exponential(1.0))),
    get(p, :gamma, get(params, :gamma, Normal(1.0, 1.0))),
    get(p, :r, get(params, :r, nothing)),
    get(p, :K, get(params, :K, nothing)),
    get(p, :beta_het, get(params, :beta_het, nothing)),
    Symbol(get(params, :method, :explicit))
)
MODEL_TO_STRUCTURE_MAP[:movement] = :spacetime

function _raster_to_graph(raster::AbstractMatrix)
    rows, cols = size(raster)
    n_units = rows * cols
    W = spzeros(Int, n_units, n_units)
    
    for r in 1:rows, c in 1:cols
        idx = (c - 1) * rows + r
        # 8-neighbor connectivity (Queen's case)
        for dr in -1:1, dc in -1:1
            if dr == 0 && dc == 0
                continue
            end
            nr, nc = r + dr, c + dc
            if 1 <= nr <= rows && 1 <= nc <= cols
                n_idx = (nc - 1) * rows + nr
                W[idx, n_idx] = 1
            end
        end
    end
    return W
end

"""
    TelemetryData <: AbstractMatrix{Float64}

Structured container for mark-recapture telemetry observation events. Supports categorical
group-stratified transition modeling while maintaining matrix indexing compatibility.

# Fields
- `releases`: Vector of unit indices at initial detection / release (1-indexed).
- `recaps`: Vector of unit indices at subsequent detection / recapture (1-indexed).
- `ks`: Elapsed discrete time intervals between consecutive detections.
- `groups`: Biological stratum / group identifiers (1-indexed integers).
- `covariates`: Individual-level continuous covariates.
- `G`: Total count of distinct biological strata / groups.
- `max_k`: Maximum observed elapsed transition step count.
- `matrix`: `Matrix{Float64}` representation of size `(N, 4)`.
"""
struct TelemetryData <: AbstractMatrix{Float64}
    releases::Vector{Int}
    recaps::Vector{Int}
    ks::Vector{Int}
    groups::Vector{Int}
    covariates::Vector{Float64}
    G::Int
    max_k::Int
    matrix::Matrix{Float64}
end

Base.size(td::TelemetryData) = size(td.matrix)
Base.size(td::TelemetryData, d::Int) = size(td.matrix, d)
Base.getindex(td::TelemetryData, i::Int, j::Int) = td.matrix[i, j]
Base.getindex(td::TelemetryData, i::Int) = td.matrix[i]
Base.IndexStyle(::Type{TelemetryData}) = IndexLinear()

"""
    _process_telemetry_data(telemetry_input; mark_recapture_G=nothing)

Transforms input telemetry data (DataFrame or Matrix) into a validated `TelemetryData`
structure for movement modeling. Supports both longitudinal event sequences and
pre-aggregated transition event pairs.
"""
function _process_telemetry_data(telemetry_input; mark_recapture_G=nothing)
    if telemetry_input isa TelemetryData
        return telemetry_input
    elseif telemetry_input isa AbstractMatrix
        mat = Matrix{Float64}(telemetry_input)
        n_rows = size(mat, 1)
        rel = n_rows > 0 ? Int.(mat[:, 1]) : Int[]
        rec = n_rows > 0 && size(mat, 2) >= 2 ? Int.(mat[:, 2]) : Int[]
        ks  = n_rows > 0 && size(mat, 2) >= 3 ? Int.(mat[:, 3]) : Int[]
        cov = n_rows > 0 && size(mat, 2) >= 4 ? mat[:, 4] : zeros(Float64, n_rows)
        grp = if size(mat, 2) >= 5
            Int.(mat[:, 5])
        elseif !isnothing(mark_recapture_G) && mark_recapture_G > 1 &&
               all(c -> isinteger(c) && c >= 1, cov)
            Int.(cov)
        else
            ones(Int, n_rows)
        end
        G_val = isnothing(mark_recapture_G) ? (isempty(grp) ? 1 : maximum(grp)) :
            Int(mark_recapture_G)
        max_k_val = isempty(ks) ? 1 : maximum(ks)
        return TelemetryData(rel, rec, ks, grp, cov, G_val, max_k_val, mat)

    elseif telemetry_input isa DataFrame
        df = telemetry_input
        # Case A: Already aggregated event pairs with release and recapture
        has_rel = hasproperty(df, :release) || hasproperty(df, :releases)
        has_rec = hasproperty(df, :recapture) || hasproperty(df, :recaps)

        if has_rel && has_rec
            rel_col = hasproperty(df, :release) ? :release : :releases
            rec_col = hasproperty(df, :recapture) ? :recapture : :recaps
            k_col = hasproperty(df, :k) ? :k : (hasproperty(df, :ks) ? :ks : nothing)
            grp_col = hasproperty(df, :group) ? :group :
                (hasproperty(df, :groups) ? :groups : nothing)
            cov_col = hasproperty(df, :covariate) ? :covariate :
                (hasproperty(df, :individual_covariate) ? :individual_covariate : nothing)

            rel = Int.(df[!, rel_col])
            rec = Int.(df[!, rec_col])
            ks  = !isnothing(k_col) ? Int.(df[!, k_col]) : ones(Int, nrow(df))
            grp = !isnothing(grp_col) ? Int.(df[!, grp_col]) : ones(Int, nrow(df))
            cov = !isnothing(cov_col) ? Float64.(df[!, cov_col]) : zeros(Float64, nrow(df))
            G_val = isnothing(mark_recapture_G) ? (isempty(grp) ? 1 : maximum(grp)) :
                Int(mark_recapture_G)
            max_k_val = isempty(ks) ? 1 : maximum(ks)
            mat = hcat(Float64.(rel), Float64.(rec), Float64.(ks), cov)
            return TelemetryData(rel, rec, ks, grp, cov, G_val, max_k_val, mat)
        end

        # Case B: Longitudinal telemetry observations
        time_col = if hasproperty(df, :time)
            :time
        elseif hasproperty(df, :timestamp)
            :timestamp
        else
            nothing
        end

        if isnothing(time_col) || !hasproperty(df, :tagid) || !hasproperty(df, :s_idx)
            error("Telemetry DataFrame must contain either event pairs (:release, :recapture) " *
                  "or longitudinal observations (:tagid, :s_idx, :time).")
        end

        releases = Int[]
        recaps = Int[]
        ks = Int[]
        groups = Int[]
        covariates = Float64[]

        tag_col = hasproperty(df, :tag) ? :tag : nothing
        grp_col = hasproperty(df, :group) ? :group :
            (hasproperty(df, :groups) ? :groups : nothing)
        cov_col = hasproperty(df, :individual_covariate) ? :individual_covariate :
            (hasproperty(df, :covariate) ? :covariate : nothing)

        gdf = groupby(df, :tagid)
        for sub_df in gdf
            if nrow(sub_df) < 2
                continue
            end
            sub_sorted = if !isnothing(tag_col)
                sort(sub_df, [order(tag_col), order(time_col)])
            else
                sort(sub_df, [order(time_col)])
            end

            for i in 1:(nrow(sub_sorted) - 1)
                row_rel = sub_sorted[i, :]
                row_rec = sub_sorted[i+1, :]
                push!(releases, Int(row_rel.s_idx))
                push!(recaps, Int(row_rec.s_idx))
                t_rel = Float64(getproperty(row_rel, time_col))
                t_rec = Float64(getproperty(row_rec, time_col))
                push!(ks, max(1, round(Int, t_rec - t_rel)))
                grp_val = !isnothing(grp_col) ? Int(getproperty(row_rel, grp_col)) : 1
                push!(groups, grp_val)
                cov_val = !isnothing(cov_col) ? Float64(getproperty(row_rel, cov_col)) : 0.0
                push!(covariates, cov_val)
            end
        end

        n_events = length(releases)
        mat = if n_events > 0
            hcat(Float64.(releases), Float64.(recaps), Float64.(ks), covariates)
        else
            Matrix{Float64}(undef, 0, 4)
        end
        G_val = isnothing(mark_recapture_G) ? (isempty(groups) ? 1 : maximum(groups)) :
            Int(mark_recapture_G)
        max_k_val = isempty(ks) ? 1 : maximum(ks)
        return TelemetryData(releases, recaps, ks, groups, covariates, G_val, max_k_val, mat)
    else
        error("Unsupported format for mark_recapture_data: $(typeof(telemetry_input)). " *
              "Expected DataFrame or Matrix.")
    end
end

function get_precomputes(m::Movement, M::NamedTuple, mod_data::Dict)::NamedTuple
    params = mod_data[:params]
    data = M.data
    variables = mod_data[:variables]
    calling_mod = get(M, :calling_module, Main)

    W = if hasproperty(M, :W) && M.W isa AbstractMatrix
        M.W
    elseif haskey(params, :W) && params[:W] isa AbstractMatrix
        params[:W]
    elseif haskey(params, :W)
        W_expr = params[:W]
        try
            Core.eval(calling_mod, W_expr)
        catch e
            error("Could not evaluate adjacency matrix `W` from `$(W_expr)`. Error: $e")
        end
    else
        nothing
    end

    if isnothing(W)
        if haskey(params, :habitat_raster)
            raster = params[:habitat_raster]
            if !(raster isa AbstractMatrix)
                error("`habitat_raster` must be a matrix.")
            end
            W = _raster_to_graph(raster)
        else
            error("The `movement` component requires either an adjacency matrix `W` or a `habitat_raster` parameter.")
        end
    end

    s_N = size(W, 1)
    if hasproperty(M, :s_N) && M.s_N != s_N
        error("Number of spatial units in data ($(M.s_N)) does not match adjacency matrix W dimension ($(s_N)).")
    end
    t_N = hasproperty(M, :t_N) ? M.t_N : (hasproperty(M, :t_idx) ? length(unique(M.t_idx)) : 1)

    habitat_data = nothing
    if hasproperty(M, :habitat) && M.habitat isa AbstractVector
        habitat_val = M.habitat
    elseif haskey(params, :habitat)
        habitat_val = params[:habitat]
        if habitat_val isa Symbol || habitat_val isa Expr
            if habitat_val isa Symbol && hasproperty(data, habitat_val)
                # Column in data, will be aggregated below
            else
                try
                    habitat_val = Core.eval(calling_mod, habitat_val)
                catch e
                    # Fallback to column lookup error below if not found
                end
            end
        end
    else
        habitat_val = nothing
    end

    if !isnothing(habitat_val)
        if habitat_val isa Symbol
            if !hasproperty(data, habitat_val)
                error("Habitat variable ':$habitat_val' not found in data.")
            end
            habitat_per_obs = data[!, habitat_val]
            habitat_aggregated = zeros(Float64, s_N)
            counts = zeros(Int, s_N)
            y_N_val = hasproperty(M, :y_N) ? M.y_N : nrow(data)
            s_idx_vec = hasproperty(M, :s_idx) ? M.s_idx :
                (hasproperty(data, :s_idx) ? data.s_idx : Int[])
            for i in 1:min(y_N_val, length(s_idx_vec))
                s_i = s_idx_vec[i]
                if 1 <= s_i <= s_N
                    habitat_aggregated[s_i] += habitat_per_obs[i]
                    counts[s_i] += 1
                end
            end
            habitat_data = habitat_aggregated ./ max.(1, counts)
        elseif habitat_val isa AbstractVector
            if length(habitat_val) != s_N
                error("Provided `habitat` vector length ($(length(habitat_val))) does not match s_N ($(s_N)).")
            end
            habitat_data = convert(Vector{Float64}, habitat_val)
        else
            error("The `habitat` parameter must be a Symbol (column name) or a Vector of length s_N.")
        end
    end

    precomputes = Dict{Symbol, Any}(
        :n_latent => s_N * t_N,
        :s_N => s_N,
        :t_N => t_N
    )
    if isnothing(habitat_data)
        habitat_data = zeros(Float64, s_N)
    end
    land_mask = if hasproperty(M, :land_mask) && M.land_mask isa AbstractVector{Bool}
        M.land_mask
    elseif haskey(params, :land_mask) && params[:land_mask] isa AbstractVector{Bool}
        params[:land_mask]
    else
        nothing
    end

    if !isnothing(land_mask)
        if length(land_mask) != s_N
            throw(DimensionMismatch("land_mask length ($(length(land_mask))) must match s_N ($s_N)."))
        end
        habitat_data[land_mask] .= 0.0
        precomputes[:land_mask] = land_mask
    end

    W_sp = sparse(W)
    if !isnothing(land_mask)
        W_sp = copy(W_sp)
        for l in findall(land_mask)
            W_sp[l, :] .= 0
            W_sp[:, l] .= 0
        end
        dropzeros!(W_sp)
    end

    W_sym = Array(max.(W_sp, W_sp'))
    deg = vec(sum(W_sym, dims=2))
    precomputes[:L_dense] = Matrix{Float64}(Diagonal(deg) - W_sym)
    precomputes[:adj_rows] = [W_sp.rowval[W_sp.colptr[i]:W_sp.colptr[i+1]-1] for i in 1:s_N]

    # Precompute topological random walk matrix T_diff (row-stochastic)
    T_diff = zeros(Float64, s_N, s_N)
    for i in 1:s_N
        if !isnothing(land_mask) && land_mask[i]
            T_diff[i, i] = 1.0
            continue
        end
        col_start = W_sp.colptr[i]
        col_end   = W_sp.colptr[i+1] - 1
        deg_i = col_end - col_start + 1
        if deg_i > 0 && col_start <= col_end
            inv_deg = 1.0 / deg_i
            for ptr in col_start:col_end
                j = W_sp.rowval[ptr]
                if isnothing(land_mask) || !land_mask[j]
                    T_diff[i, j] = inv_deg
                end
            end
        else
            T_diff[i, i] = 1.0
        end
    end
    precomputes[:T_diff] = T_diff

    tel_key = haskey(params, :telemetry_data) ? :telemetry_data : 
              (haskey(params, :mark_recapture_data) ? :mark_recapture_data : nothing)
    if !isnothing(tel_key)
        telemetry_input = params[tel_key]
        if telemetry_input isa Symbol || telemetry_input isa Expr
            try
                telemetry_input = Core.eval(calling_mod, telemetry_input)
            catch e
                error("Could not evaluate `$(tel_key)` argument `$(telemetry_input)`. Error: $e")
            end
        end

        G_hint = get(params, :mark_recapture_G, get(params, :G, nothing))
        if G_hint isa Symbol || G_hint isa Expr
            try
                G_hint = Core.eval(calling_mod, G_hint)
            catch e
                # Fallback to auto-detection from data
            end
        end

        processed_tel = _process_telemetry_data(telemetry_input; mark_recapture_G=G_hint)
        precomputes[:mark_recapture_data] = processed_tel
        precomputes[:max_k] = processed_tel.max_k
    else
        precomputes[:max_k] = 1
    end

    L_template = build_structure_template(:besag, s_N; W=W_sp).matrix
    precomputes[:L_template] = L_template

    rel_type = get(params, :relationship, get(params, :habitat_relationship, :exponential))
    local A_template
    if any(!iszero, habitat_data)
        # Directed advection operator derived from Habitat Suitability Index (HSI) gradient
        W_dir = spzeros(Float64, s_N, s_N)
        rows = rowvals(W_sp)
        vals = nonzeros(W_sp)
        for i in 1:s_N
            for j_idx in nzrange(W_sp, i)
                j = rows[j_idx]
                if i != j
                    diff_h = habitat_data[j] - habitat_data[i]
                    if diff_h > 0.0
                        if rel_type == :exponential
                            W_dir[i, j] = vals[j_idx] * exp(diff_h)
                        elseif rel_type == :logistic
                            W_dir[i, j] = vals[j_idx] / (1.0 + exp(-diff_h * 4.0))
                        else # :linear
                            W_dir[i, j] = vals[j_idx] * diff_h
                        end
                    end
                end
            end
        end
        out_degree = sum(W_dir, dims=2)[:]
        D_inv = spdiagm(0 => [od > 1e-12 ? 1.0 / od : 0.0 for od in out_degree])
        A_template = D_inv * W_dir
    else
        W_dir = tril(W_sp, -1)
        out_degree = sum(W_dir, dims=2)[:]
        D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
        A_template = D_inv * W_dir
    end
    precomputes[:A_template] = A_template
    precomputes[:habitat_data] = habitat_data

    return NamedTuple(precomputes)
end

function get_priors(
    m::Movement, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    if m.method in (:categorical, :stochastic_kernel, :density_gradient, :joint)
        G = hasproperty(spec.hyper, :mark_recapture_data) ? spec.hyper.mark_recapture_data.G : 1
        v_prior_str = _distribution_to_string(m.velocity)
        d_prior_str = _distribution_to_string(m.diffusion)
        g_prior_str = _distribution_to_string(m.gamma)
        push!(priors, "$(p_names.velocity) ~ filldist($v_prior_str, $G)")
        push!(priors, "$(p_names.diffusion) ~ filldist($d_prior_str, $G)")
        push!(priors, "$(p_names.gamma) ~ filldist($g_prior_str, $G)")
    else
        push!(priors, "$(p_names.velocity) ~ $(_distribution_to_string(m.velocity))")
        push!(priors, "$(p_names.diffusion) ~ $(_distribution_to_string(m.diffusion))")
        push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
        
        if hasproperty(spec.hyper, :habitat_data)
            push!(priors, "$(p_names.beta_habitat_diffusion) ~ Normal(0, 1.0)")
        end
        
        if !isnothing(m.r)
            push!(priors, "$(p_names.r) ~ $(_distribution_to_string(m.r))")
        end
        if !isnothing(m.K)
            push!(priors, "$(p_names.K) ~ $(_distribution_to_string(m.K))")
        end

        if hasproperty(spec.hyper, :mark_recapture_data) && !isnothing(m.beta_het)
            push!(priors, "$(p_names.beta_het) ~ $(_distribution_to_string(m.beta_het))")
        end
        
        push!(priors, "$(p_names.ure) ~ MvNormal(zeros($(spec.hyper.n_latent)), I)")
    end

    return join(priors, "\n    ")
end

"""
    get_updates(m::Movement, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing DSL code to construct the latent effect for the `Movement` component
and adds it to the linear predictor `eta`.

# Version
v1.0.0

# Arguments
- `m::Movement`: The `Movement` component instance.
- `spec::NamedTuple`: The full specification for this component instance.
- `arch::String`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx::Union{Int, Nothing}`: The index of the outcome variable.
- `M::NamedTuple`: The main model configuration.

# Returns
- A `String` containing the generated Turing code for the component's updates.
"""
function get_updates(
    m::Movement, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String

    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    hyper = spec.hyper

    if m.method in (:categorical, :stochastic_kernel, :density_gradient, :joint)
        if !hasproperty(hyper, :mark_recapture_data)
            error("The `$(m.method)` movement method requires `mark_recapture_data` or `telemetry_data`.")
        end

        hsi_calc = if m.method in (:density_gradient, :joint)
            """
            # --- Dynamic Habitat Suitability Index from Survey Density Linear Predictor ---
            eta_spatial = zeros(T, $(hyper.s_N))
            counts_spatial = zeros(Int, $(hyper.s_N))
            if hasproperty(M, :s_idx)
                @inbounds for i in 1:length(M.s_idx)
                    s_i = M.s_idx[i]
                    if 1 <= s_i <= $(hyper.s_N)
                        eta_spatial[s_i] += $(eta_target)[i]
                        counts_spatial[s_i] += 1
                    end
                end
                @inbounds for s in 1:$(hyper.s_N)
                    if counts_spatial[s] > 0
                        eta_spatial[s] /= counts_spatial[s]
                    end
                end
            end
            hsi_eff = eta_spatial
            """
        else
            """
            hsi_eff = spec_registry[:$(key)].hyper.habitat_data
            """
        end

        return """
        let
            $(hsi_calc)
            I_S_dense = Matrix{eltype($(p_names.velocity))}(I, $(hyper.s_N), $(hyper.s_N))
            T_diff = spec_registry[:$(key)].hyper.T_diff
            
            Gk_cache = map(1:$(hyper.mark_recapture_data.G)) do g
                # 1. Habitat-directed advection choice matrix A_g
                A_g = _build_A_ad(spec_registry[:$(key)].hyper.adj_rows, hsi_eff, $(p_names.gamma)[g], $(hyper.s_N))
                
                # 2. Stochastic transition kernel (strictly row-stochastic, no inversions, no clamping)
                v_g = $(p_names.velocity)[g]
                d_g = $(p_names.diffusion)[g]
                tot_g = v_g + d_g + 1e-6
                alpha_g = clamp(v_g / tot_g, 0.0, 1.0)
                rho_g   = clamp(1.0 / (1.0 + tot_g), 0.01, 0.99)
                
                w_move = 1.0 - rho_g
                w_adv  = w_move * alpha_g
                w_diff = w_move * (1.0 - alpha_g)
                
                P_g = (w_adv .* A_g) .+ (w_diff .* T_diff) .+ (rho_g .* I_S_dense)
                
                # 3. Multi-step power cache (P^1, P^2, ..., P^max_k)
                powers = Vector{Matrix{eltype(P_g)}}(undef, $(hyper.max_k))
                powers[1] = P_g
                for k_step in 2:$(hyper.max_k)
                    powers[k_step] = powers[k_step - 1] * P_g
                end
                powers
            end

            # 4. Telemetry transition log-likelihood
            for n in 1:length(spec_registry[:$(key)].hyper.mark_recapture_data.releases)
                rel = spec_registry[:$(key)].hyper.mark_recapture_data.releases[n]
                rec = spec_registry[:$(key)].hyper.mark_recapture_data.recaps[n]
                g   = spec_registry[:$(key)].hyper.mark_recapture_data.groups[n]
                k_n = spec_registry[:$(key)].hyper.mark_recapture_data.ks[n]
                
                p = Gk_cache[g][k_n][rel, :]
                ps = sum(p)
                T_el = eltype(p)
                p_norm = ps > eps(T_el) ? (p ./ ps) : fill(one(T_el) / $(hyper.s_N), $(hyper.s_N))
                
                Turing.@addlogprob! log(max(p_norm[rec], 1e-12))
            end
        end
        """
    end

    diffusion_field_code = if hasproperty(hyper, :habitat_data)
        """
        habitat_field = spec_registry[:$(key)].hyper.habitat_data
        diffusion_field = $(p_names.diffusion) .* exp.($(p_names.beta_habitat_diffusion) .* habitat_field)
        """
    else
        "diffusion_field = fill($(p_names.diffusion), $(hyper.s_N))"
    end

    common_setup = """
        # --- Movement Dynamics: $(key) ---
        $(diffusion_field_code)
        T_num_dyn = eltype(diffusion_field)
        dyn_field = zeros(T_num_dyn, $(hyper.s_N), $(hyper.t_N))
        innov_matrix = reshape($(p_names.ure), $(hyper.s_N), $(hyper.t_N))
        L_op = Matrix(spec_registry[:$(key)].hyper.L_template)
        A_op = Matrix(spec_registry[:$(key)].hyper.A_template)
    """

    has_reaction = !isnothing(m.r) && !isnothing(m.K)

    evolution_code = if m.method == :implicit
        """
        # Implicit Euler method (numerically stable, not AD-friendly, no reaction term)
        for t in 2:$(hyper.t_N)
            propagator_t = lu(Matrix(I($(hyper.s_N))) - $(p_names.velocity) * A_op - Diagonal(diffusion_field) * L_op)
            dyn_field[:, t] = (propagator_t \\ dyn_field[:, t-1]) + innov_matrix[:, t]
        end
        """
    elseif m.method == :explicit
        """
        # Explicit Euler method (AD-friendly, conditionally stable)
        propagator_t = $(p_names.velocity) .* A_op .+ Diagonal(diffusion_field) * L_op
        for t in 2:$(hyper.t_N)
            ad_diff_term = propagator_t * dyn_field[:, t-1]
            reaction_term = $(has_reaction ? "$(p_names.r) .* dyn_field[:, t-1] .* (1.0 .- dyn_field[:, t-1] ./ $(p_names.K))" : "zeros(T_num_dyn, $(hyper.s_N))")
            dyn_field[:, t] = dyn_field[:, t-1] + ad_diff_term + reaction_term + innov_matrix[:, t]
        end
        """
    else
        error("Unsupported method '$(m.method)' for Movement component.")
    end

    telemetry_likelihood_code = ""
    if hasproperty(hyper, :mark_recapture_data)
        if m.method == :explicit
          
            telemetry_likelihood_code = """
            # --- Mark-Recapture Telemetry Likelihood ---
            M_prop_tlm = Matrix(I($(hyper.s_N))) .- ($(p_names.velocity) .* A_op) .- (Diagonal(diffusion_field) * L_op)
            
            # Pre-factorize the transposed propagator for repeated solves using generic dense LU.
            F_prop_T = lu(transpose(M_prop_tlm))

            for m_idx in 1:size(spec_registry[:$(key)].hyper.mark_recapture_data, 1)
                u_rel = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 1])
                u_rec = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 2])
                time_steps = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 3])
                cov_m = spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 4]
                
                local p_unnorm
                if time_steps > 0
                    e_urel = zeros(T_num_dyn, $(hyper.s_N))
                    e_urel[u_rel] = 1.0
                    
                    y = e_urel
                    for _ in 1:time_steps
                        y = F_prop_T \\ y
                    end
                    p_unnorm = y
                else
                    p_unnorm = zeros(T_num_dyn, $(hyper.s_N))
                    p_unnorm[u_rel] = 1.0
                end

                indiv_scaling = exp($(p_names.beta_het) * cov_m)
                p_unnorm_scaled = abs.(p_unnorm) .^ indiv_scaling
                p_norm = p_unnorm_scaled / (sum(p_unnorm_scaled) + 1e-15)
                Turing.@addlogprob! log(max(p_norm[u_rec], 1e-12))
            end
            """
        elseif m.method == :implicit
            telemetry_likelihood_code = """
            # --- Implicit Mark-Recapture Telemetry Likelihood ---
            
            # Build propagator once, factorize once
            M_prop_tlm = Matrix(I($(hyper.s_N))) .- ($(p_names.velocity) .* A_op) .- (Diagonal(diffusion_field) * L_op)
            F_prop_T = lu(transpose(M_prop_tlm))

            for m_idx in 1:size(spec_registry[:$(key)].hyper.mark_recapture_data, 1)
                u_rel = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 1])
                u_rec = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 2])
                time_steps = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 3])
                cov_m = spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 4]
                
                local p_unnorm
                if time_steps > 0
                    e_urel = zeros(T_num_dyn, $(hyper.s_N))
                    e_urel[u_rel] = 1.0
                    
                    y = e_urel
                    for _ in 1:time_steps
                        y = F_prop_T \\ y
                    end
                    p_unnorm = y
                else
                    p_unnorm = zeros(T_num_dyn, $(hyper.s_N))
                    p_unnorm[u_rel] = 1.0
                end

                indiv_scaling = exp($(p_names.beta_het) * cov_m)
                p_unnorm_scaled = abs.(p_unnorm) .^ indiv_scaling
                p_norm = p_unnorm_scaled / (sum(p_unnorm_scaled) + 1e-15)
                Turing.@addlogprob! log(max(p_norm[u_rec], 1e-12))
            end
            """
        end
    end

    application_code = """
        dyn_field .*= $(p_names.sigma)
        
        # Vectorized update to the linear predictor using linear indexing
        st_idx = (M.t_idx .- 1) .* $(hyper.s_N) .+ M.s_idx
        $(p_names.sre) = vec(dyn_field)[st_idx]
        $(eta_target) = $(eta_target) .+ $(p_names.sre)
    """

    return """
    let
        $(common_setup)
        $(evolution_code)
        $(telemetry_likelihood_code)
        $(application_code)
    end
    """
end



"""
    get_effects(m::Movement, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the posterior distribution of the `Movement` component's effect.
This version is CPU-only and uses modern chain accessors.
"""
function get_effects(
    m::Movement, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end

    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    
    p_names = string.(keys(chain))
    
    key = spec.key
    hyper = spec.hyper
    L_op = hyper.L_template
    A_op = hyper.A_template
    s_N = hyper.s_N
    t_N = hyper.t_N # Training time steps

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_train = M.s_idx
    t_idx_train = M.t_idx

    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end

    t_idx_full = if !isnothing(PS) && hasproperty(PS.data, :t_idx)
        vcat(t_idx_train, PS.data.t_idx)
    else
        t_idx_train
    end
    
    N_total = length(s_idx_full)
    t_N_full = isempty(t_idx_full) ? 0 : maximum(t_idx_full)

    # Pre-calculate flat spatiotemporal index for efficient lookups
    st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names
        velocity_name = _find_parameter(p_names, string(p_names_k.velocity), k,
          is_multivariate_model)
        diffusion_name = _find_parameter(p_names, string(p_names_k.diffusion), k,
          is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
        r_name = hasproperty(p_names_k, :r) ? _find_parameter(p_names, string(p_names_k.r),
          k, is_multivariate_model) : ""
        K_name = hasproperty(p_names_k, :K) ? _find_parameter(p_names, string(p_names_k.K),
          k, is_multivariate_model) : ""

        if isempty(velocity_name) || isempty(diffusion_name) || isempty(sigma_name) || isempty(ure_name)
            @warn "Parameters for Movement component $(key) (outcome $k) not found.
              Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract all posterior samples
        velocity_samples = get_params_vector(chain, velocity_name, 1)[:, 1]
        diffusion_samples = get_params_vector(chain, diffusion_name, 1)[:, 1]
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ure_samples = get_params_vector(chain, ure_name, s_N * t_N)'
        
        r_samples = !isempty(r_name) ? get_params_vector(chain, r_name, 1)[:, 1] : zeros(n_samples)
        K_samples = !isempty(K_name) ? get_params_vector(chain, K_name, 1)[:, 1] : fill(Inf,
          n_samples)

        beta_habitat_samples = if hasproperty(hyper, :habitat_data)
            beta_name = _find_parameter(p_names,
              string(p_names_k.beta_habitat_diffusion), k, is_multivariate_model)
            if isempty(beta_name)
                beta_name = _find_parameter(p_names, "beta_habitat_diffusion_$(key)",
                  k, is_multivariate_model)
            end
            isempty(beta_name) ? nothing : get_params_vector(chain, beta_name, 1)[:, 1]
        else
            nothing
        end

        # Initialize a large matrix to hold the flattened dynamic field for all samples
        dyn_field_all_samples = zeros(Float64, s_N * t_N_full, n_samples)
        I_s = Matrix(I, s_N, s_N)

        # --- Sample-wise Reconstruction on the CPU ---
        for i in 1:n_samples
            # Construct diffusion field for the current sample
            diffusion_field = if !isnothing(beta_habitat_samples)
                habitat_field = hyper.habitat_data
                diffusion_samples[i] .* exp.(beta_habitat_samples[i] .* habitat_field)
            else
                fill(diffusion_samples[i], s_N)
            end

            # Prepare innovations matrix, extending for prediction if needed
            innov_matrix_train = reshape(ure_samples[:, i], s_N, t_N)
            innov_matrix_full = if t_N_full > t_N
                hcat(innov_matrix_train, randn(Float32, s_N, t_N_full - t_N))
            else
                innov_matrix_train[:, 1:t_N_full]
            end

            # Initialize the dynamic field for this sample
            dyn_field_sample = zeros(Float64, s_N, t_N_full)
            dyn_field_sample[:, 1] = innov_matrix_full[:, 1]

            # Time evolution loop
            if m.method == :implicit
                propagator_t = lu(I_s - velocity_samples[i] * A_op -
                  Diagonal(diffusion_field) * L_op)
                for t in 2:t_N_full
                    dyn_field_sample[:, t] = (propagator_t \ dyn_field_sample[:, t-1]) .+
                      innov_matrix_full[:, t]
                end
            else # :explicit
                propagator_t = velocity_samples[i] * A_op + Diagonal(diffusion_field) * L_op
                for t in 2:t_N_full
                    ad_diff_term = propagator_t * dyn_field_sample[:, t-1]
                    reaction_term = r_samples[i] .* dyn_field_sample[:, t-1] .* (1.0 .-
                      dyn_field_sample[:, t-1] ./ K_samples[i])
                    dyn_field_sample[:, t] = dyn_field_sample[:, t-1] + ad_diff_term +
                      reaction_term + innov_matrix_full[:, t]
                end
            end
            
            # Scale by sigma and store the flattened result
            dyn_field_sample .*= sigma_samples[i]
            dyn_field_all_samples[:, i] = vec(dyn_field_sample)
        end # end for

        # Index the full results matrix once using the pre-calculated flat indices
        effect_k = dyn_field_all_samples[st_idx_full, :]
        
        push!(structured_effects, effect_k)
    end # end for

    return (structured=structured_effects, noisy=structured_effects)
end
