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
  - `method`: `:explicit` or `:implicit`.
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
    r::Union{UnivariateDistribution, Nothing}
    K::Union{UnivariateDistribution, Nothing}
    beta_het::Union{UnivariateDistribution, Nothing}
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:movement] = Movement
COMPONENT_CONSTRUCTORS[:movement] = (p, params) -> Movement(
    p.velocity, p.diffusion, p.sigma,
    get(p, :r, nothing),
    get(p, :K, nothing),
    get(p, :beta_het, nothing),
    get(params, :method, :explicit)
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

function get_precomputes(m::Movement, M::NamedTuple, mod_data::Dict)::NamedTuple
    params = mod_data[:params]
    data = M.data
    variables = mod_data[:variables]

    W_from_params = get(params, :W, nothing)
    W_from_main = get(M, :W, nothing)
    W = isnothing(W_from_params) ? W_from_main : W_from_params

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
    t_N = M.t_N

    habitat_data = nothing
    if haskey(params, :habitat)
        habitat_val = params[:habitat]
        if habitat_val isa Symbol
            if !hasproperty(data, habitat_val)
                error("Habitat variable ':$habitat_val' not found in data.")
            end
            habitat_per_obs = data[!, habitat_val]
            habitat_aggregated = zeros(Float64, s_N)
            counts = zeros(Int, s_N)
            for i in 1:M.y_N
                s_i = M.s_idx[i]
                habitat_aggregated[s_i] += habitat_per_obs[i]
                counts[s_i] += 1
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

    L_template = build_structure_template(:besag, s_N; W=W).matrix
    
    W_dir = tril(W, -1)
    out_degree = sum(W_dir, dims=2)[:]
    D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
    A_template = D_inv * W_dir

    precomputes = Dict{Symbol, Any}(
        :L_template => L_template,
        :A_template => A_template,
        :n_latent => s_N * t_N,
        :s_N => s_N,
        :t_N => t_N
    )
    if !isnothing(habitat_data)
        precomputes[:habitat_data] = habitat_data
    end
    
    if haskey(params, :mark_recapture_data)
        telemetry_input = params[:mark_recapture_data]
        if telemetry_input isa DataFrame
            precomputes[:mark_recapture_data] = _process_telemetry_data(telemetry_input)
        else
            precomputes[:mark_recapture_data] = telemetry_input
        end
    end

    return NamedTuple(precomputes)
end

function _process_telemetry_data(telemetry_df::DataFrame)
    required_cols = [:tagid, :s_idx, :time, :tag]
    if !all(hasproperty(telemetry_df, col) for col in required_cols)
        error("Telemetry DataFrame must contain columns: :tagid, :s_idx, :time, :tag.")
    end

    transitions = []
    gdf = groupby(telemetry_df, :tagid)

    for sub_df in gdf
        if nrow(sub_df) < 2
            continue
        end
        
        # Sort observations for each individual by tag/time
        sort!(sub_df, :tag)

        for i in 1:(nrow(sub_df) - 1)
            release_row = sub_df[i, :]
            recapture_row = sub_df[i+1, :]

            release_unit = release_row.s_idx
            recapture_unit = recapture_row.s_idx
            time_steps = round(Int, recapture_row.time - release_row.time)
            
            # Use individual covariate if present, otherwise default to 0
            covariate = hasproperty(sub_df,
                :individual_covariate) ? release_row.individual_covariate : 0.0

            push!(transitions, [release_unit, recapture_unit, time_steps, covariate])
        end
    end

    if isempty(transitions)
        return Matrix{Float64}(undef, 0, 4)
    end

    return reduce(hcat, transitions)'
end

function get_priors(
    m::Movement, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.velocity) ~ $(_distribution_to_string(m.velocity))")
    push!(priors, "$(p_names.diffusion) ~ $(_distribution_to_string(m.diffusion))")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    
    if hasproperty(spec.hyper, :habitat_data)
        beta_habitat_diffusion_name = "beta_habitat_diffusion_$(spec.key)"
        push!(priors, "$(beta_habitat_diffusion_name) ~ Normal(0, 1.0)")
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
    
    push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")

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
"""

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

    diffusion_field_code = if hasproperty(hyper, :habitat_data)
        beta_habitat_diffusion_name = "beta_habitat_diffusion_$(key)"
        """
        habitat_field = spec_registry[:$(key)].hyper.habitat_data
        diffusion_field = $(p_names.diffusion) .* exp.($(beta_habitat_diffusion_name) .* habitat_field)
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
        L_op = spec_registry[:$(key)].hyper.L_template
        A_op = spec_registry[:$(key)].hyper.A_template
    """

    has_reaction = !isnothing(m.r) && !isnothing(m.K)

    evolution_code = if m.method == :implicit
        """
        # Implicit Euler method (numerically stable, not AD-friendly, no reaction term)
        for t in 2:$(hyper.t_N)
            propagator_t = lu(I($(hyper.s_N)) - $(p_names.velocity) * A_op - Diagonal(diffusion_field) * L_op)
            dyn_field[:, t] = (propagator_t \\ dyn_field[:, t-1]) + innov_matrix[:, t]
        end
        """
    elseif m.method == :explicit
        reaction_term_code = has_reaction ? "+ ($(p_names.r) .* dyn_field[:, t-1] .* (1.0 .-
          dyn_field[:, t-1] ./ $(p_names.K)))" : ""
        """
        # Explicit Euler method (AD-friendly, conditionally stable)
        propagator_t = $(p_names.velocity) * A_op + Diagonal(diffusion_field) * L_op
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
        telemetry_likelihood_code = """
        # --- Mark-Recapture Telemetry Likelihood ---
        M_prop_tlm = I($(hyper.s_N)) - $(p_names.velocity) * A_op - Diagonal(diffusion_field) * L_op
        
        # Pre-factorize the transposed propagator for repeated solves.
        F_prop_T = lu(transpose(M_prop_tlm))

        for m_idx in 1:size(spec_registry[:$(key)].hyper.mark_recapture_data, 1)
            u_rel = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 1])
            u_rec = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 2])
            time_steps = Int(spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 3])
            cov_m = spec_registry[:$(key)].hyper.mark_recapture_data[m_idx, 4]
            
            local p_unnorm
            if time_steps > 0
                # To get the u_rel-th row of inv(M_prop_tlm)^k, we solve
                # (M_prop_tlm')^k * x = e_urel, where e_urel is a basis vector.
                # This is done by k successive linear solves using the pre-computed LU
                #   factorization.
                e_urel = zeros(T_num_dyn, $(hyper.s_N))
                e_urel[u_rel] = 1.0
                
                y = e_urel
                for _ in 1:time_steps
                    y = F_prop_T \\ y
                end
                p_unnorm = y
            else
                # If time_steps is 0, transition is from a unit to itself with prob 1.
                p_unnorm = zeros(T_num_dyn, $(hyper.s_N))
                p_unnorm[u_rel] = 1.0
            end

            indiv_scaling = exp($(p_names.beta_het) * cov_m)
            p_unnorm_scaled = p_unnorm .^ indiv_scaling
            p_norm = p_unnorm_scaled / (sum(p_unnorm_scaled) + 1e-15)
            Turing.@addlogprob! log(max(p_norm[u_rec], 1e-12))
        end
        """
    end

    application_code = """
        dyn_field .*= $(p_names.sigma)
        
        # Vectorized update to the linear predictor using linear indexing
        st_idx = (M.t_idx .- 1) .* spec.hyper.s_N .+ M.s_idx
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
)::NamedTuple
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

        if isempty(velocity_name) || isempty(diffusion_name) || isempty(sigma_name) ||
          isempty(ure_name)
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
            beta_name = _find_parameter(p_names, "beta_habitat_diffusion_$(key)", k,
              is_multivariate_model)
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
        end

        # Index the full results matrix once using the pre-calculated flat indices
        effect_k = dyn_field_all_samples[st_idx_full, :]
        
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
