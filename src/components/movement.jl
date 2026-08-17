
"""
    Movement <: ComponentModel

A component for simulating population dynamics using an advection-diffusion
process on a discrete spatial graph or a continuous surface. This component models
the change in a latent field over time due to two primary processes: advection
(directional movement with a velocity field) and diffusion (random movement from
high to low concentration areas).

# Version
v1.0.4 (2026-08-14)

# Mathematical Summary
The component approximates the solution to the advection-diffusion partial
differential equation (PDE), which describes the transport of a substance or
quantity. A general reference can be found on Wikipedia's page for the
Convection-diffusion equation.

\$\\frac{\\partial C}{\\partial t} = \\nabla \\cdot (D \\nabla C) - \\nabla \\cdot (\\mathbf{v} C)\$

where:
- `C` is the concentration or density of the population.
- `D` is the diffusion coefficient, which can be spatially varying.
- `v` is the velocity field for advection.

This implementation uses a discrete state-space representation on a graph, where
the spatial operators are derived from the graph's adjacency matrix `W`. The
temporal evolution is modeled using either an explicit or implicit Euler scheme.
This approach is common in hierarchical Bayesian models for ecological processes,
such as those described by **Wikle (2003)** in "Hierarchical Bayesian models for
predicting the spread of ecological processes."

# Computational Methods
The `Movement` component supports multiple numerical methods for temporal evolution,
controlled by the `method` parameter in the `random()` call:

- **`:explicit` (Default, AD-friendly)**: Uses an explicit Euler time-stepping
  scheme. This method is fully compatible with automatic differentiation (AD) and
  thus suitable for gradient-based samplers like NUTS. However, it is only
  conditionally stable and may require small time steps or strong priors on
  `velocity` and `diffusion` to prevent numerical instability.
  The update rule is:
  `u_t = u_{t-1} + dt * (v*A*u_{t-1} + D*L*u_{t-1})`

- **`:implicit` (Didactic, Not AD-friendly)**: Uses an implicit Euler time-stepping
  scheme, which is unconditionally stable and often more robust for stiff problems
  (e.g., high diffusion). This method requires solving a linear system at each
  time step, which is done via an `lu` decomposition. This decomposition is not
  differentiable, making this method incompatible with AD. It is retained as a
  didactic alternative for use with gradient-free samplers.
  The update rule is:
  `(I - dt*(v*A + D*L)) * u_t = u_{t-1}`

# Continuous Space Formulation
If no adjacency matrix `W` is provided, the component can be configured to operate
on a continuous domain by discretizing it into a fine lattice based on a provided
`habitat_raster`. The values in this raster can then influence the diffusion
parameter, allowing for spatially-varying movement dynamics.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `s_idx`).
  - A temporal index variable (e.g., `year`).
  - Either an adjacency matrix `W` passed as a keyword argument to `@bstm`, or a
    `habitat_raster` matrix passed as a parameter to the `movement` module.
- **Optional**:
  - `habitat`: A `Symbol` pointing to a column in the data, or a `Vector` of length `s_N`.
  - `method`: A `Symbol` specifying the numerical method (`:explicit` or `:implicit`).
  - `velocity`: A `UnivariateDistribution` for the prior on the advection velocity. Default: `Normal(0, 0.5)`.
  - `diffusion`: A `UnivariateDistribution` for the prior on the diffusion rate. Default: `LogNormal(-1, 1)`.
  - `sigma`: A `UnivariateDistribution` for the prior on the process noise standard deviation. Default: `Exponential(1.0)`.

# Outputs (Parameter Names)
- `velocity_<key>`: The global advection velocity parameter.
- `diffusion_<key>`: The base diffusion parameter.
- `beta_habitat_diffusion_<key>`: The coefficient for the effect of the habitat
  covariate on diffusion (only if `habitat` is provided).
- `sigma_<key>`: The marginal standard deviation of the movement process.
- `innovations_<key>`: The latent innovations driving the process.
"""
struct Movement <: ComponentModel
    velocity::UnivariateDistribution
    diffusion::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:movement] = Movement
COMPONENT_CONSTRUCTORS[:movement] = (p, params) -> Movement(
    p.velocity, p.diffusion, p.sigma, get(params, :method, :explicit)
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
            if dr == 0 && dc == 0; continue; end
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
            if !(raster isa AbstractMatrix); error("`habitat_raster` must be a matrix."); end
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
            if !hasproperty(data, habitat_val); error("Habitat variable ':$habitat_val' not found in data."); end
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
            if length(habitat_val) != s_N; error("Provided `habitat` vector length ($(length(habitat_val))) does not match s_N ($(s_N))."); end
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

    return NamedTuple(precomputes)
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
    
    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")

    return join(priors, "\n    ")
end

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
        innov_matrix = reshape($(p_names.innovations), $(hyper.s_N), $(hyper.t_N))
        L_op = spec_registry[:$(key)].hyper.L_template
        A_op = spec_registry[:$(key)].hyper.A_template
    """

    evolution_code = if m.method == :implicit
        """
        # Implicit Euler method (numerically stable, not AD-friendly)
        for t in 2:$(hyper.t_N)
            propagator_t = lu(I($(hyper.s_N)) - $(p_names.velocity) * A_op - Diagonal(diffusion_field) * L_op)
            dyn_field[:, t] = (propagator_t \\ dyn_field[:, t-1]) + innov_matrix[:, t]
        end
        """
    elseif m.method == :explicit
        """
        # Explicit Euler method (AD-friendly, conditionally stable)
        propagator_t = $(p_names.velocity) * A_op + Diagonal(diffusion_field) * L_op
        for t in 2:$(hyper.t_N)
            dyn_field[:, t] = dyn_field[:, t-1] + propagator_t * dyn_field[:, t-1] + innov_matrix[:, t]
        end
        """
    else
        error("Unsupported method '$(m.method)' for Movement component.")
    end

    application_code = """
        dyn_field .*= $(p_names.sigma)
        for i in 1:M.y_N
            $(eta_target)[i] += dyn_field[M.s_idx[i], M.t_idx[i]]
        end
    """

    return """
    let
        $(common_setup)
        $(evolution_code)
        $(application_code)
    end
    """
end

function get_effects(
    m::Movement, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    key = spec.key
    hyper = spec.hyper
    L_op = hyper.L_template # Already on device
    A_op = hyper.A_template # Already on device
    s_N = hyper.s_N
    t_N = hyper.t_N # Training time steps

    # --- Index Handling: Combine training and prediction sets on device ---
    s_idx_train = M.s_idx # Already on device
    t_idx_train = M.t_idx # Already on device

    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, to_device(PS.data.s_idx))
    else
        s_idx_train
    end
    t_idx_full = if !isnothing(PS) && hasproperty(PS.data, :t_idx)
        vcat(t_idx_train, to_device(PS.data.t_idx))
    else
        t_idx_train
    end
    
    N_total = length(s_idx_full)
    t_N_full = isempty(t_idx_full) ? 0 : Array(maximum(t_idx_full))[] # Get max time on CPU

    # Pre-calculate flat spatiotemporal index for efficient lookups on the device
    st_idx_full = (t_idx_full .- 1) .* s_N .+ s_idx_full

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        # Find parameter names
        velocity_name = _find_parameter(p_names, string(p_names_k.velocity), k, is_multivariate_model)
        diffusion_name = _find_parameter(p_names, string(p_names_k.diffusion), k, is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)

        if isempty(velocity_name) || isempty(diffusion_name) || isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for Movement component $(key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract all posterior samples to CPU first
        velocity_samples_cpu = get_params_vector(chain, velocity_name, 1)[:, 1]
        diffusion_samples_cpu = get_params_vector(chain, diffusion_name, 1)[:, 1]
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, s_N * t_N)
        
        beta_habitat_samples_cpu = if hasproperty(hyper, :habitat_data)
            beta_name = _find_parameter(p_names, "beta_habitat_diffusion_$(key)", k, is_multivariate_model)
            isempty(beta_name) ? nothing : get_params_vector(chain, beta_name, 1)[:, 1]
        else
            nothing
        end

        # Move all sample data to the device at once
        velocity_samples_device = to_device(velocity_samples_cpu)
        diffusion_samples_device = to_device(diffusion_samples_cpu)
        sigma_samples_device = to_device(sigma_samples_cpu)
        innovations_samples_device = to_device(innovations_samples_cpu') # [s_N*t_N, n_samples]

        beta_habitat_samples_device = if !isnothing(beta_habitat_samples_cpu)
            to_device(beta_habitat_samples_cpu)
        else
            nothing
        end

        # Initialize a large matrix on the device to hold the flattened dynamic field for all samples
        dyn_field_all_samples_device = to_device(zeros(Float64, s_N * t_N_full, n_samples))
        I_s_device = to_device(Matrix(I, s_N, s_N))

        # --- Sample-wise Reconstruction on the Target Device ---
        for i in 1:n_samples
            # Construct diffusion field for the current sample on the device
            diffusion_field_device = if !isnothing(beta_habitat_samples_cpu)
                habitat_field_device = hyper.habitat_data # Already on device
                diffusion_samples_device[i] .* exp.(beta_habitat_samples_device[i] .* habitat_field_device)
            else
                fill(diffusion_samples_device[i], s_N)
            end

            # Prepare innovations matrix on the device, extending for prediction if needed
            innov_matrix_train_device = reshape(innovations_samples_device[:, i], s_N, t_N)
            innov_matrix_full_device = if t_N_full > t_N
                hcat(innov_matrix_train_device, to_device(randn(Float32, s_N, t_N_full - t_N)))
            else
                innov_matrix_train_device[:, 1:t_N_full]
            end

            # Initialize the dynamic field for this sample on the device
            dyn_field_sample_device = to_device(zeros(Float64, s_N, t_N_full))
            dyn_field_sample_device[:, 1] = innov_matrix_full_device[:, 1]

            # Time evolution loop on the device
            if m.method == :implicit
                # This method is not AD-friendly but can be run on GPU if lu is supported
                propagator_t = lu(I_s_device - velocity_samples_device[i] * A_op - Diagonal(diffusion_field_device) * L_op)
                for t in 2:t_N_full
                    dyn_field_sample_device[:, t] = (propagator_t \ dyn_field_sample_device[:, t-1]) .+ innov_matrix_full_device[:, t]
                end
            else # :explicit
                propagator_t = velocity_samples_device[i] * A_op + Diagonal(diffusion_field_device) * L_op
                for t in 2:t_N_full
                    dyn_field_sample_device[:, t] = dyn_field_sample_device[:, t-1] + propagator_t * dyn_field_sample_device[:, t-1] + innov_matrix_full_device[:, t]
                end
            end
            
            # Scale by sigma and store the flattened result
            dyn_field_sample_device .*= sigma_samples_device[i]
            dyn_field_all_samples_device[:, i] = vec(dyn_field_sample_device)
        end

        # Index the full results matrix once using the pre-calculated flat indices
        effect_k_device = dyn_field_all_samples_device[st_idx_full, :]
        
        # Move the final result for this outcome back to the CPU
        push!(structured_effects, Array(effect_k_device))
    end

    return (structured=structured_effects, noisy=structured_effects)
end