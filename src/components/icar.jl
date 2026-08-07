# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    ICAR <: ComponentModel

A component model for the Intrinsic Conditional Autoregressive (ICAR) effect,
also known as the Besag model. It implements a spatial random walk, where the
value at each location is conditionally dependent on its neighbors.

# Fields
- `sigma::Distribution`: The prior distribution for the standard deviation of the ICAR effect.
- `method::Symbol`: The computational method to use, e.g., `:spectral` (default) or `:cholesky`.
"""
struct ICAR <: ComponentModel
    sigma::Distribution
    method::Symbol
end

# Add to the central component constructor registry.
# This constructor allows specifying the computational method.
COMPONENT_CONSTRUCTORS[:icar] = (p, params) -> ICAR(p.sigma, get(params, :method, :spectral))

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[ICAR] = :spatial

"""
    get_datastructures!(m_type::Type{<:ICAR}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `ICAR` component.
It ensures that an adjacency matrix `W` is provided and sets up the spatial context
(`s_idx`, `s_N`) in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:ICAR}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    variables = mod_data[:variables]

    # Ensure W is available, either directly in params or in M
    if haskey(params, :W)
        w_val = params[:W]
        if w_val isa Expr || w_val isa Symbol
            calling_mod = get(M, :calling_module, Main)
            try
                M[:W] = Core.eval(calling_mod, w_val)
            catch e
                error("Could not evaluate `W` argument `$(w_val)` for ICAR component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("ICAR model requires an adjacency matrix `W` to be provided.")
    end

    if !hiskind(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Provided `W` for ICAR model is not a valid non-empty matrix.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        # If no variable is provided, assume s_idx is 1:s_N
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for ICAR. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for ICAR model not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::ICAR, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `ICAR` component,
specifically building the `Q_template` for the ICAR precision matrix
and its spectral decomposition if the method is `:spectral`.
"""
function get_precomputes(m::ICAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    # Build the Q_template for ICAR
    W_sym = sparse((W + W') .> 0) # Ensure symmetry and binary
    D = spdiagm(0 => vec(sum(W_sym, dims=2)))
    Q_template = D - W_sym

    rank_deficiency = 1 # ICAR has a rank deficiency of 1

    # Compute eigendecomposition for spectral sampling
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    # Compute scaling factor (geometric mean of non-zero eigenvalues)
    scaling_factor = _compute_scaling_factor(L, rank_deficiency)
    
    # Rescale Q_template and eigenvalues
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (Q_template=Q_template_scaled, scaling_factor=scaling_factor, U=U, L=L_scaled, n_latent=n)
end

"""
    get_priors(m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `ICAR` component's priors.
It defines the prior for `sigma` and the latent field `raw`.
"""
function get_priors(m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    # Latent field prior (non-centered parameterization)
    # For spectral method: raw ~ MvNormal(zeros(T, n_latent), I)
    # For cholesky method: raw ~ MvNormal(zeros(T, n_latent), I)
    
    return """
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `ICAR` component's effect
and adding it to the linear predictor (`eta`).
It supports both `:spectral` and `:cholesky` methods.
"""
function get_updates(m::ICAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    if m.method == :spectral
        # Spectral method: latent = sigma * U * Diagonal(1 ./ sqrt.(L .+ M.noise)) * raw
        # Sum-to-zero constraint is handled by setting the first eigenvalue to zero.
        return """
            # --- ICAR Component: $(spec.key) (Spectral Method) ---
            local diag_D = $(p_names.sigma) ./ sqrt.(spec_registry[:$(spec.key)].precomputes.L .+ M.noise)
            diag_D[1] = zero(T) # Enforce sum-to-zero constraint
            local $(p_names.latent) = spec_registry[:$(spec.key)].precomputes.U * (diag_D .* $(p_names.raw))
            $(eta_target) .+= $(p_names.latent)[M.s_idx]
        """
    else # :cholesky method (default or explicit)
        # Cholesky method: latent = sigma * inv(L') * raw
        # Sum-to-zero constraint is applied as a soft constraint.
        return """
            # --- ICAR Component: $(spec.key) (Cholesky Method) ---
            local Q_template = spec_registry[:$(spec.key)].precomputes.Q_template
            local F = cholesky(Symmetric(Matrix(Q_template) + M.noise * I))
            local $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
            Turing.@addlogprob! logpdf(Normal(zero(T), T(0.001) * spec_registry[:$(spec.key)].precomputes.n_latent), sum($(p_names.latent))) # Soft sum-to-zero
            $(eta_target) .+= $(p_names.latent)[M.s_idx]
        """
    end
end

"""
    get_effects(m::ICAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `ICAR` component's effect from the MCMC chain's posterior samples.
"""
function get_effects(m::ICAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)

    n_latent = spec.precomputes.n_latent
    noise = M.noise

    # Determine indices for reconstruction (training or prediction)
    idx_to_use = isnothing(PS) ? M.s_idx : PS.s_idx
    
    reconstructed_effects = zeros(n_samples, n_latent)

    if m.method == :spectral
        U = spec.precomputes.U
        L = spec.precomputes.L
        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_raw = raw_samples[i, :]
            diag_D = current_sigma ./ sqrt.(L .+ noise)
            diag_D[1] = 0.0 # Enforce sum-to-zero
            reconstructed_effects[i, :] = U * (diag_D .* current_raw)
        end
    else # :cholesky method
        Q_template = spec.precomputes.Q_template
        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_raw = raw_samples[i, :]
            F = cholesky(Symmetric(Matrix(Q_template) + noise * I))
            reconstructed_effects[i, :] = current_sigma .* (F.L' \ current_raw)
        end
    end

    mean_effect = mean(reconstructed_effects, dims=1)[:]
    lower_ci = quantile(reconstructed_effects, 0.025, dims=1)[:]
    upper_ci = quantile(reconstructed_effects, 0.975, dims=1)[:]

    indexed_mean = mean_effect[idx_to_use]
    indexed_lower = lower_ci[idx_to_use]
    indexed_upper = upper_ci[idx_to_use]

    return (structured=(mean=indexed_mean, lower=indexed_lower, upper=indexed_upper),)
end
