# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    SAR <: ComponentModel

A component model for the Simultaneous Autoregressive (SAR) effect, also known as a proper CAR model.
The value at each location is modeled as a linear combination of its neighbors plus an independent
innovation term, leading to a precision matrix of the form `(I - ρW)'(I - ρW)`.

# Fields
- `rho::Distribution`: The prior distribution for the spatial autoregressive parameter `rho`.
- `sigma::Distribution`: The prior distribution for the standard deviation of the SAR innovations.
"""
struct SAR <: ComponentModel
    rho::Distribution
    sigma::Distribution
end

# Add to the central component constructor registry.
# Includes an alias for :proper_car.
COMPONENT_CONSTRUCTORS[:sar] = (p, params) -> SAR(p.rho, p.sigma)
COMPONENT_CONSTRUCTORS[:proper_car] = (p, params) -> SAR(p.rho, p.sigma)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[SAR] = :spatial

"""
    get_datastructures!(m_type::Type{<:SAR}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SAR` component.
It ensures that an adjacency matrix `W` is provided and sets up the spatial context
(`s_idx`, `s_N`) in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:SAR}, M::Dict, mod_data::Dict)::Bool
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
                error("Could not evaluate `W` argument `$(w_val)` for SAR component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("SAR model requires an adjacency matrix `W` to be provided.")
    end

    if !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Provided `W` for SAR model is not a valid non-empty matrix.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        # If no variable is provided, assume s_idx is 1:s_N
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for SAR. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for SAR model not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::SAR, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `SAR` component.
For SAR, the `Q_template` is the row-standardized adjacency matrix `W`. The full
precision matrix is constructed dynamically within the model.
"""
function get_precomputes(m::SAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = sparse(M.W) # Ensure W is sparse

    # Row-standardize the adjacency matrix W
    row_sums = sum(W, dims=2)
    non_zero_rows = findall(x -> x > 0, vec(row_sums))
    
    W_std = spzeros(Float64, n, n)
    if !isempty(non_zero_rows)
        D_inv = spdiagm(0 => 1.0 ./ row_sums[non_zero_rows])
        W_std[non_zero_rows, :] = D_inv * W[non_zero_rows, :]
    end

    # For SAR, the Q_template is the row-standardized adjacency matrix.
    # The full precision matrix (I - rho*W)'(I - rho*W) is constructed dynamically.
    # No spectral decomposition is pre-computed as the precision matrix depends on rho.
    return (Q_template=W_std, n_latent=n)
end

"""
    get_priors(m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `SAR` component's priors.
It defines the priors for `rho`, `sigma`, and the latent field `raw`.
"""
function get_priors(m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    rho_prior_str = _distribution_to_string(m.rho)
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    # Latent field prior (non-centered parameterization)
    # raw ~ MvNormal(zeros(T, n_latent), I)
    
    return """
        $(p_names.rho) ~ NamedDist($(rho_prior_str), :$(p_names.rho))
        $(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))
        $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `SAR` component's effect
and adding it to the linear predictor (`eta`).
It uses the Cholesky decomposition method for AD-compatibility.
"""
function get_updates(m::SAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- SAR Component: $(spec.key) ---
        local W_std = spec_registry[:$(spec.key)].precomputes.Q_template
        local n_latent = spec_registry[:$(spec.key)].precomputes.n_latent
        
        # Construct the operator (I - rho*W)
        local L_op = I(n_latent) - $(p_names.rho) * W_std
        
        # Form the precision matrix Q_final = (L_op' * L_op) / sigma^2
        # Add noise for numerical stability and ensure positive definiteness.
        local Q_final = Symmetric((L_op' * L_op) / ($(p_names.sigma)^2) + M.noise * I)
        
        # Perform Cholesky decomposition for non-centered parameterization.
        # Convert to dense Matrix for AD-compatibility.
        local F = cholesky(Matrix(Q_final))
        
        # Sample latent field: latent ~ MvNormal(0, inv(Q_final))
        # which is equivalent to latent = inv(L') * raw
        local $(p_names.latent) = F.L' \\ $(p_names.raw)
        
        $(eta_target) .+= $(p_names.latent)[M.s_idx]
    """
end


"""
    get_effects(m::SAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `SAR` component's effect from the MCMC chain's posterior samples.
This version returns the raw posterior samples for each outcome, not a summary.
"""
function get_effects(m::SAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise
    W_std = spec.hyper.Q_template

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        rho_samples = get_params_vector(chain, string(p_names.rho), 1)[:, 1]
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        for i in 1:n_samples
            current_rho = rho_samples[i]
            current_sigma = sigma_samples[i]
            current_raw = raw_samples[i, :]
            
            L_op = I(n_latent) - current_rho * W_std
            Q_final = Symmetric((L_op' * L_op) / (current_sigma^2) + noise * I)
            
            F = cholesky(Matrix(Q_final))
            effect_k[:, i] = F.L' \ current_raw
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end

