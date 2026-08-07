# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    SPDE <: ComponentModel

A component model for the Stochastic Partial Differential Equation (SPDE) effect,
which provides a link to Matérn fields. It uses a graph Laplacian-based precision matrix
that is parameterized by `kappa` (related to range) and `sigma` (related to variance).

# Fields
- `sigma::Distribution`: The prior distribution for the standard deviation of the SPDE effect.
- `kappa::Union{Distribution, Vector{<:Distribution}}`: The prior distribution(s) for the `kappa` parameter,
  which controls the spatial range and smoothness. Can be a single distribution or a vector for anisotropic effects.
"""
struct SPDE <: ComponentModel
    sigma::Distribution
    kappa::Union{Distribution, Vector{<:Distribution}}
end

# Add to the central component constructor registry.
# This constructor allows specifying the kappa parameter.
COMPONENT_CONSTRUCTORS[:spde] = (p, params) -> SPDE(p.sigma, p.kappa)

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[SPDE] = :spatial

"""
    get_datastructures!(m_type::Type{<:SPDE}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `SPDE` component.
It ensures that an adjacency matrix `W` is provided and sets up the spatial context
(`s_idx`, `s_N`) in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{<:SPDE}, M::Dict, mod_data::Dict)::Bool
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
                error("Could not evaluate `W` argument `$(w_val)` for SPDE component. Error: $e")
            end
        else
            M[:W] = w_val
        end
    end

    if !haskey(M, :W)
        error("SPDE model requires an adjacency matrix `W` to be provided.")
    end

    if !isa(M[:W], AbstractMatrix) || isempty(M[:W])
        error("Provided `W` for SPDE model is not a valid non-empty matrix.")
    end

    M[:s_N] = size(M[:W], 1)

    if isempty(variables)
        # If no variable is provided, assume s_idx is 1:s_N
        M[:s_idx] = collect(1:M[:s_N])
        @warn "Spatial index variable not provided for SPDE. Assuming `s_idx = 1:s_N`."
    else
        s_var_sym = Symbol(variables[1])
        if !hasproperty(M[:data], s_var_sym)
            error("Spatial index variable ':$s_var_sym' for SPDE model not found in data.")
        end
        M[:s_idx] = M[:data][!, s_var_sym]
    end

    return true
end

"""
    get_precomputes(m::SPDE, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for the `SPDE` component,
specifically building the graph Laplacian `Q_template` and its spectral decomposition.
"""
function get_precomputes(m::SPDE, M::NamedTuple, mod_data::Dict)::NamedTuple
    n = M.s_N
    W = M.W

    # Build the graph Laplacian (Q_template)
    # For SPDE, the precision matrix is (kappa^2 * I + Q_laplacian)' * (kappa^2 * I + Q_laplacian)
    # The Q_template here is the graph Laplacian itself.
    W_sym = sparse((W + W') .> 0) # Ensure symmetry and binary
    D = spdiagm(0 => vec(sum(W_sym, dims=2)))
    Q_template = D - W_sym

    # For SPDE, the rank deficiency is typically 0 if kappa > 0.
    # However, for spectral decomposition of the Laplacian, the rank deficiency is 1.
    # We compute the eigendecomposition of the Laplacian.
    eig_decomp = eigen(Symmetric(Matrix(Q_template)))
    U = eig_decomp.vectors
    L = eig_decomp.values

    # The scaling factor is usually 1.0 for SPDE as the kappa parameter handles scaling.
    # We can still compute it for consistency with other GMRFs, but it might not be used directly.
    scaling_factor = _compute_scaling_factor(L, 1) # Laplacian has rank deficiency 1
    
    # Rescale Q_template and eigenvalues for consistency, though kappa will re-scale.
    Q_template_scaled = Q_template ./ scaling_factor
    L_scaled = L ./ scaling_factor

    return (Q_template=Q_template_scaled, scaling_factor=scaling_factor, U=U, L=L_scaled, n_latent=n)
end

"""
    get_priors(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for the `SPDE` component's priors.
It defines the priors for `sigma`, `kappa`, and the latent field `raw`.
"""
function get_priors(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ NamedDist($(sigma_prior_str), :$(p_names.sigma))")

    if m.kappa isa Vector
        kappa_priors_str = join([_distribution_to_string(p) for p in m.kappa], ", ")
        push!(priors, "$(p_names.kappa) ~ NamedDist(Product([$(kappa_priors_str)]), :$(p_names.kappa))")
    else
        kappa_prior_str = _distribution_to_string(m.kappa)
        push!(priors, "$(p_names.kappa) ~ NamedDist($(kappa_prior_str), :$(p_names.kappa))")
    end
    
    # Latent field prior (non-centered parameterization)
    # raw ~ MvNormal(zeros(T, n_latent), I)
    
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, spec_registry[:$(spec.key)].precomputes.n_latent), I), :$(p_names.raw))")

    return join(priors, "\n    ")
end

"""
    get_updates(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code string for constructing the `SPDE` component's effect
and adding it to the linear predictor (`eta`).
It uses the Cholesky decomposition of the full precision matrix.
"""
function get_updates(m::SPDE, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    return """
        # --- SPDE Component: $(spec.key) ---
        local Q_laplacian = spec_registry[:$(spec.key)].precomputes.Q_template
        local n_latent = spec_registry[:$(spec.key)].precomputes.n_latent
        
        local kappa_val = $(p_names.kappa)
        local Q_kappa_term
        if kappa_val isa AbstractVector
            Q_kappa_term = Diagonal(kappa_val.^2)
        else
            Q_kappa_term = kappa_val^2 * I(n_latent)
        end

        # The SPDE precision matrix is (kappa^2 * I + Q_laplacian)' * (kappa^2 * I + Q_laplacian)
        local L_operator = Q_kappa_term + Q_laplacian
        
        # Form the full precision matrix
        local Q_final = Symmetric(L_operator' * L_operator)
        
        # Add noise for numerical stability and ensure positive definiteness.
        # This noise is separate from M.noise, which is for the likelihood.
        local F = cholesky(Matrix(Q_final + M.noise * I))
        
        # Sample latent field: latent ~ MvNormal(0, inv(Q_final))
        # which is equivalent to latent = sigma * inv(L') * raw
        local $(p_names.latent) = $(p_names.sigma) .* (F.L' \\ $(p_names.raw))
        
        $(eta_target) .+= $(p_names.latent)[M.s_idx]
    """
end



"""
    get_effects(m::SPDE, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `SPDE` component's effect from the MCMC chain's posterior samples.
This version returns the raw posterior samples for each outcome, not a summary.
"""
function get_effects(m::SPDE, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    n_latent = spec.hyper.n_latent
    noise = M.noise
    Q_laplacian = spec.hyper.Q_template

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        kappa_samples = get_params_vector(chain, string(p_names.kappa), m.kappa isa Vector ? length(m.kappa) : 1)
        raw_samples = get_params_vector(chain, string(p_names.raw), n_latent)

        effect_k = zeros(Float64, n_latent, n_samples)

        for i in 1:n_samples
            current_sigma = sigma_samples[i]
            current_kappa = kappa_samples[i, :]
            current_raw = raw_samples[i, :]
            
            local Q_kappa_term
            if m.kappa isa Vector
                Q_kappa_term = Diagonal(current_kappa.^2)
            else
                Q_kappa_term = current_kappa[1]^2 * I(n_latent)
            end
            
            L_operator = Q_kappa_term + Q_laplacian
            Q_final = Symmetric(L_operator' * L_operator)
            
            F = cholesky(Matrix(Q_final) + noise * I)
            
            effect_k[:, i] = current_sigma .* (F.L' \ current_raw)
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end

