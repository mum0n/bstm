"""
    SVAR <: ComponentModel

A component for a Spatially Varying Autoregressive (SVAR) process. This component
models a temporal AR(1) process where the autoregressive coefficient `rho` varies
across spatial units. The spatial variation of `rho` is governed by a specified
GMRF model (e.g., ICAR, Leroux).

# Version
v1.9.4 (2026-08-08)

# Mathematical Summary
The SVAR model defines a spatiotemporal process \$\\psi_{it}\$ for spatial unit \$i\$
at time \$t\$ as a first-order autoregressive process with a spatially varying
persistence parameter \$\\rho_i\$:

\$\\psi_{it} = \\rho_i \\psi_{i,t-1} + \\epsilon_{it}, \\quad \\epsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2)\$

The spatially varying coefficient \$\\boldsymbol{\\rho} = (\\rho_1, \\dots, \\rho_{s_N})\$
is itself modeled as a latent Gaussian field, typically with a GMRF prior to
encourage spatial smoothness:

\$\\boldsymbol{\\rho}_{field} \\sim \\mathcal{N}(\\mathbf{0}, (\\tau_{\\rho} \\mathbf{Q}_{\\rho})^{-1})\$

where \$\\mathbf{Q}_{\\rho}\$ is the precision matrix of a spatial model (e.g., ICAR).
To ensure stationarity of the AR(1) process (\$-1 < \\rho_i < 1\$), the raw field
is transformed using the `tanh` function: \$\\rho_i = \\tanh(\\rho_{field, i})\$.

The process is initialized at \$t=1\$ from its stationary distribution:
\$\\psi_{i,1} \\sim \\mathcal{N}(0, \\frac{\\sigma^2}{1 - \\rho_i^2})\$

# Assumptions
- The temporal process within each spatial unit is first-order autoregressive.
- The spatial variation of the `rho` parameter can be captured by a GMRF.
- The innovations \$\\epsilon_{it}\$ are i.i.d. across space and time.

# Best Use Case
Modeling spatiotemporal phenomena where the degree of temporal persistence or
memory is expected to differ across geographical regions, such as modeling
environmental processes where local geography affects temporal dynamics.

# Key References
- Rushworth, A., Lee, D., & Mitchell, R. (2014). A Spatio-Temporal Model for
  Estimating the Long-Term Effects of Air Pollution on Respiratory Hospital
  Admissions in Greater London. *Spatial and Spatio-temporal Epidemiology*, 10, 29-38.
- Paul, M., & Held, L. (2011). Predictive assessment of a non-linear random
  effects model for multivariate time series of infectious disease counts.
  *Statistics in Medicine*, 30(10), 1118-1136.

# Fields
- `rho_model_type::Symbol`: The type of GMRF model for the spatial `rho` field.
- `rho_sigma::UnivariateDistribution`: Prior for the std. dev. of the `rho` field.
- `rho_rho::Union{UnivariateDistribution, Nothing}`: Prior for the mixing parameter
  of the `rho` field's model (used for `:bym2`, `:leroux`).
- `sigma::UnivariateDistribution`: Prior for the std. dev. of the AR(1) innovations.
"""
struct SVAR <: ComponentModel
    rho_model_type::Symbol
    rho_sigma::UnivariateDistribution
    rho_rho::Union{UnivariateDistribution, Nothing}
    sigma::UnivariateDistribution
end

COMPONENT_TYPE_REGISTRY[:svar] = SVAR

COMPONENT_CONSTRUCTORS[:svar] = (p, params) -> SVAR(
    get(params, :model, :icar),
    p.rho_sigma,
    get(p, :rho_rho, nothing),
    p.sigma
)

MODEL_TO_STRUCTURE_MAP[:svar] = :spacetime

"""
    get_datastructures!(m_type::Type{SVAR}, M::Dict, mod_data::Dict)::Bool

Data-dependent setup for the SVAR component.

This method establishes the spatiotemporal context for the model. It requires both a
spatial and a temporal index variable and sets up `s_idx`, `s_N`, `W`, `t_idx`,
and `t_N` in the main model configuration `M`.
"""
function get_datastructures!(m_type::Type{SVAR}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if length(variables) < 2
        error("SVAR requires a spatial and a temporal variable, e.g., `svar(s, t)`.")
    end

    data = M[:data]
    space_var_sym = Symbol(variables[1])
    time_var_sym = Symbol(variables[2])

    # --- Spatial Setup ---
    if !haskey(M, :s_N)
        if !hasproperty(data, space_var_sym)
            error("Spatial index ':$space_var_sym' for SVAR not found in data.")
        end
        s_idx = data[!, space_var_sym]
        M[:s_idx] = s_idx
        M[:s_N] = length(unique(s_idx))
        if !haskey(M, :W)
            @warn "Adjacency matrix `W` not provided for SVAR model."
        end
    end

    # --- Temporal Setup ---
    if !haskey(M, :t_N)
        if !hasproperty(data, time_var_sym)
            error("Time index ':$time_var_sym' for SVAR not found in data.")
        end
        time_opts = Dict(:time_method => get(mod_data[:params], :time_method, "regular"))
        tu_meta = assign_time_units(data[!, time_var_sym]; time_opts...)
        M[:t_idx] = tu_meta.idx
        M[:t_N] = tu_meta.N_cat
    end

    return true
end

"""
    get_precomputes(m::SVAR, M::NamedTuple, mod_data::Dict)::NamedTuple

Pre-computes structures for the SVAR component.
Builds the precision matrix template (`Q_rho_template`) and its spectral
decomposition for the spatial GMRF model governing the `rho` parameter. This is
essential for efficient, AD-compatible sampling of the `rho` field.
"""
function get_precomputes(m::SVAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    if s_N == 0
        error("Could not get number of spatial units for SVAR '$(mod_data[:key])'.")
    end
    W = get(M, :W, nothing)

    template = build_structure_template(m.rho_model_type, s_N; W=W)

    return (
        Q_rho_template=template.matrix,
        U_rho=template.U,
        L_rho=template.L,
        scaling_factor_rho=template.scaling_factor
    )
end

"""
    get_priors(m::SVAR, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the SVAR component's priors.
Defines priors for the hyperparameters (`rho_sigma`, `rho_rho`, `sigma`) and the
raw latent variables for the `rho` field and the AR innovations.
"""
function get_priors(m::SVAR, spec::NamedTuple, arch::String, outcome_idx, M)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    s_N = M.s_N
    n_latent_svar = M.s_N * M.t_N
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(v.rho_sigma) ~ $(_distribution_to_string(m.rho_sigma))")
        if m.rho_model_type in [:leroux, :bym2] && !isnothing(m.rho_rho)
            push!(priors_acc, "$(v.rho_rho) ~ $(_distribution_to_string(m.rho_rho))")
        end
        push!(priors_acc, "$(v.sigma) ~ $(_distribution_to_string(m.sigma))")
    end

    push!(priors_acc, "$(v.rho_raw) ~ MvNormal(zeros($(s_N)), I)")
    push!(priors_acc, "$(v.innov) ~ MvNormal(zeros($(n_latent_svar)), I)")

    return join(priors_acc, "\n    ")
end

"""
    get_updates(m::SVAR, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for the SVAR update logic.
This function generates code to:
1. Reconstruct the spatially varying `rho` field using a spectral method.
2. Evolve the state-space model over time for each spatial unit.
3. Map the resulting 2D spatiotemporal field to the 1D observation vector.
4. Add the final effect to the linear predictor `eta`.
"""
function get_updates(m::SVAR, spec::NamedTuple, arch::String, outcome_idx, M)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = is_multivariate ? "eta_latent[:, $(outcome_idx)]" : "eta"
    s_N = M.s_N
    t_N = M.t_N

    rho_recon_code = if m.rho_model_type in [:icar, :besag, :rw1, :rw2, :cyclic]
        """
        local D_rho = $(v.rho_sigma) ./ sqrt.(spec.hyper.L_rho .+ M.noise)
        local rho_field = spec.hyper.U_rho * (D_rho .* $(v.rho_raw))
        """
    elseif m.rho_model_type in [:leroux, :bym2]
        """
        local L_rho_mixed = (1 .- $(v.rho_rho)) .+ $(v.rho_rho) .* spec.hyper.L_rho
        local D_rho = $(v.rho_sigma) ./ sqrt.(L_rho_mixed .+ M.noise)
        local rho_field = spec.hyper.U_rho * (D_rho .* $(v.rho_raw))
        """
    else # Default to IID
        "local rho_field = $(v.rho_sigma) .* $(v.rho_raw)"
    end

    return """
    # --- SVAR Component: $(spec.key) ---
    # 1. Reconstruct the spatially-varying AR(1) coefficient `rho`.
    $(rho_recon_code)
    local rho_s = tanh.(rho_field) # Constrain rho to (-1, 1)

    # 2. Evolve the SVAR state-space.
    local latent_st = zeros(eltype(rho_s), $(s_N), $(t_N))
    local innov_grid = reshape($(v.innov), $(s_N), $(t_N))
    
    # Initialize t=1 with stationary variance.
    latent_st[:, 1] = ($(v.sigma) ./ sqrt.(1 .- rho_s.^2)) .* innov_grid[:, 1]
    
    # Evolve for t > 1.
    for t in 2:$(t_N)
        latent_st[:, t] = rho_s .* latent_st[:, t-1] .+ $(v.sigma) .* innov_grid[:, t]
    end
    
    # 3. Map the 2D latent field back to the 1D observation vector.
    $(v.latent) = [latent_st[M.s_idx[i], M.t_idx[i]] for i in 1:M.y_N]

    # 4. Add the effect to the linear predictor.
    $(eta_target) .+= $(v.latent)
    """
end

"""
    get_effects(m::SVAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the SVAR component's effect from posteriors.
Extracts posterior samples for all hyperparameters and latent variables. For each
sample, it reconstructs the `rho` field and re-runs the state-space evolution
to generate the full posterior distribution of the spatiotemporal effect.
"""
function get_effects(m::SVAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = []
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    s_N = M.s_N
    t_N = M.t_N
    n_latent_svar = s_N * t_N

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        v_shared = generate_full_variable_names(spec, M.model_arch, 1)
        
        rho_sigma_var = (is_shared) ? v_shared.rho_sigma : v.rho_sigma
        sigma_var = (is_shared) ? v_shared.sigma : v.sigma
        
        rho_sigma_samples = get_params_vector(chain, string(rho_sigma_var), 1)
        sigma_samples = get_params_vector(chain, string(sigma_var), 1)
        rho_raw_samples = get_params_vector(chain, string(v.rho_raw), s_N)
        innov_samples = get_params_vector(chain, string(v.innov), n_latent_svar)

        rho_rho_samples = nothing
        if m.rho_model_type in [:leroux, :bym2] && !isnothing(m.rho_rho)
            rho_rho_var = (is_shared) ? v_shared.rho_rho : v.rho_rho
            if !isnothing(rho_rho_var) && Symbol(rho_rho_var) in names(chain)
                rho_rho_samples = get_params_vector(chain, string(rho_rho_var), 1)
            end
        end
        
        T = eltype(chain.value)
        effect_k = Matrix{T}(undef, M.y_N, n_samples)

        U_rho = spec.hyper.U_rho
        L_rho = spec.hyper.L_rho

        for s in 1:n_samples
            rho_sigma_s = rho_sigma_samples[s, 1]
            sigma_s = sigma_samples[s, 1]
            rho_raw_s = rho_raw_samples[s, :]
            innov_s = innov_samples[s, :]

            local rho_field_s
            if m.rho_model_type in [:icar, :besag, :rw1, :rw2, :cyclic]
                D_rho_s = rho_sigma_s ./ sqrt.(L_rho .+ M.noise)
                rho_field_s = U_rho * (D_rho_s .* rho_raw_s)
            elseif m.rho_model_type in [:leroux, :bym2] && !isnothing(rho_rho_samples)
                rho_rho_s = rho_rho_samples[s, 1]
                L_rho_mixed = (1 .- rho_rho_s) .+ rho_rho_s .* L_rho
                D_rho_s = rho_sigma_s ./ sqrt.(L_rho_mixed .+ M.noise)
                rho_field_s = U_rho * (D_rho_s .* rho_raw_s)
            else # IID
                rho_field_s = rho_sigma_s .* rho_raw_s
            end
            rho_s_s = tanh.(rho_field_s)

            latent_st_s = zeros(T, s_N, t_N)
            innov_grid_s = reshape(innov_s, s_N, t_N)
            
            denom = sqrt.(max.(0, 1 .- rho_s_s.^2))
            latent_st_s[:, 1] = (sigma_s ./ (denom .+ M.noise)) .* innov_grid_s[:, 1]
            for t in 2:t_N
                latent_st_s[:, t] = rho_s_s .* latent_st_s[:, t-1] .+ sigma_s .* innov_grid_s[:, t]
            end
            
            for i in 1:M.y_N
                effect_k[i, s] = latent_st_s[M.s_idx[i], M.t_idx[i]]
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
