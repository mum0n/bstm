"""
    SVAR <: ComponentModel

A component for a Spatially Varying Autoregressive (SVAR) process. This component
models a temporal AR(1) process where the autoregressive coefficient `rho` varies
across spatial units. The spatial variation of `rho` is governed by a specified
GMRF model (e.g., ICAR, Leroux).

# Version
v1.9.5 (2026-08-10)

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

# Computational Methods (for the `rho` field)
- `:spectral` (default): An efficient, AD-safe method using spectral decomposition.
- `:cholesky`: An AD-safe didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse`: A non-AD-safe didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Fields
- `rho_model_type::Symbol`: The type of GMRF model for the spatial `rho` field.
- `rho_sigma::UnivariateDistribution`: Prior for the std. dev. of the `rho` field.
- `rho_rho::Union{UnivariateDistribution, Nothing}`: Prior for the mixing parameter
  of the `rho` field's model (used for `:bym2`, `:leroux`).
- `sigma::UnivariateDistribution`: Prior for the std. dev. of the AR(1) innovations.
- `method::Symbol`: The computational method for the `rho` field.
"""
struct SVAR <: ComponentModel
    rho_model_type::Symbol
    rho_sigma::UnivariateDistribution
    rho_rho::Union{UnivariateDistribution, Nothing}
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:svar] = SVAR

COMPONENT_CONSTRUCTORS[:svar] = (p, params) -> SVAR(
    get(params, :model, :icar),
    p.rho_sigma,
    get(p, :rho_rho, nothing),
    p.sigma,
    get(params, :method, :spectral)
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

Pre-computes structures for the SVAR component. Builds the precision matrix
template (`Q_rho_template`) and its spectral decomposition and Cholesky
factorization for the spatial GMRF model governing the `rho` parameter.
"""
function get_precomputes(m::SVAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = get(M, :s_N, 0)
    if s_N == 0
        error("Could not get number of spatial units for SVAR '$(mod_data[:key])'.")
    end
    W = get(M, :W, nothing)

    template = build_structure_template(m.rho_model_type, s_N; W=W)
    
    F_rho = cholesky(Symmetric(Matrix(template.matrix) + M.noise * I))

    return (
        Q_rho_template=template.matrix,
        U_rho=template.U,
        L_rho=template.L,
        scaling_factor_rho=template.scaling_factor,
        cholesky_factor_rho=F_rho
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

Generates the Turing code for the SVAR update logic, dispatching on the chosen
method for reconstructing the `rho` field.
"""
function get_updates(
    m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    v = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    s_N, t_N = M.s_N, M.t_N
    key = spec.key

    local rho_recon_code
    if m.method == :spectral
        rho_recon_code = """
            local hyper_rho = spec_registry[:$(key)].hyper
            local D_rho = $(v.rho_sigma) ./ sqrt.(hyper_rho.L_rho .+ M.noise)
            if $(m.rho_model_type) in [:icar, :besag]; D_rho[1] = 0.0; end
            local rho_field = hyper_rho.U_rho * (D_rho .* $(v.rho_raw))
        """
    elseif m.method == :cholesky
        rho_recon_code = """
            local F_rho = spec_registry[:$(key)].hyper.cholesky_factor_rho
            local rho_field_raw = F_rho.L' \\ $(v.rho_raw)
            if $(m.rho_model_type) in [:icar, :besag]; Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(s_N)), sum(rho_field_raw)); end
            local rho_field = rho_field_raw .* $(v.rho_sigma)
        """
    else # :cholesky_sparse
        rho_recon_code = """
            local Q_rho = spec_registry[:$(key)].hyper.Q_rho_template
            local F_rho = cholesky(Symmetric(Q_rho + M.noise * I))
            local rho_field_raw = F_rho.L' \\ $(v.rho_raw)
            if $(m.rho_model_type) in [:icar, :besag]; Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * $(s_N)), sum(rho_field_raw)); end
            local rho_field = rho_field_raw .* $(v.rho_sigma)
        """
    end

    return """
        # --- SVAR Component: $(key) ($(m.method)) ---
        let
            # 1. Reconstruct the spatially-varying AR(1) coefficient `rho`.
            $(rho_recon_code)
            local rho_s = tanh.(rho_field) # Constrain rho to (-1, 1)

            # 2. Evolve the SVAR state-space.
            local latent_st = zeros(eltype(rho_s), $(s_N), $(t_N))
            local innov_grid = reshape($(v.innov), $(s_N), $(t_N))
            
            latent_st[:, 1] = ($(v.sigma) ./ sqrt.(1 .- rho_s.^2)) .* innov_grid[:, 1]
            for t in 2:$(t_N)
                latent_st[:, t] = rho_s .* latent_st[:, t-1] .+ $(v.sigma) .* innov_grid[:, t]
            end
            
            # 3. Map the 2D latent field back to the 1D observation vector.
            $(v.latent) = [latent_st[M.s_idx[i], M.t_idx[i]] for i in 1:M.y_N]

            # 4. Add the effect to the linear predictor.
            $(eta_target) .+= $(v.latent)
        end
    """
end

"""
    get_effects(m::SVAR, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the SVAR component's effect from posteriors, dispatching on the
method used for sampling.
"""
function get_effects(
    m::SVAR, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate = outcomes_N > 1
    is_shared = get(spec.params, :shared, false)
    s_N, t_N, noise = M.s_N, M.t_N, M.noise
    n_latent_svar = s_N * t_N

    s_idx_full = isnothing(PS) ? M.s_idx : vcat(M.s_idx, PS.s_idx)
    t_idx_full = isnothing(PS) ? M.t_idx : vcat(M.t_idx, PS.t_idx)

    for k in 1:outcomes_N
        outcome_idx = is_multivariate ? k : nothing
        v = generate_full_variable_names(spec, M.model_arch, outcome_idx)
        v_shared = generate_full_variable_names(spec, M.model_arch, 1)
        
        rho_sigma_var = (is_shared) ? v_shared.rho_sigma : v.rho_sigma
        sigma_var = (is_shared) ? v_shared.sigma : v.sigma
        
        rho_sigma_samples = get_params_vector(chain, string(rho_sigma_var), 1)[:, 1]
        sigma_samples = get_params_vector(chain, string(sigma_var), 1)[:, 1]
        rho_raw_samples = get_params_vector(chain, string(v.rho_raw), s_N)
        innov_samples = get_params_vector(chain, string(v.innov), n_latent_svar)

        rho_rho_samples = nothing
        if m.rho_model_type in [:leroux, :bym2] && !isnothing(m.rho_rho)
            rho_rho_var = (is_shared) ? v_shared.rho_rho : v.rho_rho
            if !isnothing(rho_rho_var) && Symbol(rho_rho_var) in names(chain)
                rho_rho_samples = get_params_vector(chain, string(rho_rho_var), 1)[:, 1]
            end
        end
        
        effect_k = zeros(Float64, N_total, n_samples)
        hyper = spec.hyper

        for s in 1:n_samples
            local rho_field_s
            if m.method == :spectral
                D_rho_s = rho_sigma_samples[s] ./ sqrt.(hyper.L_rho .+ noise)
                if m.rho_model_type in [:icar, :besag]; D_rho_s[1] = 0.0; end
                rho_field_s = hyper.U_rho * (D_rho_s .* rho_raw_samples[s, :])
            else # :cholesky or :cholesky_sparse
                F_rho = hyper.cholesky_factor_rho
                rho_field_raw = F_rho.L' \ rho_raw_samples[s, :]
                if m.rho_model_type in [:icar, :besag]; rho_field_raw .-= mean(rho_field_raw); end
                rho_field_s = rho_field_raw .* rho_sigma_samples[s]
            end
            rho_s_s = tanh.(rho_field_s)

            latent_st_s = zeros(Float64, s_N, t_N)
            innov_grid_s = reshape(innov_samples[s, :], s_N, t_N)
            
            denom = sqrt.(max.(0, 1 .- rho_s_s.^2))
            latent_st_s[:, 1] = (sigma_samples[s] ./ (denom .+ noise)) .* innov_grid_s[:, 1]
            for t in 2:t_N
                latent_st_s[:, t] = rho_s_s .* latent_st_s[:, t-1] .+ sigma_samples[s] .* innov_grid_s[:, t]
            end
            
            for i in 1:N_total
                effect_k[i, s] = latent_st_s[s_idx_full[i], t_idx_full[i]]
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
