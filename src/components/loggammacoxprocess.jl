"""
    LogGammaCoxProcess <: ComponentModel

A component model for a Log-Gamma Cox Process, which models point patterns using a
latent Gamma process for the intensity. For aggregated counts over discrete areal
units, this corresponds to a Negative Binomial observation model. This implementation
approximates the log of the Gamma process's scale parameter with a Gaussian Process.

# Version
v1.0.2 (2026-08-10)

# Mathematical Summary
A Log-Gamma Cox process models point counts \$y_{st}\$ in a space-time cell \$(s,t)\$
by assuming they follow a Poisson distribution whose rate \$\\lambda_{st}\$ is itself
a random variable drawn from a Gamma distribution. This is a Poisson-Gamma mixture.

1.  **Observation Model**: \$y_{st} \\sim \\text{Poisson}(\\lambda_{st})\$
2.  **Latent Intensity**: \$\\lambda_{st} \\sim \\text{Gamma}(r, \\theta_{st})\$

This hierarchical structure marginalizes to a Negative Binomial distribution for the
counts:
\$y_{st} \\sim \\text{NegativeBinomial}(r, p)\$
where the number of successes \$r\$ is the shape parameter, and the success
probability \$p = \\theta_{st} / (1 + \\theta_{st})\$. The mean of this distribution is
\$E[y_{st}] = r \\cdot \\theta_{st}\$.

This component models the log of the mean intensity as:
\$\\log(E[y_{st}]) = \\eta_{st} + Z_{st}\$
where \$\\eta_{st}\$ is the linear predictor from other model components (intercept,
fixed effects), and \$Z_{st}\$ is a latent Gaussian Process with variance \$\\sigma^2\$.
This implies \$\\log(r \\theta_{st}) = \\eta_{st} + Z_{st}\$.

# Assumptions
- The point process intensity follows a Gamma distribution.
- The logarithm of the Gamma process's scale parameter can be approximated by a GP.

# Best Use Case
Modeling overdispersed spatial or spatiotemporal count data, such as the number of
disease cases or species occurrences in different regions over time. It provides a
more flexible alternative to a standard Poisson model when the variance of the
counts is greater than the mean.

# Key References
- Wolpert, R. L., & Ickstadt, K. (1998). *Poisson/gamma random field models for
  spatial statistics*. Biometrika, 85(2), 251-267.
- Wikipedia: Cox process

# Fields
- `model::ComponentModel`: The inner model (e.g., `ICAR`, `Leroux`) that defines the
  structure of the latent log-Gamma scale process.
- `sigma::Distribution`: The prior for the standard deviation of the latent GP.
- `shape::Distribution`: The prior for the shape parameter of the Gamma Process, which
  also acts as the dispersion parameter for the Negative Binomial likelihood.
- `method::Symbol`: The computational method. Can be `:spectral` (default, AD-safe),
  `:cholesky` (AD-safe, dense), or `:cholesky_sparse` (didactic, not AD-safe).
"""
struct LogGammaCoxProcess <: ComponentModel
    model::ComponentModel
    sigma::Distribution
    shape::Distribution
    method::Symbol
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:loggammacoxprocess] = LogGammaCoxProcess
COMPONENT_CONSTRUCTORS[:loggammacoxprocess] = (p, params) -> begin
    inner_model_obj = get(
        params, :inner_model_obj,
        error("LogGammaCoxProcess constructor requires an `inner_model_obj`.")
    )
    LogGammaCoxProcess(inner_model_obj, p.sigma, p.shape, get(params, :method, :spectral))
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[:loggammacoxprocess] = :spatial

"""
    get_datastructures!(m_type::Type{<:LogGammaCoxProcess}, M::Dict, mod_data::Dict)

Performs data-dependent setup for the `LogGammaCoxProcess` component. It delegates
spatial context setup to its inner model, processes `grid_areas`, and reshapes the
observation data into a `[space x time]` matrix.
"""
function get_datastructures!(
    m_type::Type{<:LogGammaCoxProcess}, M::Dict, mod_data::Dict
)::Bool
    params = mod_data[:params]

    # Delegate data structure setup to the inner model
    inner_model_spec_node = get(
        params, :inner_model_node,
        error("LogGammaCoxProcess model requires an `inner_model_node` parameter.")
    )
    
    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    
    inner_mod_data[:type] = :spatial
    process_spatial_module!(M, inner_mod_data, Dict(), Dict())
    
    if haskey(params, :grid_areas)
        ga_val = params[:grid_areas]
        if ga_val isa Symbol && hasproperty(M[:data], ga_val)
            M[:grid_areas] = M[:data][!, ga_val]
        elseif ga_val isa AbstractVector
            M[:grid_areas] = ga_val
        else
            calling_mod = get(M, :calling_module, Main)
            try
                M[:grid_areas] = Core.eval(calling_mod, ga_val)
            catch
                @warn "Could not resolve grid_areas. Defaulting to unit areas."
                M[:grid_areas] = ones(M[:s_N])
            end
        end
    else
        @warn "LogGammaCoxProcess model specified, but `grid_areas` is missing. " *
              "Defaulting to unit areas."
        M[:grid_areas] = ones(get(M, :s_N, 1))
    end

    if M[:outcomes_N] > 1
        error("LogGammaCoxProcess currently supports only a single outcome variable.")
    end
    
    y_obs_vec = M[:y_obs]
    s_idx, t_idx = M[:s_idx], M[:t_idx]
    s_N, t_N = M[:s_N], M[:t_N]

    y_obs_matrix = zeros(eltype(y_obs_vec), s_N, t_N)
    for i in 1:length(y_obs_vec)
        if s_idx[i] <= s_N && t_idx[i] <= t_N
            y_obs_matrix[s_idx[i], t_idx[i]] = y_obs_vec[i]
        end
    end
    M[:y_obs] = y_obs_matrix

    return true
end



"""
    get_precomputes(m::LogGammaCoxProcess, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs pre-calculations for the `LogGammaCoxProcess` component. It delegates to
its inner model to get the precision matrix structure and determines if a
spatiotemporal model is being used.
"""
function get_precomputes(
    m::LogGammaCoxProcess, M::NamedTuple, mod_data::Dict
)::NamedTuple
    params = mod_data[:params]
    
    inner_model_spec_node = get(
        params, :inner_model_node,
        error("LogGammaCoxProcess model requires an `inner_model_node` parameter.")
    )
    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)

    temporal_spec_idx = findfirst(s -> s.structure == :temporal, M.components)
    is_spatiotemporal = !isnothing(temporal_spec_idx)
    
    hyper = (
        inner_precomputes = inner_precomputes,
        grid_areas = M.grid_areas,
        is_spatiotemporal = is_spatiotemporal
    )
    
    if is_spatiotemporal
        hyper = merge(hyper, (temporal_spec = M.components[temporal_spec_idx],))
    end

    return hyper
end

"""
    get_priors(m::LogGammaCoxProcess, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code for the `LogGammaCoxProcess` component's priors.
"""
function get_priors(
    m::LogGammaCoxProcess, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = [
        "$(p_names.shape) ~ $(_distribution_to_string(m.shape))",
        "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))"
    ]
    
    if hasproperty(m.model, :rho)
        push!(priors, "$(p_names.rho) ~ $(_distribution_to_string(m.model.rho))")
    end
    
    n_latent_dims = spec.hyper.is_spatiotemporal ? "M.s_N * M.t_N" : "M.s_N"
    push!(priors, "$(p_names.raw) ~ MvNormal(zeros($(n_latent_dims)), I)")
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::LogGammaCoxProcess, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code to construct the `LogGammaCoxProcess` effect and its
custom likelihood. This component overrides the standard likelihood evaluation.
It supports three methods: `:spectral`, `:cholesky`, and `:cholesky_sparse`.
"""
function get_updates(
    m::LogGammaCoxProcess, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    # --- Common Likelihood Evaluation Block ---
    likelihood_code = """
        # 2. Assemble the full mean intensity surface.
        local mean_intensity_surface = zeros(T, M.s_N, M.t_N)
        local gamma_shape = $(p_names.shape)

        for t in 1:M.t_N, s in 1:M.s_N
            obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:M.y_N)
            base_contribution = isempty(obs_indices) ? 0.0 : mean(view(eta, obs_indices))
            
            # log(mu) = eta_base + Z_st
            local log_mu = base_contribution + (spec.hyper.is_spatiotemporal ? Z_st[s, t] : Z_st[s])
            mean_intensity_surface[s, t] = exp(log_mu)
        end

        # 3. Point Process Likelihood Evaluation using Negative Binomial
        local grid_areas = spec_registry[:$(spec.key)].hyper.grid_areas
        for t in 1:M.t_N, s in 1:M.s_N
            local y_st = M.y_obs[s, t]
            local A_s = grid_areas[s]
            local mu = mean_intensity_surface[s, t] * A_s
            
            local r_nb = gamma_shape
            local p_nb = r_nb / (r_nb + mu)
            local nb_dist = NegativeBinomial(r_nb, p_nb)
            Turing.@addlogprob! logpdf(nb_dist, y_st)
        end
        
        M[:likelihood_handled] = true
    """

    # --- Method 1: Spectral Decomposition (Default, AD-Safe) ---
    spectral_reconstruction = if spec.hyper.is_spatiotemporal
        """
        local s_spec = spec_registry[:$(spec.key)].hyper.inner_precomputes
        local t_spec_full = spec_registry[:$(spec.key)].hyper.temporal_spec
        local t_spec = t_spec_full.hyper
        
        local Us = s_spec.U
        local Ut = t_spec.U
        local Ls = s_spec.L
        local Lt = t_spec.L

        local s_rho_val = $(hasproperty(m.model, :rho) ? p_names.rho : "nothing")
        local t_rho_val = $(hasproperty(t_spec_full.component_obj, :rho) ? string(generate_full_variable_names(t_spec_full, arch, outcome_idx).rho) : "nothing")

        local diag_D_s = sqrt.(1.0 ./ ((1.0 .- s_rho_val) .+ s_rho_val .* Ls .+ M.noise))
        local diag_D_t = sqrt.(1.0 ./ ((1.0 .- t_rho_val) .+ t_rho_val .* Lt .+ M.noise))
        
        local Z_matrix = reshape($(p_names.raw), M.s_N, M.t_N)
        
        local tmp = Us' * Z_matrix * Ut
        local Z_st_raw = Us * ((diag_D_s .* tmp) .* diag_D_t') * Ut'
        local Z_st = $(p_names.sigma) .* Z_st_raw
        """
    else
        """
        local s_spec = spec_registry[:$(spec.key)].hyper.inner_precomputes
        local Us = s_spec.U
        local Ls = s_spec.L
        local rho_val = $(hasproperty(m.model, :rho) ? p_names.rho : "nothing")
        local diag_D = sqrt.(1.0 ./ ((1.0 .- rho_val) .+ rho_val .* Ls .+ M.noise))
        local spatial_component_raw = Us * (diag_D .* $(p_names.raw))
        local Z_st = $(p_names.sigma) .* spatial_component_raw
        """
    end
    spectral_code = """
    # --- LogGammaCoxProcess (Spectral): $(spec.key) ---
    let
        local Z_st
        # 1. Reconstruct the latent field Z(s,t) using spectral method.
        $(spectral_reconstruction)
        $(likelihood_code)
    end
    """

    # --- Method 2: Cholesky Decomposition (Dense, AD-Safe) ---
    cholesky_reconstruction = if spec.hyper.is_spatiotemporal
        """
        local s_spec = spec_registry[:$(spec.key)].hyper.inner_precomputes
        local t_spec_full = spec_registry[:$(spec.key)].hyper.temporal_spec
        local t_spec = t_spec_full.hyper
        
        local C_s = cholesky(Symmetric(Matrix(s_spec.Q_template) + M.noise * I))
        
        local t_model_type = t_spec_full.component_obj |> typeof |> Symbol
        local t_rho_val = $(hasproperty(t_spec_full.component_obj, :rho) ? string(generate_full_variable_names(t_spec_full, arch, outcome_idx).rho) : "nothing")
        local Q_t_final = recompose_precision(t_model_type, t_spec.Q_template, 1.0; extra_param=t_rho_val)
        local C_t = cholesky(Symmetric(Matrix(Q_t_final) + M.noise * I))
        
        local Z_matrix = reshape($(p_names.raw), M.s_N, M.t_N)
        
        local tmp_spatial = C_s.L' \\ Z_matrix
        local Z_st_raw = transpose(C_t.L' \\ transpose(tmp_spatial))
        local Z_st = $(p_names.sigma) .* Z_st_raw
        """
    else
        """
        local Q_inner = spec_registry[:$(spec.key)].hyper.inner_precomputes.Q_template
        local F_inner = cholesky(Symmetric(Matrix(Q_inner) + M.noise * I))
        local spatial_component_raw = F_inner.L' \\ $(p_names.raw)
        local Z_st = $(p_names.sigma) .* spatial_component_raw
        """
    end
    cholesky_code = """
    # --- LogGammaCoxProcess (Cholesky, AD-Safe): $(spec.key) ---
    let
        local Z_st
        # 1. Reconstruct the latent field Z(s,t) using dense Cholesky.
        $(cholesky_reconstruction)
        $(likelihood_code)
    end
    """

    # --- Method 3: Sparse Cholesky (Didactic, NOT AD-Safe) ---
    cholesky_sparse_reconstruction = replace(cholesky_reconstruction, "Matrix(" => "")
    cholesky_sparse_reconstruction = replace(cholesky_sparse_reconstruction, "Symmetric(" => "Symmetric(sparse(")
    cholesky_sparse_reconstruction = replace(cholesky_sparse_reconstruction, ") + M.noise * I)" => ") + M.noise * I))")

    cholesky_sparse_code = """
    # --- LogGammaCoxProcess (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
    # WARNING: This method is for didactic purposes and is NOT compatible with
    # automatic differentiation (e.g., NUTS sampler).
    let
        local Z_st
        # 1. Reconstruct the latent field Z(s,t) using sparse Cholesky.
        $(cholesky_sparse_reconstruction)
        $(likelihood_code)
    end
    """

    if m.method == :spectral
        return spectral_code
    elseif m.method == :cholesky
        return cholesky_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        error("Unsupported method '$(m.method)' for LogGammaCoxProcess.")
    end
end



"""
    get_effects(m::LogGammaCoxProcess, chain, M::NamedTuple, ...)

Reconstructs the `LogGammaCoxProcess` component's mean intensity surface from
posterior samples, dispatching on the method used for sampling.
"""
function get_effects(
    m::LogGammaCoxProcess, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    p_names = generate_full_variable_names(spec, M.model_arch, nothing)
    
    shape_samples = get_params_vector(chain, string(p_names.shape), 1)
    sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)
    n_latent_dims = spec.hyper.is_spatiotemporal ? M.s_N * M.t_N : M.s_N
    raw_samples = get_params_vector(chain, string(p_names.raw), n_latent_dims)
    
    s_N, t_N, noise = M.s_N, M.t_N, M.noise
    
    latent_field_samples = zeros(n_samples, s_N, t_N)

    if m.method == :spectral
        # Spectral Reconstruction
        if spec.hyper.is_spatiotemporal
            s_spec = spec.hyper.inner_precomputes
            t_spec_full = spec.hyper.temporal_spec
            t_spec = t_spec_full.hyper
            Us, Ut, Ls, Lt = s_spec.U, t_spec.U, s_spec.L, t_spec.L
            s_rho_samples = hasproperty(m.model, :rho) ? get_params_vector(chain, string(p_names.rho), 1) : nothing
            t_rho_samples = hasproperty(t_spec_full.component_obj, :rho) ? get_params_vector(chain, string(generate_full_variable_names(t_spec_full, M.model_arch, nothing).rho), 1) : nothing

            for i in 1:n_samples
                s_rho_val = isnothing(s_rho_samples) ? 0.5 : s_rho_samples[i, 1]
                t_rho_val = isnothing(t_rho_samples) ? 0.5 : t_rho_samples[i, 1]
                diag_D_s = sqrt.(1.0 ./ ((1.0 .- s_rho_val) .+ s_rho_val .* Ls .+ noise))
                diag_D_t = sqrt.(1.0 ./ ((1.0 .- t_rho_val) .+ t_rho_val .* Lt .+ noise))
                Z_matrix = reshape(raw_samples[i, :], s_N, t_N)
                tmp = Us' * Z_matrix * Ut
                Z_st_raw = Us * ((diag_D_s .* tmp) .* diag_D_t') * Ut'
                latent_field_samples[i, :, :] = sigma_samples[i] .* Z_st_raw
            end
        else
            s_spec = spec.hyper.inner_precomputes
            Us = s_spec.U
            Ls = s_spec.L
            rho_samples = hasproperty(m.model, :rho) ? get_params_vector(chain, string(p_names.rho), 1) : nothing
            for i in 1:n_samples
                rho_val = isnothing(rho_samples) ? 0.5 : rho_samples[i, 1]
                diag_D = sqrt.(1.0 ./ ((1.0 .- rho_val) .+ rho_val .* Ls .+ noise))
                spatial_component_raw = Us * (diag_D .* raw_samples[i, :])
                latent_field_samples[i, :, :] = sigma_samples[i] .* spatial_component_raw
            end
        end
    else # Cholesky or Cholesky-Sparse Reconstruction
        if spec.hyper.is_spatiotemporal
            s_spec = spec.hyper.inner_precomputes
            t_spec_full = spec.hyper.temporal_spec
            t_spec = t_spec_full.hyper
            
            # Decide whether to use dense or sparse Cholesky based on method
            Q_s_template = s_spec.Q_template
            C_s = if m.method == :cholesky_sparse
                cholesky(Symmetric(sparse(Q_s_template) + noise * I))
            else # :cholesky
                cholesky(Symmetric(Matrix(Q_s_template) + noise * I))
            end
            
            t_rho_samples = hasproperty(t_spec_full.component_obj, :rho) ? get_params_vector(chain, string(generate_full_variable_names(t_spec_full, M.model_arch, nothing).rho), 1) : nothing

            for i in 1:n_samples
                t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i, 1]
                Q_t_final = recompose_precision(Symbol(lowercase(string(typeof(t_spec_full.component_obj)))), t_spec.Q_template, 1.0; extra_param=t_rho_val)
                
                C_t = if m.method == :cholesky_sparse
                    cholesky(Symmetric(sparse(Q_t_final) + noise * I))
                else # :cholesky
                    cholesky(Symmetric(Matrix(Q_t_final) + noise * I))
                end
                
                Z_matrix = reshape(raw_samples[i, :], s_N, t_N)
                tmp_spatial = C_s.L' \ Z_matrix
                Z_st_raw = transpose(C_t.L' \ transpose(tmp_spatial))
                latent_field_samples[i, :, :] = sigma_samples[i] .* Z_st_raw
            end
        else
            Q_inner = spec.hyper.inner_precomputes.Q_template
            F_inner = if m.method == :cholesky_sparse
                cholesky(Symmetric(sparse(Q_inner) + noise * I))
            else # :cholesky
                cholesky(Symmetric(Matrix(Q_inner) + noise * I))
            end
            for i in 1:n_samples
                spatial_component_raw = F_inner.L' \ raw_samples[i, :]
                spatial_component = sigma_samples[i] .* spatial_component_raw
                latent_field_samples[i, :, :] = repeat(spatial_component', t_N, 1)'
            end
        end
    end

    # For now, we return the GP part of the log-intensity.
    # A full reconstruction would require adding the `eta` base contribution per sample.
    return (structured=[latent_field_samples], noisy=[latent_field_samples])
end
