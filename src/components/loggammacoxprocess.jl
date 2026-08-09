
"""
    LogGammaCoxProcess <: ComponentModel

A component model for a Log-Gamma Cox Process, which models point patterns using a
latent Gamma process for the intensity. For aggregated counts over discrete areal
units, this corresponds to a Negative Binomial observation model.

# Version
v1.0.0 (2026-08-08)

# Mathematical Summary
A Log-Gamma Cox process models point counts \$y_{st}\$ in a space-time cell \$(s,t)\$
by assuming they follow a Poisson distribution whose rate \$\\lambda_{st}\$ is itself
a random variable drawn from a Gamma distribution. This is a Poisson-Gamma mixture.

1.  **Observation Model**: \$y_{st} \\sim \\text{Poisson}(\\lambda_{st})\$
2.  **Latent Intensity**: \$\\lambda_{st} \\sim \\text{Gamma}(\\alpha, \\theta_{st})\$

This hierarchical structure marginalizes to a Negative Binomial distribution for the
counts:
\$y_{st} \\sim \\text{NegativeBinomial}(r, p)\$
where the number of successes \$r = \\alpha\$ (the shape parameter), and the success
probability \$p = \\theta_{st} / (1 + \\theta_{st})\$.

The model parameterizes the mean of the Poisson process, \$\\mu_{st} = \\alpha \\theta_{st}\$.
The mean intensity \$\\mu_{st}\$ is modeled on the log scale as:
\$\\log(\\mu_{st}) = \\eta_{st} + Z_{st}\$
where \$\\eta_{st}\$ is the linear predictor from other model components (intercept,
fixed effects), and \$Z_{st}\$ is a latent Gaussian Process representing the log of
the Gamma process's scale parameter.

# Assumptions
- The point process intensity follows a Gamma distribution.
- The logarithm of the Gamma process's scale parameter can be modeled as a GMRF.

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
- `shape::Distribution`: The prior for the shape parameter of the Gamma Process, which
  also acts as the dispersion parameter for the Negative Binomial likelihood.
"""
struct LogGammaCoxProcess <: ComponentModel
    model::ComponentModel
    shape::Distribution
end

# Add to the central component constructor registry.
COMPONENT_TYPE_REGISTRY[:loggammacoxprocess] = LogGammaCoxProcess
COMPONENT_CONSTRUCTORS[:loggammacoxprocess] = (p, params) -> begin
    inner_model_obj = get(
        params, :inner_model_obj,
        error("LogGammaCoxProcess constructor requires an `inner_model_obj`.")
    )
    LogGammaCoxProcess(inner_model_obj, p.shape)
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
    
    priors = ["$(p_names.shape) ~ $(_distribution_to_string(m.shape))"]
    
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
"""
function get_updates(
    m::LogGammaCoxProcess, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    reconstruction_code = if spec.hyper.is_spatiotemporal
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
        latent_field_st = exp.(transpose(C_t.L' \\ transpose(tmp_spatial)))
        """
    else
        """
        local Q_inner = spec_registry[:$(spec.key)].hyper.inner_precomputes.Q_template
        local F_inner = cholesky(Symmetric(Matrix(Q_inner) + M.noise * I))
        local spatial_component = exp.(F_inner.L' \\ $(p_names.raw))
        latent_field_st = repeat(spatial_component, 1, M.t_N)
        """
    end

    return """
        # --- LogGammaCoxProcess Model: $(spec.key) ---
        let
            local latent_field_st = zeros(T, M.s_N, M.t_N)
            
            # 1. Reconstruct the latent spatiotemporal field Z(s,t)
            $(reconstruction_code)

            # 2. Assemble the full mean intensity surface.
            local mean_intensity_surface = zeros(T, M.s_N, M.t_N)
            local gamma_shape = $(p_names.shape)

            for t in 1:M.t_N, s in 1:M.s_N
                obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:M.y_N)
                base_contribution = isempty(obs_indices) ? 0.0 : mean(view(eta, obs_indices))
                mean_intensity_surface[s, t] = exp(base_contribution) * latent_field_st[s, t]
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
        end
    """
end

"""
    get_effects(m::LogGammaCoxProcess, chain, M::NamedTuple, ...)

Reconstructs the `LogGammaCoxProcess` component's mean intensity surface from
posterior samples.
"""
function get_effects(
    m::LogGammaCoxProcess, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    p_names = generate_full_variable_names(spec, M.model_arch, nothing)
    
    shape_samples = get_params_vector(chain, string(p_names.shape), 1)
    raw_samples = get_params_vector(chain, string(p_names.raw), spec.hyper.is_spatiotemporal ? M.s_N * M.t_N : M.s_N)
    
    s_N, t_N, noise = M.s_N, M.t_N, M.noise
    
    latent_field_samples = zeros(n_samples, s_N, t_N)

    if spec.hyper.is_spatiotemporal
        s_spec = spec.hyper.inner_precomputes
        t_spec_full = spec.hyper.temporal_spec
        t_spec = t_spec_full.hyper
        
        C_s = cholesky(Symmetric(Matrix(s_spec.Q_template) + noise * I))
        
        t_rho_samples = hasproperty(t_spec_full.component_obj, :rho) ?
            get_params_vector(chain, string(generate_full_variable_names(t_spec_full, M.model_arch, nothing).rho), 1) :
            nothing

        for i in 1:n_samples
            t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i, 1]
            Q_t_final = recompose_precision(Symbol(lowercase(string(typeof(t_spec_full.component_obj)))), t_spec.Q_template, 1.0; extra_param=t_rho_val)
            C_t = cholesky(Symmetric(Matrix(Q_t_final) + noise * I))
            
            Z_matrix = reshape(raw_samples[i, :], s_N, t_N)
            tmp_spatial = C_s.L' \ Z_matrix
            latent_field_samples[i, :, :] = exp.(transpose(C_t.L' \ transpose(tmp_spatial)))
        end
    else
        Q_inner = spec.hyper.inner_precomputes.Q_template
        F_inner = cholesky(Symmetric(Matrix(Q_inner) + noise * I))
        for i in 1:n_samples
            spatial_component = exp.(F_inner.L' \ raw_samples[i, :])
            latent_field_samples[i, :, :] = repeat(spatial_component', t_N, 1)'
        end
    end

    # For now, we return the Gamma Process part of the mean intensity.
    # A full reconstruction would require adding the `eta` base contribution per sample.
    return (structured=[latent_field_samples], noisy=[latent_field_samples])
end
