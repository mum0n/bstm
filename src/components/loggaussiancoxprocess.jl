# This file contains the proposed new and updated functions for the bstm refactoring.
 

"""
    LGCP <: ComponentModel

A component model for a Log-Gaussian Cox Process (LGCP), used for modeling spatial
or spatiotemporal point patterns. The model assumes the logarithm of the point process
intensity is a Gaussian Process. This implementation models aggregated counts in discrete
areal units.

# Fields
- `model::ComponentModel`: The inner model (e.g., `ICAR`, `Leroux`) that defines the
  structure of the latent Gaussian Process.
- `sigma::Distribution`: The prior for the standard deviation of the latent GP.
"""
struct LGCP <: ComponentModel
    model::ComponentModel
    sigma::Distribution
end

# Add to the central component constructor registry.
# The `resolve_technical_primitive` function is responsible for resolving the inner model
# and then calling this constructor.
COMPONENT_CONSTRUCTORS[:lgcp] = (p, params) -> begin
    inner_model_obj = get(params, :inner_model_obj, error("LGCP constructor requires an `inner_model_obj`."))
    LGCP(inner_model_obj, p.sigma)
end

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[LGCP] = :spatial

"""
    get_datastructures!(m_type::Type{<:LGCP}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `LGCP` component. It delegates spatial context
setup to its inner model, processes `grid_areas`, and reshapes the observation data
into a `[space x time]` matrix.
"""
function get_datastructures!(m_type::Type{<:LGCP}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]

    # Delegate data structure setup to the inner model
    inner_model_spec_node = get(params, :inner_model_node, error("LGCP model requires an `inner_model_node` parameter."))
    
    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    
    # The inner model is always spatial for LGCP
    inner_mod_data[:type] = :spatial
    
    # Call the spatial processor for the inner model to set up s_N, s_idx, W, etc.
    process_spatial_module!(M, inner_mod_data, Dict(), Dict())
    
    # Handle grid_areas, which depends on s_N being set.
    if haskey(params, :grid_areas)
        ga_val = params[:grid_areas]
        if ga_val isa Symbol && hasproperty(M[:data], ga_val)
            M[:grid_areas] = M[:data][!, ga_val]
        elseif ga_val isa AbstractVector
            M[:grid_areas] = ga_val
        else
            calling_mod = get(M, :calling_module, Main)
            try; M[:grid_areas] = Core.eval(calling_mod, ga_val);
            catch; @warn "Could not resolve grid_areas for LGCP. Defaulting to unit areas."; M[:grid_areas] = ones(M[:s_N]); end
        end
    else
        @warn "LGCP model specified, but `grid_areas` parameter is missing. Defaulting to unit areas."
        M[:grid_areas] = ones(get(M, :s_N, 1))
    end

    # Reshape the observation vector into a matrix [s_N x t_N]
    if M[:outcomes_N] > 1; error("LGCP currently only supports a single outcome variable representing counts."); end
    
    y_obs_vec = M[:y_obs]
    s_idx = M[:s_idx]
    t_idx = M[:t_idx]
    s_N = M[:s_N]
    t_N = M[:t_N]

    if length(y_obs_vec) != s_N * t_N
        @warn "Length of y_obs ($(length(y_obs_vec))) does not match s_N * t_N ($(s_N * t_N)). Reshaping might be incorrect."
    end

    y_obs_matrix = zeros(eltype(y_obs_vec), s_N, t_N)
    for i in 1:length(y_obs_vec)
        if s_idx[i] <= s_N && t_idx[i] <= t_N
            y_obs_matrix[s_idx[i], t_idx[i]] = y_obs_vec[i]
        end
    end
    M[:y_obs] = y_obs_matrix # Overwrite y_obs with the matrix form

    return true
end

"""
    get_precomputes(m::LGCP, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs pre-calculations for the `LGCP` component. It delegates to its inner model
to get the precision matrix structure and determines if a spatiotemporal model is being used.
"""
function get_precomputes(m::LGCP, M::NamedTuple, mod_data::Dict)::NamedTuple
    params = mod_data[:params]
    
    # Delegate pre-computation to the inner model
    inner_model_spec_node = get(params, :inner_model_node, error("LGCP model requires an `inner_model_node` parameter."))
    inner_mod_data = Dict(
        :key => Symbol("$(mod_data[:key])_inner"),
        :type => inner_model_spec_node.module_type,
        :variables => get(inner_model_spec_node.args, :positional_args, []),
        :params => inner_model_spec_node.args
    )
    inner_precomputes = get_precomputes(m.model, M, inner_mod_data)

    # Check for a temporal component to determine if it's spatiotemporal
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
    get_priors(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the `LGCP` component's priors.
"""
function get_priors(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    priors = ["$(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))"]
    
    if hasproperty(m.model, :rho)
        push!(priors, "$(p_names.rho) ~ NamedDist($(_distribution_to_string(m.model.rho)), :$(p_names.rho))")
    end
    
    n_latent_dims = spec.hyper.is_spatiotemporal ? "M.s_N * M.t_N" : "M.s_N"
    push!(priors, "$(p_names.raw) ~ NamedDist(MvNormal(zeros(T, $(n_latent_dims)), I), :$(p_names.raw))")
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code to construct the `LGCP` effect and its custom likelihood.
This component overrides the standard likelihood evaluation.
"""
function get_updates(m::LGCP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    reconstruction_code = if spec.hyper.is_spatiotemporal
        # Spatiotemporal case with Kronecker solver
        """
        local s_spec = spec_registry[:$(spec.key)].hyper.inner_precomputes
        local t_spec = spec_registry[:$(spec.key)].hyper.temporal_spec.hyper
        
        local C_s = cholesky(Symmetric(Matrix(s_spec.Q_template) + M.noise * I))
        
        local t_model_type = spec_registry[:$(spec.key)].hyper.temporal_spec.component_obj |> typeof |> Symbol
        local t_rho_val = $(hasproperty(spec.hyper.temporal_spec.component_obj, :rho) ? generate_full_variable_names(spec.hyper.temporal_spec, arch, outcome_idx).rho : "nothing")
        local Q_t_final = recompose_precision(t_model_type, t_spec.Q_template, T(1.0); extra_param=t_rho_val)
        local C_t = cholesky(Symmetric(Matrix(Q_t_final) + M.noise * I))
        
        local Z_matrix = reshape($(p_names.raw), M.s_N, M.t_N)
        
        local tmp_spatial = C_s.L' \\ Z_matrix
        latent_field_st = (transpose(C_t.L' \\ transpose(tmp_spatial))) .* $(p_names.sigma)
        """
    else
        # Purely spatial case
        """
        local Q_inner = spec_registry[:$(spec.key)].hyper.inner_precomputes.Q_template
        local F_inner = cholesky(Symmetric(Matrix(Q_inner) + M.noise * I))
        local spatial_component = $(p_names.sigma) .* (F_inner.L' \\ $(p_names.raw))
        latent_field_st = repeat(spatial_component, 1, M.t_N)
        """
    end

    return """
        # --- LGCP Model: $(spec.key) ---
        local latent_field_st = zeros(T, M.s_N, M.t_N)
        
        # 1. Reconstruct the latent spatiotemporal field Z(s,t)
        begin
            $(reconstruction_code)
        end

        # 2. Assemble the full log-intensity surface.
        local log_intensity_surface = zeros(T, M.s_N, M.t_N)
        for t in 1:M.t_N, s in 1:M.s_N
            obs_indices = findall(i -> M.s_idx[i] == s && M.t_idx[i] == t, 1:M.y_N)
            base_contribution = isempty(obs_indices) ? zero(T) : mean(view(eta, obs_indices))
            log_intensity_surface[s, t] = base_contribution + latent_field_st[s, t]
        end

        # 3. Point Process Likelihood Evaluation
        local grid_areas = spec_registry[:$(spec.key)].hyper.grid_areas
        for t in 1:M.t_N, s in 1:M.s_N
            local y_st = M.y_obs[s, t]
            local A_s = grid_areas[s]
            local Z_st = log_intensity_surface[s, t]
            
            Turing.@addlogprob! (y_st * (Z_st + log(A_s + T(1e-9))) - A_s * exp(Z_st))
        end
    """
end

"""
    get_effects(m::LGCP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the `LGCP` component's log-intensity surface from posterior samples.
"""
function get_effects(m::LGCP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, p_names::NamedTuple, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    sigma_samples = get(chain, p_names.sigma)
    raw_samples = get(chain, p_names.raw)
    
    s_N = M.s_N
    t_N = M.t_N
    noise = M.noise
    
    log_intensity_surfaces = zeros(n_samples, s_N, t_N)

    # This reconstruction is complex as it depends on the `eta` from other components.
    # A full reconstruction would require re-running the `eta` assembly for each sample.
    # For a simplified effect, we reconstruct only the GP part of the intensity.
    
    latent_field_samples = zeros(n_samples, s_N, t_N)

    if spec.hyper.is_spatiotemporal
        s_spec = spec.hyper.inner_precomputes
        t_spec_full = spec.hyper.temporal_spec
        t_spec = t_spec_full.hyper
        
        C_s = cholesky(Symmetric(Matrix(s_spec.Q_template) + noise * I))
        
        t_rho_samples = if hasproperty(t_spec_full.component_obj, :rho)
            get(chain, generate_full_variable_names(t_spec_full, M.model_arch, nothing).rho)
        else
            nothing
        end

        for i in 1:n_samples
            t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i]
            Q_t_final = recompose_precision(Symbol(lowercase(string(typeof(t_spec_full.component_obj)))), t_spec.Q_template, 1.0; extra_param=t_rho_val)
            C_t = cholesky(Symmetric(Matrix(Q_t_final) + noise * I))
            
            Z_matrix = reshape(raw_samples[i, :], s_N, t_N)
            tmp_spatial = C_s.L' \ Z_matrix
            latent_field_samples[i, :, :] = (transpose(C_t.L' \ transpose(tmp_spatial))) .* sigma_samples[i]
        end
    else
        Q_inner = spec.hyper.inner_precomputes.Q_template
        F_inner = cholesky(Symmetric(Matrix(Q_inner) + noise * I))
        for i in 1:n_samples
            spatial_component = sigma_samples[i] .* (F_inner.L' \ raw_samples[i, :])
            latent_field_samples[i, :, :] = repeat(spatial_component, 1, t_N)
        end
    end

    # For now, we return the GP part of the log-intensity.
    # A full reconstruction would require adding the `eta` base contribution per sample.
    mean_effect = mean(latent_field_samples, dims=1)[1,:,:]
    lower_ci = mapslices(x -> quantile(x, 0.025), latent_field_samples, dims=1)[1,:,:]
    upper_ci = mapslices(x -> quantile(x, 0.975), latent_field_samples, dims=1)[1,:,:]

    return (
        log_intensity_gp_part=(mean=mean_effect, lower=lower_ci, upper=upper_ci),
    )
end 