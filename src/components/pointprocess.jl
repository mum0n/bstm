"""
    PointProcess <: ComponentModel

A component for modeling spatial or spatiotemporal point patterns. This component
acts as a wrapper for several point process models, selected via the `method` parameter.

# Version
v1.0.0 (2026-08-11)

# Mathematical Summary
Point process models describe the probability of observing events (points) in a given
domain. This component models the intensity function \$\\lambda(s)\$ of the process,
from which the number of points in a region is assumed to follow a Poisson distribution.

The component supports the following methods:

1.  **Log-Gaussian Cox Process (`:lgcp`)**:
    Models the log-intensity as a Gaussian Process (GP):
    \$\\log(\\lambda(s)) = Z(s)\$, where \$Z(s) \\sim \\mathcal{GP}(\\mu(s), k(s, s'))\$.
    The number of points in an area \$A\$ is \$N(A) \\sim \\text{Poisson}(\\int_A \\lambda(s) ds)\$.
    This is implemented by modeling \$Z(s)\$ with a GMRF on a discrete grid.

2.  **Log-Gamma Cox Process (`:lgmcp`)**:
    Models the intensity itself as a Gamma-distributed random field. This is often
    approximated by a Negative Binomial distribution for the counts, where the
    dispersion parameter is learned.

3.  **Shot-Noise Cox Process (`:sncp`)**:
    Models the intensity as the sum of kernel functions centered at a set of latent
    "parent" points:
    \$\\lambda(s) = \\sum_{i=1}^{N_p} A_i k(s, c_i; \\ell)\$
    where \$c_i\$ are the parent locations, \$A_i\$ are their amplitudes, and \$k\$ is a kernel
    with lengthscale \$\\ell\$.

# Inputs
- **Required**:
  - A spatial index variable (`s_idx`).
  - An adjacency matrix `W` (for `:lgcp`, `:lgmcp`).
  - Coordinate variables (for `:sncp`).
- **Optional (in `random()` call)**:
  - `method`: `:lgcp`, `:lgmcp`, or `:sncp`. Default: `:lgcp`.
  - `inner_model`: A `ComponentModel` for the latent field in `:lgcp` and `:lgmcp`. Default: `ICAR`.
  - `sigma`: Prior for the standard deviation of the latent field (for `:lgcp`).
  - `shape`: Prior for the shape/dispersion parameter (for `:lgmcp`).
  - `n_parents`: Number of parent points (for `:sncp`).
  - `amplitude`: Prior for parent point amplitudes (for `:sncp`).
  - `lengthscale`: Prior for the kernel lengthscale (for `:sncp`).
  - `grid_areas`: A vector of areas for each spatial unit, for integrating the intensity.

# Outputs (Parameter Names)
- Depends on the method. See individual model descriptions.
"""
struct PointProcess <: ComponentModel
    method::Symbol
    # Common for LGCP/LGMCP
    inner_model::Union{ComponentModel, Nothing}
    # LGCP
    sigma::Union{UnivariateDistribution, Nothing}
    # LGMCP
    shape::Union{UnivariateDistribution, Nothing}
    # SNCP
    n_parents::Union{Int, UnivariateDistribution, Nothing}
    amplitude::Union{UnivariateDistribution, Nothing}
    lengthscale::Union{UnivariateDistribution, Nothing}
    kernel::Union{String, Nothing}
end

# Constructor
COMPONENT_TYPE_REGISTRY[:pointprocess] = PointProcess
COMPONENT_CONSTRUCTORS[:pointprocess] = (p, params) -> begin
    method = get(params, :model, :lgcp) # User specifies model=:lgcp, etc.
    
    inner_model_obj = get(params, :inner_model_obj, nothing) # This is passed by resolve_technical_primitive
    
    PointProcess(
        method,
        inner_model_obj,
        get(p, :sigma, Exponential(1.0)),
        get(p, :shape, Exponential(1.0)),
        get(params, :n_parents, 50),
        get(p, :amplitude, Exponential(1.0)),
        get(p, :lengthscale, Gamma(2.0, 0.5)),
        string(get(params, :kernel, "se"))
    )
end

MODEL_TO_STRUCTURE_MAP[:pointprocess] = :spatial

"""
    get_datastructures!(m_type::Type{<:PointProcess}, M::Dict, mod_data::Dict)::Bool

Performs data-dependent setup for the `PointProcess` component.

# Rationale for Update
This version removes the incorrect and recursive call to `process_random_module!`.
The function's responsibility is now correctly limited to validating and processing
parameters specific to point process models, such as `grid_areas` or coordinate
data for the `:sncp` method. It assumes that the main `process_random_module!` has
already established the necessary spatial context (`s_N`).
"""
function get_datastructures!(m_type::Type{<:PointProcess}, M::Dict, mod_data::Dict)::Bool
    params = mod_data[:params]
    model_type = get(params, :model, :lgcp) # `model` is an alias for `method` here

    if model_type in [:lgcp, :lgmcp]
        if !haskey(M, :s_N)
            error("Point process models require a spatial context. Ensure a spatial index and adjacency matrix `W` are provided.")
        end
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
                    @warn "Could not resolve grid_areas for point process. Defaulting to unit areas."
                    M[:grid_areas] = ones(M[:s_N])
                end
            end
        else
            @warn "Point process model specified, but `grid_areas` parameter is missing. Defaulting to unit areas."
            M[:grid_areas] = ones(M[:s_N])
        end
    elseif model_type == :sncp
        if !hasproperty(M[:data], :s_x) || !hasproperty(M[:data], :s_y)
            error("ShotNoiseCoxProcess (`:sncp`) requires continuous spatial coordinates `s_x` and `s_y` to define the domain.")
        end
    end

    return true
end


function get_precomputes(m::PointProcess, M::NamedTuple, mod_data::Dict)::NamedTuple
    hyper_dict = Dict{Symbol, Any}()
    
    if m.method in [:lgcp, :lgmcp]
        inner_spec = build_model(m.inner_model, Dict(pairs(M)), mod_data)
        hyper_dict[:inner_spec] = inner_spec
        hyper_dict[:areas] = get(M, :grid_areas, ones(M.s_N))
        hyper_dict[:s_N] = M.s_N
        hyper_dict[:t_N] = get(M, :t_N, 1)
        
    elseif m.method == :sncp
        coords = M.data[!, [:s_x, :s_y]]
        x_min, x_max = extrema(coords.s_x)
        y_min, y_max = extrema(coords.s_y)
        
        hyper_dict[:domain_bounds] = (x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        hyper_dict[:areas] = get(M, :grid_areas, ones(M.s_N))
        hyper_dict[:s_N] = M.s_N
    end
    
    return NamedTuple(hyper_dict)
end

function get_priors(m::PointProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    if m.method == :lgcp
        return """
        $(p_names.sigma) ~ DynamicPPL.NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma)) # Prior for the standard deviation of the latent field
        $(p_names.innovations) ~ MvNormal(zeros(T, spec.hyper.s_N), I)
        """
    elseif m.method == :lgmcp
        return """
        $(p_names.shape) ~ $(_distribution_to_string(m.shape))
        $(p_names.innovations) ~ MvNormal(zeros(T, spec.hyper.s_N), I)
        """
    elseif m.method == :sncp
        n_parents_str = if m.n_parents isa Int
            string(m.n_parents)
        else
            string(p_names.n_parents)
        end
        
        priors_list = String[]
        if m.n_parents isa UnivariateDistribution
            push!(priors_list, "$(n_parents_str) ~ $(_distribution_to_string(m.n_parents))")
        end
        
        bounds = spec.hyper.domain_bounds
        push!(priors_list, "$(p_names.parent_locs_x) ~ filldist(Uniform(T($(bounds.x_min)), T($(bounds.x_max))), $(n_parents_str))")
        push!(priors_list, "$(p_names.parent_locs_y) ~ filldist(Uniform(T($(bounds.y_min)), T($(bounds.y_max))), $(n_parents_str))")
        push!(priors_list, "$(p_names.ls) ~ $(_distribution_to_string(m.lengthscale))")
        push!(priors_list, "$(p_names.amplitude) ~ filldist($(_distribution_to_string(m.amplitude)), $(n_parents_str))")
        
        return join(priors_list, "\n    ")
    end
    return ""
end

function get_updates(m::PointProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    if m.method == :lgcp
        return """
        # LGCP Model: $(spec.key)
        let
            Q_lgcp = spec.hyper.inner_spec.Q_template
            F_lgcp = cholesky(Symmetric(Matrix(Q_lgcp) + M.noise * I))
            spatial_component = $(p_names.sigma) .* (F_lgcp.U \\ $(p_names.innovations))
            
            log_intensity_surface = eta .+ spatial_component[M.s_idx]
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                A_i = spec.hyper.areas[M.s_idx[i]]
                lambda_i = exp(log_intensity_surface[i])
                
                Turing.@addlogprob! logpdf(Poisson(lambda_i * A_i), y_i)
            end
        end
        M.likelihood_handled = true
        """
    elseif m.method == :lgmcp
        return """
        # LGMCP Model: $(spec.key)
        let
            Q_lgmcp = spec.hyper.inner_spec.Q_template
            F_lgmcp = cholesky(Symmetric(Matrix(Q_lgmcp) + M.noise * I))
            spatial_component = exp.(F_lgmcp.U \\ $(p_names.innovations))
            
            mean_intensity_surface = exp.(eta) .* spatial_component[M.s_idx]
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                A_i = spec.hyper.areas[M.s_idx[i]]
                mu = mean_intensity_surface[i] * A_i
                
                r_nb = $(p_names.shape)
                p_nb = r_nb / (r_nb + mu)
                
                Turing.@addlogprob! logpdf(NegativeBinomial(r_nb, p_nb), y_i)
            end
        end
        M.likelihood_handled = true
        """
    elseif m.method == :sncp
        return """
        # SNCP Model: $(spec.key)
        let
            obs_locs = M.centroids
            parent_locs = hcat($(p_names.parent_locs_x), $(p_names.parent_locs_y))
            n_parents = length($(p_names.parent_locs_x))
            
            intensity_at_obs = zeros(T, M.s_N)
            for i in 1:M.s_N
                intensity_i = zero(T)
                for j in 1:n_parents
                    dist_sq = (obs_locs[i].x - parent_locs[j, 1])^2 + (obs_locs[i].y - parent_locs[j, 2])^2
                    kernel_val = exp(-0.5 * dist_sq / ($(p_names.ls)^2))
                    intensity_i += $(p_names.amplitude)[j] * kernel_val
                end
                intensity_at_obs[i] = intensity_i
            end
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                s_i = M.s_idx[i]
                A_s = spec.hyper.areas[s_i]
                lambda_s = intensity_at_obs[s_i] * A_s
                
                Turing.@addlogprob! logpdf(Poisson(lambda_s), y_i)
            end
        end
        M.likelihood_handled = true
        """
    end
    return ""
end


function get_effects(m::PointProcess, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    hyper = spec.hyper

    s_idx_full = if !isnothing(PS) && haskey(PS, :s_idx)
        vcat(M.s_idx, PS.s_idx)
    else
        M.s_idx
    end

    for k in 1:outcomes_N
        if m.method == :lgcp
            sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

            if isempty(sigma_name) || isempty(innovations_name)
                @warn "Parameters for LGCP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
            innovations_samples = get_params_vector(chain, innovations_name, hyper.s_N)
            
            Q_lgcp = hyper.inner_spec.Q_template
            F_lgcp = cholesky(Symmetric(Matrix(Q_lgcp) + M.noise * I))
            
            effect_k = zeros(Float64, N_total, n_samples)
            for i in 1:n_samples
                spatial_component = sigma_samples[i] .* (F_lgcp.U \ innovations_samples[i, :])
                effect_k[:, i] = view(spatial_component, s_idx_full)
            end
            push!(structured_effects, effect_k)

        elseif m.method == :lgmcp
            innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for LGMCP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            innovations_samples = get_params_vector(chain, innovations_name, hyper.s_N)
            Q_lgmcp = hyper.inner_spec.Q_template
            F_lgmcp = cholesky(Symmetric(Matrix(Q_lgmcp) + M.noise * I))

            effect_k = zeros(Float64, N_total, n_samples)
            for i in 1:n_samples
                spatial_component = exp.(F_lgmcp.U \ innovations_samples[i, :])
                effect_k[:, i] = view(spatial_component, s_idx_full)
            end
            push!(structured_effects, effect_k)

        elseif m.method == :sncp
            ls_name = _find_parameter(p_names_vec, string(spec.key), "ls", k, is_multivariate_model)
            amplitude_name = _find_parameter(p_names_vec, string(spec.key), "amplitude", k, is_multivariate_model)
            parent_locs_x_name = _find_parameter(p_names_vec, string(spec.key), "parent_locs_x", k, is_multivariate_model)
            parent_locs_y_name = _find_parameter(p_names_vec, string(spec.key), "parent_locs_y", k, is_multivariate_model)
            
            n_parents = m.n_parents isa Int ? m.n_parents : error("Dynamic n_parents not supported in reconstruction yet.")

            if isempty(ls_name) || isempty(amplitude_name) || isempty(parent_locs_x_name) || isempty(parent_locs_y_name)
                @warn "Parameters for SNCP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            ls_samples = get_params_vector(chain, ls_name, 1)[:, 1]
            amplitude_samples = get_params_vector(chain, amplitude_name, n_parents)
            parent_locs_x_samples = get_params_vector(chain, parent_locs_x_name, n_parents)
            parent_locs_y_samples = get_params_vector(chain, parent_locs_y_name, n_parents)

            obs_locs_train = M.centroids
            obs_locs_full = if !isnothing(PS) && hasproperty(PS, :centroids)
                vcat(obs_locs_train, PS.centroids)
            else
                obs_locs_train
            end

            effect_k = zeros(Float64, N_total, n_samples)
            
            for i in 1:n_samples
                parent_locs_i = hcat(parent_locs_x_samples[i, :], parent_locs_y_samples[i, :])
                intensity_at_obs = zeros(Float64, length(obs_locs_full))
                
                for obs_idx in 1:length(obs_locs_full)
                    intensity_obs = 0.0
                    for p_idx in 1:n_parents
                        dist_sq = (obs_locs_full[obs_idx].x - parent_locs_i[p_idx, 1])^2 + (obs_locs_full[obs_idx].y - parent_locs_i[p_idx, 2])^2
                        kernel_val = exp(-0.5 * dist_sq / (ls_samples[i]^2))
                        intensity_obs += amplitude_samples[i, p_idx] * kernel_val
                    end
                    intensity_at_obs[obs_idx] = intensity_obs
                end
                
                effect_k[:, i] = view(log.(intensity_at_obs .+ 1e-9), s_idx_full)
            end
            push!(structured_effects, effect_k)
        else
            @warn "Reconstruction for PointProcess method '$(m.method)' is not implemented. Returning zero effects."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
        end
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
