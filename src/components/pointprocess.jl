"""
    PointProcess <: ComponentModel

A component for modeling spatial or spatiotemporal point patterns. This component
acts as a wrapper for several point process models, selected via the `method` parameter.

# Version
v1.0.2 (2026-08-14)

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

function get_precomputes(m::PointProcess, M::NamedTuple, mod_data::Dict)::NamedTuple
    hyper_dict = Dict{Symbol, Any}()
    
    if m.method in [:lgcp, :lgmcp]
        if isnothing(m.inner_model)
            error("Point process method '$(m.method)' requires an `inner_model` to be specified.")
        end
        # Recursively get precomputes for the inner model that defines the GMRF structure.
        inner_hyper = get_precomputes(m.inner_model, M, mod_data)
        hyper_dict[:inner_hyper] = inner_hyper

        # Resolve grid areas
        if haskey(mod_data[:params], :grid_areas)
            ga_val = mod_data[:params][:grid_areas]
            if ga_val isa Symbol && hasproperty(M.data, ga_val)
                hyper_dict[:areas] = M.data[!, ga_val]
            elseif ga_val isa AbstractVector
                hyper_dict[:areas] = ga_val
            else
                calling_mod = get(M, :calling_module, Main)
                try
                    hyper_dict[:areas] = Core.eval(calling_mod, ga_val)
                catch
                    @warn "Could not resolve grid_areas for point process. Defaulting to unit areas."
                    hyper_dict[:areas] = ones(M.s_N)
                end
            end
        else
            @warn "Point process model specified, but `grid_areas` parameter is missing. Defaulting to unit areas."
            hyper_dict[:areas] = ones(M.s_N)
        end
        hyper_dict[:s_N] = M.s_N
        hyper_dict[:t_N] = get(M, :t_N, 1)
        
    elseif m.method == :sncp
        if !hasproperty(M.data, :s_x) || !hasproperty(M.data, :s_y)
            error("ShotNoiseCoxProcess (`:sncp`) requires continuous spatial coordinates `s_x` and `s_y` to define the domain.")
        end
        coords = M.data[!, [:s_x, :s_y]]
        x_min, x_max = extrema(coords.s_x)
        y_min, y_max = extrema(coords.s_y)
        
        hyper_dict[:domain_bounds] = (x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        hyper_dict[:areas] = get(M, :grid_areas, ones(M.s_N))
        hyper_dict[:s_N] = M.s_N
        hyper_dict[:centroids] = M.centroids
    end
    
    return NamedTuple(hyper_dict)
end

function get_priors(m::PointProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    if m.method == :lgcp
        n_latent = spec.hyper.inner_hyper.n_latent
        return """
        $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
        $(p_names.ure) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(p_names.ure))
        """
    elseif m.method == :lgmcp
        n_latent = spec.hyper.inner_hyper.n_latent
        return """
        $(p_names.shape) ~ $(_distribution_to_string(m.shape))
        $(p_names.ure) ~ DynamicPPL.NamedDist(MvNormal(zeros(T, $(n_latent)), I), :$(p_names.ure))
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
        push!(priors_list, "$(p_names.parent_locs_x) ~ filldist(Uniform($(bounds.x_min), $(bounds.x_max)), $(n_parents_str))")
        push!(priors_list, "$(p_names.parent_locs_y) ~ filldist(Uniform($(bounds.y_min), $(bounds.y_max)), $(n_parents_str))")
        push!(priors_list, "$(p_names.ls) ~ $(_distribution_to_string(m.lengthscale))")
        push!(priors_list, "$(p_names.amplitude) ~ filldist($(_distribution_to_string(m.amplitude)), $(n_parents_str))")
        
        return join(priors_list, "\n    ")
    end
    return ""
end

function get_updates(m::PointProcess, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    eta_target = (arch == "multivariate") ? "eta_latent" : "eta"
    
    if m.method == :lgcp
        return """
        # LGCP Model: $(key)
        let
            hyper = spec_registry[:$(key)].hyper
            Q_lgcp = hyper.inner_hyper.Q_template
            F_lgcp = cholesky(Symmetric(Matrix(Q_lgcp) + M.noise * I))
            spatial_component = $(p_names.sigma) .* (F_lgcp.L' \\ $(p_names.ure))
            
            log_intensity_surface = $(eta_target) .+ spatial_component[M.s_idx]
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                A_i = hyper.areas[M.s_idx[i]]
                lambda_i = exp(log_intensity_surface[i])
                
                Turing.@addlogprob! logpdf(Poisson(lambda_i * A_i), y_i)
            end
        end
        """
    elseif m.method == :lgmcp
        return """
        # LGMCP Model: $(key)
        let
            hyper = spec_registry[:$(key)].hyper
            Q_lgmcp = hyper.inner_hyper.Q_template
            F_lgmcp = cholesky(Symmetric(Matrix(Q_lgmcp) + M.noise * I))
            spatial_component = exp.(F_lgmcp.L' \\ $(p_names.ure))
            
            mean_intensity_surface = exp.($(eta_target)) .* spatial_component[M.s_idx]
            
            for i in 1:M.y_N
                y_i = M.y_obs[i]
                A_i = hyper.areas[M.s_idx[i]]
                mu = mean_intensity_surface[i] * A_i
                
                r_nb = $(p_names.shape)
                p_nb = r_nb / (r_nb + mu)
                
                Turing.@addlogprob! logpdf(NegativeBinomial(r_nb, p_nb), y_i)
            end
        end
        """
    elseif m.method == :sncp
        return """
        # SNCP Model: $(key)
        let
            hyper = spec_registry[:$(key)].hyper
            obs_locs = hyper.centroids
            parent_locs = hcat($(p_names.parent_locs_x), $(p_names.parent_locs_y))
            n_parents = length($(p_names.parent_locs_x))
            
            intensity_at_obs = zeros(eltype(parent_locs), hyper.s_N)
            for i in 1:hyper.s_N
                intensity_i = zero(eltype(parent_locs))
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
                A_s = hyper.areas[s_i]
                lambda_s = intensity_at_obs[s_i] * A_s
                
                Turing.@addlogprob! logpdf(Poisson(lambda_s), y_i)
            end
        end
        """
    end
    return ""
end


function get_effects(
    m::PointProcess, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    hyper = spec.hyper
    noise = M.noise

    # --- Index Handling: Combine training and prediction sets ---
    s_idx_train = M.s_idx # Spatial indices for training data
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx) # If prediction set is provided
        vcat(s_idx_train, PS.data.s_idx) # Combine training and prediction indices
    else
        s_idx_train # Otherwise, use only training indices
    end
    N_total = length(s_idx_full) # Total number of observations (training + prediction)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        
        if m.method == :lgcp
            sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)

            if isempty(sigma_name) || isempty(ure_name)
                @warn "Parameters for LGCP component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            # Extract samples (CPU)
            sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
            ure_samples = get_params_matrix(chain, ure_name, hyper.inner_hyper.n_latent) # (n_samples, n_latent)
            
            F_lgcp = hyper.inner_hyper.cholesky_factor # Cholesky factor for LGCP
            
            spatial_component_unscaled = F_lgcp.L' \ ure_samples' # Unscaled spatial component
            effect_k_latent = sigma_samples' .* spatial_component_unscaled # Scaled spatial component
            
            indexed_effects = effect_k_latent[s_idx_full, :]
            push!(structured_effects, indexed_effects)

        elseif m.method == :lgmcp
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "ure for LGMCP component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            ure_samples = get_params_matrix(chain, ure_name, hyper.inner_hyper.n_latent) # Innovations for LGMCP
            F_lgmcp = hyper.inner_hyper.cholesky_factor # Cholesky factor for LGMCP

            effect_k_latent = exp.(F_lgmcp.L' \ ure_samples') # Exponentiate to get intensity

            indexed_effects = effect_k_latent[s_idx_full, :]
            push!(structured_effects, indexed_effects)

        elseif m.method == :sncp
            ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)
            amplitude_name = _find_parameter(p_names, string(p_names_k.amplitude), k, is_multivariate_model)
            parent_locs_x_name = _find_parameter(p_names, string(p_names_k.parent_locs_x), k, is_multivariate_model)
            parent_locs_y_name = _find_parameter(p_names, string(p_names_k.parent_locs_y), k, is_multivariate_model)
            
            n_parents = m.n_parents isa Int ? m.n_parents : error("Dynamic n_parents not supported in reconstruction yet.")

            if isempty(ls_name) || isempty(amplitude_name) || isempty(parent_locs_x_name) || isempty(parent_locs_y_name)
                @warn "Parameters for SNCP component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end

            # Extract samples (CPU)
            ls_samples = get_params_vector(chain, ls_name, 1) # (n_samples, 1)
            amplitude_samples = get_params_matrix(chain, amplitude_name, n_parents) # (n_samples, n_parents)
            parent_locs_x_samples = get_params_matrix(chain, parent_locs_x_name, n_parents) # (n_samples, n_parents)
            parent_locs_y_samples = get_params_matrix(chain, parent_locs_y_name, n_parents) # (n_samples, n_parents)

            # Prepare observation locations
            obs_locs_train = hyper.centroids # Training centroids
            obs_locs_full = if !isnothing(PS) && hasproperty(PS, :centroids) # If prediction set has centroids
                vcat(obs_locs_train, PS.centroids) # Combine training and prediction centroids
            else
                obs_locs_train # Otherwise, use only training centroids
            end
            
            obs_locs_matrix = hcat([p.x for p in obs_locs_full], [p.y for p in obs_locs_full]) # Matrix of observation locations

            intensity_all_samples = zeros(Float64, length(obs_locs_full), n_samples) # Initialize intensity matrix
            
            for i in 1:n_samples # Iterate over each posterior sample
                parent_locs_i = hcat(parent_locs_x_samples[i, :], parent_locs_y_samples[i, :]) # Parent locations for current sample
                dist_sq = sum(obs_locs_matrix.^2, dims=2) .- 2 * (obs_locs_matrix * parent_locs_i') .+ sum(parent_locs_i.^2, dims=2)' # Squared distances
                kernel_vals = exp.(-0.5 .* dist_sq ./ (ls_samples[i, 1]^2)) # Kernel values
                intensity_at_obs = kernel_vals * amplitude_samples[i, :] # Intensity at observation locations
                intensity_all_samples[:, i] = intensity_at_obs # Store intensity
            end
            
            log_intensity = log.(intensity_all_samples .+ 1e-9) # Log-intensity
            indexed_effects = log_intensity[s_idx_full, :]
            push!(structured_effects, indexed_effects)
        else
            @warn "Reconstruction for PointProcess method '$(m.method)' is not implemented. Returning zero effects."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
        end
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 