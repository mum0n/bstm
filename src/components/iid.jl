"""
    IID <: ComponentModel

A simple Independent and Identically Distributed (IID) random effect, representing
unstructured noise or heterogeneity. Each latent effect is drawn independently from
the same normal distribution.

# Version
v1.2.0 (2026-08-19)

# Mathematical Summary
The IID component models a latent field \$\\phi\$ where each element \$\\phi_i\$ is drawn
independently from a zero-mean normal distribution with a shared standard deviation
\$\\sigma\$:
\$\\phi_i \\sim \\mathcal{N}(0, \\sigma^2)\$

The joint distribution is therefore a multivariate normal with a diagonal covariance
matrix:
\$\\boldsymbol{\\phi} \\sim \\mathcal{N}(\\mathbf{0}, \\sigma^2 I)\$
where \$I\$ is the identity matrix.

# Computational Methods
- `:noncentered` (Default, AD-friendly): Samples standard normal innovations and
  scales them by `sigma`. Recommended for gradient-based samplers like NUTS.
- `:centered` (Didactic, Not AD-friendly): Samples the latent effects directly from
  the `MvNormal` distribution defined by `sigma`. This can be less efficient for
  MCMC due to posterior correlations.

# Inputs
- **Required**:
  - An index variable (e.g., `group_id`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the IID effect.
- `innovations_<key>`: The raw standard normal innovations for the effect (for `:noncentered`).
- `latent_<key>`: The latent IID effect (for `:centered`).
"""
struct IID <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:iid] = IID
COMPONENT_CONSTRUCTORS[:iid] = (p, params) -> IID(
    p.sigma, get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:iid] = :any

function get_precomputes(m::IID, M::NamedTuple, mod_data::Dict)::NamedTuple
    structure = get(mod_data, :type, :any)
    
    n = if structure == :spatial
        get(M, :s_N, 0)
    elseif structure == :temporal
        get(M, :t_N, 0)
    elseif structure == :seasonal
        get(M, :u_N, 0)
    elseif structure == :mixed
        get(mod_data[:params], :n_cat, 0)
    else # smooth, etc.
        get(mod_data[:params], :nbins, 0)
    end

    if n == 0
        @warn "Could not determine dimension for IID component '$(mod_data[:key])'. " *
              "The component will have no effect."
    end

    return (n_latent=n,)
end

"""
    _iid_log_marginal_likelihood(y_residual, idx, n_latent, sigma, y_sigma, noise=1e-6)

Computes the exact closed-form log marginal likelihood for an IID random intercept component integrated out analytically.
"""
function _iid_log_marginal_likelihood(
    y_residual::AbstractVector{T},
    idx::AbstractVector{Int},
    n_latent::Int,
    sigma::T,
    y_sigma::T,
    noise::Real=1e-6
) where {T}
    N = length(y_residual)
    T_num = promote_type(T, typeof(noise))
    
    # Pre-accumulate observation counts, sums, and sum of squares per group
    N_g = zeros(T_num, n_latent)
    S_g = zeros(T_num, n_latent)
    SS_g = zeros(T_num, n_latent)
    
    for i in 1:N
        g = idx[i]
        if 1 <= g <= n_latent
            N_g[g] += one(T_num)
            S_g[g] += y_residual[i]
            SS_g[g] += y_residual[i]^2
        end
    end
    
    inv_sigma_y2 = one(T_num) / (y_sigma^2 + T_num(noise))
    var_alpha = sigma^2 + T_num(noise)
    var_y = y_sigma^2 + T_num(noise)
    
    log_lik = zero(T_num)
    
    for g in 1:n_latent
        if N_g[g] > zero(T_num)
            denom = N_g[g] * var_alpha + var_y
            log_det_g = log(denom / var_y)
            quad_g = inv_sigma_y2 * (SS_g[g] - (var_alpha / denom) * S_g[g]^2)
            log_lik += - (N_g[g] / 2) * log(2 * T_num(pi) * var_y) - (1 / 2) * log_det_g - (1 / 2) * quad_g
        end
    end
    
    return log_lik
end

function get_priors(
    m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    n_latent = spec.hyper.n_latent
    is_multivariate = (arch == "multivariate")
    is_shared = get(spec.params, :shared, false)
    is_first_outcome = (outcome_idx == 1 || isnothing(outcome_idx))

    priors_acc = String[]
    if !is_multivariate || (is_multivariate && (!is_shared || is_first_outcome))
        push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    end

    if m.method == :noncentered
        push!(priors_acc, "$(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)")
    end
    
    return join(priors_acc, "\n    ")
end

function get_updates(
    m::IID, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    index_var = if spec.structure == :spatial
        "s_idx"
    elseif spec.structure == :temporal
        "t_idx"
    elseif spec.structure == :seasonal
        "u_idx"
    elseif spec.structure == :mixed
        "mixed_idx_$(spec.var)"
    else
        string(spec.structure) * "_idx"
    end

    noncentered_code = """
        # --- IID Component (Non-Centered): $(spec.key) ---
        let
            $(p_names.sre) = $(p_names.ure) .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.sre), M.$(index_var))
        end
    """

    centered_code = """
        # --- IID Component (Centered): $(spec.key) ---
        let
            $(p_names.sre) ~ MvNormal(zeros(T, $(spec.hyper.n_latent)), $(p_names.sigma)^2 * I)
            $(eta_target) .+= view($(p_names.sre), M.$(index_var))
        end
    """

    marginalized_code = """
        # --- IID Component (Marginalized): $(spec.key) ---
        let
            y_residual = M.y_obs .- $(eta_target)
            log_lik_marginalized_$(spec.key) = _iid_log_marginal_likelihood(
                y_residual,
                M.$(index_var),
                spec_registry[:$(spec.key)].hyper.n_latent,
                $(p_names.sigma),
                y_sigma,
                M.noise
            )
            Turing.@addlogprob! log_lik_marginalized_$(spec.key)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    elseif m.method == :marginalized
        return marginalized_code
    else
        error("Unsupported method '$(m.method)' for IID component. Use :noncentered, :centered, or :marginalized.")
    end
end

"""
    get_effects(m::IID, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the IID effect from posterior samples. This version is CPU-only and
uses modern chain accessors.
"""
function get_effects(
    m::IID, chain, spec::NamedTuple, M::NamedTuple,
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
    noise = get(M, :noise, 1e-6)
    n_latent = spec.hyper.n_latent

    # --- Index Handling: Combine training and prediction sets on CPU ---
    index_var_sym = if spec.structure == :spatial
        :s_idx
    elseif spec.structure == :temporal
        :t_idx
    elseif spec.structure == :seasonal
        :u_idx
    elseif spec.structure == :mixed
        Symbol("mixed_idx_$(spec.var)")
    else
        Symbol(string(spec.structure) * "_idx")
    end

    idx_train = haskey(M, index_var_sym) ? M[index_var_sym] : ones(Int, M.y_N)
    
    idx_full = if !isnothing(PS) && hasproperty(PS.data, index_var_sym)
        idx_pred = PS.data[!, index_var_sym]
        vcat(idx_train, idx_pred)
    else
        idx_train
    end
    N_total = length(idx_full)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        
        if isempty(sigma_name)
            @warn "Sigma parameter for IID component $(spec.key) (outcome $(k)) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1) # (n_samples, 1)
        
        latent_field_samples = zeros(Float64, n_latent, n_samples)
        if m.method == :marginalized
            y_sigma_name = _find_parameter(p_names, "y_sigma", k, is_multivariate_model)
            y_sigma_samples = if !isempty(y_sigma_name)
                get_params_vector(chain, y_sigma_name, 1)[:, 1]
            else
                fill(1.0, n_samples)
            end
            
            y_vec = M.y_obs isa AbstractMatrix ? M.y_obs[:, k] : M.y_obs
            
            N_g = zeros(Float64, n_latent)
            S_g = zeros(Float64, n_latent)
            for i in 1:length(idx_train)
                g = idx_train[i]
                if 1 <= g <= n_latent
                    N_g[g] += 1.0
                    S_g[g] += y_vec[i]
                end
            end
            
            for s in 1:n_samples
                sig = sigma_samples[s, 1]
                y_sig = y_sigma_samples[s]
                
                var_alpha = sig^2 + noise
                var_y = y_sig^2 + noise
                
                for g in 1:n_latent
                    if N_g[g] > 0.0
                        denom = N_g[g] * var_alpha + var_y
                        mu_g = (var_alpha * S_g[g]) / denom
                        var_g = (var_alpha * var_y) / denom
                        latent_field_samples[g, s] = mu_g + sqrt(max(var_g, 1e-12)) * randn()
                    else
                        latent_field_samples[g, s] = sqrt(var_alpha) * randn()
                    end
                end
            end
        elseif m.method == :noncentered
            ure_name = _find_parameter(p_names, string(p_names_k.ure), k, is_multivariate_model)
            if isempty(ure_name)
                @warn "ure for IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            ure_samples = get_params_matrix(chain, ure_name, n_latent) # (n_samples, n_latent)
            
            latent_field_samples = ure_samples' .* sigma_samples' # (n_latent, n_samples)
        else # :centered
            sre_name = _find_parameter(p_names, string(p_names_k.sre), k, is_multivariate_model)
            if isempty(sre_name)
                @warn "sre for centered IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            sre_samples = get_params_matrix(chain, sre_name, n_latent)
            latent_field_samples = sre_samples'
        end
        
        effect_k = latent_field_samples[idx_full, :]
        
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end 