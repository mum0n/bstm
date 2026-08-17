"""
    IID <: ComponentModel

A simple Independent and Identically Distributed (IID) random effect, representing
unstructured noise or heterogeneity. Each latent effect is drawn independently from
the same normal distribution.

# Version
v1.1.2 (2026-08-14)

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
    structure = get(mod_data, :type, :spatial)
    
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

    template = build_structure_template(:iid, n)
    return (
        Q_template=template.matrix,
        U=template.U,
        L=template.L,
        scaling_factor=template.scaling_factor,
        n_latent=n
    )
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
        # Removed explicit `T` from `zeros` for better AD compatibility.
        push!(priors_acc, "$(p_names.innovations) ~ MvNormal(zeros($(n_latent)), I)")
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
            $(p_names.latent) = $(p_names.innovations) .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
    """

    centered_code = """
        # --- IID Component (Centered): $(spec.key) ---
        let
            # Removed explicit `T` from `zeros` for better AD compatibility.
            $(p_names.latent) ~ MvNormal(zeros($(spec.hyper.n_latent)), $(p_names.sigma)^2 * I)
            $(eta_target) .+= view($(p_names.latent), M.$(index_var))
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for IID component.")
    end
end




"""
    get_effects(m::IID, chain, spec, M, PS)

Reconstructs the IID effect from posterior samples.  
handle GPU arrays by moving sampled parameters to the device for computation and
moving the final results back to the CPU.
"""
function get_effects(
    m::IID, chain::Chains, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions and identify device ---
    n_samples = size(chain, 1) * size(chain, 3)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = names(chain)
    to_device = M.to_device
    
    n_latent = spec.hyper.n_latent

    # --- Index Handling: Combine training and prediction sets on device ---
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

    # The index in M is already on the device.
    idx_train_device = haskey(M, index_var_sym) ? M[index_var_sym] : to_device(ones(Int, M.y_N))
    
    idx_full_device = if !isnothing(PS) && hasproperty(PS.data, index_var_sym)
        idx_pred_cpu = PS.data[!, index_var_sym]
        vcat(idx_train_device, to_device(idx_pred_cpu))
    else
        idx_train_device
    end
    N_total = length(idx_full_device)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        
        if isempty(sigma_name)
            @warn "Sigma parameter for IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Samples are always on CPU
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        
        latent_field_samples_device = if m.method == :noncentered
            innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
            if isempty(innovations_name)
                @warn "Innovations for IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)
            
            # Move to device for computation
            innovations_device = to_device(innovations_samples_cpu)
            sigma_device = to_device(sigma_samples_cpu)
            
            # Broadcasting on the device
            innovations_device' .* sigma_device
        else # :centered
            latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
            if isempty(latent_name)
                @warn "Latent field for centered IID component $(spec.key) (outcome $k) not found. Returning zero-matrix."
                push!(structured_effects, zeros(Float64, N_total, n_samples))
                continue
            end
            latent_samples_cpu = get_params_matrix(chain, latent_name, n_latent)
            to_device(latent_samples_cpu')
        end
        
        # Indexing on the device
        effect_k_device = latent_field_samples_device[idx_full_device, :]
        
        # Move final result back to CPU
        push!(structured_effects, Array(effect_k_device))
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
