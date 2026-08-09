"""
    Warp <: ComponentModel

A component for a non-stationary Gaussian Process using a warping function. This
component implements a simple Deep GP structure with one hidden layer. The input
coordinates are first "warped" by a non-linear function (approximated by RFFs),
and then a standard RFF-based GP is applied to these warped coordinates.

# Version
v1.0.1 (2026-08-09)

# Mathematical Summary
A Warped Gaussian Process models a non-stationary function \$f(x)\$ by composing a
standard, stationary GP \$h(\\cdot)\$ with a non-linear warping function \$g(x)\$.
The model is defined as:

\$f(x) = h(g(x))\$

The warping function \$g(x)\$ is typically defined as an offset from the original
coordinates:

\$g(x) = x + \\text{offset}(x)\$

Both the main process \$h(\\cdot)\$ and the offset function \$\\text{offset}(x)\$ are
modeled as GPs, which are approximated using Random Fourier Features (RFFs) for
computational scalability. This allows the model to learn how to stretch and
compress the input space to best fit the data, effectively making the lengthscale
of the main GP dependent on the input location \$x\$.

# Distinction from other GP approximations
- **Stationary GP**: Assumes the correlation between two points depends only on their
  distance, not their absolute location. A Warped GP relaxes this by making the
  "effective distance" depend on location.
- **SVC (Spatially Varying Coefficients)**: Models non-stationarity by allowing a
  covariate's regression coefficient to vary spatially. Warped GPs model
  non-stationarity in the underlying smooth function itself.

# Best Use Case
Modeling complex, non-stationary spatial or temporal phenomena where the degree of
smoothness or correlation length changes across the domain. For example, modeling
an environmental process that is highly variable near a source but smooths out
further away.

# Key References
- Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS.
- Snelson, E., van Gael, J., & Ghahramani, Z. (2004). *Warped Gaussian Processes*. NIPS.

# Fields
- `lengthscale::Union{Distribution, Vector{<:Distribution}}`: Prior for the kernel lengthscale(s) of the main GP.
- `sigma::Distribution`: Prior for the std. dev. of the main GP's coefficients.
- `n_features::Int`: Number of random features for both the warping and main RFF layers.
- `kernel::String`: Name of the kernel to approximate (e.g., "se", "matern32").
"""
struct Warp <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_features::Int
    kernel::String
end

COMPONENT_TYPE_REGISTRY[:warp] = Warp

COMPONENT_CONSTRUCTORS[:warp] = (p, params) -> Warp(
    p.lengthscale,
    p.sigma,
    get(params, :n_features, 20),
    string(get(params, :kernel, "se"))
)

MODEL_TO_STRUCTURE_MAP[:warp] = :smooth

"""
    get_datastructures!(m_type::Type{<:Warp}, M::Dict, mod_data::Dict)::Bool

Ensures that coordinate variables are provided and stores them in the module data.
"""
function get_datastructures!(m_type::Type{<:Warp}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]

    if isempty(variables)
        error("The Warp model requires coordinate variables, e.g., `random(x, y, model=:warp)`.")
    end

    for var_sym in variables
        if !hasproperty(M[:data], Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Warp model not found in data.")
        end
    end

    mod_data[:params][:coords] = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    return true
end

"""
    get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple

Stores the coordinate matrix and its dimensions. All RFF parameters are learned,
so no fixed features are pre-generated.
"""
function get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = get(mod_data[:params], :coords, nothing)
    if isnothing(coords)
        error("Warp component precomputes failed: coordinates not found.")
    end
    
    in_dims = size(coords, 2)
    n_latent = size(coords, 1)

    return (
        coords=coords,
        in_dims=in_dims,
        n_latent=n_latent
    )
end

"""
    get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates priors for `sigma`, `lengthscale`, and all RFF parameters for both
the warping and main GP layers.
"""
function get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")
    beta_main_raw_name = p_names.raw

    in_dims = spec.hyper.in_dims
    n_features = m.n_features

    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end

    push!(priors, "$(W_warp_name) ~ MvNormal(zeros($(in_dims * n_features)), I)")
    push!(priors, "$(b_warp_name) ~ MvNormal(zeros($(n_features)), I)")
    push!(priors, "$(beta_warp_name) ~ MvNormal(zeros($(n_features)), I)")
    push!(priors, "$(W_main_name) ~ MvNormal(zeros($(in_dims * n_features)), I)")
    push!(priors, "$(b_main_name) ~ MvNormal(zeros($(n_features)), I)")
    push!(priors, "$(beta_main_raw_name) ~ MvNormal(zeros($(n_features)), I)")
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code for constructing the `Warp` smooth effect.
"""
function get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")
    beta_main_raw_name = p_names.raw

    precomputes = "spec.hyper"
    in_dims = spec.hyper.in_dims
    n_features = m.n_features

    ls_scaling_code = if m.lengthscale isa Vector
        "local W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(p_names.ls)'"
    else
        "local W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(p_names.ls)"
    end

    return """
        # --- Warp (Deep GP) Component: $(spec.key) ---
        local coords = $(precomputes).coords
        
        # 1. Construct and apply the warping function
        local W_warp_matrix = reshape($(W_warp_name), $(in_dims), $(n_features))
        local Phi_warp = sqrt(2.0 / $(n_features)) .* cos.((coords * W_warp_matrix) .+ $(b_warp_name)')
        local warping_effect = Phi_warp * $(beta_warp_name)
        local coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        $(ls_scaling_code)
        local Phi_main = sqrt(2.0 / $(n_features)) .* cos.((coords_warped * W_main_matrix) .+ $(b_main_name)')
        
        # 3. Scale coefficients and compute final effect
        local scaled_beta_main = $(beta_main_raw_name) .* $(p_names.sigma)
        $(p_names.latent) = Phi_main * scaled_beta_main
        
        $(eta_target) .+= $(p_names.latent)
    """
end

"""
    get_effects(m::Warp, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple

Reconstructs the `Warp` component's effect from posterior samples.
"""
function get_effects(m::Warp, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    
    coords_train = spec.hyper.coords
    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train
    end
    n_obs_full = size(coords_full, 1)

    in_dims = spec.hyper.in_dims
    n_features = m.n_features

    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        W_warp_name = Symbol("$(v.latent)_W_warp")
        b_warp_name = Symbol("$(v.latent)_b_warp")
        beta_warp_name = Symbol("$(v.latent)_beta_warp")
        W_main_name = Symbol("$(v.latent)_W_main")
        b_main_name = Symbol("$(v.latent)_b_main")
        beta_main_raw_name = v.raw

        sigma_samples = get_params_vector(chain, string(v.sigma), 1)
        ls_samples = get_params_vector(chain, string(v.ls), length(m.lengthscale))
        W_warp_samples = get_params_vector(chain, string(W_warp_name), in_dims * n_features)
        b_warp_samples = get_params_vector(chain, string(b_warp_name), n_features)
        beta_warp_samples = get_params_vector(chain, string(beta_warp_name), n_features)
        W_main_samples = get_params_vector(chain, string(W_main_name), in_dims * n_features)
        b_main_samples = get_params_vector(chain, string(b_main_name), n_features)
        beta_main_raw_samples = get_params_vector(chain, string(beta_main_raw_name), n_features)

        effect_k = Matrix{Float64}(undef, n_obs_full, n_samples)

        for i in 1:n_samples
            W_warp_i = reshape(W_warp_samples[i, :], in_dims, n_features)
            b_warp_i = b_warp_samples[i, :]
            beta_warp_i = beta_warp_samples[i, :]
            Phi_warp_i = sqrt(2.0 / n_features) .* cos.((coords_full * W_warp_i) .+ b_warp_i')
            warping_effect_i = Phi_warp_i * beta_warp_i
            coords_warped_i = coords_full .+ warping_effect_i

            ls_i = m.lengthscale isa Vector ? ls_samples[i, :]' : ls_samples[i, 1]
            W_main_i_unscaled = reshape(W_main_samples[i, :], in_dims, n_features)
            W_main_i = W_main_i_unscaled ./ ls_i

            b_main_i = b_main_samples[i, :]
            beta_main_raw_i = beta_main_raw_samples[i, :]
            Phi_main_i = sqrt(2.0 / n_features) .* cos.((coords_warped_i * W_main_i) .+ b_main_i')
            
            scaled_beta_main_i = beta_main_raw_i .* sigma_samples[i, 1]
            effect_k[:, i] = Phi_main_i * scaled_beta_main_i
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
