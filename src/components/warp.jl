 
"""
    Warp <: ComponentModel
 
A component for a non-stationary Gaussian Process using a warping function. This
component implements a simple Deep GP structure with one hidden layer. The input
coordinates are first "warped" by a non-linear function (approximated by RFFs),
and then a standard RFF-based GP is applied to these warped coordinates.

# Version
v1.0.3 (2026-08-15)

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

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where the RFF coefficients
  for the main GP are constructed from standard normal innovations. Recommended for AD.
- `:centered`: A centered parameterization where the RFF coefficients for the main GP
  are sampled directly from a scaled Normal distribution. Didactic, can be less efficient.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_features`: `Int`, number of random features for both the warping and main RFF layers. Default: `20`.
  - `kernel`: `String`, name of the kernel to approximate (e.g., `"se"`, `"matern32"`). Default: `"se"`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s) of the main GP.
  - `sigma`: `UnivariateDistribution`, prior for the std. dev. of the main GP's coefficients.
  - `method`: `Symbol`, the computational method (`:noncentered` or `:centered`). Default: `:noncentered`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the main GP's coefficients.
- `ls_<key>`: The kernel lengthscale(s).
- `W_warp_<key>`: RFF weights for the warping layer.
- `b_warp_<key>`: RFF biases for the warping layer.
- `beta_warp_<key>`: RFF coefficients for the warping layer.
- `W_main_<key>`: RFF weights for the main GP layer.
- `b_main_<key>`: RFF biases for the main GP layer.
- `innovations_<key>`: Raw standard normal innovations for main GP coefficients (non-centered).
- `latent_<key>`: Main GP coefficients (centered).
"""
struct Warp <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_features::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:warp] = Warp
 
COMPONENT_CONSTRUCTORS[:warp] = (p, params) -> Warp(
    p.lengthscale,
    p.sigma,
    get(params, :n_features, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :noncentered)
)

MODEL_TO_STRUCTURE_MAP[:warp] = :smooth
 
"""
    get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple

Performs data-independent pre-calculations for a Warp component instance.
This includes validating coordinate variables and storing the coordinate matrix
and its dimensions. All RFF parameters are learned, so no fixed features are
pre-generated.

# Arguments
- `m`: The `Warp` component instance.
- `M`: The main model configuration `NamedTuple`.
- `mod_data`: A dictionary containing parsed module data.

# Returns
- A `NamedTuple` containing precomputed items: `coords`, `in_dims`, `n_latent`.
"""
function get_precomputes(m::Warp, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The Warp model requires coordinate variables, e.g., `random(x, y, model=:warp)`.")
    end
    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for Warp model not found in data.")
        end
    end
    
    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
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

Generates the Turing code string for the Warp component's priors.
This includes priors for the overall standard deviation (`sigma`),
the kernel lengthscale(s) (`ls`), and all Random Fourier Features (RFF)
parameters for both the warping layer and the main GP layer.

# Arguments
- `m`: The `Warp` component instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for priors.
"""
function get_priors(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")
    
    key = spec.key
    in_dims = spec_registry[:$(key)].hyper.in_dims
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

    # Priors for warping layer RFF parameters (always sampled)
    push!(priors, """
        $(W_warp_name) ~ MvNormal(zeros(T, $(in_dims * n_features)), I)
    """)
    push!(priors, """
        $(b_warp_name) ~ MvNormal(zeros(T, $(n_features)), I)
    """)
    push!(priors, """
        $(beta_warp_name) ~ MvNormal(zeros(T, $(n_features)), I)
    """)
    
    # Priors for main GP layer RFF parameters (always sampled)
    push!(priors, """
        $(W_main_name) ~ MvNormal(zeros(T, $(in_dims * n_features)), I)
    """)
    push!(priors, """
        $(b_main_name) ~ MvNormal(zeros(T, $(n_features)), I)
    """)

    # Prior for main GP coefficients (depends on method)
    if m.method == :noncentered
        push!(priors, """
            $(p_names.innovations) ~ MvNormal(zeros(T, $(n_features)), I)
        """)
    elseif m.method == :centered
        push!(priors, """
            $(p_names.latent) ~ MvNormal(
                zeros(T, $(n_features)), $(p_names.sigma)^2 * I
            )
        """)
    end
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates the Turing code string for constructing the `Warp` smooth effect
and adding it to the linear predictor.

This function dispatches on the chosen computational method (`:noncentered` or
`:centered`). It first constructs a non-linear warping function using RFFs,
then applies a main GP (also RFF-based) to these warped coordinates.

# Arguments
- `m`: The `Warp` component instance.
- `spec`: A `NamedTuple` containing the component's full specification.
- `arch`: The model architecture (`"univariate"` or `"multivariate"`).
- `outcome_idx`: The index of the outcome for multivariate models, `nothing` otherwise.
- `M`: The main model configuration `NamedTuple`.

# Returns
- A `String` containing the generated Turing code for the component's update logic.
"""
function get_updates(m::Warp, spec::NamedTuple, arch::String, outcome_idx, M)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    
    W_warp_name = Symbol("$(p_names.latent)_W_warp")
    b_warp_name = Symbol("$(p_names.latent)_b_warp")
    beta_warp_name = Symbol("$(p_names.latent)_beta_warp")
    W_main_name = Symbol("$(p_names.latent)_W_main")
    b_main_name = Symbol("$(p_names.latent)_b_main")

    key = spec.key
    in_dims = spec_registry[:$(key)].hyper.in_dims
    n_features = m.n_features

    ls_scaling_code = if m.lengthscale isa Vector
        "local W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(p_names.ls)'"
    else
        "local W_main_matrix = reshape($(W_main_name), $(in_dims), $(n_features)) ./ $(p_names.ls)"
    end

    common_code = """ # Common code for both noncentered and centered methods
        local hyper = spec_registry[:$(spec.key)].hyper
        local coords = hyper.coords
        
        # 1. Construct and apply the warping function
        local W_warp_matrix = reshape($(W_warp_name), $(in_dims), $(n_features))
        local Phi_warp = sqrt(2.0 / $(n_features)) .* cos.((coords * W_warp_matrix) .+ $(b_warp_name)')
        local warping_effect = Phi_warp * $(beta_warp_name)
        local coords_warped = coords .+ warping_effect

        # 2. Construct the main GP on the warped coordinates
        $(ls_scaling_code)
        local Phi_main = sqrt(2.0 / $(n_features)) .* cos.((coords_warped * W_main_matrix) .+ $(b_main_name)')
    """

    noncentered_code = """
        # --- Warp (Deep GP) Component (Non-Centered): $(spec.key) ---
        let
            $(common_code)
            
            # 3. Scale coefficients and compute final effect
            local scaled_beta_main = $(p_names.innovations) .* $(p_names.sigma)
            $(p_names.latent) = Phi_main * scaled_beta_main
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    centered_code = """
        # --- Warp (Deep GP) Component (Centered): $(spec.key) ---
        let
            $(common_code)
            
            # 3. Sample coefficients directly and compute final effect
            $(eta_target) .+= Phi_main * $(p_names.latent)
        end
    """

    if m.method == :noncentered
        return noncentered_code
    elseif m.method == :centered
        return centered_code
    else
        error("Unsupported method '$(m.method)' for Warp component.")
    end
end

"""
    get_effects(m::Warp, chain, M, n_samples, is_multivariate_model, outcomes_N, spec, PS, N_total)::NamedTuple

Reconstructs the `Warp` component's effect from the MCMC chain's posterior samples.

This function extracts relevant parameter samples from the MCMC `chain`,
reconstructs the warping effect and the main GP effect on warped coordinates
for each sample, and returns the structured results.

# Arguments
- `m`: The `Warp` component instance.
- `chain`: The MCMC chain object containing posterior samples.
- `M`: The main model configuration `NamedTuple`.
- `n_samples`: The total number of posterior samples.
- `is_multivariate_model`: A `Bool` indicating if the model is multivariate.
- `outcomes_N`: The total number of outcomes in the model.
- `spec`: A `NamedTuple` containing the component's full specification.
- `PS`: A `Union{NamedTuple, Nothing}` containing prediction set data, or `nothing`.
- `N_total`: The total number of observations (training + prediction).

# Returns
- A `NamedTuple` (`(structured=..., noisy=...)`) containing the reconstructed effects.
"""
function get_effects(m::Warp, chain, M::NamedTuple, n_samples::Int, is_multivariate_model::Bool,
    outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
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
    p_names_vec = string.(FlexiChains.parameters(chain))
    
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        W_warp_name = _find_parameter(p_names_vec, Symbol("$(v.latent)_W_warp"), k, is_multivariate_model)
        b_warp_name = _find_parameter(p_names_vec, Symbol("$(v.latent)_b_warp"), k, is_multivariate_model)
        beta_warp_name = _find_parameter(p_names_vec, Symbol("$(v.latent)_beta_warp"), k, is_multivariate_model)
        W_main_name = _find_parameter(p_names_vec, Symbol("$(v.latent)_W_main"), k, is_multivariate_model)
        b_main_name = _find_parameter(p_names_vec, Symbol("$(v.latent)_b_main"), k, is_multivariate_model)

        sigma_samples = get_params_vector(chain, _find_parameter(p_names_vec, v.sigma, k, is_multivariate_model), 1)[:, 1]
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1
        ls_samples = get_params_vector(chain, _find_parameter(p_names_vec, v.ls, k, is_multivariate_model), ls_dim)
        W_warp_samples = get_params_vector(chain, W_warp_name, in_dims * n_features)
        b_warp_samples = get_params_vector(chain, b_warp_name, n_features)
        beta_warp_samples = get_params_vector(chain, beta_warp_name, n_features)
        W_main_samples = get_params_vector(chain, W_main_name, in_dims * n_features)
        b_main_samples = get_params_vector(chain, b_main_name, n_features)

        effect_k = Matrix{Float64}(undef, n_obs_full, n_samples)

        for i in 1:n_samples
            W_warp_i = reshape(W_warp_samples[i, :], in_dims, n_features)
            b_warp_i = b_warp_samples[i, :]
            beta_warp_i = beta_warp_samples[i, :]
            Phi_warp_i = sqrt(2.0 / n_features) .* cos.((coords_full * W_warp_i) .+ b_warp_i')
            warping_effect_i = Phi_warp_i * beta_warp_i
            coords_warped_i = coords_full .+ warping_effect_i

            ls_i = ls_dim > 1 ? ls_samples[i, :] : ls_samples[i, 1]
            W_main_i_unscaled = reshape(W_main_samples[i, :], in_dims, n_features)
            W_main_i = W_main_i_unscaled ./ (ls_i isa AbstractVector ? ls_i' : ls_i)

            b_main_i = b_main_samples[i, :]
            Phi_main_i = sqrt(2.0 / n_features) .* cos.((coords_warped_i * W_main_i) .+ b_main_i')
            
            local beta_main_i
            if m.method == :noncentered
                beta_main_raw_samples = get_params_vector(chain, _find_parameter(p_names_vec, v.innovations, k, is_multivariate_model), n_features)
                beta_main_raw_i = beta_main_raw_samples[i, :]
                beta_main_i = beta_main_raw_i .* sigma_samples[i]
            else # :centered
                beta_main_samples = get_params_vector(chain, string(v.latent), n_features)
                beta_main_i = beta_main_samples[i, :]
            end
            
            effect_k[:, i] = Phi_main_i * beta_main_i
        end
        push!(structured_effects, effect_k)
    end

    return (structured=structured_effects, noisy=structured_effects)
end
