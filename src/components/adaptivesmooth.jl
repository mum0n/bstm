"""
    AdaptiveSmooth <: ComponentModel

A component for non-linear smoothing that learns a transformation of the input
coordinates before applying a basis expansion. It uses a simple one-hidden-layer
multi-layer perceptron (MLP) to warp the coordinate space, allowing the model to
capture complex, non-stationary patterns that would be missed by standard smoothers.

# Version
v1.0.1 (2026-08-08)

# Mathematical Summary
The component models a function \$f(x)\$ by first transforming the input coordinates
\$x\$ through a non-linear warping function \$g(x)\$, and then applying a linear basis
expansion in the warped space.
1.  **Warping Function (MLP)**: The input coordinates \$x \\in \\mathbb{R}^{D_{in}}\$
    are passed through a single hidden layer with a `tanh` activation function:
    \$h = \\tanh(x W_1 + b_1)\$
    where \$W_1 \\in \\mathbb{R}^{D_{in} \\times D_{hidden}}\$ and
    \$b_1 \\in \\mathbb{R}^{D_{hidden}}\$ are learned weights and biases.

2.  **Basis Construction**: The output of the hidden layer, \$h\$, is then projected
    onto a set of \$N_{bins}\$ basis functions via a second weight matrix
    \$W_2 \\in \\mathbb{R}^{D_{hidden} \\times N_{bins}}\$:
    \$B_{adaptive} = h W_2\$

3.  **Final Effect**: The final smooth effect is a linear combination of these
    adaptive basis functions, with coefficients \$\\beta\$ scaled by a standard
    deviation \$\\sigma\$:
    \$f(x) = B_{adaptive} \\cdot (\\beta \\sigma)\$
    where \$\\beta \\sim \\mathcal{N}(0, I)\$.

# Assumptions
- The relationship between the covariates and the outcome is smooth but potentially
  highly non-linear and non-stationary.
- The `tanh` activation function is sufficient to capture the desired coordinate
  warping.

# Key References
- Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. In AISTATS.
  (Conceptual foundation for stacked non-linear transformations).

# Fields
- `hidden_dim::Int`: The number of neurons in the hidden layer of the MLP.
- `nbins::Int`: The number of adaptive basis functions to generate.
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the basis
  function coefficients.
"""
struct AdaptiveSmooth <: ComponentModel
    hidden_dim::Int
    nbins::Int
    sigma::UnivariateDistribution
end

COMPONENT_TYPE_REGISTRY[:adaptivesmooth] = AdaptiveSmooth

COMPONENT_CONSTRUCTORS[:adaptivesmooth] = (p, params) -> AdaptiveSmooth(
    get(params, :hidden_dim, 10),
    get(params, :nbins, 20),
    p.sigma
)

MODEL_TO_STRUCTURE_MAP[:adaptivesmooth] = :smooth

"""
    get_datastructures!(m_type::Type{<:AdaptiveSmooth}, M::Dict, mod_data::Dict)

Extracts the coordinate variables from the formula and stores them in the main model
configuration `M`.

# Assumptions
- The `random()` call provides one or more variables representing the coordinates to
  be smoothed.
"""
function get_datastructures!(
    m_type::Type{<:AdaptiveSmooth}, M::Dict, mod_data::Dict
)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error(
            "The AdaptiveSmooth model requires at least one coordinate variable, " *
            "e.g., `random(x, model=:adaptivesmooth)`."
        )
    end

    coords_matrix = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords_matrix
    mod_data[:params][:in_dim] = size(coords_matrix, 2)

    return true
end

"""
    get_precomputes(m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple

Stores the coordinate data and input dimension in the component's `hyper` registry
for easy access by the code generator.
"""
function get_precomputes(
    m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict
)::NamedTuple
    return (
        coords=mod_data[:params][:coords],
        in_dim=mod_data[:params][:in_dim]
    )
end

"""
    get_priors(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for the MLP weights (`W1`, `b1`, `W2`), the basis coefficients
(`innov`), and the overall scale (`sigma`).

# Assumptions
- Standard normal priors on the weights and innovations are reasonable defaults.
"""
function get_priors(
    m::AdaptiveSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    h_dim = m.hidden_dim
    in_dim = spec.hyper.in_dim
    n_bins = m.nbins

    priors = """
    # Adaptive Basis Priors
    $(p_names.W1) ~ MvNormal(zeros(T, $(in_dim * h_dim)), I)
    $(p_names.b1) ~ MvNormal(zeros(T, $(h_dim)), I)
    $(p_names.W2) ~ MvNormal(zeros(T, $(h_dim * n_bins)), I)
    $(p_names.innov) ~ MvNormal(zeros(T, $(n_bins)), I)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    """
    return priors
end

"""
    get_updates(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code to construct the adaptive basis, compute the smooth
effect, and add it to the linear predictor `eta`.

# Assumptions
- The effect is additive on the scale of the linear predictor.
- The `tanh` activation function provides sufficient non-linearity for the
  coordinate warping.
"""
function get_updates(
    m::AdaptiveSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    h_dim = m.hidden_dim
    in_dim = spec.hyper.in_dim
    n_bins = m.nbins

    return """
        # --- AdaptiveSmooth Component: $(key) ---
        let
            local X_orig = spec_registry[:$(key)].hyper.coords
            local W1 = reshape($(p_names.W1), $(in_dim), $(h_dim))
            local b1 = $(p_names.b1)
            local W2 = reshape($(p_names.W2), $(h_dim), $(n_bins))
            
            # Layer 1: Learnable coordinate transformation
            local H = tanh.((X_orig * W1) .+ b1')
            
            # Layer 2: Projection to basis space
            local B_adaptive = H * W2
            
            # Compute final effect
            local scaled_coeffs = $(p_names.innov) .* $(p_names.sigma)
            local adaptive_effect = B_adaptive * scaled_coeffs
            
            $(eta_target) .+= adaptive_effect
        end
    """
end

"""
    get_effects(m::AdaptiveSmooth, chain, M::NamedTuple, ...)

Reconstructs the `AdaptiveSmooth` component's effect from the MCMC chain's
posterior samples. It mirrors the generative logic by re-applying the learned MLP
transformation for each sample.

# Assumptions
- The MCMC `chain` contains posterior samples for the MLP weights and basis
  coefficients.
"""
function get_effects(
    m::AdaptiveSmooth, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
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

    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        
        W1_samples = get_params_vector(chain, string(p_names.W1), m.hidden_dim * spec.hyper.in_dim)
        b1_samples = get_params_vector(chain, string(p_names.b1), m.hidden_dim)
        W2_samples = get_params_vector(chain, string(p_names.W2), m.hidden_dim * m.nbins)
        innov_samples = get_params_vector(chain, string(p_names.innov), m.nbins)
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            W1 = reshape(W1_samples[i, :], spec.hyper.in_dim, m.hidden_dim)
            b1 = b1_samples[i, :]
            W2 = reshape(W2_samples[i, :], m.hidden_dim, m.nbins)
            
            scaled_coeffs = innov_samples[i, :] .* sigma_samples[i, 1]
            
            H = tanh.((coords_full * W1) .+ b1')
            B_adaptive = H * W2
            
            effect_k[:, i] = B_adaptive * scaled_coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
