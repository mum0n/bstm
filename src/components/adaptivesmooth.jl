"""
    AdaptiveSmooth <: ComponentModel

A component for non-linear smoothing that learns a transformation of the input
coordinates before applying a basis expansion. It uses a simple one-hidden-layer
multi-layer perceptron (MLP) to warp the coordinate space, allowing the model to
capture complex, non-stationary patterns that would be missed by standard smoothers.

# Version
v1.0.2 (2026-08-10)

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
    where the prior on \$\\beta\$ depends on the chosen method.

# Computational Methods
- `:noncentered` (default): A non-centered parameterization where coefficients are
  sampled from a standard normal and scaled by `sigma`. Recommended for AD.
- `:centered`: A centered parameterization where coefficients are sampled directly
  from `N(0, sigma^2)`. Didactic, can be less efficient.
- `:rw2_penalty`: Imposes a second-order random walk penalty on the basis
  coefficients, encouraging a smoother final effect.

# Fields
- `hidden_dim::Int`: The number of neurons in the hidden layer of the MLP.
- `nbins::Int`: The number of adaptive basis functions to generate.
- `sigma::UnivariateDistribution`: The prior for the standard deviation of the basis
  function coefficients.
- `method::Symbol`: The computational method, one of `:noncentered`, `:centered`,
  or `:rw2_penalty`.
"""
struct AdaptiveSmooth <: ComponentModel
    hidden_dim::Int
    nbins::Int
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:adaptivesmooth] = AdaptiveSmooth

COMPONENT_CONSTRUCTORS[:adaptivesmooth] = (p, params) -> AdaptiveSmooth(
    get(params, :hidden_dim, 10),
    get(params, :nbins, 20),
    p.sigma,
    get(params, :method, :noncentered)
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

Stores coordinate data and, for the `:rw2_penalty` method, pre-computes the
penalty matrix and its spectral decomposition.
"""
function get_precomputes(
    m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict
)::NamedTuple
    precomputes = Dict{Symbol, Any}(
        :coords => mod_data[:params][:coords],
        :in_dim => mod_data[:params][:in_dim]
    )

    if m.method == :rw2_penalty
        n_latent = m.nbins
        template = build_structure_template(:rw2, n_latent)
        Q_template = template.matrix
        
        eig_decomp = eigen(Symmetric(Matrix(Q_template)))
        U = eig_decomp.vectors
        L = eig_decomp.values
        scaling_factor = _compute_scaling_factor(L, 2)
        
        precomputes[:Q_template] = Q_template ./ scaling_factor
        precomputes[:U] = U
        precomputes[:L] = L ./ scaling_factor
    end

    return NamedTuple(precomputes)
end


"""
    get_priors(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx, M)

Generates priors for the MLP weights, basis coefficients, and scale, dispatching
on the chosen method.
"""
function get_priors(
    m::AdaptiveSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    h_dim = m.hidden_dim
    in_dim = spec.hyper.in_dim
    n_bins = m.nbins

    priors = [
        "$(p_names.W1) ~ MvNormal(zeros($(in_dim * h_dim)), I)",
        "$(p_names.b1) ~ MvNormal(zeros($(h_dim)), I)",
        "$(p_names.W2) ~ MvNormal(zeros($(h_dim * n_bins)), I)",
        "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))"
    ]

    if m.method in [:noncentered, :rw2_penalty]
        push!(priors, "$(p_names.innov) ~ MvNormal(zeros($(n_bins)), I)")
    end
    
    return join(priors, "\n    ")
end

"""
    get_updates(m::AdaptiveSmooth, spec::NamedTuple, arch::String, outcome_idx, M)

Generates the Turing code to construct the adaptive basis and compute the smooth
effect, dispatching on the chosen method.
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

    common_code = """
        local X_orig = spec_registry[:$(key)].hyper.coords
        local W1 = reshape($(p_names.W1), $(in_dim), $(h_dim))
        local b1 = $(p_names.b1)
        local W2 = reshape($(p_names.W2), $(h_dim), $(n_bins))
        
        local H = tanh.((X_orig * W1) .+ b1')
        local B_adaptive = H * W2
    """

    noncentered_code = """
        # --- AdaptiveSmooth Component (Non-Centered): $(key) ---
        let
            $(common_code)
            local scaled_coeffs = $(p_names.innov) .* $(p_names.sigma)
            local adaptive_effect = B_adaptive * scaled_coeffs
            $(eta_target) .+= adaptive_effect
        end
    """

    centered_code = """
        # --- AdaptiveSmooth Component (Centered): $(key) ---
        let
            $(common_code)
            local coeffs ~ MvNormal(zeros($(n_bins)), $(p_names.sigma)^2 * I)
            local adaptive_effect = B_adaptive * coeffs
            $(eta_target) .+= adaptive_effect
        end
    """

    rw2_penalty_code = """
        # --- AdaptiveSmooth Component (RW2 Penalty): $(key) ---
        let
            $(common_code)
            local hyper = spec_registry[:$(key)].hyper
            
            local diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0 # RW2 constraints
            
            local coeffs = hyper.U * (diag_D .* $(p_names.innov))
            local adaptive_effect = B_adaptive * coeffs
            
            $(eta_target) .+= adaptive_effect
        end
    """

    if m.method == :noncentered; return noncentered_code;
    elseif m.method == :centered; return centered_code;
    elseif m.method == :rw2_penalty; return rw2_penalty_code;
    else; error("Unsupported method '$(m.method)' for AdaptiveSmooth component."); end
end

"""
    get_effects(m::AdaptiveSmooth, chain, M::NamedTuple, ...)

Reconstructs the `AdaptiveSmooth` component's effect from posterior samples,
dispatching on the method used during sampling.
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
        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            W1 = reshape(W1_samples[i, :], spec.hyper.in_dim, m.hidden_dim)
            b1 = b1_samples[i, :]
            W2 = reshape(W2_samples[i, :], m.hidden_dim, m.nbins)
            
            H = tanh.((coords_full * W1) .+ b1')
            B_adaptive = H * W2
            
            local coeffs
            if m.method == :centered
                coeffs = get_params_vector(chain, string(p_names.latent), m.nbins)[i, :]
            else
                innov_samples = get_params_vector(chain, string(p_names.innov), m.nbins)
                if m.method == :noncentered
                    coeffs = innov_samples[i, :] .* sigma_samples[i]
                else # :rw2_penalty
                    U = spec.hyper.U
                    L = spec.hyper.L
                    diag_D = sigma_samples[i] ./ sqrt.(L .+ M.noise)
                    diag_D[1] = 0.0; diag_D[2] = 0.0
                    coeffs = U * (diag_D .* innov_samples[i, :])
                end
            end
            
            effect_k[:, i] = B_adaptive * coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
