"""
    AdaptiveSmooth <: ComponentModel

A component for non-linear smoothing that learns a transformation of the input
coordinates before applying a basis expansion. It uses a simple one-hidden-layer
multi-layer perceptron (MLP) to warp the coordinate space, allowing the model to
capture complex, non-stationary patterns that would be missed by standard smoothers.

# Version
v1.0.8 (2026-08-15)

# Mathematical Summary
The component models a function \$f(x)\$ by first transforming the input coordinates
\$x\$ through a non-linear warping function \$g(x)\$, and then applying a linear basis
expansion in the warped space.

1.  **Warping Function (MLP)**: The input coordinates \$x \\in \\mathbb{R}^{D_{in}}\$
    are passed through a single hidden layer with a `tanh` activation function:
    \$h = \\tanh(x W_1 + b_1)\$
    where \$W_1 \\in \\mathbb{R}^{D_{in} \\times D_{hidden}}\$ and
    \$b_1 \\in \\mathbb{R}^{D_{hidden}}\$ are learned weights and biases.

2.  **Adaptive Basis Construction**: The output of the hidden layer, \$h\$, is then
    projected onto a set of \$N_{bins}\$ adaptive basis functions via a second weight
    matrix \$W_2 \\in \\mathbb{R}^{D_{hidden} \\times N_{bins}}\$:
    \$B_{adaptive} = h W_2\$

3.  **Final Effect**: The final smooth effect is a linear combination of these
    adaptive basis functions, with coefficients \$\\beta\$ scaled by a standard
    deviation \$\\sigma\$:
    \$f(x) = B_{adaptive} \\cdot (\\beta \\sigma)\$
    where the prior on \$\\beta\$ depends on the chosen method.

# Computational Methods
- `:noncentered` (Default, AD-friendly): A non-centered parameterization where
  coefficients are sampled from a standard normal and scaled by `sigma`. Recommended
  for gradient-based samplers.
- `:centered` (Didactic, Not AD-friendly): A centered parameterization where
  coefficients are sampled directly from `N(0, sigma^2)`. This can be less efficient
  for MCMC and is not AD-friendly for the `MvNormal` sampling.
- `:rw2_penalty` (AD-friendly): Imposes a second-order random walk penalty on the
  basis coefficients, encouraging a smoother final effect. Uses spectral decomposition
  for AD-friendliness.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `hidden_dim`: `Int`, number of neurons in the MLP hidden layer. Default: `10`.
  - `nbins`: `Int`, number of adaptive basis functions to generate. Default: `20`.
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the basis
    function coefficients. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:noncentered`, `:centered`, `:rw2_penalty`).
    Default: `:noncentered`.

# Outputs (Parameter Names)
- `W1_<key>`: `Matrix{Float64}`, MLP input weights.
- `b1_<key>`: `Vector{Float64}`, MLP hidden biases.
- `W2_<key>`: `Matrix{Float64}`, MLP output weights.
- `sigma_<key>`: `Float64`, standard deviation of basis coefficients.
- `innovations_<key>`: `Vector{Float64}`, raw innovations for basis coefficients (for `:noncentered` and `:rw2_penalty`).
- `latent_<key>`: `Vector{Float64}`, basis coefficients (for `:centered`).

# Key References
- Bishop, C. M. (1995). *Neural Networks for Pattern Recognition*. Oxford University Press.
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
    get_precomputes(m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple

Validates and stores coordinate data and, for the `:rw2_penalty` method,
pre-computes the penalty matrix and its spectral decomposition.
"""
function get_precomputes(
    m::AdaptiveSmooth, M::NamedTuple, mod_data::Dict
)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("The AdaptiveSmooth model requires at least one coordinate variable.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for AdaptiveSmooth model not found in data.")
        end
    end

    coords = Matrix{Float64}(M.data[!, Symbol.(variables)])
    in_dim = size(coords, 2)

    precomputes = Dict{Symbol, Any}(
        :coords => coords,
        :in_dim => in_dim
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

    priors = String[]
    push!(priors, "$(p_names.W1) ~ MvNormal(zeros(T, $(in_dim * h_dim)), I)")
    push!(priors, "$(p_names.b1) ~ MvNormal(zeros(T, $(h_dim)), I)")
    push!(priors, "$(p_names.W2) ~ MvNormal(zeros(T, $(h_dim * n_bins)), I)")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.method in [:noncentered, :rw2_penalty]
        push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, $(n_bins)), I)")
    elseif m.method == :centered
        push!(priors, "$(p_names.latent) ~ MvNormal(zeros(T, $(n_bins)), I)")
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
        let
            hyper = spec_registry[:$(key)].hyper
            X_orig = hyper.coords
            W1 = reshape($(p_names.W1), $(in_dim), $(h_dim))
            b1 = $(p_names.b1)
            W2 = reshape($(p_names.W2), $(h_dim), $(n_bins))
            
            H = tanh.((X_orig * W1) .+ b1')
            B_adaptive = H * W2
    """

    noncentered_code = """
        # --- AdaptiveSmooth Component (Non-Centered): $(key) ---
        $(common_code)
            scaled_coeffs = $(p_names.innovations) .* $(p_names.sigma)
            adaptive_effect = B_adaptive * scaled_coeffs
            $(eta_target) .+= adaptive_effect
        end
    """

    centered_code = """
        # --- AdaptiveSmooth Component (Centered): $(key) ---
        $(common_code)
            coeffs = $(p_names.latent) .* $(p_names.sigma)
            adaptive_effect = B_adaptive * coeffs
            $(eta_target) .+= adaptive_effect
        end
    """

    rw2_penalty_code = """
        # --- AdaptiveSmooth Component (RW2 Penalty): $(key) ---
        $(common_code)
            hyper = spec_registry[:$(key)].hyper
            
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0
            
            coeffs = hyper.U * (diag_D .* $(p_names.innovations))
            adaptive_effect = B_adaptive * coeffs
            
            $(eta_target) .+= adaptive_effect
        end
    """

    if m.method == :noncentered; return noncentered_code;
    elseif m.method == :centered; return centered_code;
    elseif m.method == :rw2_penalty; return rw2_penalty_code;
    else; error("Unsupported method '$(m.method)' for AdaptiveSmooth component."); end
end

"""
    get_effects(m::AdaptiveSmooth, chain, M, n_samples, outcomes_N, p_names, spec, PS, N_total, is_multivariate_model)

Reconstructs the `AdaptiveSmooth` component's effect from posterior samples,
dispatching on the method used during sampling.
"""
function get_effects(
    m::AdaptiveSmooth, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    p_names::Vector{String}, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, 
    N_total::Int, is_multivariate_model::Bool
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
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        W1_name = _find_parameter(p_names, string(p_names_k.W1), k, is_multivariate_model)
        b1_name = _find_parameter(p_names, string(p_names_k.b1), k, is_multivariate_model)
        W2_name = _find_parameter(p_names, string(p_names_k.W2), k, is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        
        if isempty(W1_name) || isempty(b1_name) || isempty(W2_name) || isempty(sigma_name)
            @warn "MLP parameters for AdaptiveSmooth component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        W1_samples = get_params_matrix(chain, W1_name, m.hidden_dim * spec.hyper.in_dim)
        b1_samples = get_params_matrix(chain, b1_name, m.hidden_dim)
        W2_samples = get_params_matrix(chain, W2_name, m.hidden_dim * m.nbins)
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]

        effect_k = zeros(Float64, n_obs_full, n_samples)

        for i in 1:n_samples
            W1 = reshape(W1_samples[i, :], spec.hyper.in_dim, m.hidden_dim)
            b1 = b1_samples[i, :]
            W2 = reshape(W2_samples[i, :], m.hidden_dim, m.nbins)
            
            H = tanh.((coords_full * W1) .+ b1')
            B_adaptive = H * W2
            
            local coeffs
            if m.method == :centered
                latent_name = _find_parameter(p_names, string(p_names_k.latent), k, is_multivariate_model)
                if isempty(latent_name)
                    @warn "Latent coefficients for centered AdaptiveSmooth component $(spec.key) (outcome $k) not found. Using zeros."
                    coeffs = zeros(m.nbins)
                else
                    coeffs = get_params_matrix(chain, latent_name, m.nbins)[i, :] .* sigma_samples[i]
                end
            else # :noncentered or :rw2_penalty
                innovations_name = _find_parameter(p_names, string(p_names_k.innovations), k, is_multivariate_model)
                if isempty(innovations_name)
                    @warn "Innovations for AdaptiveSmooth component $(spec.key) (outcome $k) not found. Using zeros."
                    coeffs = zeros(m.nbins)
                else
                    innovations_samples = get_params_matrix(chain, innovations_name, m.nbins)
                    if m.method == :noncentered
                        coeffs = innovations_samples[i, :] .* sigma_samples[i]
                    else # :rw2_penalty
                        U = spec.hyper.U
                        L = spec.hyper.L
                        diag_D = sigma_samples[i] ./ sqrt.(L .+ M.noise)
                        diag_D[1] = 0.0; diag_D[2] = 0.0
                        coeffs = U * (diag_D .* innovations_samples[i, :])
                    end
                end
            end
            
            effect_k[:, i] = B_adaptive * coeffs
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
