"""
    WaveletGP <: ComponentModel

A component for a Gaussian Process modeled in the wavelet domain. This approach
leverages the Discrete Wavelet Transform (DWT) to model a latent field with a
flexible, data-driven covariance structure. It is particularly effective at
capturing processes with multi-scale features and non-stationarities.

# Version
v1.1.2 (2026-08-19)

# Mathematical Summary
This component models a latent field \$f(s)\$ by defining the statistical properties
of its wavelet coefficients. Based on the work of Whittle (1956) and others, for
many stationary processes, the wavelet coefficients at different scales and locations
are approximately uncorrelated.

The model works as follows:
1.  **Discretization**: The continuous spatial domain is discretized onto a regular grid.
2.  **Wavelet Decomposition**: The latent field \$f(s)\$ on the grid is decomposed into
    wavelet coefficients \$d_{j,k}\$ using the DWT, where \$j\$ is the scale and \$k\$ is the location.
3.  **Priors on Coefficients**: The wavelet coefficients are modeled as independent zero-mean
    Gaussian random variables, with a variance that depends on the scale \$j\$:
    \$d_{j,k} \\sim \\mathcal{N}(0, \\sigma_j^2)\$
4.  **Variance Model**: The variance across scales is modeled using a power law, which is
    controlled by two hyperparameters: an overall scale \$\\sigma_0\$ and a smoothness/decay
    parameter \$\\alpha\$:
    \$\\sigma_j^2 = \\sigma_0^2 \\cdot 2^{-\\alpha j}\$
    Estimating \$\\alpha\$ allows the model to learn the smoothness of the underlying process from the data.
5.  **Synthesis**: The latent field is reconstructed by applying the inverse DWT (IDWT) to the
    sampled wavelet coefficients.
6.  **Interpolation**: The values of the latent field at the original observation locations are
    obtained by interpolating from the reconstructed grid.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `resolution`: `Int`, the grid resolution for discretization (must be a power of 2). Default: `32`.
  - `wavelet`: `Symbol`, the wavelet family to use (e.g., `:db4`, `:sym6`). Default: `:db4`.
  - `sigma0`: `UnivariateDistribution`, prior for the overall scale of the wavelet variances. Default: `Exponential(1.0)`.
  - `alpha`: `UnivariateDistribution`, prior for the smoothness/decay parameter. Default: `Normal(1.5, 0.5)`.

# Outputs (Parameter Names)
- `sigma0_<key>`: The overall scale of the wavelet coefficient variances.
- `alpha_<key>`: The smoothness/decay parameter.
- `innovations_<key>`: The raw standard normal innovations for the wavelet coefficients.
- `latent_<key>`: The reconstructed latent effect at the observation coordinates.

# Key References
- Nason, G. P. (2008). *Wavelet Methods in Statistics with R*. Springer.
- Whittle, P. (1956). *On the variation of yield variance with plot size*. Biometrika, 43(3/4), 337-343.
"""
struct WaveletGP <: ComponentModel
    sigma0::UnivariateDistribution
    alpha::UnivariateDistribution
    wavelet::Symbol
    resolution::Int
end

COMPONENT_TYPE_REGISTRY[:waveletgp] = WaveletGP
COMPONENT_CONSTRUCTORS[:waveletgp] = (p, params) -> WaveletGP(
    p.sigma0,
    p.alpha,
    get(params, :wavelet, :db4),
    get(params, :resolution, 32)
)
MODEL_TO_STRUCTURE_MAP[:waveletgp] = :smooth
 
"""
    _get_wavelet_scale_indices_2d(res::Int, wt)

Computes the scale level for each coefficient in a 2D DWT.
"""
function _get_wavelet_scale_indices_2d(res::Int, wt)
    scale_indices_matrix = zeros(Int, res, res)
    max_level = dwt_levels(zeros(res, res), wt)
    
    current_res = res
    for level in 1:max_level
        half_res = current_res ÷ 2
        if half_res == 0; break; end
        
        # Assign level to the detail coefficient quadrants
        scale_indices_matrix[1:half_res, (half_res+1):current_res] .= level # Horizontal details
        scale_indices_matrix[(half_res+1):current_res, 1:half_res] .= level # Vertical details
        scale_indices_matrix[(half_res+1):current_res, (half_res+1):current_res] .= level # Diagonal details
        
        current_res = half_res
    end
    # Assign the highest level to the remaining approximation coefficients
    scale_indices_matrix[1:current_res, 1:current_res] .= max_level
    
    return vec(scale_indices_matrix)
end

function get_precomputes(m::WaveletGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation
    variables = mod_data[:variables]
    if isempty(variables)
        error("WaveletGP model requires coordinate variables.")
    end
    
    if !ispow2(m.resolution)
        error("Resolution for WaveletGP must be a power of 2. Got: $(m.resolution)")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for WaveletGP model not found in data.")
        end
    end

    # All computations are on the CPU.
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    res = m.resolution
    n_dims = size(coords_cpu, 2)
    
    if n_dims > 2
        error("WaveletGP currently supports only 1D and 2D spatial inputs.")
    end

    n_latent = res^n_dims
    
    min_coords = minimum(coords_cpu, dims=1)
    max_coords = maximum(coords_cpu, dims=1)
    grid_ranges = [range(min_coords[d], stop=max_coords[d], length=res) for d in 1:n_dims]

    wt = Wavelets.wavelet(m.wavelet)
    local scale_indices_cpu
    if n_dims == 1
        x_dummy = zeros(res)
        c, l = wavedec(x_dummy, wt)
        scale_indices_cpu = zeros(Int, length(c))
        start_idx = 1
        
        # Approximation coefficients scale
        scale_indices_cpu[start_idx : start_idx + l[1] - 1] .= dwt_levels(x_dummy)
        start_idx += l[1]
        
        # Detail coefficients scales
        for i in 2:length(l)-1
            current_level = dwt_levels(x_dummy) - (i - 1)
            scale_indices_cpu[start_idx : start_idx + l[i] - 1] .= current_level
            start_idx += l[i]
        end
    else # 2D
        scale_indices_cpu = _get_wavelet_scale_indices_2d(res, wt)
    end
    
    return (
        resolution = res,
        n_dims = n_dims,
        n_latent = n_latent,
        coords = coords_cpu,
        grid_ranges = grid_ranges,
        scale_indices = scale_indices_cpu
    )
end


function get_priors(
    m::WaveletGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    priors = String[]

    push!(priors, "$(p_names.sigma0) ~ $(_distribution_to_string(m.sigma0))")
    push!(priors, "$(p_names.alpha) ~ $(_distribution_to_string(m.alpha))")
    push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)")

    return join(priors, "\n    ")
end

function get_updates(
    m::WaveletGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    res = m.resolution
    n_dims = spec.hyper.n_dims

    return """
    # --- WaveletGP Component: $(key) ---
    let
        hyper = spec_registry[:$(key)].hyper
        wt = Wavelets.wavelet(Symbol("$(m.wavelet)"))
        
        scale_variances = $(p_names.sigma0)^2 .* (2.0 .^ (-$(p_names.alpha) .* hyper.scale_indices))
        wavelet_coeffs = $(p_names.ure) .* sqrt.(scale_variances)
        
        local latent_field_grid
        if $(n_dims) == 1
            latent_field_grid = idwt(wavelet_coeffs, wt)
        else
            coeffs_reshaped = reshape(wavelet_coeffs, $(res), $(res))
            latent_field_grid = idwt(coeffs_reshaped, wt)
        end
        
        itp = linear_interpolation(hyper.grid_ranges, latent_field_grid, extrapolation_bc=Flat())
        coords_for_itp = ntuple(d -> hyper.coords[:, d], $(n_dims))
        $(p_names.sre) = itp(coords_for_itp...)
        
        $(eta_target) .+= $(p_names.sre)
    end
    """
end



function get_effects(
    m::WaveletGP, chain, spec::NamedTuple, M::NamedTuple,
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
    
    # --- Get precomputed data (all on CPU) ---
    hyper = spec.hyper
    res = hyper.resolution
    n_dims = hyper.n_dims
    n_latent = hyper.n_latent
    scale_indices_cpu = hyper.scale_indices

    # --- Coordinate and Grid Handling on CPU for Interpolation ---
    coord_vars = get(spec.params, :positional_args, [])
    coords_train_cpu = hyper.coords
    
    coords_full_cpu = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(coords_train_cpu, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        coords_train_cpu
    end
    N_total_eff = size(coords_full_cpu, 1)
    
    grid_ranges_cpu = hyper.grid_ranges

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        sigma0_name = _find_parameter(p_names, string(v.sigma0), k, is_multivariate_model)
        alpha_name = _find_parameter(p_names, string(v.alpha), k, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(v.ure), k, is_multivariate_model)

        if isempty(sigma0_name) || isempty(alpha_name) || isempty(ure_name)
            @warn "Parameters for WaveletGP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total_eff, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma0_samples_cpu = get_params_vector(chain, sigma0_name, 1)[:, 1]
        alpha_samples_cpu = get_params_vector(chain, alpha_name, 1)[:, 1]
        ure_samples_cpu = get_params_matrix(chain, ure_name, n_latent)

        effect_k = zeros(Float64, N_total_eff, n_samples)
        wt = Wavelets.wavelet(m.wavelet)
        
        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples
            sigma0_s = sigma0_samples_cpu[i]
            alpha_s = alpha_samples_cpu[i]
            innov_s = ure_samples_cpu[i, :]

            scale_variances = sigma0_s^2 .* (2.0 .^ (-alpha_s .* scale_indices_cpu))
            wavelet_coeffs = innov_s .* sqrt.(scale_variances)
            
            local latent_field_grid_cpu
            if n_dims == 1
                latent_field_grid_cpu = idwt(wavelet_coeffs, wt)
            else
                coeffs_reshaped = reshape(wavelet_coeffs, res, res)
                latent_field_grid_cpu = idwt(coeffs_reshaped, wt)
            end
            
            itp_s = linear_interpolation(grid_ranges_cpu, latent_field_grid_cpu, extrapolation_bc=Flat())
            
            coords_for_itp = ntuple(d -> view(coords_full_cpu, :, d), n_dims)
            effect_k[:, i] = itp_s(coords_for_itp...)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

