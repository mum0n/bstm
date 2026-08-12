"""
    SpectralGP <: ComponentModel

A component for a Gaussian Process modeled in the frequency domain. This approach
leverages the Fast Fourier Transform (FFT) to efficiently model stationary
covariance structures, making it highly scalable for data on regular grids.

# Version
v1.2.0 (2026-08-11)

# Mathematical Summary
This component models a latent field \$f(s)\$ by defining its properties in the
frequency domain, based on the Wiener-Khinchin theorem. The theorem states that
the power spectral density (PSD), \$S(\\omega)\$, of a stationary process is the
Fourier transform of its autocorrelation function (the kernel).

Instead of defining the kernel \$k(s, s')\$ in the spatial domain, we define a
parametric model for the PSD, \$S(\\omega)\$. For example, the Matern kernel has a known
analytical form for its spectral density. The latent field \$f(s)\$ is then
realized by:
1.  Sampling the complex Fourier coefficients \$\\tilde{f}(\\omega)\$ from a Gaussian
    distribution whose variance is given by the PSD:
    \$\\tilde{f}(\\omega) \\sim \\mathcal{CN}(0, S(\\omega))\$
2.  Transforming these coefficients back to the spatial domain using the inverse
    Fast Fourier Transform (iFFT) to get a latent field on a regular grid.
3.  Using multilinear interpolation to obtain the latent field values at the
    original, potentially irregular, observation coordinates.

This approach is computationally efficient, scaling as \$O(N \\log N)\$ for a grid of
\$N\$ points, compared to \$O(N^3)\$ for a standard GP.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `resolution`: `Int`, the grid resolution for discretization. Default: `32`.
  - `kernel`: `String`, the parametric family for the power spectrum (e.g., `"matern"`). Default: `"matern"`.
  - `sigma`: `UnivariateDistribution`, prior for the overall marginal standard deviation. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `nu` (smoothness): `UnivariateDistribution`, prior for the Matern smoothness parameter. Default: `LogNormal(log(1.5), 0.5)`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s).
- `nu_<key>`: The Matern smoothness parameter.
- `innovations_<key>`: The raw standard normal innovations for the Fourier coefficients.
- `latent_<key>`: The interpolated latent effect at the observation coordinates.
"""
struct SpectralGP <: ComponentModel
    sigma::UnivariateDistribution
    lengthscale::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}}
    nu::UnivariateDistribution
    kernel::String
    resolution::Int
end

COMPONENT_TYPE_REGISTRY[:spectral_gp] = SpectralGP
COMPONENT_CONSTRUCTORS[:spectral_gp] = (p, params) -> SpectralGP(
    p.sigma,
    p.lengthscale,
    get(p, :nu, LogNormal(log(1.5), 0.5)),
    string(get(params, :kernel, "matern")),
    get(params, :resolution, 32)
)
MODEL_TO_STRUCTURE_MAP[:spectral_gp] = :smooth

function get_datastructures!(m_type::Type{<:SpectralGP}, M::Dict, mod_data::Dict)::Bool
    variables = mod_data[:variables]
    if isempty(variables)
        error("SpectralGP model requires coordinate variables.")
    end
    coords = Matrix{Float64}(M[:data][!, Symbol.(variables)])
    mod_data[:params][:coords] = coords
    return true
end

function get_precomputes(m::SpectralGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    coords = mod_data[:params][:coords]
    res = m.resolution
    n_dims = size(coords, 2)

    min_coords = minimum(coords, dims=1)
    max_coords = maximum(coords, dims=1)
    
    # Grid ranges for interpolation
    grid_ranges = [range(min_coords[d], stop=max_coords[d], length=res) for d in 1:n_dims]

    freqs = [fftfreq(res, res / (max_coords[d] - min_coords[d])) for d in 1:n_dims]
    
    # Create a meshgrid of frequencies
    freq_grids = [reshape(f, (d == i ? res : 1 for i in 1:n_dims)...) for (d, f) in enumerate(freqs)]
    
    n_latent = res^n_dims

    return (
        coords = coords,
        resolution = res,
        n_dims = n_dims,
        n_latent = n_latent,
        freq_grids = freq_grids,
        grid_ranges = grid_ranges
    )
end

function get_priors(
    m::SpectralGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")
    push!(priors, "$(p_names.nu) ~ $(_distribution_to_string(m.nu))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        push!(priors, "$(p_names.ls) ~ $(_distribution_to_string(m.lengthscale))")
    end
    
    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, spec.hyper.n_latent), I)")

    return join(priors, "\n    ")
end

function get_updates(
    m::SpectralGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    res = m.resolution
    n_dims = spec.hyper.n_dims

    return """
    # --- SpectralGP Component: $(key) ---
    let
        hyper = spec_registry[:$(key)].hyper
        
        # 1. Compute the Power Spectral Density (PSD) on the frequency grid
        S_w = anisotropic_matern_spectral_density(
            hyper.freq_grids,
            $(p_names.sigma),
            $(p_names.ls),
            $(p_names.nu),
            $(n_dims)
        )
        
        # 2. Construct complex Fourier coefficients from standard normal innovations
        innov_reshaped = reshape($(p_names.innovations), $(fill(res, n_dims)...))
        f_tilde_complex = complex.(innov_reshaped)
        f_tilde_scaled = f_tilde_complex .* sqrt.(S_w)

        # 3. Transform back to spatial domain using inverse FFT
        latent_field_grid = real.(ifft(f_tilde_scaled)) .* ($(res^(n_dims/2)))
        
        # 4. Interpolate the grid values to the original observation coordinates
        itp = linear_interpolation(hyper.grid_ranges, latent_field_grid, extrapolation_bc=Flat())
        coords_for_itp = ntuple(d -> hyper.coords[:, d], $(n_dims))
        $(p_names.latent) = itp(coords_for_itp...)
        
        $(eta_target) .+= $(p_names.latent)
    end
    """
end

function get_effects(
    m::SpectralGP, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int,
    spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int
)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    is_multivariate_model = M.model_arch == "multivariate"
    p_names_vec = string.(FlexiChains.parameters(chain))
    hyper = spec.hyper
    res = hyper.resolution
    n_dims = hyper.n_dims
    n_latent = hyper.n_latent

    coord_vars = get(spec.params, :positional_args, [])
    coords_full = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        vcat(hyper.coords, Matrix{Float64}(PS.data[!, Symbol.(coord_vars)]))
    else
        hyper.coords
    end

    for k in 1:outcomes_N
        sigma_name = _find_parameter(p_names_vec, string(spec.key), "sigma", k, is_multivariate_model)
        ls_name = _find_parameter(p_names_vec, string(spec.key), "ls", k, is_multivariate_model)
        nu_name = _find_parameter(p_names_vec, string(spec.key), "nu", k, is_multivariate_model)
        innovations_name = _find_parameter(p_names_vec, string(spec.key), "innovations", k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(nu_name) || isempty(innovations_name)
            @warn "Parameters for SpectralGP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_samples = get_params_vector(chain, ls_name, m.lengthscale isa Vector ? n_dims : 1)
        nu_samples = get_params_vector(chain, nu_name, 1)[:, 1]
        innovations_samples = get_params_vector(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, N_total, n_samples)

        for i in 1:n_samples
            S_w = anisotropic_matern_spectral_density(
                hyper.freq_grids,
                sigma_samples[i],
                ls_samples[i, :],
                nu_samples[i],
                n_dims
            )
            
            innov_reshaped = reshape(innovations_samples[i, :], fill(res, n_dims)...)
            f_tilde_complex = complex.(innov_reshaped)
            f_tilde_scaled = f_tilde_complex .* sqrt.(S_w)
            
            latent_field_grid = real.(ifft(f_tilde_scaled)) .* (res^(n_dims/2))
            
            itp_s = linear_interpolation(hyper.grid_ranges, latent_field_grid, extrapolation_bc=Flat())
            coords_for_itp = ntuple(d -> coords_full[:, d], n_dims)
            effect_k[:, i] = itp_s(coords_for_itp...)
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 

"""
    anisotropic_matern_spectral_density(freq_grids, sigma, ls, nu, n_dims)

Computes the power spectral density of an anisotropic Matérn kernel on a frequency grid.

# Mathematical Summary
The spectral density \$S(f)\$ for an anisotropic Matérn kernel is given by:
\$S(\\mathbf{f}) = \\sigma^2 (\\prod_i \\ell_i) \\frac{2^d \\pi^{d/2} \\Gamma(\\nu+d/2)(2\\nu)^\\nu}{\\Gamma(\\nu)} \\left(2\\nu + 4\\pi^2 \\sum_i (\\ell_i f_i)^2 \\right)^{-(\\nu+d/2)}\$
where \$d\$ is the number of dimensions, \$\\sigma\$ is the marginal standard deviation,
\$\\nu\$ is the smoothness, \$\\ell_i\$ are the lengthscales, and \$f_i\$ are the frequencies.

# Arguments
- `freq_grids`: A vector of frequency grids for each dimension.
- `sigma`: The marginal standard deviation of the process.
- `ls`: A vector of lengthscales for each dimension.
- `nu`: The smoothness parameter of the Matérn kernel.
- `n_dims`: The number of dimensions.

# Returns
- A matrix representing the power spectral density on the grid.
"""
function anisotropic_matern_spectral_density(freq_grids, sigma, ls, nu, n_dims)
    T = promote_type(typeof(sigma), eltype(ls), typeof(nu))
    ls_vec = ls isa Real ? fill(convert(T, ls), n_dims) : convert(Vector{T}, ls)
    
    freq_norm_sq = zeros(T, size(freq_grids[1]))
    for d in 1:n_dims
        freq_norm_sq .+= (2 * T(pi) .* ls_vec[d] .* freq_grids[d]).^2
    end
    
    const_factor = (2^n_dims * T(pi)^(n_dims/2) * gamma(nu + n_dims/2) * (2*nu)^nu) / gamma(nu)
    
    total_scaling = sigma^2 * prod(ls_vec) * const_factor

    power_val = nu + n_dims/2
    
    base_term = (2*nu) .+ freq_norm_sq

    S_w = total_scaling .* base_term.^(-power_val)
    
    return S_w
end
