"""
    WaveletGP <: ComponentModel

A component for a Gaussian Process modeled in the wavelet domain. This approach
leverages the Discrete Wavelet Transform (DWT) to model a latent field with a
flexible, data-driven covariance structure. It is particularly effective at
capturing processes with multi-scale features and non-stationarities.

# Version
v1.0.0

# Mathematical Summary
This component models a latent field \$f(s)\$ by defining the statistical properties
of its wavelet coefficients. Based on the work of Whittle (1956) and others, for
many stationary processes, the wavelet coefficients at different scales and locations
are approximately uncorrelated.

The model works as follows:
1.  **Discretization**: The continuous spatial domain is discretized onto a regular grid.
2.  **Wavelet Decomposition**: The latent field \$f(s)\$ on the grid is decomposed into
    wavelet coefficients \$d_{j,k}\$ using the DWT, where \$j\$ is the scale and \$k\$ is
      the location.
3.  **Priors on Coefficients**: The wavelet coefficients are modeled as independent zero-mean
    Gaussian random variables, with a variance that depends on the scale \$j\$:
    \$d_{j,k} \\sim \\mathcal{N}(0, \\sigma_j^2)\$
4.  **Variance Model**: The variance across scales is modeled using a power law, which is
    controlled by two hyperparameters: an overall scale \$\\sigma_0\$ and a smoothness/decay
    parameter \$\\alpha\$:
    \$\\sigma_j^2 = \\sigma_0^2 \\cdot 2^{-\\alpha j}\$
    Estimating \$\\alpha\$ allows the model to learn the smoothness of the underlying
      process from the data.
5.  **Synthesis**: The latent field is reconstructed by applying the inverse DWT (IDWT) to the
    sampled wavelet coefficients.
6.  **Interpolation**: The values of the latent field at the original observation locations are
    obtained by interpolating from the reconstructed grid.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `resolution`: `Int`, the grid resolution for discretization (must be a power of 2).
    Default: `32`.
  - `wavelet`: `Symbol`, the wavelet family to use (e.g., `:db4`, `:sym6`). Default: `:db4`.
  - `sigma`: `UnivariateDistribution`, prior for the overall scale of the wavelet
    variances. Default: `Exponential(1.0)`.
  - `alpha`: `UnivariateDistribution`, prior for the smoothness/decay parameter. Default:
    `Normal(1.5, 0.5)`.

# Outputs (Parameter Names)
- `sigma_<key>`: The overall scale of the wavelet coefficient variances.
- `alpha_<key>`: The smoothness/decay parameter.
- `innovations_<key>`: The raw standard normal innovations for the wavelet coefficients.
- `latent_<key>`: The reconstructed latent effect at the observation coordinates.

# Key References
- Nason, G. P. (2008). *Wavelet Methods in Statistics with R*. Springer.
- Whittle, P. (1956). *On the variation of yield variance with plot size*. Biometrika,
  43(3/4), 337-343.
"""
struct WaveletGP <: ComponentModel
    sigma::Union{UnivariateDistribution, Real}
    alpha::UnivariateDistribution
    wavelet::Symbol
    resolution::Int
end

COMPONENT_TYPE_REGISTRY[:waveletgp] = WaveletGP
COMPONENT_CONSTRUCTORS[:waveletgp] = (p, params) -> WaveletGP(
    get(p, :sigma, get(p, :sigma, Exponential(1.0))),
    get(p, :alpha, Exponential(1.0)),
    get(params, :wavelet, :db4),
    get(params, :resolution, 32)
)
MODEL_TO_STRUCTURE_MAP[:waveletgp] = :smooth
 
"""
    _wavelet_max_level(res::Integer)

Number of wavelet decomposition levels usable for a `res`-point grid.

This is `log2(res)`, i.e. `Wavelets.maxtransformlevels(res)`, and requires a
power-of-two resolution because the multiresolution transform installed here is
strictly dyadic and length-preserving (see [`_wavelet_decomposition_lengths`](@ref)).
"""
function _wavelet_max_level(res::Integer)
    res >= 2 || error("waveletgp requires resolution >= 2, got $res")
    ispow2(res) || error(
        "waveletgp requires a power-of-2 resolution for a dyadic wavelet basis, got $res. " *
        "Use one of 4, 8, 16, 32, 64, ...")
    return trailing_zeros(res)
end

"""
    _get_wavelet_scale_indices_2d(res::Integer)

Scale index of every coefficient of the 2-D `res`x`res` wavelet basis, as a vector
of length `res^2`.

Index `0` marks the coarsest (residual approximation) block; larger indices mark
progressively finer detail subbands. This is the same convention as the 1-D path
(see [`_wavelet_scale_indices_1d`](@ref)), which matters because the component turns
these indices into variances as `sigma^2 * 2^(-alpha * j)`: with `alpha > 0` the
variance must *decay* towards fine scales for the Whittle power law the component
documents.

At decomposition step `k` the three detail subbands live on a `res/2^k` grid, so their
scale index is `max_level - k + 1`; the surviving approximation block gets `0`.

The previous implementation instead assigned `k` -- growing with *coarseness* -- and gave
the approximation block the *largest* index. That inverted the decay, and did so in the
opposite direction from the 1-D path, so the same component shrank fine-scale variances
in 2-D and fine-scale variances in 1-D meant the opposite thing.
"""
function _get_wavelet_scale_indices_2d(res::Integer)
    scale_indices_matrix = zeros(Int, res, res)
    max_level = _wavelet_max_level(res)

    current_res = res
    for k in 1:max_level
        half_res = current_res ÷ 2
        half_res == 0 && break

        level = max_level - k + 1   # finest first, so the index grows as scales refine
        scale_indices_matrix[1:half_res, (half_res+1):current_res] .= level # Horizontal details
        scale_indices_matrix[(half_res+1):current_res, 1:half_res] .= level # Vertical details
        scale_indices_matrix[(half_res+1):current_res, (half_res+1):current_res] .= level # Diagonal details

        current_res = half_res
    end
    # Coarsest block: index 0, so it carries the unattenuated overall variance.
    scale_indices_matrix[1:current_res, 1:current_res] .= 0

    return vec(scale_indices_matrix)
end

function _resolve_wavelet(w)
    if w isa Wavelets.WT.OrthoWaveletClass || w isa Wavelets.WT.BiOrthoWaveletClass
        return Wavelets.wavelet(w)
    elseif w isa Symbol || w isa AbstractString
        str = lowercase(string(w))
        if str in ["db4", "daubechies4"]
            return Wavelets.wavelet(Wavelets.WT.db4)
        elseif str in ["db2", "daubechies2"]
            return Wavelets.wavelet(Wavelets.WT.db2)
        elseif str in ["db6", "daubechies6"]
            return Wavelets.wavelet(Wavelets.WT.db6)
        elseif str in ["db8", "daubechies8"]
            return Wavelets.wavelet(Wavelets.WT.db8)
        elseif str in ["haar", "db1"]
            return Wavelets.wavelet(Wavelets.WT.haar)
        elseif str in ["coif2"]
            return Wavelets.wavelet(Wavelets.WT.coif2)
        elseif str in ["coif4"]
            return Wavelets.wavelet(Wavelets.WT.coif4)
        elseif str in ["sym4"]
            return Wavelets.wavelet(Wavelets.WT.sym4)
        elseif str in ["sym8"]
            return Wavelets.wavelet(Wavelets.WT.sym8)
        else
            return Wavelets.wavelet(Wavelets.WT.db4)
        end
    else
        return Wavelets.wavelet(Wavelets.WT.db4)
    end
end

"""
    _wavelet_decomposition_lengths(res::Integer)

Band lengths of the length-preserving multiresolution wavelet coefficient vector that
Wavelets 0.10 produces for a `res`-point signal.

**Why this exists.** `waveletgp` previously called `c, l = wavedec(x_dummy, wt)` and was
therefore **unimplementable**: `wavedec` does not exist in the installed `Wavelets` 0.10 or
`WaveletsExt` 0.2, so the component could not be built at all. The decomposition itself was
never needed -- `x_dummy` is an all-zero signal, the coefficients `c` were discarded, and
only `length(c)` and the entries of `l` were read, purely to label which scale each basis
column belongs to. The scale boundaries are a property of the wavelet *family*, not of the
data, so they are computed directly here.

The installed transform is **length-preserving**, which is what makes the reconstruction
self-consistent: `dwt(x, wt, L)` returns exactly `length(x)` coefficients, `idwt(c, wt, L)`
inverts it to machine precision, and the `Phi[:, j] = idwt(e_j, wt, L)` basis built from it
is exactly orthonormal (`Phi' * Phi == I`). That is why `n_latent == res` holds.

Coefficients are stored coarsest-first, so the bands are

    1, 1, 2, 4, ..., 2^(L-1)        with  sum(lengths) == res == 2^L

Verified against the installed library: `dwt(ones(res), wt, L)` is nonzero *only* at position
1 (the approximation band), and the remaining positions group into bands of `2^(l-1)` -- the
groups are homogeneous in `sum(abs)/max(abs)`, e.g. for `res == 16`, `L == 4` the groups are
`{1}`, `{2}`, `{3:4}`, `{5:8}`, `{9:16}` with ratios `16.0`, `9.77`, `6.27`, `5.52`, `2.61`.

A classic MATLAB-style `wavedec` length vector (`[1, 8, 8, 4, 4, 2, 2, 1, 1]`, summing to 31
for a 16-point signal) does **not** apply here: it belongs to the boundary-extending variant,
which this Wavelets version does not implement.
"""
function _wavelet_decomposition_lengths(res::Integer)
    max_lvl = _wavelet_max_level(res)
    return [1; [2^(l - 1) for l in 1:max_lvl]]
end

"""
    _wavelet_scale_indices_1d(res::Integer)

Scale index of every coefficient of the 1-D length-`res` wavelet basis: a vector of length
`res` whose entries are the band level `0, 1, 2, ..., max_lvl` in coarsest-first order.

Index `0` is the coarsest (approximation) band, which carries the unattenuated overall
variance; each finer band gets a larger index and therefore a smaller variance under
`var_j = sigma^2 * 2^(-alpha * j)`.

The previous implementation had three defects here: it sized the vector as `length(c)` from a
standard `wavedec` layout (31 entries for `res == 16`) while the innovations vector has
`n_latent == res` entries, so `innovations .* sqrt.(scale_variances)` in `get_updates` was a
dimension mismatch that could never succeed; it looped over `2:length(l)-1`, skipping the
final band; and it assigned `max_lvl - (i - 1)`, which runs **negative** for the later bands
and inflates their variances above `sigma^2`.
"""
function _wavelet_scale_indices_1d(res::Integer)
    lengths = _wavelet_decomposition_lengths(res)
    idx = Vector{Int}(undef, sum(lengths))
    pos = 1
    for (band, n) in enumerate(lengths)
        idx[pos:(pos + n - 1)] .= band - 1
        pos += n
    end
    return idx
end

function get_precomputes(m::WaveletGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    if isempty(variables)
        error("WaveletGP model requires coordinate variables.")
    end
    for var_sym in variables
        if !hasproperty(M.data, var_sym)
            error("Coordinate variable ':$var_sym' for WaveletGP model not found in data.")
        end
    end
    
    res = m.resolution
    coords_cpu = Matrix{Float64}(M.data[:, variables])
    n_dims = size(coords_cpu, 2)
    
    if n_dims > 2
        error("WaveletGP currently supports only 1D and 2D spatial inputs.")
    end

    n_latent = res^n_dims
    
    min_coords = minimum(coords_cpu, dims=1)
    max_coords = maximum(coords_cpu, dims=1)
    grid_ranges = [range(min_coords[d], stop=max_coords[d], length=res) for d in 1:n_dims]

    wt = _resolve_wavelet(m.wavelet)
    max_lvl = _wavelet_max_level(res)
    local scale_indices_cpu
    if n_dims == 1
        scale_indices_cpu = _wavelet_scale_indices_1d(res)
    else # 2D
        scale_indices_cpu = _get_wavelet_scale_indices_2d(res)
    end
    # The innovations vector has exactly `n_latent` entries, and `get_updates` multiplies it
    # elementwise by `sqrt.(scale_variances)`, so these must agree exactly.
    length(scale_indices_cpu) == n_latent || error(
        "internal waveletgp error: got $(length(scale_indices_cpu)) scale indices " *
        "for $(n_dims)D resolution $res but n_latent is $n_latent.")

    # Precompute wavelet synthesis basis matrix: [n_latent, n_latent]
    # `idwt(c, wt, max_lvl)` is the exact inverse of the installed length-preserving
    # `dwt`, so these columns are an orthonormal wavelet basis for an orthonormal filter.
    Phi_wavelet = zeros(Float64, n_latent, n_latent)
    for j in 1:n_latent
        e_j = zeros(Float64, n_latent)
        e_j[j] = 1.0
        if n_dims == 1
            Phi_wavelet[:, j] = idwt(e_j, wt, max_lvl)
        else
            e_j_2d = reshape(e_j, res, res)
            Phi_wavelet[:, j] = vec(idwt(e_j_2d, wt, max_lvl))
        end
    end

    # Precompute interpolation projection to observation coordinates: [N_obs, n_latent]
    N_obs = size(coords_cpu, 1)
    B_obs = zeros(Float64, N_obs, n_latent)
    for j in 1:n_latent
        grid_j = reshape(Phi_wavelet[:, j], fill(res, n_dims)...)
        itp_j = linear_interpolation(Tuple(grid_ranges), grid_j, extrapolation_bc=Interpolations.Flat())
        B_obs[:, j] = [itp_j(coords_cpu[i, :]...) for i in 1:N_obs]
    end
    
    return (
        resolution = res,
        max_lvl = max_lvl,
        n_dims = n_dims,
        n_latent = n_latent,
        coords = coords_cpu,
        grid_ranges = grid_ranges,
        scale_indices = scale_indices_cpu,
        wt = wt,
        Phi_wavelet = Phi_wavelet,
        B_obs = B_obs
    )
end


function get_priors(
    m::WaveletGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    priors = String[]

    push!(priors, "$(_prior_or_constant(p_names.sigma, m.sigma))")
    push!(priors, "$(_prior_or_constant(p_names.alpha, m.alpha))")
    push!(priors, "$(p_names.innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)")

    return join(priors, "\n    ")
end

function get_updates(
    m::WaveletGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    return """
    # --- WaveletGP Component: $(key) ---
    let
        hyper = spec_registry[:$(key)].hyper
        
        scale_variances = $(p_names.sigma)^2 .* (2.0 .^ (-$(p_names.alpha) .* hyper.scale_indices))
        wavelet_coeffs = $(p_names.innovations) .* sqrt.(scale_variances)
        $(p_names.latent_field) = hyper.B_obs * wavelet_coeffs
        
        $(eta_target) = $(eta_target) .+ $(p_names.latent_field)
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
    max_lvl = hyper.max_lvl
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
        innovations_name = _find_parameter(p_names, string(v.innovations), k, is_multivariate_model)

        # Resolve the hyperparameters through the shared helper so a pinned constant (e.g.
        # `random(x, model=waveletgp, sigma=1.0)`) falls back to its value. The previous code
        # demanded a chain variable for `sigma`, so every pinned-sigma model matched none of
        # the three names and reconstruction silently returned an all-zero effect matrix --
        # the fit looked empty rather than failing.
        sigma_samples_cpu = _resolve_hyper_samples(
            chain, p_names, v.sigma, m.sigma, k, is_multivariate_model, n_samples)
        alpha_samples_cpu = _resolve_hyper_samples(
            chain, p_names, v.alpha, m.alpha, k, is_multivariate_model, n_samples)

        if isempty(innovations_name) || isnothing(sigma_samples_cpu) || isnothing(alpha_samples_cpu)
            @warn "Parameters for WaveletGP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total_eff, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = vec(sigma_samples_cpu)
        alpha_samples_cpu = vec(alpha_samples_cpu)
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent)

        effect_k = zeros(Float64, N_total_eff, n_samples)
        wt = _resolve_wavelet(m.wavelet)
        
        # --- Sample-wise Reconstruction ---
        for i in 1:n_samples
            sigma_s = sigma_samples_cpu[i]
            alpha_s = alpha_samples_cpu[i]
            innov_s = innovations_samples_cpu[i, :]

            scale_variances = sigma_s^2 .* (2.0 .^ (-alpha_s .* scale_indices_cpu))
            wavelet_coeffs = innov_s .* sqrt.(scale_variances)
            
            local latent_field_grid_cpu
            if n_dims == 1
                latent_field_grid_cpu = idwt(wavelet_coeffs, wt, max_lvl)
            else
                coeffs_reshaped = reshape(wavelet_coeffs, res, res)
                latent_field_grid_cpu = idwt(coeffs_reshaped, wt, max_lvl)
            end
            
            itp_s = linear_interpolation(Tuple(grid_ranges_cpu), latent_field_grid_cpu,
                extrapolation_bc=Interpolations.Flat())
            
            for j in 1:N_total_eff
                pt = ntuple(d -> coords_full_cpu[j, d], n_dims)
                effect_k[j, i] = itp_s(pt...)
            end
        end
        push!(structured_effects, effect_k)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end

