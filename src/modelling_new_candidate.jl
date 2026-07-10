### BSTM Modular Fragment Dispatch v10.0.0
# Timestamp: 2026-07-10
# Rationale: New naming scheme: [domain]_[variable]_[parameter]_[outcome_idx?]

import Statistics: eigen


function clean_dist_str(d)
    s = string(d)
    m = match(r"^([a-zA-Z0-9]+)\{?[^}]*\}?\(([^)]*)\)", s)
    if isnothing(m) return s end
    name = m.captures[1]
    args = m.captures[2]
    clean_args = replace(args, r"[a-zA-Zę-ωΑ-Ω]+\s*=\s*" => "")
    return "$(name)($(clean_args))"
end

# --- 1. Spatial Manifold Dispatch ---

function spatial(m_obj::IID, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "spatial_$(variable)_sigma$(suffix)"
    latent_field = "spatial_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(latent_field) ~ NamedDist(filldist(Normal(0, $(param_sigma)), M.s_N), :$(latent_field))
    """
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.s_idx)" :
        "eta .+= view($(latent_field), M.s_idx)"
    return (priors = priors, update = update)
end

function spatial(m_obj::Besag, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "spatial_$(variable)_sigma$(suffix)"
    latent_field = "spatial_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    Q_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template + noise * I
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_$(variable)$(suffix)), :$(latent_field))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum($(latent_field)))
    """
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.s_idx)" :
        "eta .+= view($(latent_field), M.s_idx)"
    return (priors = priors, update = update)
end

function spatial(m_obj::BYM2, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "spatial_$(variable)_sigma$(suffix)"
    param_rho = "spatial_$(variable)_rho$(suffix)"
    latent_struct = "spatial_$(variable)_struct$(suffix)"
    latent_iid = "spatial_$(variable)_iid$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_rho) ~ NamedDist($(clean_dist_str(m_obj.rho_prior)), :$(param_rho))
    $(latent_struct) ~ NamedDist(MvNormalCanon(zeros(M.s_N), spec_registry["$(variable)"].Q_template + noise * I), :$(latent_struct))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum($(latent_struct)))
    $(latent_iid) ~ NamedDist(MvNormal(zeros(M.s_N), I), :$(latent_iid))
    """
    
    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= view($(param_sigma) .* (sqrt($(param_rho)) .* $(latent_struct) .+ sqrt(1.0 - $(param_rho)) .* $(latent_iid)), M.s_idx)"
    
    return (priors = priors, update = update)
end

function spatial(m_obj::ICAR, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "spatial_$(variable)_sigma$(suffix)"
    latent_field = "spatial_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    Q_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template + noise * I
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_$(variable)$(suffix)), :$(latent_field))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum($(latent_field)))
    """
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.s_idx)" :
        "eta .+= view($(latent_field), M.s_idx)"
    return (priors = priors, update = update)
end

function spatial(m_obj::Leroux, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "spatial_$(variable)_sigma$(suffix)"
    param_rho = "spatial_$(variable)_rho$(suffix)"
    latent_field = "spatial_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_rho) ~ NamedDist($(clean_dist_str(m_obj.rho_prior)), :$(param_rho))
    Q_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* ($(param_rho) .* spec_registry["$(variable)"].Q_template + (1.0 - $(param_rho)) .* I(M.s_N)) + noise * I
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_$(variable)$(suffix)), :$(latent_field))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum($(latent_field)))
    """
    update = architecture == "multivariate" ? "eta_latent[:, $k] .+= view($(latent_field), M.s_idx)" : "eta .+= view($(latent_field), M.s_idx)"
    return (priors = priors, update = update)
end

function spatial(m_obj::SAR, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "spatial_$(variable)_sigma$(suffix)"
    param_rho = "spatial_$(variable)_rho$(suffix)"
    latent_field = "spatial_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_rho) ~ NamedDist($(clean_dist_str(m_obj.rho_prior)), :$(param_rho))
    Q_innov_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* I(M.s_N)
    L_op_$(variable)$(suffix) = I(M.s_N) - $(param_rho) * spec_registry["$(variable)"].Q_template
    Q_sar_$(variable)$(suffix) = Symmetric(L_op_$(variable)$(suffix)' * Q_innov_$(variable)$(suffix) * L_op_$(variable)$(suffix) + noise * I)
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros(M.s_N), Q_sar_$(variable)$(suffix)), :$(latent_field))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.s_N), sum($(latent_field)))
    """
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.s_idx)" :
        "eta .+= view($(latent_field), M.s_idx)"
    return (priors = priors, update = update)
end

# --- 2. Temporal Manifold Dispatch ---

function temporal(m_obj::AR1, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    param_rho = "temporal_$(variable)_rho$(suffix)"
    param_innov = "temporal_$(variable)_innov$(suffix)"
    field_name = "temporal_$(variable)_field$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_rho) ~ NamedDist($(clean_dist_str(m_obj.rho_prior)), :$(param_rho))
    $(param_innov) ~ NamedDist(MvNormal(zeros(M.t_N), I), :$(param_innov))
    """
    target_eta = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = """
    begin
        $(field_name) = Vector{T}(undef, M.t_N)
        $(field_name)[1] = $(param_innov)[1] / sqrt(1.0 - $(param_rho)^2 + noise)
        for i in 2:M.t_N
            $(field_name)[i] = $(param_rho) * $(field_name)[i-1] + $(param_innov)[i]
        end
        $(target_eta) .+= view($(field_name) .* $(param_sigma), M.t_idx)
    end
    """
    return (priors = priors, update = update)
end

function temporal(m_obj::RW1, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    latent_field = "temporal_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    Q_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template + noise * I
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros(M.t_N), Q_$(variable)$(suffix)), :$(latent_field))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.t_N), sum($(latent_field)))
    """
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.t_idx)" :
        "eta .+= view($(latent_field), M.t_idx)"
    return (priors = priors, update = update)
end

function temporal(m_obj::RW2, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    latent_field = "temporal_$(variable)_latent$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    Q_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template + noise * I
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros(M.t_N), Q_$(variable)$(suffix)), :$(latent_field))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * M.t_N), sum($(latent_field)))
    """
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.t_idx)" :
        "eta .+= view($(latent_field), M.t_idx)"
    return (priors = priors, update = update)
end

function temporal(m_obj::Cyclic, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    latent_field = "temporal_$(variable)_latent$(suffix)"
    n_units_var = "n_units_temporal_$(variable)$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(n_units_var) = size(spec_registry["$(variable)"].Q_template, 1)
    Q_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template + noise * I
    $(latent_field) ~ NamedDist(MvNormalCanon(zeros($(n_units_var)), Q_$(variable)$(suffix)), :$(latent_field))
    """
    # A cyclic model does not require a sum-to-zero constraint.
    # It uses the seasonal index `M.u_idx` set by the formula parser.
    update = architecture == "multivariate" ?
        "eta_latent[:, $k] .+= view($(latent_field), M.u_idx)" :
        "eta .+= view($(latent_field), M.u_idx)"
    return (priors = priors, update = update)
end

function temporal(m_obj::Harmonic, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    # This function generates a model fragment for a harmonic (seasonal) component.
    # It creates a basis of sine and cosine functions and samples coefficients for them.
    
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    param_coeffs = "temporal_$(variable)_coeffs$(suffix)"
    
    # The number of harmonics determines the complexity of the seasonal pattern.
    n_harmonics = get(m_obj.params, :n_harmonics, 2) # Default to 2 harmonics
    n_basis = 2 * n_harmonics
    period = m_obj.period

    priors = """
    # Priors for harmonic component: $(variable)
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_coeffs) ~ NamedDist(filldist(Normal(0, $(param_sigma)), $(n_basis)), :$(param_coeffs))
    """

    # The update string constructs the harmonic basis matrix on-the-fly.
    # This avoids storing large basis matrices in the configuration object.
    # It uses the seasonal index `M.u_idx` set by the formula parser, here aliased as u_vals for clarity.
    update = """
    begin
        local u_vals = M.u_idx
        local B_harmonic = hcat([ (i % 2 == 1 ? sin : cos).(2pi * ceil(i/2) .* u_vals ./ $(period)) for i in 1:$(n_basis)]...)
        $(architecture == "multivariate" ? "eta_latent[:, $k]" : "eta") .+= B_harmonic * $(param_coeffs)
    end
    """
    return (priors = priors, update = update)
end

function temporal(m_obj::Union{PSpline, TPS}, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    param_coeffs = "temporal_$(variable)_coeffs$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    n_basis_$(variable) = size(spec_registry["$(variable)"].Q_template, 1)
    Q_penalty_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template
    $(param_coeffs)_raw ~ NamedDist(MvNormalCanon(zeros(n_basis_$(variable)), Q_penalty_$(variable)$(suffix)), :($(param_coeffs)_raw))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_basis_$(variable)), sum($(param_coeffs)_raw))
    $(param_coeffs) = vec($(param_coeffs)_raw)
    """

    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= M.basis_matrices[Symbol('$(variable)')] * $(param_coeffs)"
    
    return (priors = priors, update = update)
end

function temporal(m_obj::Union{BSpline, FFT, Wavelet, Spherical, ExponentialDecay, Barycentric}, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    param_coeffs = "temporal_$(variable)_coeffs$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    n_basis_$(variable) = size(M.basis_matrices[Symbol("$(variable)")], 2)
    $(param_coeffs) ~ NamedDist(filldist(Normal(0, $(param_sigma)), n_basis_$(variable)), :$(param_coeffs))
    """

    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= M.basis_matrices[Symbol('$(variable)')] * $(param_coeffs)"
    
    return (priors = priors, update = update)
end

function temporal(m_obj::RFF, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "temporal_$(variable)_sigma$(suffix)"
    param_ls = "temporal_$(variable)_ls$(suffix)"
    param_coeffs = "temporal_$(variable)_coeffs$(suffix)"
    param_W = "temporal_$(variable)_W$(suffix)"
    param_b = "temporal_$(variable)_b$(suffix)"
    
    n_features = m_obj.n_features

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_ls) ~ NamedDist($(clean_dist_str(m_obj.lengthscale_prior)), :$(param_ls))
    
    d_rff_$(variable) = size(spec_registry["$(variable)"].params.coords, 2)
    sigma_spectral_$(variable) = 1.0 / $(param_ls)
    $(param_W)_raw ~ NamedDist(filldist(Normal(0, 1), d_rff_$(variable) * $(n_features)), :($(param_W)_raw))
    $(param_W) = reshape($(param_W)_raw, d_rff_$(variable), $(n_features)) .* sigma_spectral_$(variable)
    $(param_b) ~ NamedDist(filldist(Uniform(0, 2*pi), $(n_features)), :$(param_b))

    B_rff_$(variable) = sqrt(2.0 / $(n_features)) .* cos.((spec_registry["$(variable)"].params.coords * $(param_W)) .+ $(param_b)')

    $(param_coeffs) ~ NamedDist(filldist(Normal(0, $(param_sigma)), $(n_features)), :$(param_coeffs))
    """

    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= B_rff_$(variable) * $(param_coeffs)"
    
    return (priors = priors, update = update)
end

function temporal(m_obj::Union{GP, FITC, SVGP, Nystrom}, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    # This logic is identical to the smooth() implementation for GPs.
    # The variable names are updated to reflect the temporal domain.
    smooth_frag = smooth(m_obj, variable, architecture, k)
    priors = replace(smooth_frag.priors, "smooth_" => "temporal_")
    update = replace(smooth_frag.update, "smooth_" => "temporal_")
    return (priors=priors, update=update)
end

# --- 3. Smooth Manifold Dispatch ---

function smooth(m_obj::Union{PSpline, TPS}, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "smooth_$(variable)_sigma$(suffix)"
    param_coeffs = "smooth_$(variable)_coeffs$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    n_basis_$(variable) = size(spec_registry["$(variable)"].Q_template, 1)
    Q_penalty_$(variable)$(suffix) = (1.0 / ($(param_sigma)^2 + noise)) .* spec_registry["$(variable)"].Q_template
    $(param_coeffs)_raw ~ NamedDist(MvNormalCanon(zeros(n_basis_$(variable)), Q_penalty_$(variable)$(suffix)), :($(param_coeffs)_raw))
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * n_basis_$(variable)), sum($(param_coeffs)_raw))
    $(param_coeffs) = vec($(param_coeffs)_raw)
    """

    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= M.basis_matrices[Symbol('$(variable)')] * $(param_coeffs)"
    
    return (priors = priors, update = update)
end

function smooth(m_obj::Union{BSpline, FFT, Wavelet, Moran, Spherical, ExponentialDecay, Barycentric}, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "smooth_$(variable)_sigma$(suffix)"
    param_coeffs = "smooth_$(variable)_coeffs$(suffix)"

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    n_basis_$(variable) = size(M.basis_matrices[Symbol("$(variable)")], 2)
    $(param_coeffs) ~ NamedDist(filldist(Normal(0, $(param_sigma)), n_basis_$(variable)), :$(param_coeffs))
    """

    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= M.basis_matrices[Symbol('$(variable)')] * $(param_coeffs)"
    
    return (priors = priors, update = update)
end

function smooth(m_obj::RFF, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "smooth_$(variable)_sigma$(suffix)"
    param_ls = "smooth_$(variable)_ls$(suffix)"
    param_coeffs = "smooth_$(variable)_coeffs$(suffix)"
    param_W = "smooth_$(variable)_W$(suffix)"
    param_b = "smooth_$(variable)_b$(suffix)"
    
    n_features = m_obj.n_features

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_ls) ~ NamedDist($(clean_dist_str(m_obj.lengthscale_prior)), :$(param_ls))
    
    # RFF parameters
    d_rff_$(variable) = size(spec_registry["$(variable)"].params.coords, 2)
    sigma_spectral_$(variable) = 1.0 / $(param_ls)
    $(param_W)_raw ~ NamedDist(filldist(Normal(0, 1), d_rff_$(variable) * $(n_features)), :($(param_W)_raw))
    $(param_W) = reshape($(param_W)_raw, d_rff_$(variable), $(n_features)) .* sigma_spectral_$(variable)
    $(param_b) ~ NamedDist(filldist(Uniform(0, 2*pi), $(n_features)), :$(param_b))

    # RFF basis matrix construction
    B_rff_$(variable) = sqrt(2.0 / $(n_features)) .* cos.((spec_registry["$(variable)"].params.coords * $(param_W)) .+ $(param_b)')

    # Coefficients
    $(param_coeffs) ~ NamedDist(filldist(Normal(0, $(param_sigma)), $(n_features)), :$(param_coeffs))
    """

    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"
    update = "$(target) .+= B_rff_$(variable) * $(param_coeffs)"
    
    return (priors = priors, update = update)
end

function smooth(m_obj::Union{GP, FITC, SVGP, Nystrom}, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    suffix = architecture == "multivariate" ? "_$(k)" : ""
    param_sigma = "smooth_$(variable)_sigma$(suffix)"
    param_ls = "smooth_$(variable)_ls$(suffix)"
    param_u_raw = "smooth_$(variable)_u_raw$(suffix)"
    param_f_innov = "smooth_$(variable)_f_innov$(suffix)"

    n_inducing = m_obj.n_inducing
    kernel_str = m_obj.kernel

    priors = """
    $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
    $(param_ls) ~ NamedDist($(clean_dist_str(m_obj.lengthscale_prior)), :$(param_ls))
    
    # Latent values at inducing points
    $(param_u_raw) ~ NamedDist(MvNormal(zeros($(n_inducing)), I), :$(param_u_raw))
    
    # Innovation for non-centered parameterization of f
    n_obs_$(variable) = size(spec_registry["$(variable)"].params.coords, 1)
    $(param_f_innov) ~ NamedDist(MvNormal(zeros(n_obs_$(variable)), I), :$(param_f_innov))
    """

    update = """
    begin
        # Reconstruct GP field for smooth_$(variable)
        let spec = spec_registry["$(variable)"],
            sigma = $(param_sigma),
            ls = $(param_ls),
            u_raw = $(param_u_raw),
            f_innov = $(param_f_innov)

            kernel = get_kernel_from_string("$(kernel_str)")
            kernel_scaled = sigma^2 * (kernel ∘ ScaleTransform(1.0 / ls))
            
            Z_inducing = spec.params.Z_inducing
            coords = spec.params.coords
            
            K_uu = kernelmatrix(kernel_scaled, RowVecs(Z_inducing)) + noise * I
            K_uf = kernelmatrix(kernel_scaled, RowVecs(Z_inducing), RowVecs(coords))
            k_ff_diag = diag(kernelmatrix(kernel_scaled, RowVecs(coords)))
            L_uu = cholesky(Symmetric(K_uu)).L
            u_latent = L_uu * u_raw
            A = L_uu' \\ K_uf
            mean_f = A' * u_latent
            var_f = k_ff_diag - vec(sum(A.^2, dims=1))
            gp_effect = mean_f + sqrt.(max.(var_f, 0.0) .+ noise) .* f_innov
            $(architecture == "multivariate" ? "eta_latent[:, $k]" : "eta") .+= gp_effect
        end
    end
    """
    
    return (priors = priors, update = update)
end

 

function get_kernel_from_string(kernel_name::String)
    k_name = lowercase(kernel_name)
    if k_name == "constant"
        return ConstantKernel()
    elseif k_name == "linear"
        return LinearKernel()
    elseif k_name == "matern12" || k_name == "exponential"
        return Matern12Kernel()
    elseif k_name == "matern32"
        return Matern32Kernel()
    elseif k_name == "matern52"
        return Matern52Kernel()
    elseif k_name == "spherical"
        return SphericalKernel()
    elseif k_name == "squared_exponential" || k_name == "se" || k_name == "gaussian" || k_name == "rbf"
        return SqExponentialKernel()
    elseif k_name == "periodic"
        return PeriodicKernel()
    else
        @warn "Kernel '$kernel_name' not recognized. Defaulting to SqExponentialKernel."
        return SqExponentialKernel()
    end
end




# --- 4. Spatially Varying Coefficients (SVC) Dispatch ---

function svc(m_obj::SVCManifold, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    cov_var = m_obj.covariate
    inner_model = m_obj.model
    
    # Generate the fragment for the inner spatial model.
    inner_frag = spatial(inner_model, variable, architecture, k)
    
    # Modify the update string to include the covariate multiplication.
    modified_update = replace(inner_frag.update, "M.s_idx)" => "M.s_idx) .* M.data[!, :$(cov_var)]")

    return (priors = inner_frag.priors, update = modified_update)
end

# --- 5. Multifidelity-Specific Fragments ---

function multifidelity_z(m_obj::Manifold, variable::String, architecture::String)
    priors = """
    z_$(variable)_ls ~ NamedDist(Gamma(2, 2), :z_$(variable)_ls)
    z_$(variable)_beta ~ NamedDist(filldist(Normal(0, 1), M.M_rff), :z_$(variable)_beta)
    z_$(variable)_beta_eta ~ NamedDist(Normal(0, 1), :z_$(variable)_beta_eta)
    """
    update = """
    begin
        z_proj_$(variable) = (M.z_coords_s * (M.W_fixed[1:size(M.z_coords_s, 2), :] ./ z_$(variable)_ls)) .+ M.b_fixed'
        z_latent_field_$(variable) = M.rff_scale .* (cos.(z_proj_$(variable)) * z_$(variable)_beta)
        eta .+= z_latent_field_$(variable) .* z_$(variable)_beta_eta
    end
    """
    return (priors = priors, update = update)
end

function multifidelity_w(m_obj::Manifold, variable::String, architecture::String)
    priors = """
    w_$(variable)_ls ~ NamedDist(Gamma(2, 2), :w_$(variable)_ls)
    w_beta_flat_$(variable) ~ NamedDist(filldist(Normal(0, 1), M.M_rff * 3), :w_beta_$(variable))
    w_beta_$(variable) = reshape(w_beta_flat_$(variable), M.M_rff, 3)
    w_beta_eta_$(variable) ~ NamedDist(MvNormal(zeros(3), I), :w_beta_eta_$(variable))
    """
    update = """
    begin
        w_coords_augmented_$(variable) = hcat(M.w_coords_st, z_latent_field_$(variable)[1:size(M.w_coords_st, 1)])
        w_proj_$(variable) = (w_coords_augmented_$(variable) * (M.W_fixed[1:size(w_coords_augmented_$(variable), 2), :] ./ w_$(variable)_ls)) .+ M.b_fixed'
        w_latent_field_$(variable) = M.rff_scale .* (cos.(w_proj_$(variable)) * w_beta_$(variable))
        eta .+= w_latent_field_$(variable) * w_beta_eta_$(variable)
    end
    """
    return (priors = priors, update = update)
end

function observation_volatility(M)
    if get(M, :use_sv, false) == true
        priors = """
        volatility_sigma_log_var ~ NamedDist(Exponential(1.0), :volatility_sigma_log_var)
        volatility_beta_latent ~ NamedDist(filldist(Normal(0, 1), M.M_rff_sigma), :volatility_beta_latent)
        """
        calculation = """
        vol_proj_field = M.vol_proj * volatility_beta_latent
        vol_latent_field = sqrt(2.0 / M.M_rff_sigma) .* cos.(vol_proj_field)
        y_sigma = exp.((volatility_sigma_log_var .* vol_latent_field[1:M.y_N]) ./ 2.0)
        """
    else
        priors = ""
        calculation = "y_sigma = fill(y_sigma_const, M.y_N)"
    end
    return (priors = priors, calculation = calculation)
end

# --- 6. Mixed Effects Manifold ---

function mixed(m_obj::MixedManifold, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    # This function generates a model fragment for mixed-effects components.
    # It handles simple random intercepts `(1 | g)`, simple random slopes `(z | g)`,
    # and correlated random intercepts and slopes `(1 + z | g)`.

    lhs_terms = [strip(t) for t in split(m_obj.lhs, '+')]
    n_terms = length(lhs_terms)

    suffix = architecture == "multivariate" ? "_$(k)" : ""
    n_cat_var = "spec_registry[\"$(variable)\"].params.n_cat"
    indices_var = "spec_registry[\"$(variable)\"].params.indices"
    target = architecture == "multivariate" ? "eta_latent[:, $k]" : "eta"

    if n_terms == 1
        # --- Independent Random Intercept or Slope ---
        lhs_covariate = lhs_terms[1]
        param_sigma = "mixed_$(variable)_sigma$(suffix)"
        latent_coeffs = "mixed_$(variable)_coeffs$(suffix)"

        priors = """
        # Priors for mixed effect: $(variable)
        $(param_sigma) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(param_sigma))
        $(latent_coeffs) ~ NamedDist(filldist(Normal(0, $(param_sigma)), $(n_cat_var)), :$(latent_coeffs))
        """
        
        update = if lhs_covariate == "1"
            # Random Intercept
            "$(target) .+= view($(latent_coeffs), $(indices_var))"
        else
            # Random Slope
            slope_cov_data = "M.data[!, Symbol(\"$(lhs_covariate)\")]"
            "$(target) .+= view($(latent_coeffs), $(indices_var)) .* $(slope_cov_data)"
        end
        
        return (priors = priors, update = update)

    else
        # --- Correlated Random Intercepts and Slopes ---
        priors_acc = ["# Priors for correlated mixed effect: $(variable)"]
        updates_acc = String[]
        
        # Priors for standard deviations of each term
        sigmas = []
        for (i, term) in enumerate(lhs_terms)
            term_name = term == "1" ? "intercept" : "slope_$(term)"
            sigma_param = "mixed_$(variable)_sigma_$(term_name)$(suffix)"
            push!(sigmas, sigma_param)
            push!(priors_acc, "$(sigma_param) ~ NamedDist($(clean_dist_str(m_obj.sigma_prior)), :$(sigma_param))")
        end

        # Prior for the correlation matrix
        l_corr_param = "mixed_$(variable)_L_corr$(suffix)"
        push!(priors_acc, "$(l_corr_param) ~ NamedDist(LKJCholesky($(n_terms), 2.0), :$(l_corr_param))")

        # Construct the scale matrix D
        scale_matrix_D = "D_$(variable)$(suffix) = Diagonal([$(join(sigmas, ", "))])"
        push!(priors_acc, scale_matrix_D)

        # Sample raw coefficients and transform them
        raw_coeffs_param = "mixed_$(variable)_coeffs_raw$(suffix)"
        push!(priors_acc, "$(raw_coeffs_param) ~ NamedDist(MvNormal(zeros($(n_cat_var) * $(n_terms)), I), :$(raw_coeffs_param))")
        
        # Reshape and apply transformation
        correlated_coeffs_param = "mixed_$(variable)_coeffs_correlated$(suffix)"
        transform_block = """
        local raw_matrix = reshape($(raw_coeffs_param), $(n_cat_var), $(n_terms))
        local $(correlated_coeffs_param) = (($(l_corr_param).L * raw_matrix')' * D_$(variable)$(suffix))
        """

        push!(priors_acc, transform_block)

        # Generate update strings for each term
        for (i, term) in enumerate(lhs_terms)
            term_coeffs = "$(correlated_coeffs_param)[:, $(i)]"
            if term == "1"
                push!(updates_acc, "$(target) .+= view($(term_coeffs), $(indices_var))")
            else
                slope_cov_data = "M.data[!, Symbol(\"$(term)\")]"
                push!(updates_acc, "$(target) .+= view($(term_coeffs), $(indices_var)) .* $(slope_cov_data)")
            end
        end

        return (priors = join(priors_acc, "\n"), update = join(updates_acc, "\n"))
    end
end


# --- 7. Eigen Manifold (Bayesian PCA) ---

function eigen(m_obj::Eigen, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    if architecture == "multivariate"
        @warn "Eigen manifold is not fully supported in multivariate architecture yet. Generating univariate fragment."
    end
    
    key = variable
    param_pca_sd = "eigen_$(key)_pca_sd"
    param_pdef_sd = "eigen_$(key)_pdef_sd"
    param_v_vec = "eigen_$(key)_v"
    param_z_latent = "eigen_$(key)_z_latent"
    param_eigen_coeffs = "eigen_$(key)_coeffs"
    
    n_factors = m_obj.n_factors
    vars_str = join([":$(s)" for s in m_obj.variables], ", ")
    
    priors = """
    # --- Eigen Manifold: $(key) ---
    $(param_pca_sd) ~ NamedDist($(clean_dist_str(m_obj.pca_sd_prior)), :$(param_pca_sd))
    $(param_pdef_sd) ~ NamedDist($(clean_dist_str(m_obj.pdef_sd_prior)), :$(param_pdef_sd))
    
    ltri_indices_$(key) = spec_registry["$(key)"].params.ltri_indices
    n_vars_$(key) = length(spec_registry["$(key)"].variables)
    
    $(param_v_vec) ~ NamedDist(filldist(Normal(0, 1), length(ltri_indices_$(key))), :$(param_v_vec))
    
    v_mat_$(key) = zeros(T, n_vars_$(key), $(n_factors))
    v_mat_$(key)[ltri_indices_$(key)] .= $(param_v_vec)
    
    U_mat_$(key) = householder_to_eigenvector(v_mat_$(key), n_vars_$(key), $(n_factors))
    
    $(param_z_latent) ~ NamedDist(filldist(Normal(0, 1), $(n_factors), M.y_N), :$(param_z_latent))
    $(param_eigen_coeffs) ~ NamedDist(filldist(Normal(0, 1), $(n_factors)), :$(param_eigen_coeffs))
    
    # Likelihood for the multivariate data being decomposed
    cov_data_$(key) = Matrix(M.data[!, [$(vars_str)]])'
    reconstructed_cov_$(key) = U_mat_$(key) * $(param_z_latent)
    for i in 1:M.y_N
        Turing.@addlogprob! logpdf(MvNormal(reconstructed_cov_$(key)[:, i], $(param_pdef_sd)^2 * I), cov_data_$(key)[:, i])
    end
    """
    
    update = "eta .+= $(param_z_latent)' * $(param_eigen_coeffs)"
    
    return (priors = priors, update = update)
end

# --- 8. Composed Manifold (Interact) ---

function interact(m_obj::ComposedManifold, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    # This function handles manifolds created by algebraic operators.
    # The implementation strategy depends on the operator.
    op = m_obj.operator
    key = variable
    suffix = architecture == "multivariate" ? "_$(k)" : ""

    if op == :pipe
        # Handles state-space models, e.g., spatial() |> temporal(model=ar1).
        # Assumes the first component is the state and the second is the dynamic model.
        if length(m_obj.components) != 2
            @warn "Pipe operator for '$(key)' requires exactly two components (state |> dynamic). Skipping."
            return (priors="", update="")
        end

        state_manifold = m_obj.components[1]
        dynamic_manifold = m_obj.components[2]

        # The parser must create a spec for the state manifold, which we look up.
        # We assume a naming convention like "state_of_<piped_manifold_key>".
        state_manifold_key = "state_of_$(key)"

        if dynamic_manifold isa Union{AR1, RW1, RW2}
            # State-space model with GMRF dynamics.
            param_rho_pipe = "interaction_$(key)_rho$(suffix)"
            param_sigma_innov = "interaction_$(key)_sigma_innov$(suffix)"
            priors = [
                "$(param_sigma_innov) ~ NamedDist($(clean_dist_str(dynamic_manifold.sigma_prior)), :$(param_sigma_innov))"
            ]
            if dynamic_manifold isa AR1
                push!(priors, "$(param_rho_pipe) ~ NamedDist($(clean_dist_str(dynamic_manifold.rho_prior)), :$(param_rho_pipe))")
            end

            update = """
            begin
                # State-space model for $(key)
                let state_spec = spec_registry["$(state_manifold_key)"],
                    sigma = $(param_sigma_innov)

                    Q_innov = recompose_precision_new(state_spec.manifold_obj, M, sigma)
                    L_innov = cholesky(Symmetric(Q_innov)).L
                    n_state = size(Q_innov, 1)
                    pipe_field = Matrix{T}(undef, n_state, M.t_N)

                    if $(dynamic_manifold isa AR1)
                        rho = $(param_rho_pipe)
                        innov_base_1 ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_$(key)_1$(suffix)"))
                        pipe_field[:, 1] = (L_innov \\ innov_base_1) ./ sqrt(1.0 - rho^2 + noise)
                        for t in 2:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_$(key)_", t, "$(suffix)"))
                            pipe_field[:, t] = rho .* pipe_field[:, t-1] .+ (L_innov \\ innov_t)
                        end
                    elseif $(dynamic_manifold isa RW1)
                        pipe_field[:, 1] ~ MvNormal(zeros(n_state), 100.0 * I)
                        for t in 2:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_$(key)_", t, "$(suffix)"))
                            pipe_field[:, t] = pipe_field[:, t-1] .+ (L_innov \\ innov_t)
                        end
                    else # RW2
                        pipe_field[:, 1] ~ MvNormal(zeros(n_state), 100.0 * I)
                        pipe_field[:, 2] ~ MvNormal(zeros(n_state), 100.0 * I)
                        for t in 3:M.t_N
                            innov_t ~ NamedDist(MvNormal(zeros(n_state), I), Symbol("innov_base_$(key)_", t, "$(suffix)"))
                            pipe_field[:, t] = 2.0 .* pipe_field[:, t-1] .- pipe_field[:, t-2] .+ (L_innov \\ innov_t)
                        end
                    end

                    for i in 1:N; eta[i] += pipe_field[M.s_idx[i], M.t_idx[i]]; end
                end
            end
            """
            return (priors=join(priors, "\n"), update=update)
        elseif dynamic_manifold isa Union{RFF, TPS, PSpline, BSpline, FFT, Wavelet, Spherical, ExponentialDecay, Barycentric}
            @warn "State-space models with smooth dynamic components ('$(typeof(dynamic_manifold))') are experimental. Returning empty fragment for '$(key)'."
            return (priors="", update="")
        end
    elseif op == :kronecker_product
        # This is handled by the main assembler's spatiotemporal interaction block,
        # which has the necessary context of both the main spatial and temporal manifolds.
        # This fragment generator serves as a pass-through. The `spacetime()` module
        # in the formula is the user-facing way to trigger this logic.
        return (priors="", update="")
    else
        @warn "ComposedManifold operator ':$op' is not fully supported by the fragment system. Returning empty fragment."
        return (priors="", update="")
    end

    @warn "Unhandled ComposedManifold structure for '$(key)'. Returning empty fragment."
    return (priors="", update="")
end


# --- 9. Dynamics Manifold ---

function dynamics(m_obj::DynamicsManifold, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    if architecture == "multivariate"
        @warn "Dynamics manifold is not fully supported in multivariate architecture yet. Generating univariate fragment."
    end

    model_type = m_obj.model
    key = variable
    
    priors = ""
    update = ""

    if model_type == "advection" || model_type == "diffusion"
        param_v = "dynamics_$(key)_v"
        param_sig = "dynamics_$(key)_sigma"
        dyn_field = "dynamics_$(key)_field"

        priors = """
        $(param_v) ~ NamedDist(Normal(0, 0.5), :$(param_v))
        $(param_sig) ~ NamedDist(Exponential(1.0), :$(param_sig))
        """
        
        update = """
        begin
            # Advection/Diffusion Dynamics for $(key)
            let L_mat = spec_registry["$(key)"].Q_template,
                v = $(param_v),
                sig = $(param_sig)

                $(dyn_field) = Matrix{T}(undef, M.s_N, M.t_N)
                $(dyn_field)[:, 1] ~ MvNormal(zeros(M.s_N), I)

                for t in 2:M.t_N
                    drift = if "$(model_type)" == "advection"
                        -v .* (L_mat * $(dyn_field)[:, t-1])
                    else # diffusion
                        v .* (L_mat * $(dyn_field)[:, t-1])
                    end
                    $(dyn_field)[:, t] ~ MvNormal($(dyn_field)[:, t-1] .+ drift, sig^2 * I)
                end

                for i in 1:N
                    eta[i] += $(dyn_field)[M.s_idx[i], M.t_idx[i]]
                end
            end
        end
        """
    elseif model_type == "gompertz" || model_type == "logistic_basic"
        param_r = "dynamics_$(key)_r"
        param_K = "dynamics_$(key)_K"
        pop_state = "dynamics_$(key)_state"

        priors = """
        $(param_r) ~ NamedDist(LogNormal(-1.5, 0.5), :$(param_r))
        $(param_K) ~ NamedDist(Normal(150, 50), :$(param_K))
        """

        update = """
        begin
            # Biological Dynamics for $(key)
            let r = $(param_r), K = $(param_K)
                $(pop_state) = Vector{T}(undef, M.t_N)
                $(pop_state)[1] ~ Normal(log(K / 2.0), 1.0)
                for t in 2:M.t_N
                    growth = if "$(model_type)" == "gompertz"
                        r * (log(K) - $(pop_state)[t-1])
                    else # logistic
                        r * $(pop_state)[t-1] * (1.0 - exp($(pop_state)[t-1]) / K)
                    end
                    $(pop_state)[t] ~ Normal($(pop_state)[t-1] + growth, 0.1)
                end
                eta .+= $(pop_state)[M.t_idx]
            end
        end
        """
    else
        @warn "Dynamics model type '$(model_type)' not recognized. Returning empty fragment."
    end

    return (priors=priors, update=update)
end

# --- 10. Spacetime Interaction Module ---

function spacetime(m_obj::Manifold, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    # This is a placeholder fragment generator. The actual logic for spacetime interactions
    # is handled by the assembler based on the M.model_st flag, which is set by the
    # process_spacetime_module! in the formula parser.
    # This function ensures that the module is recognized by the system but delegates
    # the complex implementation to the assembler where context from both spatial and
    # temporal manifolds is available.
    return (priors="", update="")
end

# --- 11. Custom User-Defined Manifold ---

struct CustomManifold <: ManifoldModel
    code_fragment::String
    params::Dict{Symbol, Any}
end

function custom(m_obj::CustomManifold, variable::String, architecture::String, k::Union{Int, Nothing}=nothing)
    # This manifold allows the user to inject a raw code string into the model.
    # The user's code is responsible for sampling its own parameters and updating `eta`.
    # The code fragment is expected to be a valid `begin ... end` block.
    
    # Priors are expected to be defined within the user's code block.
    priors = "" 
    
    # The update is the user's code fragment itself.
    # The assembler will place this block directly into the model.
    update = m_obj.code_fragment
    
    return (priors=priors, update=update)
end


### BSTM Unified Text Assembler v10.0.0
# Rationale: A unified assembler for all model architectures.

function bstm_text_assembler(M::NamedTuple)
    arch = get(M, :model_arch, "univariate")
    is_multivariate = arch == "multivariate"
    
    # Determine model function name based on architecture
    model_func_name = if arch == "multivariate"
        :bstm_text_generated_multivariate
    elseif arch == "multifidelity"
        :bstm_text_generated_multifidelity
    else # univariate
        :bstm_text_generated_univariate
    end

    spec_registry = Dict{String, Any}()
    priors_acc = String[]
    updates_acc = String[]
    outcomes_N = get(M, :outcomes_N, 1)
    
    # Track main spatial and temporal components for interactions
    main_spatial_spec = nothing
    main_temporal_spec = nothing
    main_spatial_Q = "sparse(I(M.s_N))"
    main_temporal_Q = "sparse(I(M.t_N))"

    for spec in M.manifolds
        spec_registry[string(spec.var)] = spec
        domain_fn = getfield(Main, spec.domain)

        for k in 1:outcomes_N
            outcome_idx = arch == "multivariate" ? k : nothing
            frag = domain_fn(spec.manifold_obj, string(spec.var), arch, outcome_idx)
            
            update_str = frag.update # Placeholder substitution is handled by the fragment functions now

            push!(priors_acc, frag.priors)
            push!(updates_acc, update_str)
        end

        # Capture main components for ST interactions
        if spec.domain == :spatial && isnothing(main_spatial_spec)
            main_spatial_spec = spec
        end
        if spec.domain == :temporal && isnothing(main_temporal_spec)
            main_temporal_spec = spec
        end
    end

    # --- Likelihood and Global Priors Section ---
    likelihood_section = if is_multivariate
        """
        # --- Global Multivariate Priors ---
        L_corr ~ NamedDist(LKJCholesky(K, 1.0), :L_corr)
        y_sigma ~ NamedDist(filldist(Exponential(1.0), K), :y_sigma)
        """
    else # univariate or multifidelity
        """
        # --- Observation Noise & Volatility ---
        y_sigma_const ~ NamedDist(Exponential(1.0), :y_sigma_const)
        $(observation_volatility(M).priors)
        $(observation_volatility(M).calculation)
        """
    end

    # --- Linear Predictor and Updates Section ---
    eta_name = is_multivariate ? "eta_latent" : "eta"
    eta_init = is_multivariate ? "zeros(T, N, K)" : "zeros(T, N)"

    intercept_block = if get(M, :add_intercept, false)
        dist = is_multivariate ? "filldist(get(M, :intercept_prior, Normal(0,5)), K)" : "get(M, :intercept_prior, Normal(0, 5))"
        update = is_multivariate ? "for k in 1:K; $(eta_name)[:, k] .+= intercept[k]; end" : "$(eta_name) .+= intercept"
        """
        intercept ~ NamedDist($(dist), :intercept)
        $(update)
        """
    else "" end

    offset_block = if haskey(M, :log_offset)
        is_multivariate ? "$(eta_name) .+= get(M, :log_offset, zeros(T, N, 1))" : "$(eta_name) .+= get(M, :log_offset, zeros(T, N))"
    else "" end

    fixed_effects_block = if M.Xfixed_N > 0
        if is_multivariate
            beta_count = "M.Xfixed_N * K"
            update = "$(eta_name) .+= M.Xfixed * reshape(Xfixed_beta, M.Xfixed_N, K)"
            """
            Xfixed_beta ~ NamedDist(MvNormal(zeros($(beta_count)), 5.0 * I), :Xfixed_beta)
            $(update)
            """
        else # Univariate case with per-coefficient priors
            """
            begin
                local Xfixed_beta = Vector{T}(undef, M.Xfixed_N)
                local priors = get(M, :Xfixed_priors_vec, [Normal(0, 5) for _ in 1:M.Xfixed_N])
                for i in 1:M.Xfixed_N
                    Xfixed_beta[i] ~ NamedDist(priors[i], Symbol("Xfixed_beta[", i, "]"))
                end
                eta .+= M.Xfixed * Xfixed_beta
            end
            """
        end
    else "" end

    # --- Spatiotemporal Interaction Block ---
    st_interaction_block = ""
    model_st = get(M, :model_st, "none")
    if model_st != "none"
        if isnothing(main_spatial_spec) || isnothing(main_temporal_spec)
            @warn "Spacetime interaction specified, but could not find both a primary spatial and temporal manifold. Interaction term will be ignored."
        else
            # This block replicates the logic from the reference bstm_univariate model
            # for Knorr-Held spatiotemporal interactions.
            st_interaction_block = """
            # --- Spatiotemporal Interaction (Type $(model_st)) ---
            st_sigma ~ NamedDist(Exponential(0.5), :st_sigma)
            st_raw ~ NamedDist(MvNormal(zeros(T, M.s_N * M.t_N), I), :st_raw)
            
            let s_Q = spec_registry["$(main_spatial_spec.var)"].Q_template,
                t_Q = spec_registry["$(main_temporal_spec.var)"].Q_template,
                # Dynamically get the rho parameter for the main temporal effect.
                # This check ensures that a Type IV interaction is only attempted if the
                # temporal model is an AR1 process.
                t_rho = if $(main_temporal_spec.manifold_obj isa AR1)
                            $(Symbol("temporal_$(main_temporal_spec.var)_rho"))
                        else
                            # This value signals that a non-AR1 model is being used.
                            -999.0 
                        end

                st_inter = zeros(T, M.s_N, M.t_N)
                st_innov_matrix = reshape(st_raw, M.s_N, M.t_N)

                if "$(model_st)" == "I"
                    st_inter = st_innov_matrix .* st_sigma
                elseif "$(model_st)" == "II"
                    C_t = cholesky(Symmetric(t_Q + noise * I))
                    st_inter = (C_t.U \\ st_innov_matrix')' .* st_sigma
                elseif "$(model_st)" == "III"
                    C_s = cholesky(Symmetric(s_Q + noise * I))
                    st_inter = (C_s.U \\ st_innov_matrix) .* st_sigma
                elseif "$(model_st)" == "IV"
                    if t_rho == -999.0
                        @warn "Type IV interaction specified, but main temporal effect is not AR1. Defaulting to Type III."
                        C_s = cholesky(Symmetric(s_Q + noise * I))
                        st_inter = (C_s.U \\ st_innov_matrix) .* st_sigma
                    else
                        C_s = cholesky(Symmetric(s_Q + noise * I))
                        st_inter[:, 1] = (C_s.U \\ st_innov_matrix[:, 1]) ./ sqrt(1.0 - t_rho^2 + noise)
                        for t in 2:M.t_N; st_inter[:, t] = t_rho .* st_inter[:, t-1] .+ (C_s.U \\ st_innov_matrix[:, t]); end
                        st_inter .*= st_sigma
                    end
                end

                for i in 1:N; eta[i] += st_inter[M.s_idx[i], M.t_idx[i]]; end
            end
            """
        end
    end

    # --- Final Likelihood Evaluation Section ---
    final_likelihood = if is_multivariate
        """
        eta = eta_latent * L_corr.L
        for k in 1:K
            family_k = M.likelihood_specs[k][:family]
            for i in 1:N
                d_lik = bstm_Likelihood(family_k, [T(M.y_obs[i, k])]; sigma_y=[y_sigma[k]], trial=[Int(get(M.trials, (i,k), 1))])
                Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i, k])
            end
        end
        """
    else
        """
        family = M.likelihood_specs[1][:family]
        for i in 1:N
            d_lik = bstm_Likelihood(family, [T(M.y_obs[i])]; sigma_y=[y_sigma[i]], trial=[Int(get(M.trials, i, 1))])
            Turing.@addlogprob! Distributions.logpdf(d_lik, eta[i])
        end
        """
    end

    # --- Model String Assembly ---
    model_string = """
    @model function $(model_func_name)(M, spec_registry, ::Type{T}=Float64) where {T}
        noise = get(M, :noise, 1e-6)
        N = M.y_N
        K = $(outcomes_N)

        $(likelihood_section)

        # --- Manifold Priors ---
        $(join(priors_acc, "\n        "))

        # --- Linear Predictor ---
        $(eta_name) = $(eta_init)
        $(intercept_block)
        $(offset_block)
        $(fixed_effects_block)

        # --- Latent Field Updates ---
        $(join(updates_acc, "\n        "))

        $(st_interaction_block)

        # --- Likelihood ---
        $(final_likelihood)
    end
    """

    println(model_string)
    
    try
        return Meta.parse(model_string), spec_registry
    catch e
        println("BSTM Assembler Error: Failed to parse the generated model string.")
        rethrow(e)
    end
end

### BSTM Dynamic Model Execution Bridge v10.0.0
# Rationale: A unified entry point for compiling and instantiating any dynamically generated model.

function bstm_dynamic_model(config::NamedTuple)
    # 1. Generate model expression and technical registry
    expr, registry = bstm_text_assembler(config)
    
    # 2. Evaluate expression in Main scope to define the model function
    Base.invokelatest(eval, expr)
    
    # 3. Instantiate the Turing model object
    arch = get(config, :model_arch, "univariate")
    model_func_name = if arch == "multivariate"
        :bstm_text_generated_multivariate
    elseif arch == "multifidelity"
        :bstm_text_generated_multifidelity
    else
        :bstm_text_generated_univariate
    end
    model_func = getfield(Main, model_func_name)
    return Base.invokelatest(model_func, config, registry)
end

function householder_to_eigenvector(v_mat::AbstractMatrix{T}, nU, n_factors) where {T}
    U = Matrix{T}(I, nU, nU)

    for k in 1:n_factors
        vk = v_mat[:, k]
        norm_v = LinearAlgebra.norm(vk)
        
        if norm_v > 1e-9
            vk = vk / norm_v
            v_transpose_U = vk' * U
            U = U - 2.0 .* vk * v_transpose_U
        end
    end

    return U[:, 1:n_factors]
end
 
function recompose_precision_new(m_obj, M, sigma; noise=1e-6)
    # Placeholder for a new, text-based-model-compatible precision recomposition engine
    # for state-space models. A full implementation would be recursive and handle
    # various manifold types passed as `m_obj`. For now, it returns a simple identity
    # matrix, which is incorrect for structured state-space models.
    return Symmetric(I(M.s_N))
end




### Likelihood Family Trait System
abstract type AbstractBSTM_Family end
struct PoissonFamily <: AbstractBSTM_Family end
struct GaussianFamily <: AbstractBSTM_Family end
struct LogNormalFamily <: AbstractBSTM_Family end
struct NegativeBinomialFamily <: AbstractBSTM_Family end
struct BinomialFamily <: AbstractBSTM_Family end
struct GammaFamily <: AbstractBSTM_Family end
struct ExponentialFamily <: AbstractBSTM_Family end
struct BetaFamily <: AbstractBSTM_Family end
struct InverseGaussianFamily <: AbstractBSTM_Family end
struct StudentTFamily <: AbstractBSTM_Family end
struct HalfNormalFamily <: AbstractBSTM_Family end
struct HalfStudentTFamily <: AbstractBSTM_Family end
struct LaplaceFamily <: AbstractBSTM_Family end
struct ParetoFamily <: AbstractBSTM_Family end
struct DirichletFamily <: AbstractBSTM_Family end
struct InverseWishartFamily <: AbstractBSTM_Family end

const BSTM_FAMILY_REGISTRY = Dict{String, AbstractBSTM_Family}(
    "poisson" => PoissonFamily(),
    "gaussian" => GaussianFamily(),
    "lognormal" => LogNormalFamily(),
    "bernoulli" => BinomialFamily(),
    "binomial" => BinomialFamily(),
    "negbin" => NegativeBinomialFamily(),
    "gamma" => GammaFamily(),
    "exponential" => ExponentialFamily(),
    "beta" => BetaFamily(),
    "inverse_gaussian" => InverseGaussianFamily(),
    "student_t" => StudentTFamily(),
    "half_normal" => HalfNormalFamily(),
    "half_student_t" => HalfStudentTFamily(),
    "laplace" => LaplaceFamily(),
    "pareto" => ParetoFamily(),
    "dirichlet" => DirichletFamily(),
    "inverse_wishart" => InverseWishartFamily()
)

abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end

abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end

function get_model_family(model_family::String)
    family_key = lowercase(model_family)
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$model_family'. Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end

function get_dist_ref(::PoissonFamily, d, eta, sig); return Poisson(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::GaussianFamily, d, eta, sig); return Normal(eta, max(sig, 1e-9)); end
function get_dist_ref(::LogNormalFamily, d, eta, sig); return LogNormal(eta, max(sig, 1e-9)); end
function get_dist_ref(::NegativeBinomialFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); return NegativeBinomial(d.r_nb, d.r_nb/(d.r_nb + mu)); end
function get_dist_ref(::BinomialFamily, d, eta, sig); n = d.trial isa AbstractVector ? d.trial[1] : d.trial; return Binomial(Int(n), logistic(eta)); end
function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, clamp(exp(eta), 1e-9, 1e9)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(clamp(mu*phi, 1e-9, Inf), clamp((1.0-mu)*phi, 1e-9, Inf)); end
function get_dist_ref(::InverseGaussianFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return InverseGaussian(mu, lambda); end
function get_dist_ref(::StudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return LocationScale(eta, max(sig, 1e-9), TDist(nu)); end
function get_dist_ref(::HalfNormalFamily, d, eta, sig); return truncated(Normal(0.0, max(sig, 1e-9)), 0.0, Inf); end
function get_dist_ref(::HalfStudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return truncated(LocationScale(0.0, max(sig, 1e-9), TDist(nu)), 0.0, Inf); end
function get_dist_ref(::LaplaceFamily, d, eta, sig); return Laplace(eta, max(sig, 1e-9)); end
function get_dist_ref(::ParetoFamily, d, eta, sig)
    shape = d.extra_params isa Number && d.extra_params > 1.0 ? d.extra_params : 1.1
    mean_val = clamp(exp(eta), 1e-9, 1e9)
    scale = mean_val * (shape - 1.0) / shape
    return Pareto(shape, scale)
end

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end
function is_discrete_family(::AbstractBSTM_Family)
    return false
end

function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zi::AbstractZIState, d, eta, sig, y)
    dist = get_dist_ref(fam, d, eta, sig)
    lp_branch = logpdf(dist, y)
    lp_final = lp_branch
    if zi isa ZeroInflated
        log_phi = log(d.phi_zi + 1e-15)
        log_one_minus_phi = log(1.0 - d.phi_zi + 1e-15)
        if y == 0.0
            if is_discrete_family(fam)
                p_base_zero = pdf(dist, 0.0)
                lp_final = logsumexp(log_phi, log_one_minus_phi + log(p_base_zero + 1e-15))
            else
                lp_final = log_phi
            end
        else
            lp_final = log_one_minus_phi + lp_branch
        end
    end
    if d.hurdle > -Inf
        lp_final = lp_final - logccdf(dist, d.hurdle)
    end
    return lp_final
end

### Likelihood Factory v1.2.9
# Timestamp: 2026-07-17 12:45:00
# Rationale: Resolving MethodError by implementing consistent scalar and vector logpdf overloads.

struct bstm_Likelihood{F, Z, C, W, P, R, S, T, TR, TL, TU, HT, EX} <: ContinuousMultivariateDistribution
    family::F
    y_obs::TR
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    r_nb::R
    sigma_y::S
    trial::T
    y_L::TL
    y_U::TU
    hurdle::HT
    extra_params::EX
end

# Keyword-aware constructor
function bstm_Likelihood(family_input::Union{String, Symbol}, y_obs;
    zi_state=nothing, censoring_state=nothing, weight=nothing,
    phi_zi=-Inf, r_nb=1.0, sigma_y=1.0, trial=1,
    y_L=-Inf, y_U=Inf, hurdle=nothing, extra_params=nothing
)
    f_trait = get_model_family(string(family_input)) # This was the previous fix.
    # For simplicity in this fix, we assume Uncensored and NonZeroInflated.
    # A more robust implementation would parse these from the likelihood spec.
    zi_trait = phi_zi > -Inf ? ZeroInflated() : NonZeroInflated()
    censor_trait = Uncensored()
    y_vec = y_obs isa AbstractVector ? y_obs : [y_obs]
    return bstm_Likelihood(f_trait, y_vec, zi_trait, censor_trait, weight, phi_zi, r_nb, sigma_y, trial, y_L, y_U, hurdle, extra_params)
end

# Required interface methods
Base.length(d::bstm_Likelihood) = length(d.y_obs)
Base.size(d::bstm_Likelihood) = (length(d.y_obs),)

# Internal logpdf implementation for vector-based observations
function Distributions._logpdf(d::bstm_Likelihood, eta::AbstractVector{<:Real})
    logp = 0.0
    for i in 1:length(eta)
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[i] : d.weight
        logp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
    end
    return logp
end

# Public scalar overload
function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end


# Public vector overload to maintain MultivariateDistribution compliance
function Distributions.logpdf(d::bstm_Likelihood, y::AbstractVector{<:Real})
    return Distributions._logpdf(d, y)
end
