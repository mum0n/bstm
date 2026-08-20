"""
    SparseGP <: ComponentModel

A component for sparse Gaussian Process approximations, supporting FITC (Fully
Independent Training Conditional), VFE (Variational Free Energy), and PIC
(Partially Independent Conditional) methods. It approximates a full GP using a
small set of `n_inducing` points to make it scalable for larger datasets.

# Version
v1.3.2 (2026-08-19)

# Mathematical Summary
All methods approximate a full GP posterior by introducing a set of \$M\$ inducing
points, \$\\mathbf{Z}\$. The latent GP values \$f\$ are modeled as:
\$f \\sim \\mathcal{N}(\\mu_f, \\Sigma_f)\$
where the conditional mean is \$\\mu_f = K_{XZ} K_{ZZ}^{-1} u\$, with \$u \\sim \\mathcal{N}(0, K_{ZZ})\$.

The methods differ in their covariance approximation:
- **`:fitc` (default)**: Includes a diagonal correction to account for the variance
  of data points not captured by the inducing points.
  \$\\Sigma_f = \\text{diag}(K_{XX} - Q_{XX}) + \\sigma_n^2 I\$, where \$Q_{XX} = K_{XZ} K_{ZZ}^{-1} K_{ZX}\$.
- **`:vfe` (didactic)**: A pure low-rank approximation, equivalent to DTC.
  \$\\Sigma_f = Q_{XX}\$. This is simpler but can underestimate variance.
- **`:pic`**: The Partially Independent Conditional approximation, which uses a
  block-diagonal correction for improved variance estimation. The correction term is
  a block-diagonal matrix where each block is \$K_{C_i, C_i} - Q_{C_i, C_i}\$ for
  cluster \$i\$.

# Computational Methods
- `:fitc` (Default, AD-friendly): The Fully Independent Training Conditional approximation.
  It is generally preferred for its more accurate variance estimates.
- `:vfe` (Didactic, AD-friendly): The Variational Free Energy approximation, also known
  as DTC. It is a pure low-rank approximation that can be faster but may
  underestimate variance. Retained for didactic purposes.
- `:pic` (AD-friendly): The Partially Independent Conditional method, using
  block-diagonal corrections for better variance estimates.

# Inputs
- **Required**:
  - One or more coordinate variables (e.g., `x`, `y`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `n_inducing`: `Int`, the number of inducing points. Default: `20`.
  - `n_clusters`: `Int`, the number of clusters for the PIC method. Default: `10`.
  - `kernel`: `String`, the name of the kernel function (e.g., `"se"`, `"matern32"`). Default: `"se"`.
  - `sigma`: `UnivariateDistribution`, prior for the marginal standard deviation of the GP. Default: `Exponential(1.0)`.
  - `lengthscale`: `UnivariateDistribution` or `Vector{<:UnivariateDistribution}`, prior for the kernel lengthscale(s). Default: `Gamma(2, 0.5)`.
  - `method`: `Symbol`, approximation method (`:fitc`, `:vfe`, or `:pic`). Default: `:fitc`.
  - `knot_method`: `Symbol`, method for placing inducing points (`:kmeans`, `:random`, `:quantile`, `:range`). Default: `:kmeans`.

# Outputs (Parameter Names)
- `sigma_<key>`: The marginal standard deviation of the GP.
- `ls_<key>`: The kernel lengthscale(s).
- `inducing_innovations_<key>`: Raw standard normal innovations for the inducing points.
- `diag_innovations_<key>`: Raw standard normal innovations for the diagonal correction (for `:fitc` method).
- `pic_innovations_<key>`: Raw standard normal innovations for the block correction (for `:pic` method).
- `latent_<key>`: The reconstructed latent GP effect.

# Key References
- Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using
  Pseudo-inputs*. In Advances in neural information processing systems, 18.
- Titsias, M. (2009). *Variational learning of inducing variables in sparse
  Gaussian processes*. In AISTATS.
- Vanhatalo, J., & Vehtari, A. (2007). *Sparse log-Gaussian process approximations
  for spatial epidemiology*. In Proceedings of the 7th International Workshop on
  Bayesian Inference in the Health Sciences.
"""
struct SparseGP <: ComponentModel
    lengthscale::Union{Distribution, Vector{<:Distribution}}
    sigma::Distribution
    n_inducing::Int
    kernel::String
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:svgp] = SparseGP
COMPONENT_TYPE_REGISTRY[:sparsegp] = SparseGP

COMPONENT_CONSTRUCTORS[:svgp] = (p, params) -> SparseGP(
    p.lengthscale,
    p.sigma,
    get(params, :n_inducing, 20),
    string(get(params, :kernel, "se")),
    get(params, :method, :fitc)
)
COMPONENT_CONSTRUCTORS[:sparsegp] = COMPONENT_CONSTRUCTORS[:svgp]

MODEL_TO_STRUCTURE_MAP[:svgp] = :smooth
MODEL_TO_STRUCTURE_MAP[:sparsegp] = :smooth

function get_precomputes(m::SparseGP, M::NamedTuple, mod_data::Dict)::NamedTuple
    variables = mod_data[:variables]
    params = mod_data[:params]

    if isempty(variables)
        error("The SparseGP model requires coordinate variables, e.g., `random(x, y, model=:svgp)`.")
    end

    for var_sym in variables
        if !hasproperty(M.data, Symbol(var_sym))
            error("Coordinate variable ':$var_sym' for SparseGP model not found in data.")
        end
    end

    # All computations are performed on the CPU.
    coords_cpu = Matrix{Float64}(M.data[!, Symbol.(variables)])
    knot_method = get(params, :knot_method, :kmeans)
    Z_inducing_cpu = generate_inducing_points(coords_cpu, m.n_inducing; method=string(knot_method))

    precomputes = Dict{Symbol, Any}(
        :coords => coords_cpu,
        :Z_inducing => Z_inducing_cpu,
        :n_latent => size(coords_cpu, 1)
    )

    # If using PIC, perform k-means clustering on CPU.
    if m.method == :pic
        n_clusters = get(params, :n_clusters, 10)
        if n_clusters > precomputes[:n_latent]
            n_clusters = precomputes[:n_latent]
            @warn "n_clusters ($(params[:n_clusters])) is greater than the number of data points. Setting n_clusters to $(n_clusters)."
        end
        kmeans_res = kmeans(coords_cpu', n_clusters)
        precomputes[:cluster_assignments] = assignments(kmeans_res)
        precomputes[:n_clusters] = nclusters(kmeans_res)
    end

    return NamedTuple(precomputes)
end
 
function get_priors(
    m::SparseGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    priors = String[]
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    if m.lengthscale isa Vector
        ls_priors_str = join([_distribution_to_string(p) for p in m.lengthscale], ", ")
        push!(priors, "$(p_names.ls) ~ Product([$(ls_priors_str)])")
    else
        ls_prior_str = _distribution_to_string(m.lengthscale)
        push!(priors, "$(p_names.ls) ~ $(ls_prior_str)")
    end
    
    push!(priors, "$(p_names.inducing_innovations) ~ MvNormal(zeros(T, $(m.n_inducing)), I)")
    
    if m.method == :fitc
        push!(
            priors,
            "$(p_names.diag_innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)"
        )
    elseif m.method == :pic
        # For PIC, innovations are for the entire latent field, then partitioned by block.
        push!(
            priors,
            "$(p_names.pic_innovations) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent), I)"
        )
    end

    return join(priors, "\n    ")
end

function get_updates(
    m::SparseGP, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    common_code = """
        local hyper = spec_registry[:$(key)].hyper
        local X_coords = hyper.coords
        local Z_coords = hyper.Z_inducing
        local kernel_type = Symbol("$(m.kernel)")
        
        local K_UU = evaluate_kernel_matrix(
            Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
        )
        local K_XU = evaluate_cross_kernel_matrix(
            X_coords, Z_coords, $(p_names.sigma), $(p_names.ls), kernel_type
        )
        
        local L_UU = cholesky(Symmetric(K_UU)).L
        local u_latent = L_UU * $(p_names.inducing_innovations)
    """

    fitc_code = """
        # --- FITC Sparse GP Component: $(key) ---
        let
            $(common_code)
            
            local K_UU_inv_u = K_UU \\ u_latent
            local mean_f = K_XU * K_UU_inv_u
            
            local diag_K_XX = fill($(p_names.sigma)^2, hyper.n_latent)
            local tmp = (L_UU' \\ K_XU')'
            local diag_Q_ff = sum(tmp.^2, dims=2)
            local lambda_diag = diag_K_XX - vec(diag_Q_ff)
            
            $(p_names.latent) = mean_f .+
                sqrt.(max.(lambda_diag, 0.0) .+ M.noise) .* $(p_names.diag_innovations)
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    vfe_code = """
        # --- VFE/DTC Sparse GP Component: $(key) ---
        let
            $(common_code)
            
            local K_UU_inv_u = K_UU \\ u_latent
            $(p_names.latent) = K_XU * K_UU_inv_u
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    pic_code = """
        # --- PIC Sparse GP Component: $(key) ---
        let
            $(common_code)
            
            local K_UU_inv_u = K_UU \\ u_latent
            local mean_f = K_XU * K_UU_inv_u
            
            # Initialize latent field with the mean component
            $(p_names.latent) = deepcopy(mean_f)
            
            # Loop over each cluster to apply the block-diagonal correction
            for g in 1:hyper.n_clusters
                # Find indices for the current block
                local block_indices = findall(==(g), hyper.cluster_assignments)
                if isempty(block_indices)
                    continue
                end
                
                # Extract coordinates and cross-covariance for the block
                local X_coords_block = X_coords[block_indices, :]
                local K_XU_block = K_XU[block_indices, :]
                
                # Compute the exact kernel matrix for the block
                local K_block = evaluate_kernel_matrix(
                    X_coords_block, $(p_names.sigma), $(p_names.ls), kernel_type, M.noise
                )
                
                # Compute the low-rank approximation for the block
                local Q_block = K_XU_block * (K_UU \\ K_XU_block')
                
                # Compute the correction matrix and its Cholesky factor
                local C_block = K_block - Q_block
                local L_C_block = cholesky(Symmetric(C_block + I * M.noise)).L
                
                # Apply the correction to the latent field for this block
                $(p_names.latent)[block_indices] .+= L_C_block * $(p_names.pic_innovations)[block_indices]
            end
            
            $(eta_target) .+= $(p_names.latent)
        end
    """

    if m.method == :fitc
        return fitc_code
    elseif m.method == :vfe
        return vfe_code
    elseif m.method == :pic
        return pic_code
    else
        error("Unsupported method '$(m.method)' for SparseGP component.")
    end
end

function get_effects(
    m::SparseGP, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = size(chain, 1) * FlexiChains.nchains(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    # --- Get precomputed data ---
    hyper = spec.hyper
    coords_train_cpu = hyper.coords
    Z_inducing_cpu = hyper.Z_inducing
    n_obs_train = hyper.n_latent

    # --- Coordinate Handling: Combine training and prediction sets on CPU ---
    coord_vars = get(spec.params, :positional_args, [])
    coords_full_cpu = if !isnothing(PS) && all(hasproperty(PS.data, Symbol(v)) for v in coord_vars)
        coords_pred_cpu = Matrix{Float64}(PS.data[!, Symbol.(coord_vars)])
        vcat(coords_train_cpu, coords_pred_cpu)
    else
        coords_train_cpu
    end
    n_obs_full = size(coords_full_cpu, 1)

    kernel_type = Symbol(m.kernel)
    noise = M.noise
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(p_names_k.sigma), k, is_multivariate_model)
        ls_name = _find_parameter(p_names, string(p_names_k.ls), k, is_multivariate_model)
        inducing_innov_name = _find_parameter(p_names, string(p_names_k.inducing_innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(ls_name) || isempty(inducing_innov_name)
            @warn "Parameters for SparseGP component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, n_obs_full, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        ls_dim = m.lengthscale isa Vector ? length(m.lengthscale) : 1
        ls_samples_cpu = get_params_matrix(chain, ls_name, ls_dim)
        inducing_innov_samples_cpu = get_params_matrix(chain, inducing_innov_name, m.n_inducing)

        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, n_obs_full, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        for i in 1:n_samples
            current_sigma = sigma_samples_cpu[i]
            current_ls = ls_dim > 1 ? ls_samples_cpu[i, :] : ls_samples_cpu[i, 1]
            current_u_raw = inducing_innov_samples_cpu[i, :]
            
            # Kernel evaluation and Cholesky happen on the CPU
            K_UU = evaluate_kernel_matrix(Z_inducing_cpu, current_sigma, current_ls, kernel_type, noise)
            K_XU_full = evaluate_cross_kernel_matrix(coords_full_cpu, Z_inducing_cpu, current_sigma, current_ls, kernel_type)
            
            L_UU = cholesky(Symmetric(K_UU)).L
            u_latent = L_UU * current_u_raw
            K_UU_inv_u = K_UU \ u_latent
            mean_f = K_XU_full * K_UU_inv_u

            if m.method == :fitc
                diag_innov_name = _find_parameter(p_names, string(p_names_k.diag_innovations), k, is_multivariate_model)
                if isempty(diag_innov_name)
                    @warn "Diagonal innovations for FITC component $(spec.key) (outcome $k) not found. Using zero for correction."
                    effect_k_cpu[:, i] = mean_f
                    continue
                end
                
                diag_innov_samples_cpu = get_params_matrix(chain, diag_innov_name, n_obs_train)
                
                # Handle prediction innovations
                diag_innov_i_cpu = if n_obs_full > n_obs_train
                    innov_train_cpu = diag_innov_samples_cpu[i, :]
                    innov_pred_cpu = randn(Float32, n_obs_full - n_obs_train)
                    vcat(innov_train_cpu, innov_pred_cpu)
                else
                    diag_innov_samples_cpu[i, :]
                end

                diag_K_XX = fill(current_sigma^2, n_obs_full)
                tmp = (L_UU' \ K_XU_full')'
                diag_Q_ff = sum(tmp.^2, dims=2)
                lambda_diag = diag_K_XX - vec(diag_Q_ff)
                
                effect_k_cpu[:, i] = mean_f .+ sqrt.(max.(lambda_diag, 0.0) .+ noise) .* diag_innov_i_cpu
            elseif m.method == :pic
                effect_k_cpu[:, i] = mean_f
                pic_innov_name = _find_parameter(p_names, string(p_names_k.pic_innovations), k, is_multivariate_model)
                if isempty(pic_innov_name)
                    @warn "PIC innovations for component $(spec.key) (outcome $k) not found. Using mean-only prediction."
                    continue
                end
                pic_innov_samples_cpu = get_params_matrix(chain, pic_innov_name, n_obs_train)
                pic_innov_i_cpu = pic_innov_samples_cpu[i, :]

                K_XU_train = K_XU_full[1:n_obs_train, :]
                
                # Note: PIC correction is only applied to the training data points.
                # Prediction points only get the mean effect.
                for g in 1:hyper.n_clusters
                    block_indices = findall(==(g), hyper.cluster_assignments)
                    if isempty(block_indices)
                        continue
                    end
                    
                    coords_block = coords_train_cpu[block_indices, :]
                    K_XU_block = K_XU_train[block_indices, :]
                    
                    K_block = evaluate_kernel_matrix(coords_block, current_sigma, current_ls, kernel_type, noise)
                    Q_block = K_XU_block * (K_UU \ K_XU_block')
                    C_block = K_block - Q_block
                    L_C_block = cholesky(Symmetric(C_block + I * noise)).L
                    
                    block_correction = L_C_block * pic_innov_i_cpu[block_indices]
                    effect_k_cpu[block_indices, i] .+= block_correction
                end
            else # :vfe
                effect_k_cpu[:, i] = mean_f
            end
        end
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 