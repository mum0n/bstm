"""
    RW1 <: ComponentModel

A component for a first-order random walk (RW1) model. This model assumes that the
current value of a latent temporal field is the previous value plus a random
innovation. It is an intrinsic Gaussian Markov Random Field (GMRF) with a rank
deficiency of 1, implying a sum-to-zero constraint for identifiability.

# Version
v2.1.0 (2026-08-17)

# Mathematical Summary
The RW1 model defines a latent temporal field \$\\phi\$ where the value at time \$t\$ is
conditionally dependent on its immediate predecessor:
\$\\phi_t | \\phi_{t-1} \\sim \\mathcal{N}(\\phi_{t-1}, \\sigma^2)\$
This can be written as \$\\phi_t - \\phi_{t-1} = \\epsilon_t\$, where
\$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$.

The joint precision matrix \$\\mathbf{Q}\$ for this process is singular (rank-deficient),
making it an "intrinsic" GMRF. To ensure the model is identifiable from a global
intercept, a sum-to-zero constraint (\$\\sum_i \\phi_i = 0\$) is imposed on the
latent field.

# Computational Methods
- `:statespace` (Default, AD-friendly): The most efficient method, constructing the RW1 process
  via a cumulative sum of innovations.
- `:spectral` (AD-friendly): An efficient method using spectral decomposition of the
  precision matrix.
- `:cholesky` (AD-friendly): A didactic alternative using dense Cholesky factorization.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A didactic method using sparse Cholesky
  factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A temporal index variable (e.g., `year`) passed to `random()`.
- **Optional (in `random()` call)**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    innovations. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:statespace`, `:spectral`, `:cholesky`,
    or `:cholesky_sparse`). Default: `:statespace`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the innovations.
- `innovations_<key>`: The raw standard normal innovations for the latent field.
- `latent_<key>`: The reconstructed latent temporal field.

# Key References
- Rue, H., & Held, L. (2005). *Gaussian Markov Random Fields: Theory and
  Applications*. CRC Press.
- Wikipedia: Random walk
"""
struct RW1 <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rw1] = RW1

COMPONENT_CONSTRUCTORS[:rw1] = (p, params) -> RW1(
    p.sigma, get(params, :method, :statespace)
)

MODEL_TO_STRUCTURE_MAP[:rw1] = :temporal

function get_precomputes(m::RW1, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation
    variables = mod_data[:variables]
    if isempty(variables)
        error("The RW1 model requires a time index variable, e.g., `random(year, model=:rw1)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M.data, time_var_sym)
        error("Time index variable ':$time_var_sym' for RW1 model not found in data.")
    end

    t_N = get(M, :t_N, nothing)
    if isnothing(t_N)
        error(
            "RW1 component '$(mod_data[:key])' failed: t_N not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end
    
    if t_N == 0
        @warn "Number of time steps for RW1 component '$(mod_data[:key])' is zero. " *
              "The component will have no effect."
    end

    # build_structure_template returns CPU arrays
    template = build_structure_template(:rw1, t_N)
    
    # All arrays remain on the CPU.
    Q_template_cpu = template.matrix
    U_cpu = template.U
    L_cpu = template.L
    
    # Pre-compute the dense Cholesky factor for the :cholesky method on the CPU
    F_cpu = cholesky(Symmetric(Matrix(Q_template_cpu) + M.noise * I))
    
    return (
        Q_template=Q_template_cpu,
        U=U_cpu,
        L=L_cpu,
        scaling_factor=template.scaling_factor,
        n_latent=t_N,
        cholesky_factor=F_cpu
    )
end

function get_priors(
    m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    sigma_prior_str = _distribution_to_string(m.sigma)
    
    return """
        $(p_names.sigma) ~ $(sigma_prior_str)
        $(p_names.innovations) ~ MvNormal(
            zeros(T, spec_registry[:$(key)].hyper.n_latent), I
        )
    """
end

function get_updates(
    m::RW1, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    statespace_code = """
        # --- RW1 Component: $(key) (State-Space Method) ---
        let
            innovations = $(p_names.innovations)
            latent_field_raw = cumsum(innovations)
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * spec_registry[:$(key)].hyper.n_latent), 
                sum(latent_field_raw)
            )
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    spectral_code = """
        # --- RW1 Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[1] = 0.0
            $(p_names.latent) = hyper.U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    cholesky_code = """
        # --- RW1 Component: $(key) (Cholesky Method, AD-Safe) ---
        let
            F = spec_registry[:$(key)].hyper.cholesky_factor
            latent_field_raw = F.L' \\ $(p_names.innovations)
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * spec_registry[:$(key)].hyper.n_latent), 
                sum(latent_field_raw)
            )
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    cholesky_sparse_code = """
        # --- RW1 Component: $(key) (Sparse Cholesky, Not AD-Safe) ---
        let
            Q = spec_registry[:$(key)].hyper.Q_template
            F = cholesky(Symmetric(Q + M.noise * I))
            latent_field_raw = F.L' \\ $(p_names.innovations)
            Turing.@addlogprob! logpdf(
                Normal(0.0, 0.001 * spec_registry[:$(key)].hyper.n_latent), 
                sum(latent_field_raw)
            )
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    if m.method == :statespace; return statespace_code;
    elseif m.method == :spectral; return spectral_code;
    elseif m.method == :cholesky; return cholesky_code;
    elseif m.method == :cholesky_sparse; return cholesky_sparse_code;
    else; error("Unsupported method '$(m.method)' for RW1. Use :statespace, :spectral, :cholesky, or :cholesky_sparse."); end
end


"""
    get_effects(m::RW1, chain, spec, M, PS)

Reconstructs the RW1 effect from posterior samples. This version is CPU-only.
"""
function get_effects(
    m::RW1, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = size(chain, 1) * FlexiChains.nchains(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    noise = M.noise

    # --- Get precomputed data ---
    hyper = spec.hyper
    n_latent_train = hyper.n_latent

    # --- Index Handling: Combine training and prediction sets ---
    t_idx_train_cpu = M.t_idx
    t_idx_full_cpu = if !isnothing(PS) && haskey(PS.data, :t_idx)
        vcat(t_idx_train_cpu, PS.data.t_idx)
    else
        t_idx_train_cpu
    end
    t_N_full = isempty(t_idx_full_cpu) ? 0 : maximum(t_idx_full_cpu)
    N_total = length(t_idx_full_cpu)

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
        innovations_name = _find_parameter(p_names, string(v.innovations), k, is_multivariate_model)

        if isempty(sigma_name) || isempty(innovations_name)
            @warn "Parameters for RW1 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        innovations_samples_cpu = get_params_matrix(chain, innovations_name, n_latent_train)

        # Initialize output matrix for the full latent field on the CPU
        effect_k_latent_cpu = zeros(Float64, t_N_full, n_samples)
        
        # --- Vectorized Reconstruction on CPU ---
        local latent_field_train_cpu
        if m.method == :statespace
            latent_field_raw_cpu = cumsum(innovations_samples_cpu', dims=1)
            latent_field_centered_cpu = latent_field_raw_cpu .- mean(latent_field_raw_cpu, dims=1)
            latent_field_train_cpu = latent_field_centered_cpu .* sigma_samples_cpu'
        elseif m.method == :spectral
            U_cpu = hyper.U
            L_cpu = hyper.L
            diag_D = (sigma_samples_cpu' ./ sqrt.(L_cpu .+ noise))
            diag_D[1, :] .= 0.0 # Enforce sum-to-zero constraint for all samples
            latent_field_train_cpu = U_cpu * (diag_D .* innovations_samples_cpu')
        else # :cholesky or :cholesky_sparse
            F_cpu = hyper.cholesky_factor
            latent_field_raw_cpu = F_cpu.L' \ innovations_samples_cpu'
            latent_field_centered_cpu = latent_field_raw_cpu .- mean(latent_field_raw_cpu, dims=1)
            latent_field_train_cpu = latent_field_centered_cpu .* sigma_samples_cpu'
        end
        effect_k_latent_cpu[1:n_latent_train, :] = latent_field_train_cpu

        # Forecasting step (vectorized over samples)
        if t_N_full > n_latent_train
            innov_pred_cpu = randn(Float32, t_N_full - n_latent_train, n_samples)
            
            for t in (n_latent_train + 1):t_N_full
                innov_t = view(innov_pred_cpu, t - n_latent_train, :)
                effect_k_latent_cpu[t, :] = effect_k_latent_cpu[t-1, :] .+ innov_t' .* sigma_samples_cpu'
            end
        end

        # Indexing on the CPU
        indexed_effects_cpu = effect_k_latent_cpu[t_idx_full_cpu, :]
        push!(structured_effects, indexed_effects_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 