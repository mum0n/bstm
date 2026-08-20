"""
    RW2 <: ComponentModel

A component for a second-order random walk (RW2) model. This model assumes that the
second differences of a latent temporal field follow a random innovation. It is an
intrinsic Gaussian Markov Random Field (GMRF) with a rank deficiency of 2, implying
two sum-to-zero constraints for identifiability. It produces a smoother field than
an RW1 model.

# Version
v2.1.1 (2026-08-19)

# Mathematical Summary
The RW2 model defines a latent temporal field \$\\phi\$ where the value at time \$t\$ is
a linear extrapolation from its two immediate predecessors, plus a random innovation:
\$\\phi_t | \\phi_{t-1}, \\phi_{t-2} \\sim \\mathcal{N}(2\\phi_{t-1} - \\phi_{t-2}, \\sigma^2)\$
This can be written as \$\\phi_t - 2\\phi_{t-1} + \\phi_{t-2} = \\epsilon_t\$, where
\$\\epsilon_t \\sim \\mathcal{N}(0, \\sigma^2)\$.

The joint precision matrix \$\\mathbf{Q}\$ for this process is singular (rank-deficient),
making it an "intrinsic" GMRF. To ensure the model is identifiable from a global
intercept and linear trend, two sum-to-zero constraints are imposed on the latent
field.

# Computational Methods
- `:statespace` (Default, AD-friendly): The most efficient method, constructing the RW2 process
  via a state-space recurrence relation.
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
struct RW2 <: ComponentModel
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:rw2] = RW2

COMPONENT_CONSTRUCTORS[:rw2] = (p, params) -> RW2(
    p.sigma, get(params, :method, :statespace)
)

MODEL_TO_STRUCTURE_MAP[:rw2] = :temporal

function get_precomputes(m::RW2, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Data validation
    variables = mod_data[:variables]
    if isempty(variables)
        error("The RW2 model requires a time index variable, e.g., `random(year, model=:rw2)`.")
    end

    time_var_sym = Symbol(variables[1])
    if !hasproperty(M.data, time_var_sym)
        error("Time index variable ':$time_var_sym' for RW2 model not found in data.")
    end

    t_N = get(M, :t_N, nothing)
    if isnothing(t_N)
        error(
            "RW2 component '$(mod_data[:key])' failed: t_N not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end
    
    if t_N == 0
        @warn "Number of time steps for RW2 component '$(mod_data[:key])' is zero. " *
              "The component will have no effect."
    end

    # build_structure_template returns CPU arrays
    template = build_structure_template(:rw2, t_N)
    
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
    m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
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
    m::RW2, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    statespace_code = """
        # --- RW2 Component: $(key) (State-Space Method) ---
        let
            innovations = $(p_names.innovations)
            T_num = eltype(innovations)
            n_latent = spec_registry[:$(key)].hyper.n_latent
            latent_field_raw = similar(innovations, T_num, n_latent)
            
            if n_latent > 0; latent_field_raw[1] = innovations[1]; end
            if n_latent > 1; latent_field_raw[2] = 2 * latent_field_raw[1] + innovations[2]; end
            for t in 3:n_latent
                latent_field_raw[t] = 2 * latent_field_raw[t-1] - latent_field_raw[t-2] + innovations[t]
            end
            if n_latent > 0
                Turing.@addlogprob! logpdf(
                    Normal(0.0, 0.001 * n_latent), sum(latent_field_raw)
                )
            end
            
            $(p_names.latent) = latent_field_raw .* $(p_names.sigma)
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    spectral_code = """
        # --- RW2 Component: $(key) (Spectral Method) ---
        let
            hyper = spec_registry[:$(key)].hyper
            diag_D = $(p_names.sigma) ./ sqrt.(hyper.L .+ M.noise)
            diag_D[1] = 0.0; diag_D[2] = 0.0
            $(p_names.latent) = hyper.U * (diag_D .* $(p_names.innovations))
            $(eta_target) .+= view($(p_names.latent), M.t_idx)
        end
    """

    cholesky_code = """
        # --- RW2 Component: $(key) (Cholesky Method, AD-Safe) ---
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
        # --- RW2 Component: $(key) (Sparse Cholesky, Not AD-Safe) ---
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
    else; error("Unsupported method '$(m.method)' for RW2. Use :statespace, :spectral, :cholesky, or :cholesky_sparse."); end
end

"""
    get_effects(m::RW2, chain, spec::NamedTuple, M::NamedTuple, PS)

Reconstructs the RW2 effect from posterior samples. This version is CPU-only and
vectorized for efficiency.
"""
function get_effects(
    m::RW2, chain, spec::NamedTuple, M::NamedTuple,
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
            @warn "Parameters for RW2 component $(spec.key) (outcome $k) not found. Returning zero-matrix."
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
            innovations_T = innovations_samples_cpu' # [n_latent_train, n_samples]
            latent_field_raw_cpu = similar(innovations_T)
            if n_latent_train > 0; latent_field_raw_cpu[1, :] = innovations_T[1, :]; end
            if n_latent_train > 1; latent_field_raw_cpu[2, :] = 2 .* latent_field_raw_cpu[1, :] .+ innovations_T[2, :]; end
            for t in 3:n_latent_train
                latent_field_raw_cpu[t, :] = 2 .* latent_field_raw_cpu[t-1, :] .- latent_field_raw_cpu[t-2, :] .+ innovations_T[t, :]
            end
            latent_field_centered_cpu = latent_field_raw_cpu .- mean(latent_field_raw_cpu, dims=1)
            latent_field_train_cpu = latent_field_centered_cpu .* sigma_samples_cpu'
        elseif m.method == :spectral
            U_cpu = hyper.U
            L_cpu = hyper.L
            diag_D = (sigma_samples_cpu' ./ sqrt.(L_cpu .+ noise))
            diag_D[1, :] .= 0.0; diag_D[2, :] .= 0.0 # Enforce sum-to-zero constraints
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
                effect_k_latent_cpu[t, :] = 2 .* effect_k_latent_cpu[t-1, :] .- effect_k_latent_cpu[t-2, :] .+ innov_t' .* sigma_samples_cpu'
            end
        end

        # Indexing on the CPU
        indexed_effects_cpu = effect_k_latent_cpu[t_idx_full_cpu, :]
        push!(structured_effects, indexed_effects_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 