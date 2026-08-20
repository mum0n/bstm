"""
    SVAR <: ComponentModel

A component for a Spatially Varying Autoregressive (SVAR) process. This component
models a temporal AR(1) process where the autoregressive coefficient `rho` varies
across spatial units. The spatial variation of `rho` is governed by a specified
GMRF model (e.g., ICAR, Leroux).

# Version
v2.0.2 (2026-08-19)

# Mathematical Summary
The SVAR model defines a spatiotemporal process \$\\psi_{it}\$ for spatial unit \$i\$
at time \$t\$ as a first-order autoregressive process with a spatially varying
persistence parameter \$\\rho_i\$:

\$\\psi_{it} = \\rho_i \\psi_{i,t-1} + \\epsilon_{it}, \\quad \\epsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2)\$

The spatially varying coefficient \$\\boldsymbol{\\rho} = (\\rho_1, \\dots, \\rho_{s_N})\$
is itself modeled as a latent Gaussian field, typically with a GMRF prior to
encourage spatial smoothness:

\$\\boldsymbol{\\rho}_{field} \\sim \\mathcal{N}(\\mathbf{0}, (\\tau_{\\rho} \\mathbf{Q}_{\\rho})^{-1})\$

where \$\\mathbf{Q}_{\\rho}\$ is the precision matrix of a spatial model (e.g., ICAR).
To ensure stationarity of the AR(1) process (\$-1 < \\rho_i < 1\$), the raw field
is transformed using the `tanh` function: \$\\rho_i = \\tanh(\\rho_{field, i})\$.

# Computational Methods (for the `rho` field)
- `:spectral` (Default, AD-friendly): Regularizes coefficients using a spectral
  decomposition of the precision matrix. Recommended for NUTS.
- `:cholesky` (AD-friendly): Uses a dense Cholesky factorization of the precision
  matrix.
- `:cholesky_sparse` (Didactic, Not AD-friendly): Uses sparse Cholesky factorization,
  which is not compatible with most AD backends.

# Inputs
- **Required**:
  - A spatial index variable (e.g., `region`) passed to `random()`.
  - A temporal index variable (e.g., `year`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm`.
- **Optional (in `random()` call)**:
  - `model`: `Symbol`, the GMRF model for the `rho` field (e.g., `:icar`, `:leroux`). Default: `:icar`.
  - `rho_sigma`: `UnivariateDistribution`, prior for the std. dev. of the `rho` field. Default: `Exponential(1.0)`.
  - `rho_rho`: `UnivariateDistribution`, prior for the mixing parameter of the `rho` field's model (if applicable). Default: `Beta(1,1)`.
  - `sigma`: `UnivariateDistribution`, prior for the std. dev. of the AR(1) innovations. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method for the `rho` field. Default: `:spectral`.

# Outputs (Parameter Names)
- `rho_sigma_<key>`: Standard deviation of the spatial `rho` field.
- `rho_rho_<key>`: Mixing parameter of the spatial `rho` field (if applicable).
- `sigma_<key>`: Standard deviation of the AR(1) innovations.
- `rho_innovations_<key>`: Raw innovations for the spatial `rho` field.
- `innovations_<key>`: Raw innovations for the temporal AR(1) processes.
- `latent_<key>`: The reconstructed spatiotemporal SVAR effect.
"""
struct SVAR <: ComponentModel
    rho_model_type::Symbol
    rho_sigma::UnivariateDistribution
    rho_rho::Union{UnivariateDistribution, Nothing}
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:svar] = SVAR

COMPONENT_CONSTRUCTORS[:svar] = (p, params) -> SVAR(
    get(params, :model, :icar),
    p.rho_sigma,
    get(p, :rho_rho, nothing),
    p.sigma,
    get(params, :method, :spectral)
)

MODEL_TO_STRUCTURE_MAP[:svar] = :spacetime

function get_precomputes(m::SVAR, M::NamedTuple, mod_data::Dict)::NamedTuple
    # Validation moved from get_datastructures!
    variables = mod_data[:variables]
    if length(variables) < 2
        error("SVAR requires a spatial and a temporal variable, e.g., `random(s, t, model=svar)`.")
    end

    # The processor is now responsible for creating s_idx, t_idx, s_N, t_N.
    # We just need to get them from the main config M.
    s_N = get(M, :s_N, nothing)
    t_N = get(M, :t_N, nothing)
    if isnothing(s_N) || isnothing(t_N)
        error(
            "SVAR component '$(mod_data[:key])' failed: s_N or t_N not found in model " *
            "configuration. This should have been set by the model processor."
        )
    end
    W = get(M, :W, nothing)
    if isnothing(W) && m.rho_model_type != :iid
        @warn "Adjacency matrix `W` not provided for SVAR model's rho field."
    end

    # This returns CPU arrays
    template = build_structure_template(m.rho_model_type, s_N; W=W)

    return (
        Q_rho_template=template.matrix,
        U_rho=template.U,
        L_rho=template.L,
        scaling_factor_rho=template.scaling_factor,
        n_latent_rho=s_N,
        n_latent_svar=s_N * t_N
    )
end


function get_priors(
    m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key

    priors_acc = String[]
    push!(priors_acc, "$(p_names.rho_sigma) ~ $(_distribution_to_string(m.rho_sigma))")
    if m.rho_model_type in [:leroux, :bym2] && !isnothing(m.rho_rho)
        push!(priors_acc, "$(p_names.rho_rho) ~ $(_distribution_to_string(m.rho_rho))")
    end
    push!(priors_acc, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    push!(priors_acc, "$(p_names.ure_rho) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent_rho), I)")
    push!(priors_acc, "$(p_names.ure) ~ MvNormal(zeros(T, spec_registry[:$(key)].hyper.n_latent_svar), I)")

    return join(priors_acc, "\n    ")
end

function get_updates(
    m::SVAR, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing},
    M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    local rho_recon_code
    if m.method == :spectral
        rho_recon_code = """
            local hyper_rho = spec_registry[:$(key)].hyper
            local D_rho = $(p_names.rho_sigma) ./ sqrt.(hyper_rho.L_rho .+ M.noise)
            if "$(string(m.rho_model_type))" in ["icar", "besag"]; D_rho[1] = 0.0; end
            local rho_field = hyper_rho.U_rho * (D_rho .* $(p_names.ure_rho))
        """
    else # :cholesky or :cholesky_sparse
        rho_recon_code = """
            local hyper_rho = spec_registry[:$(key)].hyper
            local Q_rho = hyper_rho.Q_rho_template
            local F_rho = cholesky(Symmetric(Matrix(Q_rho) + M.noise * I))
            local rho_field_unscaled = F_rho.L' \\ $(p_names.ure_rho)
            if "$(string(m.rho_model_type))" in ["icar", "besag"]
                Turing.@addlogprob! logpdf(Normal(0.0, 0.001 * hyper_rho.n_latent_rho), sum(rho_field_unscaled))
            end
            local rho_field = rho_field_unscaled .* $(p_names.rho_sigma)
        """
    end

    return """
        # --- SVAR Component: $(key) ($(m.method)) ---
        let
            # 1. Reconstruct the spatially-varying AR(1) coefficient `rho`.
            $(rho_recon_code)
            local rho_s = tanh.(rho_field) # Constrain rho to (-1, 1)

            # 2. Evolve the SVAR state-space.
            local latent_st = zeros(eltype(rho_s), M.s_N, M.t_N)
            local innovations_grid = reshape($(p_names.ure), M.s_N, M.t_N)
            
            latent_st[:, 1] = ($(p_names.sigma) ./ sqrt.(1 .- rho_s.^2 .+ M.noise)) .* innovations_grid[:, 1]
            for t in 2:M.t_N
                latent_st[:, t] = rho_s .* latent_st[:, t-1] .+ $(p_names.sigma) .* innovations_grid[:, t]
            end
            
            # 3. Map the 2D latent field back to the 1D observation vector.
            $(p_names.sre) = [latent_st[M.s_idx[i], M.t_idx[i]] for i in 1:M.y_N]

            # 4. Add the effect to the linear predictor.
            $(eta_target) .+= $(p_names.sre)
        end
    """
end


function get_effects(
    m::SVAR, chain, spec::NamedTuple, M::NamedTuple,
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
    
    # --- Get precomputed data ---
    hyper = spec.hyper
    s_N, t_N, noise = M.s_N, M.t_N, M.noise
    n_latent_svar = s_N * t_N
    Q_rho_template_cpu = hyper.Q_rho_template
    U_rho_cpu = hyper.U_rho
    L_rho_cpu = hyper.L_rho

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_cpu = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS.data, :s_idx, []))
    t_idx_cpu = isnothing(PS) ? M.t_idx : vcat(M.t_idx, get(PS.data, :t_idx, []))
    N_total = length(s_idx_cpu)
    t_N_full = isempty(t_idx_cpu) ? 0 : maximum(t_idx_cpu)
    
    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        
        rho_sigma_name = _find_parameter(p_names, string(v.rho_sigma), k, is_multivariate_model)
        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
        ure_rho_name = _find_parameter(p_names, string(v.ure_rho), k, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(v.ure), k, is_multivariate_model)
        
        if isempty(rho_sigma_name) || isempty(sigma_name) || isempty(ure_rho_name) || isempty(ure_name)
            @warn "Parameters for SVAR component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (these are on the CPU)
        rho_sigma_samples_cpu = get_params_vector(chain, rho_sigma_name, 1)[:, 1]
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        rho_innovations_samples_cpu = get_params_matrix(chain, ure_rho_name, s_N)
        innovations_samples_cpu = get_params_matrix(chain, ure_name, n_latent_svar)
        
        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, N_total, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        for s in 1:n_samples
            rho_sigma_s = rho_sigma_samples_cpu[s]
            sigma_s = sigma_samples_cpu[s]
            rho_innovations_s = rho_innovations_samples_cpu[s, :]
            innovations_s = innovations_samples_cpu[s, :]

            local rho_field_s
            if m.method == :spectral
                D_rho_s = rho_sigma_s ./ sqrt.(L_rho_cpu .+ noise)
                if m.rho_model_type in [:icar, :besag]; D_rho_s[1] = 0.0; end
                rho_field_s = U_rho_cpu * (D_rho_s .* rho_innovations_s)
            else # :cholesky or :cholesky_sparse
                F_rho_cpu = cholesky(Symmetric(Matrix(Q_rho_template_cpu) + noise * I))
                rho_field_unscaled = F_rho_cpu.L' \ rho_innovations_s
                if m.rho_model_type in [:icar, :besag]; rho_field_unscaled .-= mean(rho_field_unscaled); end
                rho_field_s = rho_field_unscaled .* rho_sigma_s
            end
            rho_s_s = tanh.(rho_field_s)

            latent_st_s = zeros(Float64, s_N, t_N_full)
            innovations_grid_train = reshape(innovations_s, s_N, t_N)
            innovations_grid_full = if t_N_full > t_N
                hcat(innovations_grid_train, randn(Float32, s_N, t_N_full - t_N))
            else
                innovations_grid_train[:, 1:t_N_full]
            end
            
            denom = sqrt.(max.(0, 1 .- rho_s_s.^2))
            latent_st_s[:, 1] = (sigma_s ./ (denom .+ noise)) .* innovations_grid_full[:, 1]
            for t in 2:t_N_full
                latent_st_s[:, t] = rho_s_s .* latent_st_s[:, t-1] .+ sigma_s .* innovations_grid_full[:, t]
            end
            
            linear_indices = s_idx_cpu .+ (t_idx_cpu .- 1) .* s_N
            effect_k_cpu[:, s] = vec(latent_st_s)[linear_indices]
        end
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
