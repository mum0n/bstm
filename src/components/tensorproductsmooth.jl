"""
    TensorProductSmooth <: ComponentModel

A component for creating inseparable spatiotemporal effects by taking the tensor
product of marginal (e.g., spatial and temporal) components. This is typically
specified in the formula via the Kronecker product operator `⊗`.

# Version
v1.0.0

# Mathematical Summary
This component models an inseparable spatiotemporal random effect \$\\boldsymbol{\\delta}\$
as a zero-mean Gaussian Process with a separable covariance structure. Given a
spatial precision matrix \$\\mathbf{Q}_s\$ and a temporal precision matrix \$\\mathbf{Q}_t\$,
the joint spatiotemporal precision matrix is the Kronecker product of the marginals:

\$\\mathbf{Q}_{st} = \\mathbf{Q}_t \\otimes \\mathbf{Q}_s\$

This structure implies that the covariance function is separable, i.e.,
\$K_{st}((s_1, t_1), (s_2, t_2)) = K_s(s_1, s_2) \\cdot K_t(t_1, t_2)\$. This is the
basis for the Knorr-Held Type IV interaction model.

# Computational Methods
- `:cholesky` (Default, AD-friendly): An AD-safe method using dense Cholesky factorization
  of the marginal precision matrices. Recommended for gradient-based samplers.
- `:cholesky_sparse` (Didactic, Not AD-friendly): A more memory-efficient method using
  sparse Cholesky factorization, suitable for gradient-free samplers.

# Inputs
- **Required**:
  - A composition of two `random()` modules using the `⊗` operator, e.g.,
    `random(s_idx, model=icar) ⊗ random(year, model=ar1)`.
- **Optional**:
  - `sigma`: `UnivariateDistribution`, prior for the standard deviation of the
    interaction effect. Default: `Exponential(1.0)`.
  - `method`: `Symbol`, computational method (`:cholesky` or `:cholesky_sparse`).
    Default: `:cholesky`.

# Outputs (Parameter Names)
- `sigma_<key>`: The standard deviation of the interaction effect.
- `innovations_<key>`: The raw standard normal innovations for the interaction field.

# Key References
- Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
  in disease risk. *Statistical Methods in Medical Research*, 9(3), 205-220.
"""
struct TensorProductSmooth <: ComponentModel
    components::Vector{ComponentModel}
    sigma::Distribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:tensorproductsmooth] = TensorProductSmooth
COMPONENT_CONSTRUCTORS[:tensorproductsmooth] = (p, params) -> begin
    components = get(params, :components, error("TensorProductSmooth requires child components."))
    TensorProductSmooth(components, p.sigma, get(params, :method, :spectral))
end
COMPONENT_CONSTRUCTORS[:interaction] = COMPONENT_CONSTRUCTORS[:tensorproductsmooth]

MODEL_TO_STRUCTURE_MAP[:tensorproductsmooth] = :spacetime

function get_precomputes(m::TensorProductSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple
    if isempty(m.components) || length(m.components) != 2
        error("TensorProductSmooth requires exactly two child components.")
    end

    if !hasproperty(M, :st_idx)
        error("TensorProductSmooth requires spatiotemporal index `st_idx` to be pre-computed by the model processor.")
    end

    child_nodes = get(mod_data[:params], :components, [])
    child_specs_list = []
    n_latent_total = 1

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = m.components[i]
        
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args
        )
        
        precomputes = get_precomputes(child_component_obj, M, child_mod_data)
        n_latent_total *= precomputes.n_latent

        child_spec = (
            key = child_mod_data[:key],
            structure = get_component_structure(child_component_obj),
            var = join(child_mod_data[:variables], "_"),
            component_obj = child_component_obj,
            params = child_mod_data[:params],
            hyper = precomputes
        )
        push!(child_specs_list, child_spec)
    end

    return (
        child_specs = child_specs_list,
        n_latent = n_latent_total
    )
end

function get_priors(
    m::TensorProductSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    key = spec.key
    
    return """
    # Priors for Spatiotemporal Interaction: $(spec.key)
    $(p_names.sigma) ~ $(_distribution_to_string(m.sigma))
    $(p_names.ure) ~ MvNormal(
        zeros(T, spec_registry[:$(key)].hyper.n_latent), I
    )
    """
end

function get_updates(
    m::TensorProductSmooth, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key
    
    child_specs = spec.hyper.child_specs
    s_spec, t_spec = child_specs[1], child_specs[2]
    
    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    s_rho_val = hasproperty(s_spec.component_obj,
        :rho) ? string(generate_full_variable_names(s_spec, arch, outcome_idx).rho) : "nothing"
    t_rho_val = hasproperty(t_spec.component_obj,
        :rho) ? string(generate_full_variable_names(t_spec, arch, outcome_idx).rho) : "nothing"

    s_N = s_spec.hyper.n_latent
    t_N = t_spec.hyper.n_latent

    spectral_code = """
        # --- Spatiotemporal Interaction (Spectral, Fast AD-Safe): $(spec.key) ---
        let
            local parent_hyper = spec_registry[:$(key)].hyper
            local s_hyper = parent_hyper.child_specs[1].hyper
            local t_hyper = parent_hyper.child_specs[2].hyper
            
            local diag_Ls = $(s_rho_val == "nothing" ? "s_hyper.L" : "(1.0 .- $(s_rho_val))
              .+ $(s_rho_val) .* s_hyper.L")
            local diag_Lt = $(t_rho_val == "nothing" ? "t_hyper.L" : "(1.0 .- $(t_rho_val))
              .+ $(t_rho_val) .* t_hyper.L")
            
            local diag_D_s = $(p_names.sigma) ./ sqrt.(diag_Ls .+ M.noise)
            local diag_D_t = 1.0 ./ sqrt.(diag_Lt .+ M.noise)
            
            local Z_matrix = reshape($(p_names.ure), $(s_N), $(t_N))
            local transformed = (diag_D_s .* Z_matrix) .* diag_D_t'
            local st_field = s_hyper.U * transformed * t_hyper.U'
            $(eta_target) = $(eta_target) .+ view(st_field, M.st_idx)
        end
    """

    cholesky_base_code = """
        local parent_hyper = spec_registry[:$(key)].hyper
        local s_spec_hyper = parent_hyper.child_specs[1].hyper
        local t_spec_hyper = parent_hyper.child_specs[2].hyper
        local Q_s = recompose_precision(:$(s_model_type), s_spec_hyper.Q_template, 1.0;
          extra_param=$(s_rho_val))
        local Q_t = recompose_precision(:$(t_model_type), t_spec_hyper.Q_template, 1.0;
          extra_param=$(t_rho_val))
    """

    cholesky_dense_code = """
        # --- Spatiotemporal Interaction (Cholesky, AD-Safe): $(spec.key) ---
        let
            $(cholesky_base_code)
            local C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
            local C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
            local Z_matrix = reshape($(p_names.ure), $(s_N), $(t_N))
            local tmp_spatial = C_s.L' \\ Z_matrix
            local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * ($(s_N) * $(t_N))),
              sum(st_field_unscaled))
            local st_field = st_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view(st_field, M.st_idx)
        end
    """

    cholesky_sparse_code = """
        # --- Spatiotemporal Interaction (Sparse Cholesky, Not AD-Safe): $(spec.key) ---
        let
            $(cholesky_base_code)
            local C_s = cholesky(Symmetric(Q_s + M.noise * I))
            local C_t = cholesky(Symmetric(Q_t + M.noise * I))
            local Z_matrix = reshape($(p_names.ure), $(s_N), $(t_N))
            local tmp_spatial = C_s.L' \\ Z_matrix
            local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
            Turing.@addlogprob! logpdf(Normal(0, 0.001 * ($(s_N) * $(t_N))),
              sum(st_field_unscaled))
            local st_field = st_field_unscaled .* $(p_names.sigma)
            $(eta_target) = $(eta_target) .+ view(st_field, M.st_idx)
        end
    """

    has_spectral = hasproperty(s_spec.hyper, :U) && hasproperty(s_spec.hyper, :L) &&
                   hasproperty(t_spec.hyper, :U) && hasproperty(t_spec.hyper, :L)

    if m.method == :spectral && has_spectral
        return spectral_code
    elseif m.method == :cholesky || (m.method == :spectral && !has_spectral)
        return cholesky_dense_code
    elseif m.method == :cholesky_sparse
        return cholesky_sparse_code
    else
        return has_spectral ? spectral_code : cholesky_dense_code
    end
end


function get_effects(
    m::TensorProductSmooth, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    # --- Setup: Extract dimensions ---
    n_samples = size(chain, 1) * FlexiChains.nchains(chain)
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))
    
    # --- Get precomputed data ---
    child_specs = spec.hyper.child_specs
    s_spec, t_spec = child_specs[1], child_specs[2]
    s_N, t_N = s_spec.hyper.n_latent, t_spec.hyper.n_latent
    noise = M.noise

    Q_s_template_cpu = s_spec.hyper.Q_template
    Q_t_template_cpu = t_spec.hyper.Q_template

    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))

    # --- Index Handling: Combine training and prediction sets on CPU ---
    s_idx_full_cpu = isnothing(PS) ? M.s_idx : vcat(M.s_idx, get(PS.data, :s_idx, []))
    t_idx_full_cpu = isnothing(PS) ? M.t_idx : vcat(M.t_idx, get(PS.data, :t_idx, []))
    N_total = length(s_idx_full_cpu)
    st_idx_full_cpu = (t_idx_full_cpu .- 1) .* s_N .+ s_idx_full_cpu

    structured_effects = Vector{Matrix{Float64}}()

    # --- Reconstruction Loop: Iterate over each outcome variable ---
    for k in 1:outcomes_N
        v = generate_full_variable_names(spec, M.model_arch, k)
        s_v = generate_full_variable_names(s_spec, M.model_arch, k)
        t_v = generate_full_variable_names(t_spec, M.model_arch, k)

        sigma_name = _find_parameter(p_names, string(v.sigma), k, is_multivariate_model)
        ure_name = _find_parameter(p_names, string(v.ure), k, is_multivariate_model)
        
        s_rho_name = hasproperty(s_spec.component_obj, :rho) ? _find_parameter(p_names,
            string(s_v.rho), k, is_multivariate_model) : ""
        t_rho_name = hasproperty(t_spec.component_obj, :rho) ? _find_parameter(p_names,
            string(t_v.rho), k, is_multivariate_model) : ""

        if isempty(sigma_name) || isempty(ure_name)
            @warn "Parameters for TensorProductSmooth component $(spec.key) (outcome $k) not found. Returning zero-matrix."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        # Extract posterior samples (CPU)
        sigma_samples_cpu = get_params_vector(chain, sigma_name, 1)[:, 1]
        ure_samples_cpu = get_params_matrix(chain, ure_name, s_N * t_N)
        
        s_rho_samples_cpu = !isempty(s_rho_name) ? get_params_vector(chain, s_rho_name, 1)[:,
            1] : nothing
        t_rho_samples_cpu = !isempty(t_rho_name) ? get_params_vector(chain, t_rho_name, 1)[:,
            1] : nothing

        # Initialize the output matrix for the full effect on the CPU
        effect_k_cpu = zeros(Float64, N_total, n_samples)

        # --- Sample-wise Reconstruction on the CPU ---
        for i in 1:n_samples
            sigma_i = sigma_samples_cpu[i]
            innovations_i = ure_samples_cpu[i, :]
            s_rho_val = isnothing(s_rho_samples_cpu) ? nothing : s_rho_samples_cpu[i]
            t_rho_val = isnothing(t_rho_samples_cpu) ? nothing : t_rho_samples_cpu[i]
            
            # Recompose precision matrices on the CPU
            Q_s = recompose_precision(s_model_type, Q_s_template_cpu, 1.0; extra_param=s_rho_val)
            Q_t = recompose_precision(t_model_type, Q_t_template_cpu, 1.0; extra_param=t_rho_val)
            
            # Perform Cholesky and back-solve on the CPU
            C_s = cholesky(Symmetric(Matrix(Q_s) + noise * I))
            C_t = cholesky(Symmetric(Matrix(Q_t) + noise * I))
            
            Z_matrix = reshape(innovations_i, s_N, t_N)
            tmp_spatial = C_s.L' \ Z_matrix
            st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
            
            st_field_unscaled .-= mean(st_field_unscaled)
            st_field = st_field_unscaled .* sigma_i
            
            effect_k_cpu[:, i] = vec(st_field)[st_idx_full_cpu]
        end
        
        push!(structured_effects, effect_k_cpu)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
 
