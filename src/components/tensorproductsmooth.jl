# This file contains the proposed new and updated functions for the bstm refactoring.


"""
    TensorProductSmooth <: ComponentModel

A component for creating inseparable spatiotemporal effects by taking the tensor
product of marginal (spatial and temporal) components. This is typically specified
in the formula via the Kronecker product operator `⊗`.

# Fields
- `components::Vector{ComponentModel}`: The child components being combined (e.g., a spatial and a temporal model).
- `sigma::Distribution`: The prior for the standard deviation of the interaction effect.
"""
struct TensorProductSmooth <: ComponentModel
    components::Vector{ComponentModel}
    sigma::Distribution
end

# Add to the central component constructor registry.
# This is called by `resolve_technical_primitive` when it encounters a `⊗` operator.
# The parser should map `⊗` to the `:tensorproductsmooth` model type.
COMPONENT_CONSTRUCTORS[:tensorproductsmooth] = (p, params) -> begin
    components = get(params, :components, error("TensorProductSmooth requires child components."))
    TensorProductSmooth(components, p.sigma)
end
COMPONENT_CONSTRUCTORS[:interaction] = COMPONENT_CONSTRUCTORS[:tensorproductsmooth] # Alias

# Add to the model-to-structure map.
MODEL_TO_STRUCTURE_MAP[TensorProductSmooth] = :spacetime

"""
    get_datastructures!(m_type::Type{<:TensorProductSmooth}, M::Dict, mod_data::Dict)::Bool

Delegates data structure setup to child components and then computes the combined
spatiotemporal index `st_idx` required for mapping the effect.
"""
function get_datastructures!(m_type::Type{<:TensorProductSmooth}, M::Dict, mod_data::Dict)::Bool
    child_nodes = get(mod_data[:params], :components, [])
    
    if !haskey(mod_data, :component_obj)
        error("TensorProductSmooth's get_datastructures! requires a temporary component object in mod_data.")
    end

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = mod_data[:component_obj].components[i]
        child_component_type = typeof(child_component_obj)
        
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :type => child_node.module_type,
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args,
            :component_obj => child_component_obj
        )
        
        get_datastructures!(child_component_type, M, child_mod_data)
    end

    # After children have run, s_idx, t_idx, and s_N should be in M.
    if haskey(M, :s_idx) && haskey(M, :t_idx) && haskey(M, :s_N)
        M[:st_idx] = (M[:t_idx] .- 1) .* M[:s_N] .+ M[:s_idx]
    else
        @warn "Could not compute spatiotemporal index `st_idx` for component '$(mod_data[:key])'. Ensure spatial and temporal components are defined."
    end
    
    return true
end

"""
    get_precomputes(m::TensorProductSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple

Delegates pre-computation to child components and aggregates their results.
"""
function get_precomputes(m::TensorProductSmooth, M::NamedTuple, mod_data::Dict)::NamedTuple
    child_nodes = get(mod_data[:params], :components, [])
    child_precomputes_list = []
    child_specs_list = []

    for (i, child_node) in enumerate(child_nodes)
        child_component_obj = m.components[i]
        child_mod_data = Dict(
            :key => Symbol("$(mod_data[:key])_child_$(i)"),
            :type => child_node.module_type,
            :variables => get(child_node.args, :positional_args, []),
            :params => child_node.args
        )
        precomputes = get_precomputes(child_component_obj, M, child_mod_data)
        push!(child_precomputes_list, precomputes)

        child_spec = (
            key = child_mod_data[:key],
            structure = MODEL_TO_STRUCTURE_MAP[typeof(child_component_obj)],
            var = join(child_mod_data[:variables], "_"),
            component_obj = child_component_obj,
            params = child_mod_data[:params],
            hyper = precomputes
        )
        push!(child_specs_list, child_spec)
    end

    return (child_specs = child_specs_list,)
end

"""
    get_priors(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates priors for the interaction scale `sigma` and the latent innovations `raw`.
"""
function get_priors(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    child_specs = spec.hyper.child_specs
    
    # Assuming the first child is spatial and the second is temporal
    s_N = child_specs[1].hyper.n_latent
    t_N = child_specs[2].hyper.n_latent
    
    return """
    # Priors for Spatiotemporal Interaction: $(spec.key)
    $(p_names.sigma) ~ NamedDist($(_distribution_to_string(m.sigma)), :$(p_names.sigma))
    $(p_names.raw) ~ NamedDist(MvNormal(zeros(T, $(s_N * t_N)), I), :$(p_names.raw))
    """
end

"""
    get_updates(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String

Generates the Turing code for the spatiotemporal interaction effect using a Kronecker solver.
"""
function get_updates(m::TensorProductSmooth, spec::NamedTuple, arch::String, outcome_idx::Union{Int, Nothing}, M::NamedTuple)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    child_specs = spec.hyper.child_specs

    s_spec = child_specs[1]
    t_spec = child_specs[2]
    
    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    s_rho_val = hasproperty(s_spec.component_obj, :rho) ? generate_full_variable_names(s_spec, arch, outcome_idx).rho : "nothing"
    t_rho_val = hasproperty(t_spec.component_obj, :rho) ? generate_full_variable_names(t_spec, arch, outcome_idx).rho : "nothing"

    return """
    # --- Spatiotemporal Interaction Update: $(spec.key) ---
    local Q_s = recompose_precision(:$(s_model_type), spec_registry[:$(s_spec.key)].hyper.Q_template, T(1.0); extra_param=$(s_rho_val))
    local Q_t = recompose_precision(:$(t_model_type), spec_registry[:$(t_spec.key)].hyper.Q_template, T(1.0); extra_param=$(t_rho_val))
    
    local C_s = cholesky(Symmetric(Matrix(Q_s) + M.noise * I))
    local C_t = cholesky(Symmetric(Matrix(Q_t) + M.noise * I))
    
    local Z_matrix = reshape($(p_names.raw), M.s_N, M.t_N)
    
    local tmp_spatial = C_s.L' \\ Z_matrix
    local st_field_unscaled = transpose(C_t.L' \\ transpose(tmp_spatial))
    
    Turing.@addlogprob! logpdf(Normal(0, 0.001 * (M.s_N * M.t_N)), sum(st_field_unscaled))
    
    local st_field = st_field_unscaled .* $(p_names.sigma)
    
    $(eta_target) .+= st_field[M.st_idx]
    """
end


"""
    get_effects(m::TensorProductSmooth, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple

Reconstructs the posterior of the spatiotemporal interaction field.
This version returns the raw posterior samples for each outcome, not a summary.
"""
function get_effects(m::TensorProductSmooth, chain, M::NamedTuple, n_samples::Int, outcomes_N::Int, spec::NamedTuple, PS::Union{NamedTuple, Nothing}, N_total::Int)::NamedTuple
    structured_effects = Vector{Matrix{Float64}}()
    child_specs = spec.hyper.child_specs
    s_spec = child_specs[1]
    t_spec = child_specs[2]
    
    s_N = s_spec.hyper.n_latent
    t_N = t_spec.hyper.n_latent
    noise = M.noise

    s_model_type = Symbol(lowercase(string(typeof(s_spec.component_obj))))
    t_model_type = Symbol(lowercase(string(typeof(t_spec.component_obj))))
    
    for k in 1:outcomes_N
        p_names = generate_full_variable_names(spec, M.model_arch, k)
        s_p_names = generate_full_variable_names(s_spec, M.model_arch, k)
        t_p_names = generate_full_variable_names(t_spec, M.model_arch, k)

        sigma_samples = get_params_vector(chain, string(p_names.sigma), 1)[:, 1]
        raw_samples = get_params_vector(chain, string(p_names.raw), s_N * t_N)
        
        s_rho_samples = hasproperty(s_spec.component_obj, :rho) ? get_params_vector(chain, string(s_p_names.rho), 1)[:, 1] : nothing
        t_rho_samples = hasproperty(t_spec.component_obj, :rho) ? get_params_vector(chain, string(t_p_names.rho), 1)[:, 1] : nothing

        st_field_samples = zeros(Float64, s_N * t_N, n_samples)

        for i in 1:n_samples
            s_rho_val = isnothing(s_rho_samples) ? nothing : s_rho_samples[i]
            t_rho_val = isnothing(t_rho_samples) ? nothing : t_rho_samples[i]

            Q_s = recompose_precision(s_model_type, s_spec.hyper.Q_template, 1.0; extra_param=s_rho_val)
            Q_t = recompose_precision(t_model_type, t_spec.hyper.Q_template, 1.0; extra_param=t_rho_val)
            
            C_s = cholesky(Symmetric(Matrix(Q_s) + noise * I))
            C_t = cholesky(Symmetric(Matrix(Q_t) + noise * I))
            
            Z_matrix = reshape(raw_samples[i, :], s_N, t_N)
            
            tmp_spatial = C_s.L' \ Z_matrix
            st_field_unscaled = transpose(C_t.L' \ transpose(tmp_spatial))
            
            st_field = st_field_unscaled .* sigma_samples[i]
            st_field_samples[:, i] = vec(st_field)
        end
        push!(structured_effects, st_field_samples)
    end
    
    return (structured=structured_effects, noisy=structured_effects)
end
