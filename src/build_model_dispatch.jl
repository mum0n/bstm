function _build_pass_through_model(m::ManifoldModel, data_inputs::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # v1.0.1 (2026-06-29 16:13:05)
    # Purpose: A generic constructor for manifold configurations that do not require complex template generation.
    # Inputs: m - The manifold model struct.
    #         data_inputs - The main model configuration dictionary.
    #         model_type_sym - Optional symbol to override the model type.
    #         Q_template_val - Optional pre-computed precision template.
    #         sf_val - Optional scaling factor.
    # Outputs: A NamedTuple representing the manifold's configuration.
    model_sym = isnothing(model_type_sym) ? Symbol(lowercase(string(typeof(m)))) : model_type_sym
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        # Extract all fields from the manifold struct that are not types or matrices
        # to populate the hyperparameter tuple.
        field_val = getfield(m, fn)
        if !(field_val isa DataType) && !(field_val isa AbstractMatrix)
             hyper_dict[fn] = field_val
        end
    end
    
    return (
        Q_template = Q_template_val,
        scaling_factor = sf_val,
        model_type = model_sym,
        hyper = NamedTuple(hyper_dict)
    )
end

function _build_from_template(m::ManifoldModel, data_inputs::Dict, domain::Symbol)
    model_sym = Symbol(lowercase(string(typeof(m))))
    
    n, W_mat = if domain == :spatial
        (get(data_inputs, :s_N, 1), get(data_inputs, :W, nothing))
    elseif domain == :temporal
        (get(data_inputs, :t_N, 10), nothing)
    else 
        @warn "Unrecognized domain '$domain'. Defaulting to spatial context."
        (get(data_inputs, :s_N, 1), get(data_inputs, :W, nothing))
    end

    template = build_structure_template(model_sym, n; W=W_mat)

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        field_val = getfield(m, fn)
        if !(field_val isa DataType) && !(field_val isa AbstractMatrix)
             hyper_dict[fn] = field_val
        end
    end
    
    for p in [:rho_prior, :lengthscale_prior, :kappa_prior]
        if !haskey(hyper_dict, p)
            hyper_dict[p] = nothing
        end
    end

    return (
        Q_template = template.matrix,
        scaling_factor = template.scaling_factor,
        model_type = model_sym,
        hyper = NamedTuple(hyper_dict)
    )
end

function build_model(m::Union{IID, ICAR, Besag, BYM2, Leroux, SAR}, data_inputs::Dict)
    return _build_from_template(m, data_inputs, :spatial)
end

function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict)
    return _build_from_template(m, data_inputs, :temporal)
end

function build_model(m::Union{GP, FITC, RFF, SVGP, Warp, Nystrom, Harmonic, Hyperbolic, ExponentialDecay}, data_inputs::Dict)
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict)
    n = 20 # Default value
    if hasproperty(m, :domain)
        domain = get(m, :domain, :spatial)
        if domain == :spatial; n = get(data_inputs, :s_N, 20);
        elseif domain == :temporal; n = get(data_inputs, :t_N, 20);
        elseif domain == :seasonal; n = get(data_inputs, :u_N, 20);
        else; n = get(m, :nbins, get(data_inputs, :s_N, 20)); end
    else
        n = get(m, :nbins, get(data_inputs, :s_N, 20))
    end

    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Wavelet, data_inputs::Dict)
    n = 20 # Default value
    if hasproperty(m, :domain)
        domain = get(m, :domain, :spatial)
        if domain == :spatial; n = get(data_inputs, :s_N, 20);
        elseif domain == :temporal; n = get(data_inputs, :t_N, 20);
        elseif domain == :seasonal; n = get(data_inputs, :u_N, 20);
        else; n = get(m, :nbins, get(data_inputs, :s_N, 20)); end
    else
        n = get(m, :nbins, get(data_inputs, :s_N, 20))
    end

    template = build_structure_template(:iid, n)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:spectral, Q_template_val=template.matrix)
end

function build_model(m::FFT, data_inputs::Dict)
    n = get(m, :nbins, get(data_inputs, :t_N, get(data_inputs, :s_N, 20)))
    template = build_structure_template(:iid, n)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:spectral, Q_template_val=template.matrix)
end

function build_model(m::SPDE, data_inputs::Dict)
    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    template = build_structure_template(:besag, n; W=W)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Eigen, data_inputs::Dict)
    n = get(data_inputs, :s_N, 1)
    template = build_structure_template(:eigen, n)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::BCGN, data_inputs::Dict)
    n_groups = size(m.bipartite_adj, 2)
    template = build_structure_template(:iid, n_groups)
    return _build_pass_through_model(m, data_inputs, Q_template_val=template.matrix)
end

function build_model(m::NetworkFlow, data_inputs::Dict)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:network, Q_template_val=m.adjacency_matrix)
end

function build_model(m::LocalAdaptive, data_inputs::Dict)
    template = build_structure_template(:besag, get(data_inputs, :s_N, 1); W=get(data_inputs, :W, nothing))
    return _build_pass_through_model(m, data_inputs, model_type_sym=:local_adaptive, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Mosaic, data_inputs::Dict)
    return _build_pass_through_model(m, data_inputs)
end

function build_model(m::Cyclic, data_inputs::Dict)
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::TensorProductSmooth, data_inputs::Dict)
    return (
        Q_template = m.Q_template,
        scaling_factor = 1.0,
        model_type = :tensor_product_smooth,
        hyper = (sigma_prior = m.sigma_prior,)
    )
end

function build_model(m::SoftConstraintManifold, data_inputs::Dict)
    inner_config = build_model(m.manifold, data_inputs)
    new_hyper = merge(inner_config.hyper, (soft_constraint_type=m.type, soft_constraint_weight=m.weight))
    return merge(inner_config, (hyper = new_hyper,))
end

function build_model(m::RegularizationGroupManifold, data_inputs::Dict)
    return (
        Q_template = nothing,
        scaling_factor = 1.0,
        model_type = :regularization_group,
        hyper = (
            penalty = m.penalty,
            lambda_prior = m.lambda_prior,
            alpha_prior = m.alpha_prior, 
            sub_manifolds = m.manifolds
        )
    )
end

function build_model(m::Manifold, data_inputs::Dict)
    @warn "No specific builder for $(typeof(m)). Using IID identity template."
    template = build_structure_template(:iid, get(data_inputs, :s_N, 1))
    return (
        Q_template = template.matrix,
        scaling_factor = 1.0,
        model_type = :iid,
        hyper = (sigma_prior = hasproperty(m, :sigma_prior) ? m.sigma_prior : Exponential(1.0),)
    )
end
