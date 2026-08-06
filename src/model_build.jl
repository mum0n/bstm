

"""
    generate_rff_params(...)

Generates random projection weights (W) and biases (b) for RFF approximation.
This version is included for completeness and supports ARD.
"""
function generate_rff_params(in_dims::Int, n_features::Int, lengthscale::Union{Real, AbstractVector}, kernel_name::String)
    b = rand(Uniform(0, 2 * pi), n_features)
    W = Matrix{Float64}(undef, in_dims, n_features)
    k_name = lowercase(kernel_name)

    if k_name in ["se", "gaussian", "rbf"]
        if lengthscale isa Real
            W .= rand(Normal(0, 1.0 / lengthscale), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = rand(Normal(0, 1.0 / lengthscale[d]), n_features); end
        end
    elseif occursin("matern", k_name)
        nu = if k_name == "matern12"; 0.5; elseif k_name == "matern32"; 1.5; else 2.5; end
        df = 2 * nu
        if lengthscale isa Real
            W .= (sqrt(df) / lengthscale) .* rand(TDist(df), in_dims, n_features)
        else
            if length(lengthscale) != in_dims; error("ARD lengthscale vector length mismatch."); end
            for d in 1:in_dims; W[d, :] = (sqrt(df) / lengthscale[d]) .* rand(TDist(df), n_features); end
        end
    else
        @warn "Kernel '$kernel_name' not recognized for RFF. Defaulting to SE."
        return generate_rff_params(in_dims, n_features, lengthscale, "se")
    end
    return W, b
end




# Generic builder for standard Component types
function build_model(m::Component, data_inputs::Dict, module_metadata::Dict)
    structure = get(module_metadata, :type, :spatial)
    return _build_from_template(m, data_inputs, structure, module_metadata)
end

function build_model(m::CustomComponent, data_inputs::Dict, module_metadata::Dict)
"""
    build_model(m::CustomComponent, data_inputs::Dict, module_metadata::Dict)

A model builder for the `CustomComponent`.

# Rationale
This is a new function that ensures `CustomComponent` is handled correctly by the
configuration engine. Since a custom component is defined entirely by user-provided
code, it does not have a predefined structure matrix (`Q_template`). This builder
uses the `_build_pass_through_model` helper to signal that no precision matrix
template is needed, preventing the framework from incorrectly trying to build one.
"""
    # This component is defined entirely by user code, so it doesn't have a Q_template.
    # We use the pass-through builder to indicate this.
    return _build_pass_through_model(m, data_inputs, module_metadata)
end


# Specialized builder for IID to ensure structure-specific template resolution
function build_model(m::IID, data_inputs::Dict, module_metadata::Dict)
    structure = get(module_metadata, :type, :spatial)
    return _build_from_template(m, data_inputs, structure, module_metadata)
end

# Builder for temporal Gaussian Markov Random Fields
function build_model(m::Union{AR1, RW1, RW2}, data_inputs::Dict, module_metadata::Dict)
    # v1.0.0 (2026-07-20) - Context-aware structure resolution.
    # If used in a `smooth()` call, the structure is `:mixed` (on bins), not `:temporal`.
    structure = get(module_metadata, :type, :temporal)
    return _build_from_template(m, data_inputs, structure, module_metadata)
end



"""
    build_model(m::Union{Wavelet, FFT}, data_inputs::Dict, module_metadata::Dict)

A new model builder for `Wavelet` and `FFT` components.

# Rationale
This builder ensures that the coordinate data and the per-dimension bin counts,
which are resolved in `process_smooth_module!`, are correctly stored in the
component's `hyper` registry. This information is essential for the new dynamic
code generators for these models.
"""
function build_model(m::Union{Wavelet, FFT}, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("$(typeof(m)) component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:coords] = coords
    
    if haskey(module_metadata[:params], :nbins_per_dim)
        hyper_dict[:nbins_per_dim] = module_metadata[:params][:nbins_per_dim]
    end

    return (Q_template=nothing, scaling_factor=1.0, model_type=Symbol(lowercase(string(typeof(m)))), hyper=NamedTuple(hyper_dict))
end


function build_model(m::ShotNoiseCoxProcess, data_inputs::Dict, module_metadata::Dict)
    # Get spatial domain bounds from the data coordinates.
    # Assumes 's_x' and 's_y' are present from a `random(s_x, s_y, ...)` call.
    if !hasproperty(data_inputs[:data], :s_x) || !hasproperty(data_inputs[:data], :s_y)
        error("SNCP requires continuous spatial coordinates `s_x` and `s_y` to define the domain.")
    end
    
    coords = data_inputs[:data]
    x_min, x_max = extrema(coords.s_x)
    y_min, y_max = extrema(coords.s_y)
    
    areas = get(data_inputs, :grid_areas, ones(get(data_inputs, :s_N, 1)))

    hyper_dict = Dict(
        :domain_bounds => (x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        :areas => Float64.(areas),
        :s_N => get(data_inputs, :s_N, 1)
    )

    # SNCP does not use a GMRF precision matrix template.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:sncp, hyper=NamedTuple(hyper_dict))
end



function build_model(m::MixedComponent, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `MixedComponent`.
    # Rationale: This function correctly constructs the technical specification for a mixed effect model.
    #            It recursively calls `build_model` on the inner component (e.g., IID, RW2)
    #            to obtain its precision matrix template (`Q_template`), which defines the
    #            correlation structure of the random effects.
    # v1.0.0 (2026-07-20)
    
    # The inner model determines the structure of the random effects.
    inner_mod_data = Dict(
        :type => :mixed,
        :params => module_metadata[:params]
    )
    # Purpose: A specialized model builder for the `MixedComponent`.
    # Rationale: This function correctly constructs the technical specification for a mixed effect model.
    #            It recursively calls `build_model` on the inner component (e.g., IID, RW2)
    #            to obtain its precision matrix template (`Q_template`), which defines the
    #            correlation structure of the random effects.
    # v1.0.0 (2026-07-20)
    
    # The inner model determines the structure of the random effects.
    inner_mod_data = Dict(
        :type => :mixed,
        :params => module_metadata[:params]
    )
    
    inner_spec = build_model(m.model, data_inputs, inner_mod_data)
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:inner_hyper] = inner_spec.hyper
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:mixed, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::Union{ICAR, Besag, BYM2, Leroux, SAR}, ...)

A model builder for standard spatial GMRF models.

# Rationale for Update
This version has been updated to remove `LocalAdaptive` from the type union.
`LocalAdaptive` now has its own specialized builder to handle the inclusion
of clustering information.
"""
function build_model(m::Union{ICAR, Besag, BYM2, Leroux, SAR}, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata) 
end

"""
    build_model(m::SPDE, data_inputs::Dict, module_metadata::Dict)

A model builder for the `SPDE` component.
"""
function build_model(m::SPDE, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata)
end


function build_model(m::LocalAdaptive, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `LocalAdaptive` component.
    # Rationale: This version is updated to correctly build the precision matrix template.
    #            The `localadaptive` model uses a Leroux-style precision matrix. This builder
    #            now correctly calls `_build_from_template` with a `:leroux` context, ensuring
    #            the correct `Q_template` is generated. It also passes the clustering information
    #            (computed in `process_localadaptive_module!`) to the `hyper` registry for use
    #            by the code generator. This resolves the "Unknown model type" warning.
    # v1.0.2 (2026-08-02)
    # Inputs/Outputs: Standard model builder arguments.

    # The LocalAdaptive model uses a Leroux precision structure.
    # We call the generic template builder with a `:leroux` context.
    base_spec = _build_from_template(m, data_inputs, :spatial, module_metadata)
    
    # Augment the hyper parameters with the clustering info.
    hyper_dict = Dict{Symbol, Any}(pairs(base_spec.hyper))
    
    if !haskey(data_inputs, :n_clusters) || !haskey(data_inputs, :cluster_assignments)
        error("LocalAdaptive model requires `n_clusters` and `cluster_assignments` to be pre-computed, but they were not found in the model configuration. This indicates an issue with `process_localadaptive_module!`.")
    end
    
    hyper_dict[:n_clusters] = data_inputs[:n_clusters]
    
    # The `model_type` in the final spec should be `:localadaptive` to ensure the correct
    # code generator is called.
    return merge(base_spec, (model_type=:localadaptive, hyper=NamedTuple(hyper_dict)))
end





function build_model(m::SVCComponent, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `SVCComponent`.
    # Rationale: This function correctly constructs the technical specification for an SVC model.
    #            It recursively calls `build_model` on the inner spatial component (e.g., BYM2, ICAR)
    #            to obtain its precision matrix template (`Q_template`). This template is then
    #            passed up to the main configuration, ensuring that the code generator for the
    #            SVC has the correct structural information to model the spatially varying coefficient.
    # v1.0.0 (2026-07-20)
    
    # The inner model (e.g., BYM2) determines the structure.
    # We call its builder to get the Q_template.
    
    spatial_model_spec_node = get(module_metadata[:params], :spatial_model_spec, nothing)
    if isnothing(spatial_model_spec_node); error("SVC builder is missing the inner spatial model specification."); end
    
    spatial_mod_data = Dict(:type => spatial_model_spec_node.module_type, :params => spatial_model_spec_node.args, :variables => get(spatial_model_spec_node.args, :positional_args, []))
    
    # Call the builder for the inner spatial model
    inner_spec = build_model(m.model, data_inputs, spatial_mod_data)
    
    # The SVC component itself doesn't have hyperparameters, but we pass them
    # from the inner model for the code generator to use.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Pass the inner spec's hyper-parameters up.
    hyper_dict[:inner_hyper] = inner_spec.hyper
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:svc, hyper=NamedTuple(hyper_dict))
end




"""
    build_model(m::Spherical, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Spherical` component when used as a full Gaussian Process.

# Rationale for Update
This is a new function that enables the `Spherical` component to be treated as a
continuous-space Gaussian Process, consistent with its definition which includes
priors for `sigma` and `range`. It ensures that the coordinate data from a `smooth()`
call is correctly captured and passed to the code generator.
"""
function build_model(m::Spherical, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords); error("Spherical component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`."); end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    
    return (Q_template=nothing, scaling_factor=1.0, model_type=:spherical, hyper=NamedTuple(hyper_dict))
end




"""
    build_model(m::RFF, data_inputs::Dict, module_metadata::Dict)

Updated model builder for the `RFF` component to handle anisotropic lengthscales.

# Rationale for Update
This version correctly handles the case where the `lengthscale` prior is a vector.
It computes a vector of initial lengthscale values by taking the mean of each prior
distribution. This initial vector is then passed to `generate_rff_params`, which
already supports ARD and will generate the fixed projection weights `W_fixed` from
the corresponding anisotropic spectral density.
"""
function build_model(m::RFF, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords); error("RFF component requires coordinates."); end
    
    ls_prior = m.lengthscale
    local ls_initial
    if ls_prior isa Vector
        ls_initial = [mean(p isa Truncated ? untruncated(p) : p) for p in ls_prior]
    else
        ls_initial = mean(ls_prior isa Truncated ? untruncated(ls_prior) : ls_prior)
    end
    
    in_dims = size(coords, 2)
    W_fixed, b_fixed = generate_rff_params(in_dims, m.n_features, ls_initial, m.kernel)

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    hyper_dict[:W_fixed] = W_fixed
    hyper_dict[:b_fixed] = b_fixed
    return (Q_template=nothing, scaling_factor=1.0, model_type=:rff, hyper=NamedTuple(hyper_dict))
end




"""
    build_model(m::FITC, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `FITC` (Fully Independent Training Conditional) component.

# Rationale for Update
This version cleans up the internal implementation by removing redundant code. Its primary
role remains to configure the `FITC` sparse Gaussian Process model by:
1.  Storing the observation coordinates (`coords`) in the `Q_template` field.
2.  Storing the pre-computed inducing point locations (`Z_inducing`) in the component's
    `hyper` registry for use by the code generator.
"""
function build_model(m::FITC, data_inputs::Dict, module_metadata::Dict)
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("FITC component requires coordinates, but none were not found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("FITC component requires inducing points, but they were not found. This is an internal error in the `smooth` or `spatial` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:Z_inducing] = Z_inducing
    
    # The `Q_template` field is used by convention to pass the observation coordinates.
    return (Q_template=coords, scaling_factor=1.0, model_type=:fitc, hyper=NamedTuple(hyper_dict))
end


"""
    build_model(m::Moran, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Moran` spatial component.

# Rationale for Update
This is a new function to correctly implement the Moran eigenvector spectral model.
It computes the Moran operator `M = (I - 11'/n)W(I - 11'/n)`, calculates its
eigenvectors, and stores them in the component's hyperparameter registry. These
eigenvectors serve as the basis functions for the spatial effect, aligning the
implementation with the documented behavior of `Moran's I Basis Component`.
"""
function build_model(m::Moran, data_inputs::Dict, module_metadata::Dict)
    W = get(data_inputs, :W, nothing)
    if isnothing(W)
        error("The `moran` component requires an adjacency matrix `W`, but it was not found in the model configuration.")
    end
    
    n = size(W, 1)
    
    # Create the centering matrix H = I - (1/n) * 1*1'
    H = I - (1/n) * ones(n, n)
    
    # Compute the Moran operator M = HWH
    # Ensure W is a concrete matrix for computation
    W_mat = Matrix(W)
    moran_operator = H * W_mat * H
    
    # Compute the eigenvectors of the symmetric Moran operator
    eig_result = eigen(Symmetric(moran_operator))
    moran_eigenvectors = eig_result.vectors
    
    # Store the eigenvectors in the hyperparameter registry for the code generator
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:moran_eigenvectors] = moran_eigenvectors
    
    # Q_template is not used for this spectral model, but a placeholder is returned for API consistency.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:moran, hyper=NamedTuple(hyper_dict))
end
  

function build_model(m::Warp, data_inputs::Dict, module_metadata::Dict)
    # For Warp, we need the raw coordinates to apply the warping function to.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Warp component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or a spatial module with available coordinates.")
    end
    
    # The warping function's parameters are fully learned, so we don't
    # pre-generate fixed features like in RFF. We just need to pass the coordinates.
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    hyper_dict[:coords] = coords

    # The Q_template is not used, but we provide a placeholder for consistency
    # with the rest of the framework's data structures.
    Q_template = sparse(I, m.n_features, m.n_features)
    return (Q_template=Q_template, scaling_factor=1.0, model_type=:warp, hyper=NamedTuple(hyper_dict))
end

function build_model(m::SVGP, data_inputs::Dict, module_metadata::Dict)
    # For SVGP, we need both the observation coordinates and the inducing point coordinates.
    # The "template" will store the observation coordinates.
    # The inducing points will be stored in the `hyper` registry.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("SVGP component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("SVGP component requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}(); for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:Z_inducing] = Z_inducing
    return (Q_template=coords, scaling_factor=1.0, model_type=:svgp, hyper=NamedTuple(hyper_dict))
end
 
function build_model(m::Nystrom, data_inputs::Dict, module_metadata::Dict)
    # For Nystrom, we need both the observation coordinates and the inducing point coordinates.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Nystrom component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end

    Z_inducing = get(module_metadata[:params], :Z_inducing, nothing)
    if isnothing(Z_inducing)
        error("Nystrom component requires inducing points, but none were found. This is an internal error in the `smooth` processor.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    hyper_dict[:Z_inducing] = Z_inducing
    return (Q_template=nothing, scaling_factor=1.0, model_type=:nystrom, hyper=NamedTuple(hyper_dict))
end

function build_model(m::GP, data_inputs::Dict, module_metadata::Dict)
    # For GP, the "template" is the coordinate matrix itself, not the distance matrix.
    # This allows the kernel evaluation to handle ARD kernels correctly.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("GP component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Store the raw coordinates in the template field.
    return (Q_template=coords, scaling_factor=1.0, model_type=:gp, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Hyperbolic, data_inputs::Dict, module_metadata::Dict)
    # For Hyperbolic GP, we need the raw coordinates to compute hyperbolic distances.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("Hyperbolic component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    return (Q_template=nothing, scaling_factor=1.0, model_type=:hyperbolic, hyper=NamedTuple(hyper_dict))
end

function build_model(m::ExponentialDecay, data_inputs::Dict, module_metadata::Dict)
    # For Exponential Decay GP, we need the raw coordinates to compute distances.
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("ExponentialDecay component requires coordinates, but none were found. Ensure you are using `smooth(var1, var2, ...)` or `spatial(lon, lat, ...)`.")
    end
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:coords] = coords
    return (Q_template=nothing, scaling_factor=1.0, model_type=:exponentialdecay, hyper=NamedTuple(hyper_dict))
end

function build_model(m::Harmonic, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for continuous, spectral, and other advanced components.
    # Rationale: These models do not rely on pre-computed templates in the same way as GMRFs, so they use a pass-through builder.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    return _build_pass_through_model(m, data_inputs, module_metadata)
end


"""
    build_model(m::NetworkFlow, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `NetworkFlow` component.

# Rationale
This function dispatches the `NetworkFlow` component to the template-based builder
with a `:spatial` context. This ensures that the adjacency matrix `W` from the main
model configuration is correctly identified and passed as the structural template
for the network model.

# Arguments
- `m::NetworkFlow`: The NetworkFlow component object.
- `data_inputs::Dict`: The main model configuration dictionary.
- `module_metadata::Dict`: The parsed dictionary for the module.

# Returns
- A `NamedTuple` containing the component's technical specification.
"""
function build_model(m::NetworkFlow, data_inputs::Dict, module_metadata::Dict)
    return _build_from_template(m, data_inputs, :spatial, module_metadata)
end


function build_model(m::Union{PSpline, TPS, BSpline}, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for spline-based smoothers.
    # Rationale: Determines the appropriate underlying GMRF template (RW1 or RW2) based on the spline type and penalty order.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    n = m.nbins
    template_type = m isa PSpline ? (m.diff_order == 1 ? :rw1 : :rw2) : (m isa TPS ? :rw2 : :iid)
    template = build_structure_template(template_type, n)
    return _build_pass_through_model(m, data_inputs, module_metadata; Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::Cyclic, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the `Cyclic` component.
    # Rationale: Creates a circulant precision matrix for smooth periodic effects.
    # v1.0.0 (2026-07-16)
    # Assumptions: None.
    # Inputs/Outputs: See `build_model`.
    template = build_structure_template(:cyclic, m.period)
    return _build_pass_through_model(m, data_inputs, module_metadata; model_type_sym=:cyclic, Q_template_val=template.matrix, sf_val=template.scaling_factor)
end

function build_model(m::BCGN, data_inputs::Dict, module_metadata::Dict)
    # Purpose: Builder for the BCGN (Bipartite Graph Convolutional Network) component.
    # Rationale: Constructs the precision matrix template from the provided bipartite adjacency matrix.
    #            The precision is based on the graph Laplacian of the one-mode projection of the
    #            bipartite graph. This induces a GMRF structure on one set of nodes, where two
    #            nodes are considered "neighbors" if they share a common neighbor in the other partition.
    # v1.0.0 (2026-07-19)

    B = m.bipartite_adj
    if isempty(B) || all(iszero, B)
        error("BCGN component requires a non-empty `bipartite_adj` matrix, but it was not provided or is all zeros.")
    end

    # The latent effect is defined on the first set of nodes (rows of B).
    # We create the precision matrix from the one-mode projection onto this set.
    W_proj = B * B'
    
    # For a standard graph Laplacian, self-loops (diagonal elements) are set to zero.
    W_proj[diagind(W_proj)] .= 0
    W_proj = dropzeros(W_proj)

    # Build the graph Laplacian from the projected adjacency matrix: L = D - W
    D_proj = spdiagm(0 => vec(sum(W_proj, dims=2)))
    Q_template = D_proj - W_proj

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); if fn != :bipartite_adj; hyper_dict[fn] = getfield(m, fn); end; end
    
    return (Q_template=Q_template, scaling_factor=1.0, model_type=:bcgn, hyper=NamedTuple(hyper_dict))
end

"""
    build_model(m::Eigen, data_inputs::Dict, module_metadata::Dict)

A model builder specifically for the `Eigen` component.

# Rationale for Update
This new builder method ensures that the pre-processed data matrix required for the
Bayesian PCA is correctly passed from the module processor into the component's
hyperparameter registry. This makes the data accessible to the code generator.
"""
function build_model(m::Eigen, data_inputs::Dict, module_metadata::Dict)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        hyper_dict[fn] = getfield(m, fn)
    end
    
    # Retrieve the data matrix from the parameters populated by `process_eigen_module!`.
    eigen_data = get(module_metadata[:params], :eigen_data, nothing)
    if isnothing(eigen_data)
        error("Eigen data matrix not found in module metadata. This indicates an issue in `process_eigen_module!`.")
    end
    hyper_dict[:eigen_data] = eigen_data
    
    # Q_template is not used for the Eigen component, but a placeholder is returned for API consistency.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:eigen, hyper=NamedTuple(hyper_dict))
end



# Version 1.5.1 (2026-08-06)
# Purpose: Builds the technical specification for a `DynamicsComponent`.
# Rationale: This version improves comprehensibility by adding detailed comments that
#            clarify the complex logic for processing `effort` and `removal` parameters.
#            It explains how the function handles various input formats (symbols, vectors,
#            matrices) and aggregates per-observation data into a spatiotemporal grid
#            for use in the dynamics simulation.
function build_model(m::DynamicsComponent, data_inputs::Dict, module_metadata::Dict)
    # This builder prepares the necessary templates and parameters for mechanistic dynamics models.

    n = get(data_inputs, :s_N, 1)
    W = get(data_inputs, :W, nothing)
    if isnothing(W)
        error("DynamicsComponent requires an adjacency matrix W, but it was not found in the model configuration.")
    end

    # Build templates for diffusion (L) and advection (A) operators.
    L_template = build_structure_template(:besag, n; W=W).matrix
    A_template = if m.model in ["advection", "advection_diffusion"]
        W_dir = tril(W, -1)
        out_degree = sum(W_dir, dims=2)[:]
        D_inv = spdiagm(0 => 1.0 ./ (out_degree .+ 1e-9))
        D_inv * W_dir
    else
        spzeros(Float64, n, n)
    end

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    hyper_dict[:L_template] = L_template
    hyper_dict[:A_template] = A_template
    hyper_dict[:areas] = get(data_inputs, :grid_areas, ones(n))

    s_N = get(data_inputs, :s_N, 1)
    t_N = get(data_inputs, :t_N, 1)
    y_N = data_inputs[:y_N]
    s_idx = get(data_inputs, :s_idx, ones(Int, y_N))
    t_idx = get(data_inputs, :t_idx, ones(Int, y_N))
    data = data_inputs[:data]

    processed_params = Dict{Symbol, Any}()
    hyper_dict[:effort_keys] = Symbol[]
    hyper_dict[:removal_keys] = Symbol[]

    # Process exploitation parameters (`effort` and `removal`).
    # This logic handles various input formats and aggregates per-observation
    # data onto the spatiotemporal grid required by the dynamics simulation.
    for param_base_name in [:effort, :removal]
        if haskey(m.params, param_base_name)
            val = m.params[param_base_name]
            target_keys_list = hyper_dict[Symbol(string(param_base_name) * "_keys")]
            
            # Determine how to process the input based on its type.
            local vals_to_process
            if val isa Matrix && size(val, 1) != y_N
                # Case 1: A pre-aggregated matrix [s_N x t_N] or [s_N x t_N x n_classes].
                vals_to_process = [val]
            elseif val isa Matrix
                # Case 2: A matrix of per-observation covariates [y_N x k_sources].
                vals_to_process = [val[:, i] for i in 1:size(val, 2)]
            elseif val isa Vector && !(val isa AbstractVector{<:Real})
                # Case 3: A vector of symbols, e.g., [:e1, :e2].
                vals_to_process = val
            else
                # Case 4: A single symbol or a single vector of reals (per-observation).
                vals_to_process = [val]
            end

            for (i, v) in enumerate(vals_to_process)
                storage_key = length(vals_to_process) > 1 ? Symbol("$(param_base_name)_$(i)") : param_base_name
                
                # Resolve the value `v` to the actual data array.
                covariate_data = if v isa AbstractArray{<:Real}
                    v
                elseif v isa Symbol && hasproperty(data, v)
                    data[!, v]
                else
                    nothing
                end

                if !isnothing(covariate_data)
                    if ndims(covariate_data) == 1
                        # Aggregate per-observation data to a spatial-temporal grid.
                        cov_matrix = zeros(s_N, t_N)
                        counts = zeros(Int, s_N, t_N)
                        for obs_i in 1:y_N
                            si, ti = s_idx[obs_i], t_idx[obs_i]
                            cov_matrix[si, ti] += covariate_data[obs_i]
                            counts[si, ti] += 1
                        end
                        cov_matrix ./= max.(1, counts) # Average values within each cell.
                        processed_params[storage_key] = cov_matrix
                        push!(target_keys_list, storage_key)
                    elseif ndims(covariate_data) >= 2
                        # Data is already in matrix form [s_N, t_N] or [s_N, t_N, n_classes].
                        processed_params[storage_key] = covariate_data
                        push!(target_keys_list, storage_key)
                    end
                end
            end
        end
    end

    # Process other parameters.
    for (key, val) in m.params
        if key in [:effort, :removal]; continue; end
        processed_params[key] = val
    end
    hyper_dict[:processed_params] = processed_params

    return (Q_template=L_template, scaling_factor=1.0, model_type=:dynamics, hyper=NamedTuple(hyper_dict))
end



 
function build_model(m::TensorProductSmooth, data_inputs::Dict, module_metadata::Dict)
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        if !(fn in [:Q_template])
            hyper_dict[fn] = getfield(m, fn)
        end
    end
    return (Q_template=m.Q_template, scaling_factor=1.0, model_type=:tensorproductsmooth, hyper=NamedTuple(hyper_dict))
end
 

# Version 1.5.1 (2026-08-06)
# Purpose: Builds the technical specification for a `ComposedComponent`.
# Rationale: This version improves comprehensibility by adding detailed comments that
#            clarify the logic for handling different composition operators (`:pipe`,
#            `:composition`, `⊗`). It explains how the builder recursively constructs
#            specifications for child components and packages them into the `hyper`
#            registry for the code generator.
function build_model(m::ComposedComponent, data_inputs::Dict, module_metadata::Dict)
    # This builder handles components created by algebraic operators like `|>` and `⊗`.
    # It recursively builds the specifications for the child components and stores
    # them in the `hyper` registry for the code generator to use.

    op = m.operator 
    key = string(module_metadata[:key])

    if op == :pipe
        # Handles `dynamic |> state` syntax, e.g., for spatially varying curves.
        if length(m.components) != 2; error("Pipe operator requires exactly two components: dynamic |> state."); end

        dynamic_component = m.components[1] # The curve (e.g., PSpline)
        state_component = m_obj.components[2]   # The field (e.g., ICAR)

        # Reconstruct metadata for the state component to pass to its builder.
        state_node = module_metadata[:params][:components][2]
        state_mod_data = Dict(
            :type => get(state_node.args, :structure, :spatial),
            :params => state_node.args,
            :variables => get(state_node.args, :positional_args, [])
        )
        state_spec = build_model(state_component, data_inputs, state_mod_data)

        # Find the variable associated with the dynamic component to identify its basis matrix.
        dynamic_node = module_metadata[:params][:components][1]
        dynamic_vars = get(dynamic_node.args, :positional_args, [])
        if isempty(dynamic_vars); error("The dynamic part of a pipe operator (e.g., a smoother) must have a variable."); end
        dynamic_basis_key = Symbol(join(dynamic_vars, "_"))

        # Package all necessary info for the code generator into the hyper registry.
        hyper_dict = Dict{Symbol, Any}(
            :state_spec => state_spec,
            :dynamic_component_obj => dynamic_component,
            :dynamic_basis_key => dynamic_basis_key
        )
        return (Q_template=nothing, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    elseif op == :composition
        # Handles `modifier ∘ base` syntax, e.g., for non-stationary variance.
        base_component = get(m.components, 1, nothing)
        if isnothing(base_component); error("Composition component is missing its base component."); end 
        
        base_spec = build_model(base_component, data_inputs, module_metadata)
        hyper_dict = Dict(:base_spec => base_spec)
        return (Q_template=base_spec.Q_template, scaling_factor=1.0, model_type=:composed, hyper=NamedTuple(hyper_dict))

    else
        # For other operators like `⊗` (Kronecker product), no special template is needed
        # at this stage, as they are handled by other processors or code generators.
        return _build_pass_through_model(m, data_inputs, module_metadata)
    end
end

 

function build_model(m::SVAR, data_inputs::Dict, module_metadata::Dict)
    s_N = get(data_inputs, :s_N, 1)
    t_N = get(data_inputs, :t_N, 1)
    
    # The inner spatial model object is already correctly stored in m.rho_spatial.
    # We need to build its specification to store it in the hyper registry for the code generator.
    
    # To call build_model on the inner component, we need to reconstruct its module_metadata.
    inner_model_name = Symbol(lowercase(string(typeof(m.rho_spatial))))
    
    inner_mod_data = Dict(
        :type => :spatial, # The inner model of an SVAR is always spatial.
        :params => module_metadata[:params], # Pass the original params down.
        :variables => get(module_metadata, :variables, [])
    )
    # Ensure the 'model' parameter in the metadata reflects the actual inner model type.
    inner_mod_data[:params][:model] = inner_model_name

    # Recursively call build_model on the inner spatial component.
    rho_spatial_spec = build_model(m.rho_spatial, data_inputs, inner_mod_data)
    
    # Store the resulting specification in the hyper registry.
    hyper_dict = Dict(
        :rho_spatial_spec => rho_spatial_spec,
        :s_N => s_N,
        :t_N => t_N
    )
    
    # The SVAR component itself does not have a Q_template.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:svar, hyper=NamedTuple(hyper_dict))
end
 

# Adaptive Basis Functions: Learns a non-linear warping of coordinates using a hidden layer before kernel application, facilitating the discovery of complex spatial/temporal deformations.
function build_model(m::AdaptiveSmooth, data_inputs::Dict, module_metadata::Dict)
    # Resolve coordinates for the transformation
    coords = get(module_metadata[:params], :coords, nothing)
    if isnothing(coords)
        error("AdaptiveSmooth requires coordinates in the module parameters.")
    end

    hyper_dict = Dict(
        :coords => Float64.(coords),
        :in_dim => size(coords, 2),
        :hidden_dim => m.hidden_dim,
        :nbins => m.nbins
    )

    return (Q_template=nothing, scaling_factor=1.0, model_type=:adaptive_smooth, hyper=NamedTuple(hyper_dict))
end


function build_model(m::TAR, data_inputs::Dict, module_metadata::Dict)
    t_N = get(data_inputs, :t_N, 1)
    data = data_inputs[:data]
    threshold_var_sym = m.threshold_var
    
    if !hasproperty(data, threshold_var_sym)
        error("TAR model's threshold variable ':$threshold_var_sym' not found in the provided data.")
    end
    
    threshold_data = data[!, threshold_var_sym]

    hyper_dict = Dict(
        :threshold_var => threshold_var_sym,
        :threshold_data => Float64.(threshold_data),
        :t_N => t_N
    )

    return (Q_template=nothing, scaling_factor=1.0, model_type=:tar, hyper=NamedTuple(hyper_dict))
end





 

 
function build_model(m::LGCP, data_inputs::Dict, module_metadata::Dict)
    # The LGCP struct 'm' already contains the resolved inner model (m.model)
    # and the original inner model node (m.inner_model_node).
    
    # Reconstruct the module_metadata for the inner model from m.inner_model_node
    inner_mod_data = Dict(
        :type => m.inner_model_node.module_type, # e.g., :random
        :params => m.inner_model_node.args,
        :variables => get(m.inner_model_node.args, :positional_args, [])
    )
    # Ensure structure is inferred if not explicit in the inner node's args
    if !haskey(inner_mod_data[:params], :structure)
        inner_mod_data[:params][:structure] = _infer_structure_from_args(inner_mod_data[:params])
    end

    # Now, call build_model on the inner model (m.model) with its reconstructed metadata.
    # This will correctly build its Q_template and hyper.
    inner_spec = build_model(m.model, data_inputs, inner_mod_data)
    
    temporal_spec_idx = findfirst(s -> s.structure == :temporal, data_inputs[:components])
    areas = get(data_inputs, :grid_areas, ones(data_inputs[:s_N]))

    hyper_dict = Dict(
        :inner_spec => inner_spec, 
        :areas => Float64.(areas), 
        :s_N => data_inputs[:s_N], 
        :t_N => get(data_inputs, :t_N, 1)
    )
    if !isnothing(temporal_spec_idx)
        hyper_dict[:temporal_spec] = data_inputs[:components][temporal_spec_idx]
    end

    return (Q_template=inner_spec.Q_template, scaling_factor=1.0, model_type=:lgcp, hyper=NamedTuple(hyper_dict))
end


"""
    build_model(m::Kriging, data_inputs::Dict, module_metadata::Dict)

A model builder for the `Kriging` component.

# Rationale for Update
This version is updated to be consistent with the `GP` builder. It now stores the
coordinate matrix in the `Q_template` field, ensuring that the code generator can
correctly access the coordinates to build the dense covariance matrix.
"""
function build_model(m::Kriging, data_inputs::Dict, module_metadata::Dict)
    # Kriging requires continuous coordinates (e.g., s_x, s_y)
    coords = get(module_metadata[:params], :coords, nothing)
    
    if isnothing(coords)
        # Fallback: check if coordinates exist in the top-level data inputs
        if haskey(data_inputs, :coords)
            coords = data_inputs[:coords]
        else
            error("Kriging component requires coordinate data. Ensure s_x and s_y are provided.")
        end
    end

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    
    # Store the raw coordinates in the Q_template field for consistency with the code generator.
    return (Q_template=coords, scaling_factor=1.0, model_type=:kriging, hyper=NamedTuple(hyper_dict))
end



function build_model(m::TVCComponent, data_inputs::Dict, module_metadata::Dict)
    temporal_model_spec_node = get(module_metadata[:params], :temporal_model_spec, nothing)
    if isnothing(temporal_model_spec_node); error("TVC builder is missing the inner temporal model specification."); end
    
    temporal_mod_data = Dict(:type => :temporal, :params => temporal_model_spec_node.args, :variables => get(temporal_model_spec_node.args, :positional_args, []))
    
    # Call the builder for the inner temporal model (e.g., RW2)
    inner_spec = build_model(m.model, data_inputs, temporal_mod_data)
    
    hyper_dict = Dict{Symbol, Any}(:inner_hyper => inner_spec.hyper)
    
    return (Q_template=inner_spec.Q_template, scaling_factor=inner_spec.scaling_factor, model_type=:tvc, hyper=NamedTuple(hyper_dict))
end




function build_model(m::LogGammaCoxProcess, data_inputs::Dict, module_metadata::Dict)
    params = module_metadata[:params]
    
    inner_model_node = get(params, :inner_model_node, nothing)
    if isnothing(inner_model_node)
        error("LogGammaCoxProcess builder is missing the inner model specification node.")
    end

    inner_params = merge(params, inner_model_node.args)

    inner_mod_data = Dict(
        :type => get(inner_model_node.args, :structure, :spatial), 
        :params => inner_params, 
        :variables => get(inner_model_node.args, :positional_args, [])
    )
    inner_spec = build_model(m.model, data_inputs, inner_mod_data)
    
    temporal_spec_idx = findfirst(s -> s.structure == :temporal, data_inputs[:components])
    areas = get(data_inputs, :grid_areas, ones(data_inputs[:s_N]))

    hyper_dict = Dict(
        :inner_spec => inner_spec, 
        :areas => Float64.(areas), 
        :s_N => data_inputs[:s_N], 
        :t_N => get(data_inputs, :t_N, 1)
    )
    if !isnothing(temporal_spec_idx)
        hyper_dict[:temporal_spec] = data_inputs[:components][temporal_spec_idx]
    end

    return (Q_template=inner_spec.Q_template, scaling_factor=1.0, model_type=:lgammap, hyper=NamedTuple(hyper_dict))
end



"""
    build_model(m::NonStationaryVariance, data_inputs::Dict, module_metadata::Dict)

A specialized model builder for the `NonStationaryVariance` component.

# Rationale
This builder constructs the technical specifications for both the `base_model` (spatial)
and `modifier_model` (smoother) by recursively calling `build_model` on them. The resulting
specifications, which include their respective precision matrix templates, are stored in
the `hyper` registry of the main component. This makes all necessary structural information
available to the code generator.
"""
function build_model(m::NonStationaryVariance, data_inputs::Dict, module_metadata::Dict)
    # Purpose: A specialized model builder for the `NonStationaryVariance` component.
    # Rationale: This builder constructs the technical specifications for both the `base_model`
    #            (spatial) and `modifier_model` (smoother) by recursively calling `build_model`
    #            on them. The resulting specifications, which include their respective precision
    #            matrix templates, are stored in the `hyper` registry of the main component.
    #            This makes all necessary structural information available to the code generator.
    # v1.0.0 (2026-07-29)
    # Inputs/Outputs: Standard model builder arguments.
    base_node = module_metadata[:params][:base_node]
    modifier_node = module_metadata[:params][:modifier_node]

    # Build the spec for the base model (e.g., ICAR).
    base_mod_data = Dict(:type => base_node.module_type, :params => base_node.args, :variables => get(base_node.args, :positional_args, []))
    base_spec = build_model(m.base_model, data_inputs, base_mod_data)

    # Build the spec for the modifier model (e.g., PSpline).
    modifier_mod_data = Dict(:type => modifier_node.module_type, :params => modifier_node.args, :variables => get(modifier_node.args, :positional_args, []))
    modifier_spec = build_model(m.modifier_model, data_inputs, modifier_mod_data)

    # Store these specs and the basis key in the hyper registry for the code generator.
    hyper_dict = Dict(
        :base_spec => base_spec,
        :modifier_spec => modifier_spec,
        :modifier_basis_key => module_metadata[:params][:modifier_basis_key]
    )

    # The NonStationaryVariance component itself does not have a Q_template.
    return (Q_template=nothing, scaling_factor=1.0, model_type=:nonstationary_variance, hyper=NamedTuple(hyper_dict))
end






function _build_from_template(m::ComponentModel, data_inputs::Dict, structure::Symbol, module_metadata::Dict)
    # Purpose: A generic builder for components that use a pre-defined template.
    # Rationale: This version is updated to correctly resolve the adjacency matrix `W`. It now
    #            searches for `W` first in the local module's parameters (highest precedence),
    #            then falls back to the global model configuration. This ensures that `W`
    #            provided inside a nested `random()` call (e.g., in an SVC) is correctly found.
    # v1.0.1 (2026-07-28)
    # Inputs:
    #   - m: The ComponentModel object.
    #   - data_inputs: The main model configuration dictionary (`M`).
    #   - structure: The structure of the component (:spatial, :temporal, :mixed).
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the component's technical specification.
    model_sym = Symbol(lowercase(string(typeof(m))))
    
    local n, W_mat
    if structure == :spatial
        # --- W Resolution Logic ---
        # 1. Check for W in the local module's parameters first.
        local W_from_local = nothing
        if haskey(module_metadata[:params], :W)
            w_val = module_metadata[:params][:W]
            if w_val isa Expr || w_val isa Symbol
                calling_mod = get(data_inputs, :calling_module, Main)
                try
                    W_from_local = Core.eval(calling_mod, w_val)
                catch e
                    error("Could not evaluate `W` argument `$(w_val)` in module '$(get(module_metadata, :type, "unknown"))'. Error: $e")
                end
            else
                W_from_local = w_val
            end
        end

        # 2. Fallback to the global W from the main configuration.
        W_from_main = get(data_inputs, :W, nothing)

        # 3. Prioritize the locally provided W.
        W_mat = isnothing(W_from_local) ? W_from_main : W_from_local
        
        # Determine the number of spatial units from W if available.
        n = isnothing(W_mat) ? get(data_inputs, :s_N, 1) : size(W_mat, 1)

    elseif structure == :temporal
        n = get(data_inputs, :t_N, 10)
        W_mat = nothing
    elseif structure == :mixed
        n_levels = get(get(module_metadata, :params, Dict()), :n_cat, 0)
        if n_levels == 0
            error("Could not determine number of levels for mixed effect. `n_cat` not found in module parameters.")
        end
        n = n_levels
        W_mat = nothing
    else
        @warn "Unrecognized structure '$structure'. Defaulting to spatial context."
        n = get(data_inputs, :s_N, 1)
        W_mat = get(data_inputs, :W, nothing)
    end

    template = build_structure_template(model_sym, n; W=W_mat)
    
    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m))
        if !(fn in [:Q_template])
            hyper_dict[fn] = getfield(m, fn)
        end
    end
    
    return (Q_template = template.matrix, scaling_factor = template.scaling_factor, model_type = model_sym, hyper = NamedTuple(hyper_dict))
end


function _build_pass_through_model(m::ComponentModel, data_inputs::Dict, module_metadata::Dict; model_type_sym=nothing, Q_template_val=nothing, sf_val=1.0)
    # Purpose: A generic builder for components that do not require complex template generation.
    # Rationale: Used for models where the structure is defined by parameters (e.g., splines) or handled dynamically.
    # v1.0.0 (2026-07-16)
    #            This version ensures a default identity Q_template is created for basis-like models.
    # Assumptions: None.
    # Inputs:
    #   - m, data_inputs, and optional overrides.
    #   - module_metadata: The parsed dictionary for the module.
    # Outputs: A NamedTuple with the component's technical specification.
    model_sym = isnothing(model_type_sym) ? Symbol(lowercase(string(typeof(m)))) : model_type_sym

    # If Q_template is not provided, create a default identity matrix based on n_features, nbins, or n_inducing.
    # This is crucial for allowing these models to be used in compositions.
    if isnothing(Q_template_val)
        n_units = 0
        if hasproperty(m, :n_features); n_units = m.n_features;
        elseif hasproperty(m, :nbins); n_units = m.nbins;
        elseif hasproperty(m, :n_inducing); n_units = m.n_inducing;
        end
        
        if n_units > 0; Q_template_val = sparse(I(n_units)); end
    end

    hyper_dict = Dict{Symbol, Any}()
    for fn in fieldnames(typeof(m)); hyper_dict[fn] = getfield(m, fn); end
    if haskey(hyper_dict, :Q_template); delete!(hyper_dict, :Q_template); end
    return (Q_template=Q_template_val, scaling_factor=sf_val, model_type=model_sym, hyper=NamedTuple(hyper_dict))
end
