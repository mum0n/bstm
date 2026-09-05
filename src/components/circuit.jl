"""
    Circuit <: ComponentModel

A component model for Bayesian circuit-theoretic spatial processes and migration
corridor inference. It models spatial random effects on an irregular marine or
landscape graph parameterized by habitat conductance, with native support for
errors-in-variables propagation of habitat suitability index (HSI) uncertainty.

# Version
v1.0.0

# Mathematical Summary
Classical spatial models assume isotropic Euclidean covariance across space, ignoring
coastlines, islands, and landscape resistance. The `Circuit` component operates on the
weighted graph Laplacian:
```math
\\mathbf{L}_C = \\mathbf{D}_C - \\mathbf{C}
```
where ``\\mathbf{C}`` is the symmetric conductance matrix parameterized by latent or
observed habitat quality ``\\mathbf{H}`` and conductance scaling sensitivity ``\\beta``:
```math
C_{uv} = \\exp\\left( \\beta \\cdot \\frac{H_u^{\\text{eff}} + H_v^{\\text{eff}}}{2} \\right)
```
When HSI standard errors ``\\boldsymbol{\\sigma}_H`` are provided (`habitat_se`), the
model incorporates an errors-in-variables observation error structure:
```math
H_u^{\\text{eff}} = H_u^{\\text{obs}} + \\sigma_{H, u} \\cdot u_{H, u}
```
with latent innovations ``u_{H, u} \\sim \\mathcal{N}(0, 1)``.
The latent spatial field ``\\mathbf{s} \\in \\mathbb{R}^S`` is modeled as a Gaussian Markov
Random Field (GMRF) with precision matrix proportional to the circuit Laplacian:
```math
\\mathbf{s} \\sim \\mathcal{N}\\left( \\mathbf{0},
  \\left( \\frac{1}{\\sigma^2} (\\mathbf{L}_C + \\epsilon \\mathbf{I}) \\right)^{-1} \\right)
```

# Posterior Inference on Migratory Circuit Paths
When migratory source and sink nodes are provided (via `sources` and `sinks` arguments or
queried post-hoc via `get_circuit_paths`), the component propagates the posterior
distribution of ``\\beta`` and latent habitat ``\\mathbf{H}^{\\text{eff}}`` across all MCMC
draws through the electrical network solver:
```math
\\mathbf{L}_{C}^{(m)} \\mathbf{V}^{(m)} = \\mathbf{I}_{\\text{ext}}
```
This yields the full posterior distribution of migratory current density ``\\bar{J}_i``,
95% credible intervals, and bottleneck pinch-point occurrence probabilities:
```math
P(\\text{PinchPoint}_i \\mid \\text{data}) =
  \\frac{1}{M} \\sum_{m=1}^M \\mathbb{I}(J_i^{(m)} \\ge q_{0.90}^{(m)})
```

# Inputs
- **Required**:
  - A spatial index variable (e.g. `region`, `s_idx`) passed to `random()`.
  - An adjacency matrix `W` passed as a keyword argument to `@bstm(..., W=W)`.
  - `habitat`: A `Symbol` pointing to a column in the data, or a `Vector` of length `s_N`.
- **Optional (in `random()` call)**:
  - `habitat_se`: `Symbol`, `Vector`, or `Real`, standard error / observation error on HSI.
  - `beta`: `UnivariateDistribution`, prior for conductance sensitivity
    (default: `Normal(1.0, 1.0)`).
  - `sigma`: `UnivariateDistribution`, prior for spatial scale (default: `Exponential(1.0)`).
  - `sources`: `Vector{Int}`, release / origin node indices for migratory path inference.
  - `sinks`: `Vector{Int}`, recapture / destination node indices.
  - `weights`: `Vector{<:Real}`, event weights (e.g. tag observation counts).
  - `land_mask`: `Vector{Bool}`, indicator vector for land units.
  - `top_quantile`: `Real`, quantile cutoff for pinch-points (default: 0.90).
  - `method`: `Symbol`, computational method (`:cholesky`, default).

# Outputs (Parameter Names)
- `beta_<key>`: Conductance sensitivity parameter.
- `sigma_<key>`: Marginal standard deviation of the latent spatial field.
- `ure_<key>`: Standard normal innovations for the spatial field.
- `ure_hab_<key>`: Latent habitat innovation terms (when `habitat_se` is supplied).
- `sre_<key>`: Structured spatial effect vector.
- `circuit_paths`: `PosteriorCircuitResult` containing posterior current density, credible
  intervals, and bottleneck certainty probabilities (if `sources` and `sinks` are given).

# Key References
- McRae, B. H., Dickson, B. G., Keitt, T. H., & Shah, V. B. (2008). Using circuit theory to
  model connectivity in ecology, evolution, and conservation. Ecology, 89(10), 2712-2724.
- Hanks, E. M., & Hooten, M. B. (2013). Circuit theory and model-based inference for
  landscape connectivity. Journal of the American Statistical Association, 108(501), 22-33.
"""
struct Circuit <: ComponentModel
    beta::UnivariateDistribution
    sigma::UnivariateDistribution
    method::Symbol
end

COMPONENT_TYPE_REGISTRY[:circuit] = Circuit
COMPONENT_CONSTRUCTORS[:circuit] = (p, params) -> Circuit(
    get(p, :beta, Normal(1.0, 1.0)),
    get(p, :sigma, Exponential(1.0)),
    Symbol(get(params, :method, :cholesky))
)

MODEL_TO_STRUCTURE_MAP[:circuit] = :spatial

"""
    get_precomputes(m::Circuit, M::NamedTuple, mod_data::Dict)::NamedTuple

Precomputes graph connectivity, habitat covariates, and optional observation error terms.
"""
function get_precomputes(m::Circuit, M::NamedTuple, mod_data::Dict)::NamedTuple
    s_N = M.s_N
    W = M.W
    params = mod_data[:params]
    data = M.data

    if !haskey(params, :habitat)
        error(
            "The `circuit` model requires a `habitat` parameter specifying the " *
            "conductivity/resistivity data."
        )
    end

    habitat_val = params[:habitat]
    local habitat_data::Vector{Float64}

    if habitat_val isa Symbol
        if !hasproperty(data, habitat_val)
            error("Habitat variable ':$habitat_val' not found in data frame.")
        end
        habitat_per_obs = data[!, habitat_val]
        habitat_aggregated = zeros(Float64, s_N)
        counts = zeros(Int, s_N)
        for i in 1:M.y_N
            s_i = M.s_idx[i]
            habitat_aggregated[s_i] += habitat_per_obs[i]
            counts[s_i] += 1
        end
        habitat_data = habitat_aggregated ./ max.(1, counts)
    elseif habitat_val isa AbstractVector
        if length(habitat_val) == s_N
            habitat_data = convert(Vector{Float64}, habitat_val)
        elseif length(habitat_val) == M.y_N
            habitat_aggregated = zeros(Float64, s_N)
            counts = zeros(Int, s_N)
            for i in 1:M.y_N
                s_i = M.s_idx[i]
                habitat_aggregated[s_i] += habitat_val[i]
                counts[s_i] += 1
            end
            habitat_data = habitat_aggregated ./ max.(1, counts)
        else
            error(
                "Provided `habitat` length ($(length(habitat_val))) must match " *
                "s_N ($s_N) or observations ($(M.y_N))."
            )
        end
    else
        error("`habitat` parameter must be a Symbol or Vector of length s_N.")
    end

    # Habitat observation error (errors-in-variables)
    habitat_se_data = zeros(Float64, s_N)
    has_error = false
    if haskey(params, :habitat_se)
        se_val = params[:habitat_se]
        if se_val isa Symbol
            if hasproperty(data, se_val)
                se_per_obs = data[!, se_val]
                se_agg = zeros(Float64, s_N)
                counts = zeros(Int, s_N)
                for i in 1:M.y_N
                    s_i = M.s_idx[i]
                    se_agg[s_i] += se_per_obs[i]
                    counts[s_i] += 1
                end
                habitat_se_data = se_agg ./ max.(1, counts)
                has_error = any(s -> s > 1e-8, habitat_se_data)
            end
        elseif se_val isa AbstractVector
            if length(se_val) == s_N
                habitat_se_data = convert(Vector{Float64}, se_val)
                has_error = any(s -> s > 1e-8, habitat_se_data)
            elseif length(se_val) == M.y_N
                se_agg = zeros(Float64, s_N)
                counts = zeros(Int, s_N)
                for i in 1:M.y_N
                    s_i = M.s_idx[i]
                    se_agg[s_i] += se_val[i]
                    counts[s_i] += 1
                end
                habitat_se_data = se_agg ./ max.(1, counts)
                has_error = any(s -> s > 1e-8, habitat_se_data)
            else
                @warn "habitat_se vector length mismatch ($(length(se_val)) != $s_N). Ignored."
            end
        elseif se_val isa Real
            habitat_se_data = fill(Float64(se_val), s_N)
            has_error = se_val > 1e-8
        end
    end

function _resolve_node_indices(val, M)::Vector{Int}
    if val isa AbstractVector{<:Integer}
        return collect(Int, val)
    elseif val isa Integer
        return [Int(val)]
    elseif val isa Symbol
        if hasproperty(M, val) && getproperty(M, val) isa AbstractVector{<:Integer}
            return collect(Int, getproperty(M, val))
        elseif hasproperty(M, val) && getproperty(M, val) isa Integer
            return [Int(getproperty(M, val))]
        elseif haskey(M, val) && M[val] isa AbstractVector{<:Integer}
            return collect(Int, M[val])
        elseif haskey(M, val) && M[val] isa Integer
            return [Int(M[val])]
        elseif hasproperty(M.data, val)
            col = M.data[!, val]
            if eltype(col) == Bool
                return findall(col)
            elseif eltype(col) <: Integer
                return unique(filter(x -> x > 0, col))
            end
        elseif isdefined(Main, val)
            main_val = getfield(Main, val)
            if main_val isa AbstractVector{<:Integer}
                return collect(Int, main_val)
            elseif main_val isa Integer
                return [Int(main_val)]
            end
        end
    elseif val isa Expr
        try
            eval_val = Core.eval(Main, val)
            if eval_val isa AbstractVector{<:Integer}
                return collect(Int, eval_val)
            elseif eval_val isa Integer
                return [Int(eval_val)]
            end
        catch
        end
    end
    return Int[]
end

    sources_raw = if hasproperty(M, :sources) && !isnothing(M.sources)
        M.sources
    elseif haskey(params, :sources)
        params[:sources]
    else
        Int[]
    end

    sinks_raw = if hasproperty(M, :sinks) && !isnothing(M.sinks)
        M.sinks
    elseif haskey(params, :sinks)
        params[:sinks]
    else
        Int[]
    end

    sources = _resolve_node_indices(sources_raw, M)
    sinks = _resolve_node_indices(sinks_raw, M)
    weights = get(params, :weights, nothing)
    top_quant = get(params, :top_quantile, 0.90)
    land_mask = get(params, :land_mask, nothing)

    I, J, _ = findnz(W)
    return (
        W_I = I,
        W_J = J,
        n_latent = s_N,
        s_N = s_N,
        habitat_data = habitat_data,
        habitat_se_data = habitat_se_data,
        has_error = has_error,
        sources = sources,
        sinks = sinks,
        weights = weights,
        top_quantile = top_quant,
        land_mask = land_mask
    )
end

"""
    get_priors(m::Circuit, spec::NamedTuple, arch::String, outcome_idx, M)::String

Defines Bayesian prior distributions for conductance sensitivity, spatial scale,
and latent habitat innovations under errors-in-variables.
"""
function get_priors(
    m::Circuit, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    priors = String[]

    push!(priors, "$(p_names.beta) ~ $(_distribution_to_string(m.beta))")
    push!(priors, "$(p_names.sigma) ~ $(_distribution_to_string(m.sigma))")

    s_N = spec.hyper.s_N
    n_latent = spec.hyper.n_latent

    if spec.hyper.has_error
        push!(priors, "$(p_names.ure_hab) ~ MvNormal(zeros(T, $(s_N)), I)")
    end

    push!(priors, "$(p_names.ure) ~ MvNormal(zeros(T, $(n_latent)), I)")

    return join(priors, "\n    ")
end

"""
    get_updates(m::Circuit, spec::NamedTuple, arch::String, outcome_idx, M)::String

Generates DynamicPPL code for assembling the circuit Laplacian precision matrix and
updating the linear predictor eta.
"""
function get_updates(
    m::Circuit, spec::NamedTuple, arch::String,
    outcome_idx::Union{Int, Nothing}, M::NamedTuple
)::String
    p_names = generate_full_variable_names(spec, arch, outcome_idx)
    eta_target = (arch == "multivariate") ? "eta_latent[:, $(outcome_idx)]" : "eta"
    key = spec.key

    code = if spec.hyper.has_error
        """
        # --- Circuit Component (Errors-in-Variables): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            W_I = hyper.W_I
            W_J = hyper.W_J
            s_N = hyper.s_N
            hab_mean = hyper.habitat_data
            hab_se = hyper.habitat_se_data

            hab_latent = hab_mean .+ hab_se .* $(p_names.ure_hab)

            C_vals = exp.($(p_names.beta) .* (hab_latent[W_I] .+ hab_latent[W_J]) ./ 2.0)
            W_C = sparse(W_I, W_J, C_vals, s_N, s_N)
            D_C = Diagonal(vec(sum(W_C, dims=2)))
            Q_C = D_C - W_C

            F = cholesky(Symmetric(Matrix(Q_C) + M.noise * I))
            $(p_names.sre) = $(p_names.sigma) .* (F.L' \\ $(p_names.ure))

            $(eta_target) = $(eta_target) .+ view($(p_names.sre), M.s_idx)
        end
        """
    else
        """
        # --- Circuit Component (Conductance GMRF): $(key) ---
        let
            hyper = spec_registry[:$(key)].hyper
            W_I = hyper.W_I
            W_J = hyper.W_J
            s_N = hyper.s_N
            habitat = hyper.habitat_data

            C_vals = exp.($(p_names.beta) .* (habitat[W_I] .+ habitat[W_J]) ./ 2.0)
            W_C = sparse(W_I, W_J, C_vals, s_N, s_N)
            D_C = Diagonal(vec(sum(W_C, dims=2)))
            Q_C = D_C - W_C

            F = cholesky(Symmetric(Matrix(Q_C) + M.noise * I))
            $(p_names.sre) = $(p_names.sigma) .* (F.L' \\ $(p_names.ure))

            $(eta_target) = $(eta_target) .+ view($(p_names.sre), M.s_idx)
        end
        """
    end

    return code
end

"""
    get_effects(m::Circuit, chain, spec::NamedTuple, M::NamedTuple, PS)::NamedTuple

Reconstructs the latent spatial field and evaluates posterior migratory circuit path
inference across all MCMC draws.
"""
function get_effects(
    m::Circuit, chain, spec::NamedTuple, M::NamedTuple,
    PS::Union{NamedTuple, Nothing}
)::NamedTuple
    n_samples = if occursin("FlexiChain", string(typeof(chain)))
        size(chain, 1) * FlexiChains.nchains(chain)
    else
        size(chain, 1) * size(chain, 3)
    end
    outcomes_N = M.outcomes_N
    is_multivariate_model = M.model_arch == "multivariate"
    p_names = string.(keys(chain))

    key = spec.key
    hyper = spec.hyper
    W_I = hyper.W_I
    W_J = hyper.W_J
    s_N = hyper.s_N
    habitat_data = hyper.habitat_data
    habitat_se_data = hyper.habitat_se_data
    has_error = hyper.has_error
    noise = M.noise

    s_idx_train = M.s_idx
    s_idx_full = if !isnothing(PS) && hasproperty(PS.data, :s_idx)
        vcat(s_idx_train, PS.data.s_idx)
    else
        s_idx_train
    end
    N_total = length(s_idx_full)

    structured_effects = Vector{Matrix{Float64}}()
    h_latent_matrix = zeros(Float64, s_N, n_samples)

    for k_outcome in 1:outcomes_N
        p_names_k = generate_full_variable_names(spec, M.model_arch, k_outcome)
        beta_name = _find_parameter(
            p_names, string(p_names_k.beta), k_outcome, is_multivariate_model
        )
        sigma_name = _find_parameter(
            p_names, string(p_names_k.sigma), k_outcome, is_multivariate_model
        )
        ure_name = _find_parameter(
            p_names, string(p_names_k.ure), k_outcome, is_multivariate_model
        )

        if isempty(beta_name) || isempty(sigma_name) || isempty(ure_name)
            @warn "Parameters for Circuit component $(spec.key) not found."
            push!(structured_effects, zeros(Float64, N_total, n_samples))
            continue
        end

        beta_samples = get_params_vector(chain, beta_name, 1)[:, 1]
        sigma_samples = get_params_vector(chain, sigma_name, 1)[:, 1]
        ure_samples = get_params_matrix(chain, ure_name, s_N)

        ure_hab_samples = if has_error
            hab_name = _find_parameter(
                p_names, string(p_names_k.ure_hab), k_outcome, is_multivariate_model
            )
            !isempty(hab_name) ? get_params_matrix(chain, hab_name, s_N) : nothing
        else
            nothing
        end

        reconstructed_effects_k = zeros(Float64, s_N, n_samples)

        for i in 1:n_samples
            hab_i = if has_error && !isnothing(ure_hab_samples)
                habitat_data .+ habitat_se_data .* ure_hab_samples[i, :]
            else
                habitat_data
            end
            if k_outcome == 1
                h_latent_matrix[:, i] = hab_i
            end

            V_beta_i = exp.(beta_samples[i] .* (hab_i[W_I] .+ hab_i[W_J]) ./ 2.0)
            W_beta_i = sparse(W_I, W_J, V_beta_i, s_N, s_N)
            D_beta_i = Diagonal(vec(sum(W_beta_i, dims=2)))
            Q_beta_i = D_beta_i - W_beta_i

            F_i = cholesky(Symmetric(Matrix(Q_beta_i) + noise * I))
            innov_i = ure_samples[i, :]
            reconstructed_effects_k[:, i] = sigma_samples[i] .* (F_i.L' \ innov_i)
        end

        push!(structured_effects, reconstructed_effects_k[s_idx_full, :])
    end

    # If sources and sinks are available, compute stochastic circuit path inference
    circuit_res = if !isempty(hyper.sources) && !isempty(hyper.sinks)
        posterior_circuit_inference(
            M.W,
            hyper.sources,
            hyper.sinks;
            hsi_samples = h_latent_matrix,
            weights = hyper.weights,
            land_mask = hyper.land_mask,
            top_quantile = hyper.top_quantile
        )
    else
        nothing
    end

    return (
        structured = structured_effects,
        noisy = structured_effects,
        circuit_paths = circuit_res,
        habitat_latent = h_latent_matrix
    )
end

"""
    get_circuit_paths(
        model_result;
        key::Symbol = :circuit
    ) -> Union{Nothing, PosteriorCircuitResult}

    get_circuit_paths(
        model_result,
        sources::AbstractVector{Int},
        sinks::AbstractVector{Int};
        key::Symbol = :circuit,
        weights = nothing,
        land_mask = nothing,
        top_quantile::Real = 0.90
    ) -> PosteriorCircuitResult

Extracts or computes stochastic circuit path inference from a fitted BSTM model
containing a `Circuit` component.
"""
function get_circuit_paths(model_result; key::Symbol = :circuit)
    if hasproperty(model_result, :effects)
        eff = model_result.effects
        if haskey(eff, key)
            comp = eff[key]
            if hasproperty(comp, :circuit_paths)
                return comp.circuit_paths
            end
        end
    end
    return nothing
end

function get_circuit_paths(
    model_result,
    sources::AbstractVector{Int},
    sinks::AbstractVector{Int};
    key::Symbol = :circuit,
    weights = nothing,
    land_mask = nothing,
    top_quantile::Real = 0.90
)
    h_latent = if hasproperty(model_result, :effects) && haskey(model_result.effects, key)
        comp = model_result.effects[key]
        hasproperty(comp, :habitat_latent) ? comp.habitat_latent : nothing
    else
        nothing
    end

    if isnothing(h_latent)
        error(
            "Could not find `:habitat_latent` draws in component effects for key :$key. " *
            "Ensure the model was fitted with `model=:circuit`."
        )
    end

    W = if hasproperty(model_result, :M) && hasproperty(model_result.M, :W)
        model_result.M.W
    else
        error("Model result must contain graph adjacency matrix `M.W`.")
    end

    return posterior_circuit_inference(
        W,
        sources,
        sinks;
        hsi_samples = h_latent,
        weights = weights,
        land_mask = land_mask,
        top_quantile = top_quantile
    )
end
