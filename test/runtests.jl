# ==============================================================================
# BSTM Comprehensive Test Suite Orchestrator
# ==============================================================================
# Usage:
#   julia --project=. tests/runtests.jl                  # Run all test segments
#   julia --project=. tests/runtests.jl fast             # Run ultra-fast unit tests
#   julia --project=. tests/runtests.jl core             # Run formula & registry tests
#   julia --project=. tests/runtests.jl likelihoods      # Run likelihood taxonomy tests
#   julia --project=. tests/runtests.jl partitioning     # Run spatial graph tests
#   julia --project=. tests/runtests.jl components       # Run component model tests
#   julia --project=. tests/runtests.jl derivatives      # Run surface derivatives & BPI
#   julia --project=. tests/runtests.jl eiv              # Run Errors-in-Variables tests
#   julia --project=. tests/runtests.jl pipeline         # Run DAG pipeline orchestrator
#   julia --project=. tests/runtests.jl par              # Run PAR / PAF engine tests
#   julia --project=. tests/runtests.jl models           # Run model MCMC inference tests
#   julia --project=. tests/runtests.jl multinomial      # Run multinomial & categorical models
#   julia --project=. tests/runtests.jl persistence      # Run DuckDB & JLD2 persistence
#   julia --project=. tests/runtests.jl nested           # Run nested multi-fidelity tests
#
# Standalone execution:
#   julia --project=. tests/test_core_formula.jl
#   julia --project=. tests/test_likelihoods.jl
#   julia --project=. tests/test_partitioning.jl
#   julia --project=. tests/test_components.jl
#   julia --project=. tests/test_derivatives.jl
#   julia --project=. tests/test_eiv.jl
#   julia --project=. tests/test_pipeline.jl
#   julia --project=. tests/test_par.jl
#   julia --project=. tests/test_models.jl
#   julia --project=. tests/test_multinomial.jl
#   julia --project=. tests/test_data_persistence_plots.jl
#   julia --project=. tests/test_nested.jl
# ==============================================================================

include(joinpath(@__DIR__, "test_helpers.jl"))

const AVAILABLE_SEGMENTS = Dict{Symbol, @NamedTuple{file::String, desc::String}}(
    :core => (
        file = "test_core_formula.jl",
        desc = "Core Formula Parsing, ParamRegistry & Manifolds"
    ),
    :likelihoods => (
        file = "test_likelihoods.jl",
        desc = "Likelihood Engine Taxonomy & Distributions"
    ),
    :partitioning => (
        file = "test_partitioning.jl",
        desc = "Spatial Partitioning, Graph Engine & Polygon Topology"
    ),
    :components => (
        file = "test_components.jl",
        desc = "ComponentModel Interface, NNGP, MCAR & Movement"
    ),
    :derivatives => (
        file = "test_derivatives.jl",
        desc = "Surface Derivatives, Topographic Metrics & Bessel BPI"
    ),
    :eiv => (
        file = "test_eiv.jl",
        desc = "Errors-in-Variables (EIV) Covariate Measurement Error Priors"
    ),
    :pipeline => (
        file = "test_pipeline.jl",
        desc = "Modular DAG Pipeline Orchestrator & Cross-Mesh Resharding"
    ),
    :par => (
        file = "test_par.jl",
        desc = "Population Attributable Risk (PAR / PAF) Engine"
    ),
    :models => (
        file = "test_models.jl",
        desc = "Model Instantiation, Smoke Tests & Complex Inference"
    ),
    :multinomial => (
        file = "test_multinomial.jl",
        desc = "Multinomial, Categorical & Dirichlet Formulations"
    ),
    :persistence => (
        file = "test_data_persistence_plots.jl",
        desc = "DuckDB Analytics, JLD2 Bundles, GeoJSON & Plot Validation"
    ),
    :nested => (
        file = "test_nested.jl",
        desc = "Nested Multi-Fidelity Models & Sub-Model Linkage Architecture"
    )
)

const ALIAS_MAP = Dict{String, Vector{Symbol}}(
    "all"          => [:core, :likelihoods, :partitioning, :components, :derivatives,
                       :eiv, :pipeline, :par, :models, :multinomial, :persistence, :nested],
    "full"         => [:core, :likelihoods, :partitioning, :components, :derivatives,
                       :eiv, :pipeline, :par, :models, :multinomial, :persistence, :nested],
    "fast"         => [:core, :likelihoods, :partitioning, :eiv, :pipeline, :par],
    "quick"        => [:core, :likelihoods, :partitioning, :eiv, :pipeline, :par],
    "core"         => [:core],
    "formula"      => [:core],
    "registry"     => [:core],
    "manifolds"    => [:core],
    "likelihoods"  => [:likelihoods],
    "likelihood"   => [:likelihoods],
    "taxonomy"     => [:likelihoods],
    "partitioning" => [:partitioning],
    "partition"    => [:partitioning],
    "spatial"      => [:partitioning],
    "components"   => [:components],
    "component"    => [:components],
    "nngp"         => [:components],
    "movement"     => [:components],
    "derivatives"  => [:derivatives],
    "derivative"   => [:derivatives],
    "slope"        => [:derivatives],
    "bpi"          => [:derivatives],
    "eiv"          => [:eiv],
    "errors"       => [:eiv],
    "measurement"  => [:eiv],
    "pipeline"     => [:pipeline],
    "workflow"     => [:pipeline],
    "orchestrator" => [:pipeline],
    "dag"          => [:pipeline],
    "resharding"   => [:pipeline],
    "par"          => [:par],
    "paf"          => [:par],
    "attributable" => [:par],
    "models"       => [:models],
    "model"        => [:models],
    "instantiation"=> [:models],
    "gibbs"        => [:models],
    "smoke"        => [:models],
    "multinomial"  => [:multinomial],
    "categorical"  => [:multinomial],
    "dirichlet"    => [:multinomial],
    "persistence"  => [:persistence],
    "data"         => [:persistence],
    "plots"        => [:persistence],
    "duckdb"       => [:persistence],
    "geojson"      => [:persistence],
    "nested"       => [:nested],
    "transfer"     => [:nested],
    "submodel"     => [:nested],
    "multifidelity"=> [:nested]
)

"""
    parse_requested_segments(args::Vector{String}) -> Vector{Symbol}

Parses CLI test segment arguments, resolving aliases and fast-test shortcuts.
"""
function parse_requested_segments(args::Vector{String})::Vector{Symbol}
    if isempty(args)
        return [:core, :likelihoods, :partitioning, :components, :derivatives,
                :eiv, :pipeline, :par, :models, :multinomial, :persistence, :nested]
    end

    selected = Symbol[]
    for a in args
        key_str = lowercase(strip(a))
        if haskey(ALIAS_MAP, key_str)
            append!(selected, ALIAS_MAP[key_str])
        else
            avail = sort(collect(keys(AVAILABLE_SEGMENTS)))
            @warn "Unrecognized test segment: '$a'. Available segments: $(join(avail, ", "))"
        end
    end

    return isempty(selected) ? [:core, :likelihoods, :partitioning, :eiv, :pipeline, :par] : unique(selected)
end

requested_segments = parse_requested_segments(ARGS)

@testset "BSTM Comprehensive Test Suite" begin
    for seg_key in requested_segments
        info = AVAILABLE_SEGMENTS[seg_key]
        println("\n" * "="^80)
        println(">>> Running Segment: $(info.desc) ($(info.file))")
        println("="^80)
        include(joinpath(@__DIR__, info.file))
    end
end
