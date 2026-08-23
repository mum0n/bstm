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
#   julia --project=. tests/runtests.jl models           # Run model MCMC inference tests
#   julia --project=. tests/runtests.jl persistence      # Run DuckDB & JLD2 persistence
#
# Standalone execution:
#   julia --project=. tests/test_core_formula.jl
#   julia --project=. tests/test_likelihoods.jl
#   julia --project=. tests/test_partitioning.jl
#   julia --project=. tests/test_components.jl
#   julia --project=. tests/test_derivatives.jl
#   julia --project=. tests/test_eiv.jl
#   julia --project=. tests/test_pipeline.jl
#   julia --project=. tests/test_models.jl
#   julia --project=. tests/test_data_persistence_plots.jl
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
    :models => (
        file = "test_models.jl",
        desc = "Model Instantiation, Smoke Tests & Complex Inference"
    ),
    :persistence => (
        file = "test_data_persistence_plots.jl",
        desc = "DuckDB Analytics, JLD2 Bundles, GeoJSON & Plot Validation"
    )
)

const ALIAS_MAP = Dict{String, Vector{Symbol}}(
    "all"          => [:core, :likelihoods, :partitioning, :components, :derivatives, :eiv, :pipeline, :models, :persistence],
    "full"         => [:core, :likelihoods, :partitioning, :components, :derivatives, :eiv, :pipeline, :models, :persistence],
    "fast"         => [:core, :likelihoods, :partitioning, :eiv, :pipeline],
    "quick"        => [:core, :likelihoods, :partitioning, :eiv, :pipeline],
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
    "models"       => [:models],
    "model"        => [:models],
    "instantiation"=> [:models],
    "gibbs"        => [:models],
    "smoke"        => [:models],
    "persistence"  => [:persistence],
    "data"         => [:persistence],
    "plots"        => [:persistence],
    "duckdb"       => [:persistence],
    "geojson"      => [:persistence]
)

"""
    parse_requested_segments(args::Vector{String}) -> Vector{Symbol}

Parses CLI test segment arguments, resolving aliases and fast-test shortcuts.
"""
function parse_requested_segments(args::Vector{String})::Vector{Symbol}
    if isempty(args)
        return [:core, :likelihoods, :partitioning, :components, :derivatives, :eiv, :pipeline, :models, :persistence]
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

    return isempty(selected) ? [:core, :likelihoods, :partitioning, :eiv, :pipeline] : unique(selected)
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
