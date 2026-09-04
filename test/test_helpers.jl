# ==============================================================================
# BSTM Test Suite: Shared Test Helpers and Fixtures
# ==============================================================================

using Test
try
    using bstm
catch
    include(joinpath(@__DIR__, "..", "src", "bstm.jl"))
    using .bstm
end

using DynamicPPL
using Plots
using Distributions
using LinearAlgebra
using DataFrames
using CategoricalArrays
using Turing
using Random
using SparseArrays
using Clustering
using LogExpFunctions
using Graphs
using StatsModels

const bstm_Likelihood = bstm.bstm_Likelihood

"""
    create_chain_adj_matrix(n::Int)

Helper function for creating a simple 1D chain spatial adjacency matrix of size `n × n`.
"""
function create_chain_adj_matrix(n::Int)
    W = spzeros(Int, n, n)
    for i in 1:(n - 1)
        W[i, i + 1] = 1
        W[i + 1, i] = 1
    end
    return W
end

"""
    stable_logdiffexp(a, b)

Helper function for stable numerical calculation of log(exp(a) - exp(b)).
"""
stable_logdiffexp(a, b) = a + LogExpFunctions.log1mexp(b - a)

"""
    mock_M_config(N_obs, N_areas, N_time, model_arch="univariate")

Mock model configuration object for isolated unit testing of component primitives.
"""
function mock_M_config(N_obs, N_areas, N_time, model_arch="univariate")
    return (
        data = DataFrame(
            y = rand(N_obs),
            s_idx = repeat(1:N_areas, inner=N_time)[1:N_obs],
            t_idx = repeat(1:N_time, outer=N_areas)[1:N_obs],
            group_var = repeat(1:N_areas, inner=N_obs ÷ N_areas)[1:N_obs]
        ),
        model_arch = model_arch,
        technical = Dict(
            :component_levels => Dict(),
            :component_indices => Dict()
        )
    )
end

"""
    mock_spec(key, hyper_obj=NamedTuple(), params=Dict(), structure=:any)

Mock component specification NamedTuple.
"""
function mock_spec(key, hyper_obj=NamedTuple(), params=Dict(), structure=:any)
    return (
        key = Symbol(key),
        structure = structure,
        var = string(key),
        hyper = hyper_obj,
        params = params
    )
end

"""
    mock_chain(param_names_and_values::Dict, n_samples=10)

Mock chain dictionary mapping parameter symbols to sample matrices.
"""
function mock_chain(param_names_and_values::Dict, n_samples=10)
    mock_params = Dict{Symbol, Matrix{Float64}}()
    for (name, val) in param_names_and_values
        if val isa Vector
            mock_params[Symbol(name)] = reshape(val, 1, :)
        elseif val isa Matrix
            mock_params[Symbol(name)] = val
        else
            mock_params[Symbol(name)] = fill(Float64(val), 1, n_samples)
        end
    end
    return mock_params
end
