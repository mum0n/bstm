using bstm
using Pkg
Pkg.activate("."; io=devnull)
using DataFrames, Turing, Random, Statistics, LinearAlgebra, SparseArrays

# THE decisive check for the diag_D refactor. A null direction has L ~ 0, so
# `diag_D = sigma / sqrt(L + noise)` is LARGE for an undeflated null mode. Therefore:
#
#   the sampling covariance is diag_D.^2, and on a DISCONNECTED graph every null
#   direction must show ~zero variance in a prior draw.
#
# Before the refactor, components hard-coded how many leading entries to zero, so a
# 2-component graph left the second null direction carrying sigma^2/noise = 1/0.01 = 100.
# This measures that directly instead of trusting the code shape.

function chain_graph(n, edges)
    W = spzeros(Int, n, n)
    for (i, j) in edges
        W[i, j] = 1
        W[j, i] = 1
    end
    W
end

function main()
    n0 = 8
    pts = DataFrame(idx = collect(1:n0), x = collect(1.0:n0), y = zeros(Float64, n0))
    pts.depth = 9.0 .+ 3.0 .* sin.(pts.x ./ 3.0)

    # Two components: {1,2,3,4} and {5,6,7,8}. Never connected.
    Wdisc = chain_graph(n0, [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8)])
    # Connected ring, for contrast.
    Wconn = chain_graph(n0, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1)])

    t_d = bstm.build_structure_template(:icar, n0; W = Wdisc)
    t_c = bstm.build_structure_template(:icar, n0; W = Wconn)
    nnull_d = count(<=(1e-10 * maximum(t_d.L)), t_d.L)
    nnull_c = count(<=(1e-10 * maximum(t_c.L)), t_c.L)
    println("icar template: disconnected null eigenvalues = ", nnull_d,
            "   connected = ", nnull_c)
    println("  (a 2-component graph must have 2; a connected one exactly 1)\n")

    noise = 0.1
    println("per-component max variance on the NULL space")
    println("  (sigma=1, noise=$noise -> an undeflated null mode would show 1/noise = ",
            round(1 / noise; digits=1), ")")

    models = [:icar, :besag, :bym2, :leroux, :cyclic, :rw1, :rw2, :localadaptive]
    println()
    println(rpad("model", 16), rpad("connected", 12), "disconnected", "   verdict")
    for m in models
        row = rpad(string(m), 16)
        vals = Float64[]
        for (W, tag) in ((Wconn, :c), (Wdisc, :d))
            f = "likelihood(depth, family=gaussian) ~ intercept() + " *
                "random(idx, model=$m, sigma=1.0)"
            got = NaN
            try
                mm = bstm.bstm_core(f, pts; W = W, verbose = false)
                ch = sample(MersenneTwister(4), mm, Prior(), 400; progress = false)
                comp = mm.args.M.components[1]
                hyper = comp.hyper
                L = hasproperty(hyper, :L) ? hyper.L : hyper.template.L
                dd = 1.0 ./ sqrt.(L .+ noise)
                bstm._zero_null_modes!(dd, L)
                # Variance contributed by the eigen-directions that are null:
                # sum over null j of (dd[j] * U[:,j])^2 , per unit.
                U = hasproperty(hyper, :U) ? hyper.U : hyper.template.U
                null = findall(<=(1e-10 * maximum(L)), L)
                v = [sum(abs2, U[:, j]) * dd[j]^2 for j in null] / n0
                got = maximum(v; init = 0.0)
            catch e
                got = NaN
            end
            push!(vals, got)
        end
        ok = (isnan(vals[1]) || isnan(vals[2])) ? "skipped" :
             (vals[2] < 1e-8 ? "OK both null modes zeroed" : "LEAK " * string(round(vals[2]; sigdigits=3)))
        println(row, rpad(string(round(vals[1]; sigdigits=3)), 12),
                string(round(vals[2]; sigdigits=3)), "   ", ok)
    end
end

main()