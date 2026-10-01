using Pkg
Pkg.activate("."; io=devnull)
using bstm, DataFrames, Turing, Random, Statistics, LinearAlgebra, SparseArrays

chain_graph(n, edges) = begin
    W = spzeros(Int, n, n)
    for (i, j) in edges
        W[i, j] = 1
        W[j, i] = 1
    end
    W
end

function main()
    # Disconnected: two components, sizes 2 and 3.
    W = chain_graph(5, [(1, 2), (3, 4), (4, 5)])
    t = bstm.build_structure_template(:icar, 5; W=W)

    println("n_components   = ", t.n_components)
    println("rank_deficiency= ", t.rank_deficiency)
    println("L              = ", round.(t.L; sigdigits=4))
    println("null_vectors   = size ", size(t.null_vectors))
    nv = Matrix(t.null_vectors)
    if size(nv, 1) == 5
        for c in axes(nv, 2)
            println("   null dir ", c, " = ", round.(nv[:, c]; digits=4))
        end
    end

    # THE decisive test. The GMRF sampling precision is built from the eigenbasis:
    #   Q = U * Diagonal(L) * U'
    # and diag_D zeroes the null eigenvalues. A direction is a genuine null direction of
    # the *model* iff Q * v == 0 for it. If deflation is right, EVERY column of
    # null_vectors must be annihilated by the deflated matrix -- not just the first.
    Q = Matrix(t.matrix)
    println("\nQ * null_vectors  (should be ~0 in every column)")
    Qn = Q * nv
    for c in axes(Qn, 2)
        println("   col ", c, ": max|.| = ", maximum(abs, Qn[:, c]))
    end

    # Reproduce what the model does: zero the null eigenvalues in the eigenbasis, rebuild.
    Lc = copy(t.L)
    dD = fill(1.0, length(Lc))
    zed = bstm._zero_null_modes!(dD, Lc)
    println("\n_zero_null_modes! zeroed indices = ", zed, "   (n_components = ", t.n_components, ")")
    dD .*= Lc                      # eigenvalues
    dD[zed] .= 0.0
    if hasproperty(t, :U)
        U = Matrix(t.U)
        Qd = U * Diagonal(dD) * U'
        Qdn = Qd * nv
        println("deflated Q * null_vectors")
        for c in axes(Qdn, 2)
            println("   col ", c, ": max|.| = ", maximum(abs, Qdn[:, c]))
        end
        # The field mean along each component must be ~0: the sum-to-zero constraint.
        println("\nper-component mean of the null directions (must be 0)")
        for c in axes(nv, 2)
            v = nv[:, c]
            for comp in [[1, 2], [3, 4, 5]]
                println("   dir ", c, " comp ", comp, ": mean = ",
                        round(mean(v[comp]); sigdigits=3))
            end
        end
    else
        println("template has no U field; fields = ", collect(keys(t)))
    end
end

main()