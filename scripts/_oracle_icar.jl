using Pkg
Pkg.activate("."; io=devnull)
using bstm, LinearAlgebra, Statistics, DataFrames, SparseArrays

# ICAR oracle. The GMRF prior is a proper GMRF only if the intrinsic
# sum-to-zero constraint is enforced correctly, and the spectral path enforces it by
# zeroing `diag_D[1]`. That is right ONLY if `U[:, 1]` spans the null space -- true for a
# connected graph (one zero eigenvalue, eigenvector = the constant vector), FALSE for a
# disconnected graph, which has as many zero eigenvalues as there are components.
#
# Reference: the intrinsic GMRF on the subspace orthogonal to 1. Removing the null space
# must leave a covariance with
#     (1) mean exactly 0 (the sum-to-zero constraint),
#     (2) Q' = -Q  (where Q is the pseudoinverse of the Laplacian),
#     (3) the same implied correlations as the analytic ICAR, on a connected graph.

function show_case(name, W, n; island_handling=:none)
    tmpl = bstm.build_structure_template(:icar, n; W=W, island_handling=island_handling)
    U, L = tmpl.U, tmpl.L
    Q = Matrix(tmpl.matrix)

    # Number of structurally zero eigenvalues (numerically).
    tol = 1e-8 * max(maximum(abs, L), 1.0)
    n_zero = count(<(tol), L)
    println("\n  $name  (n=$n)")
    println("    numerically-zero eigenvalues of L: $n_zero")
    println("    smallest 3 eigenvalues: ", round.(sort(L)[1:min(3, end)], sigdigits=5))
    println("    is U[:,1] the constant vector? ",
            isapprox(vec(U[:, 1]), fill(1 / sqrt(n), n); atol=1e-8))

    # Reproduce the model-side spectral field with sigma = 1.
    sigma = 1.0
    diag_D = sigma ./ sqrt.(L .+ 1e-9)
    diag_D[1] = 0.0                       # <-- the line under test
    C = U * Diagonal(diag_D .^ 2) * U'   # implied covariance

    # Property 1: the field must have mean exactly zero (sum-to-zero).
    field_mean = maximum(abs.(diag(C) .* 0 .+ vec(sum(C, dims=1)) ./ n))
    println("    implied mean of the field: ", field_mean)

    # Property 2: the covariance must be consistent with the pseudo-inverse of Q on the
    # non-null subspace, i.e. C ~= Qp * sigma^2 for a CONNECTED graph.
    Qp = pinv(Symmetric(Q))
    resid = norm(C - Qp) / norm(Qp)
    println("    ||C - pinv(Q)|| / ||pinv(Q)||: ", round(resid, sigdigits=4))

    # Property 3: variance lying ON THE NULL SPACE. U's columns are orthonormal, so the
    # variance contributed by eigen-direction i is exactly diag_D[i]^2 -- NOT a diagonal
    # entry of C. (An earlier version of this script summed C[i,i] over null eigenvalue
    # indices, which is meaningless: C's diagonal is not indexed by eigenvalue.)
    null_var = sum(diag_D[i]^2 for i in 1:n if L[i] < tol)
    tot_var = sum(diag_D .^ 2)
    println("    variance on the NULL space: ", round(null_var, sigdigits=5),
            "  of total ", round(tot_var, sigdigits=5),
            "  -> ", null_var / tot_var > 1e-6 ? "LEAK" : "ok")
    # And the headline: is the sum-to-zero constraint actually satisfied?
    println("    max |mean of the implied field| = ", round(field_mean, sigdigits=4))
    return n_zero, null_var / tot_var, field_mean
end

function main()
    println("="^74)
    println("  ICAR: does diag_D[1] = 0 enforce sum-to-zero on every graph?")
    println("="^74)

    # 1. Connected chain graph 1-2-3-4-5.
    n = 5
    W1 = spzeros(n, n)
    for i in 1:n-1
        W1[i, i+1] = 1.0; W1[i+1, i] = 1.0
    end
    r1 = show_case("connected chain", W1, n)

    # 2. Disconnected: two components {1,2} and {3,4,5}, no edge between them.
    n2 = 5
    W2 = spzeros(n2, n2)
    W2[1, 2] = 1.0; W2[2, 1] = 1.0
    W2[3, 4] = 1.0; W2[4, 3] = 1.0
    W2[4, 5] = 1.0; W2[5, 4] = 1.0
    r2 = show_case("TWO components", W2, n2)

    # 3. Three isolated units -- the degenerate but legal case.
    n3 = 3
    W3 = spzeros(n3, n3)
    r3 = show_case("three isolated units", W3, n3)

    # 4. Same disconnected graph, but asking for island handling.
    println("\n  --- with island_handling ---")
    for mode in (:none, :offset, :deflation, :all)
        try
            r = show_case("TWO components ($mode)", W2, n2; island_handling=mode)
        catch e
            println("\n  TWO components ($mode): ERROR ",
                    first(split(sprint(showerror, e), '\n')))
        end
    end

    println("\n" * "="^74)
    println("  Verdict")
    println("="^74)
    if r1[2] <= 1e-6
        println("  connected graph  : no null-space leak -> the constraint holds")
    else
        println("  connected graph  : LEAK")
    end
    for (nm, r) in [("2 components", r2), ("3 isolated", r3)]
        nzero, frac = r
        if nzero > 1
            print("  ", rpad(nm, 15), ": ", nzero, " zero eigenvalues, only index 1 zeroed")
            println(frac > 1e-6 ?
                    "  -> LEAK, " * string(round(frac * 100, sigdigits=4)) * "% of the field variance sits off the constraint" :
                    "  -> no leak")
        end
    end
end

main()
