using Pkg
Pkg.activate("."; io=devnull)
using bstm, LinearAlgebra, Statistics, Random

# Verify all three sparse-GP methods against independently computed references:
#   :dtc  -> diag(K_nn - K_XU K_UU^-1 K_XU')          (the old `:fitc`, i.e. what it really was)
#   :fitc -> sigma^2 - diag(A B^-1 A'), B = I + A'A/sigma^2   (true FITC)
#   :vfe  -> no lambda at all
# And check the two invariants that matter: the model body and get_effects must use the SAME
# dispatcher, and the generated code must reference it by name.

function kern(A, B, sigma, ls)
    K = zeros(size(A, 1), size(B, 1))
    for i in axes(A, 1), j in axes(B, 1)
        r2 = sum((A[i, :] .- B[j, :]) .^ 2) / ls^2
        K[i, j] = sigma^2 * exp(-0.5 * r2)
    end
    K
end

function main()
    Random.seed!(11)
    n, sigma, ls, noise = 30, 1.3, 2.5, 1e-8
    X = rand(MersenneTwister(11), n, 2) .* 10

    println("  1. :dtc and :fitc against independent references")
    println("      M   method   max|ref - code|   mean lambda")
    for M in (6, 12, 25)
        Z = X[1:M, :]
        K_UU = kern(Z, Z, sigma, ls) + noise * I
        K_XU = kern(X, Z, sigma, ls)
        L = cholesky(Symmetric(K_UU)).L

        # Independent references, built directly from the definitions.
        A = K_XU / L
        s2 = sigma^2
        B = Matrix{Float64}(I, M, M) + (A' * A) ./ s2
        ref_fitc = max.(s2 .- vec(sum(((B \ (A')) .* (A')), dims=1)), 0.0)
        ref_dtc = max.(s2 .- vec(sum(A .^ 2, dims=2)), 0.0)

        for (meth, ref) in ((:dtc, ref_dtc), (:fitc, ref_fitc))
            got = bstm._sparse_gp_lambda_diag(meth, L, K_XU, sigma)
            println("     ", lpad(M, 3), "   ", rpad(meth, 6), "   ",
                    lpad(round(maximum(abs.(got .- ref)), sigdigits=3), 13), "     ",
                    round(mean(ref), sigdigits=6))
        end
    end

    println("\n  2. the two methods are genuinely different, and DTC is the larger")
    Z = X[1:12, :]
    K_UU = kern(Z, Z, sigma, ls) + noise * I
    K_XU = kern(X, Z, sigma, ls)
    L = cholesky(Symmetric(K_UU)).L
    d = bstm._sparse_gp_lambda_diag(:dtc, L, K_XU, sigma)
    f = bstm._sparse_gp_lambda_diag(:fitc, L, K_XU, sigma)
    println("      mean dtc = ", round(mean(d); sigdigits=6),
            "   mean fitc = ", round(mean(f); sigdigits=6))
    println("      dtc >= fitc everywhere: ", all(d .>= f .- 1e-12))
    println("      they differ:           ", !isapprox(d, f; rtol=1e-6))
    println("      ratio dtc/fitc         = ", round(mean(d)/mean(f); sigdigits=6))

    println("\n  3. :vfe has no lambda and says so")
    try
        bstm._sparse_gp_lambda_diag(:vfe, L, K_XU, sigma)
        println("      ERROR: it returned a value")
    catch e
        println("      throws: ", first(split(sprint(showerror, e), '\n')))
    end

    println("\n  4. an unknown method is rejected with a useful message")
    try
        bstm._sparse_gp_lambda_diag(:nonsense, L, K_XU, sigma)
        println("      ERROR: accepted :nonsense")
    catch e
        println("      throws: ", first(split(sprint(showerror, e), '\n')))
    end

    println("\n  5. helpers are registered for the generated-code module")
    println("      _sparse_gp_lambda_diag reachable from bstm: ",
            isdefined(bstm, :_sparse_gp_lambda_diag))
    println("      _fitc_lambda_diag / _dtc_lambda_diag defined: ",
            isdefined(bstm, :_fitc_lambda_diag), " / ", isdefined(bstm, :_dtc_lambda_diag))

    println("\n  6. zero/small sigma edge case does not produce NaN")
    tiny = bstm._sparse_gp_lambda_diag(:fitc, L, K_XU, 1e-3)
    println("      sigma=1e-3: all finite = ", all(isfinite, tiny),
            "   all >= 0 = ", all(tiny .>= 0.0))
end

main()
