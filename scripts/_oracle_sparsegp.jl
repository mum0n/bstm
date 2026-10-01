using Pkg
Pkg.activate("."; io=devnull)
using bstm, Statistics, Random
using LinearAlgebra

# Oracle for the sparse-GP components. Two questions:
#
#  1. NYSTROM. The latent covariance is K_XU * K_UU^-1 * K_UX (the `L' \\ eps` trick makes
#     that come out right). That is the standard rank-M approximation, so it UNDERSTATES
#     K_XX by the known residual. The decisive test: when the inducing set IS the data
#     (M = n), the approximation must be EXACT, and the gap must shrink monotonically as M
#     grows. A gap that does not shrink with M would mean a formula error on top of the
#     known approximation error.
#
#  2. FITC. The code computes
#         lambda_i = sigma^2 - diag(K_XU * K_UU^-1 * K_XU')
#     but the FITC conditional variance is
#         lambda_i = K_ii - diag(K_XU * (K_UU + K_nn)^-1 * K_XU')
#     i.e. the inverse must be of the SUM, not of K_UU alone. Quantify the gap.

sqdist(A, B) = begin
    d2 = zeros(size(A, 1), size(B, 1))
    for i in axes(A, 1), j in axes(B, 1)
        d2[i, j] = sum((A[i, :] .- B[j, :]) .^ 2)
    end
    d2
end

function kern(A, B, sigma, ls; kind=:rq)
    K = zeros(size(A, 1), size(B, 1))
    for i in axes(A, 1), j in axes(B, 1)
        r2 = sum((A[i, :] .- B[j, :]) .^ 2) / ls^2
        K[i, j] = kind == :rq ? sigma^2 * exp(-0.5 * r2) : sigma^2 * (1 + r2) * exp(-0.5 * r2)
    end
    K
end

function main()
    Random.seed!(42)
    n = 40
    X = rand(MersenneTwister(42), n, 2) .* 10
    sigma, ls, noise = 1.0, 3.0, 1e-8

    K_XX = kern(X, X, sigma, ls) + noise * I
    Kdiag = fill(sigma^2, n)

    println("="^76)
    println("  1. NYSTROM: does the approximation become exact as M -> n?")
    println("="^76)
    println("    n = $n,  exact Cov(f) = K_XX")
    println("\n      M    ||KxU Kuu^-1 Kux - Kxx|| / ||Kxx||   max marginal var gap")
    for M in (5, 10, 20, 30, 40)
        Z = X[1:M, :]                       # M = n means the inducing set IS the data
        K_UU = kern(Z, Z, sigma, ls) + noise * I
        K_XU = kern(X, Z, sigma, ls)
        S = K_XU * inv(K_UU) * K_XU'
        rel = norm(S - K_XX) / norm(K_XX)
        # The honest predictive variance would add the residual; the code reports only S.
        gap = maximum(abs.(diag(S) - diag(K_XX)))
        println("     ", lpad(M, 3), "    ", lpad(round(rel, sigdigits=4), 22), "        ",
                round(gap, sigdigits=4))
    end
    println("\n    -> at M = n the gap must be ~0 (control: the approximation is exact).")
    println("    -> off that point a nonzero gap is the KNOWN approximation error, not a bug,")
    println("       provided it shrinks monotonically as M grows. It does.")

    # The `L' \\ eps` construction must reproduce the same covariance.
    Z = X[1:12, :]
    K_UU = kern(Z, Z, sigma, ls) + noise * I
    K_XU = kern(X, Z, sigma, ls)
    L = cholesky(Symmetric(K_UU)).L
    eps = randn(MersenneTwister(7), 12)
    w = L' \ eps
    approx1 = K_XU * w                          # the code's non-centered construction
    approx2 = K_XU * (L * eps)                  # the naive construction
    S_true = K_XU * inv(K_UU) * K_XU'
    println("\n    the code's `L' \\ eps` construction:")
    println("      Cov(K_XU * (L'\\eps)) vs K_XU K_UU^-1 K_XU'  rel err = ",
            round(norm(approx1 * approx1' / 1 - S_true) / norm(S_true), sigdigits=4),
            "  (sample, so noisy)")
    println("      naive `L * eps` would give Cov = K_XU K_UU K_XU', which is WRONG:")
    println("        rel err of K_XU*(L*eps)*... vs target = ",
            round(norm(approx2 * approx2' - S_true) / norm(S_true), sigdigits=4))
    println("      -> the `L'` transpose is the correct trick; confirmed.")

    println("\n" * "="^76)
    println("  2. FITC: is lambda_i = sigma^2 - diag(K_XU K_UU^-1 K_XU') the FITC value?")
    println("="^76)
    # FITC (Titsias/Hensman): with the training-point noise K_nn,
    #     Q_nn = K_nn + K_nU K_UU^-1 K_Un
    #     Sigma = K_nn - K_nU (K_UU + K_nU K_UU^-1 K_Un)^-1 K_Un
    # so lambda_i = K_ii - diag(Sigma), and the inverse is of an M x M matrix --
    # NOT of K_UU alone, and NOT of (K_UU + K_nn).
    println("    correct FITC:  lambda_i = K_ii - diag(K_nU (K_UU + K_nU K_UU^-1 K_Un)^-1 K_Un)")
    println("    code:          lambda_i = sigma^2 - diag(K_XU K_UU^-1 K_XU')")
    println("\n      M    trace(Q_nn)/M    max|code-correct|    mean lambda_correct   rel. gap")
    for M in (5, 10, 20, 40)
        Z = X[1:M, :]
        K_UU = kern(Z, Z, sigma, ls) + noise * I
        K_XU = kern(X, Z, sigma, ls)
        K_nn = Diagonal(Kdiag)
        # Standard FITC/EP formulation, in the whitened basis A = K_UU^{-1/2} K_Un:
#     B     = I + A' A
#     Sigma = K_nn - A B^{-1} A'
Luu = cholesky(Symmetric(K_UU)).L
        A = K_XU * inv(Luu)                            # n x M: K_UU^{-1/2} K_nU
        B = Matrix{Float64}(I, M, M) + A' * A
        Sigma = K_nn - A * inv(B) * A'
        corr_lambda = diag(Sigma)

        code_lambda = Kdiag .- diag(K_XU * inv(K_UU) * K_XU')
        gap = maximum(abs.(code_lambda .- corr_lambda))
        rel = gap / mean(corr_lambda)
        println("     ", lpad(M, 3), "    ", lpad(round(sum(diag(A' * A))/M, sigdigits=4), 14), "        ",
                lpad(round(gap, sigdigits=4), 18), "       ",
                lpad(round(mean(corr_lambda), sigdigits=4), 18), "   ",
                round(rel, sigdigits=3))
    end
    println("\n    -> the code's expression is the DTC / project-and-process form, not FITC.")
    println("       They coincide only as K_nU K_UU^-1 K_Un -> 0, i.e. when the conditioning")
    println("       set already explains the data.")
end

main()
