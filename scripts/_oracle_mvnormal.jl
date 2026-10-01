using DataFrames, Random, Statistics, LinearAlgebra, Distributions

# Joint-log-density oracle for the multivariate Gaussian (M2 / M2a).
#
# Purpose: give the multivariate path a reference it can be tested against, so "is this the
# right density?" is a number rather than an opinion. It FIRES today; it is written to be
# silent once M2a wires MvNormalFamily into the codegen, which makes it the regression test
# for that fix.
#
#   julia --project=. scripts/_oracle_mvnormal.jl
#
# The comparison needs no fitting. Both densities are evaluated at the SAME, TRUE
# parameters, so the gap isolates model form from parameter estimation:
#
#   joint MVN(mu, D R D)   - what the package documents for a multivariate Gaussian
#   sum of marginals       - what the codegen actually emits
#                            (@addlogprob! per outcome, no joint term)
#
# These agree exactly when the residual correlation is zero. Any positive correlation
# separates them by the correlation term, which is the defect.

function case(; rho = 0.7, s1 = 0.8, s2 = 1.0, m1 = 2.0, m2 = 5.0, N = 500, seed = 99)
    Random.seed!(seed)
    R = [1.0 rho; rho 1.0]
    S = Diagonal([s1, s2]) * R * Diagonal([s1, s2])
    S = (S + S') / 2
    d = [rand(MvNormal([m1, m2], S)) for _ in 1:N]
    DataFrame(y = [x[1] for x in d], y2 = [x[2] for x in d])
end

function report(rho_true)
    df = case(rho = rho_true)
    N = nrow(df)
    mu1, mu2 = 2.0, 5.0
    s1, s2 = 0.8, 1.0
    R = [1.0 rho_true; rho_true 1.0]
    S = Diagonal([s1, s2]) * R * Diagonal([s1, s2])
    S = (S + S') / 2

    ref = sum(logpdf(MvNormal([mu1, mu2], S), [df.y[i], df.y2[i]]) for i in 1:N)
    fac = sum(logpdf(Normal(mu1, s1), df.y[i]) + logpdf(Normal(mu2, s2), df.y2[i])
              for i in 1:N)

    println("  true rho = ", rpad(round(rho_true, digits=3), 5),
            "   realised cor = ", round(cor(df.y, df.y2), digits=3))
    println("    joint MVN(mu, D R D)  = ", rpad(round(ref, digits=3), 12),
            "  <- what the package documents")
    println("    sum of marginals      = ", rpad(round(fac, digits=3), 12),
            "  <- what the codegen emits")
    println("    difference            = ", round(ref - fac, digits=3))
    if iszero(rho_true)
        ok = isapprox(ref, fac; atol = 1e-6)
        println("    expect 0 (factorising is exact at rho=0): ", ok ? "OK" : "FAIL")
    else
        fires = !isapprox(ref, fac; atol = 1e-6)
        println("    expect NONZERO (correlation term missing): ",
                fires ? "oracle fires - model is the factorised one" : "unexpectedly equal")
    end
    println()
end

println("=== multivariate joint-density oracle (M2 / M2a) ===")
for r in (0.0, 0.3, 0.6, 0.85)
    report(r)
end
println("At rho = 0 the two densities are identical, because a zero correlation makes the")
println("residual independent and factorising is exact. At rho > 0 they separate, and that")
println("separation is precisely the cross-outcome information the model discards.")
