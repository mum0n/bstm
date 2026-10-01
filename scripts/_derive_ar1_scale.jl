using LinearAlgebra

# Re-derive the AR(1) prior scaling from the generative definition, because the first
# attempt at the fix inverted the factor. The correct check is the diagonal of the implied
# covariance, which the conditional (Dirichlet) AR(1) prior pins exactly:
#   phi_1 ~ N(0, sigma^2/(1-rho^2))  =>  Var(phi_1) = sigma^2/(1-rho^2).

function main()
    n, s2 = 12, 1.0
    println("  rho   Var(phi_1) implied   Var(phi_1) implied   agreement")
    println("        Q = Qb*(1-r2)/s2     Q = Qb/(s2*(1-r2))")
    for r in (0.0, 0.3, 0.6, 0.9)
        r2 = r * r
        Qb = Matrix(Tridiagonal(fill(-r, n - 1), fill(1 + r2, n), fill(-r, n - 1)))
        correct = inv(Qb * (1 - r2) / s2)
        ascode = inv(Qb / (s2 * (1 - r2)))
        println("  ", r, "    ", round(correct[1,1], sigdigits=8),
                "              ", round(ascode[1,1], sigdigits=8),
                "           ", isapprox(correct[1,1], ascode[1,1]; rtol=1e-10))
        println("        target sigma^2/(1-rho^2) = ", round(s2/(1-r2), sigdigits=8))
    end
    println("\n  Conclusion:")
    println("  Q = Q_base * (1-rho^2)/sigma^2  reproduces Var(phi_1) = sigma^2/(1-rho^2).")
    println("  Q = Q_base / (sigma^2*(1-rho^2)) is the RECIPROCAL and does not.")
    println("  => the spectral field SD is sigma / sqrt((1-rho^2) * lambda), i.e. the")
    println("     sqrt(1-rho^2) goes in the DENOMINATOR, not the numerator.")
end

main()
