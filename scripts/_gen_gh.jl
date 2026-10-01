using LinearAlgebra, Printf, Statistics, Random

# Golub-Welsch: nodes/weights for the PHYSICISTS' Hermite rule
#   int f(x) exp(-x^2) dx  ~=  sum_i w_i f(x_i)
# The Jacobi matrix is the symmetric tridiagonal with off-diagonal sqrt(i/2), i=1..n-1.
# Its eigenpairs give the nodes (eigenvalues) and weights (mu1 * v[1,:]^2).
function gauss_hermite_physicists(n::Int)
    J = zeros(Float64, n, n)
    for i in 1:n-1
        J[i, i+1] = sqrt(i / 2)
        J[i+1, i] = sqrt(i / 2)
    end
    E = eigen(Symmetric(J))
    x = E.values
    w = E.vectors[1, :].^2 .* sqrt(pi)
    return x, w
end

const N = 32
x, w = gauss_hermite_physicists(N)

# Verify against exact moments of the physicists' weight exp(-x^2).
# int x^{2k} exp(-x^2) dx = Gamma(k + 1/2)
println("Gauss-Hermite (physicists'), n = ", N)
@printf("  sum(w)          = %.15f   exact sqrt(pi) = %.15f  err=%.2e\n",
        sum(w), sqrt(pi), abs(sum(w) - sqrt(pi)))
@printf("  sum(w x^2)      = %.15f   exact sqrt(pi)/2 = %.15f  err=%.2e\n",
        sum(w .* x.^2), sqrt(pi)/2, abs(sum(w .* x.^2) - sqrt(pi)/2))
@printf("  sum(w x^4)      = %.15f   exact 3sqrt(pi)/4 = %.15f  err=%.2e\n",
        sum(w .* x.^4), 3 * sqrt(pi)/4, abs(sum(w .* x.^4) - 3 * sqrt(pi)/4))
@printf("  sum(w x^6)      = %.15f   exact 15sqrt(pi)/8 = %.15f  err=%.2e\n",
        sum(w .* x.^6), 15 * sqrt(pi)/8, abs(sum(w .* x.^6) - 15 * sqrt(pi)/8))

# The quantity we actually need: E[g(Z)], Z ~ N(0,1), equals (1/sqrt(pi)) sum w_i g(sqrt2 x_i).
logistic(z) = 1 / (1 + exp(-z))
for sig in (0.0, 0.5, 0.9, 1.5, 2.0)
    eta = -0.7
    gh = (1 / sqrt(pi)) * sum(w .* logistic.(eta .+ sig .* sqrt(2) .* x))
    # Monte-Carlo cross-check with a fixed seed.
    mc = mean(logistic.(eta .+ sig .* randn(MersenneTwister(2024), 4_000_000)))
    @printf("  sigma=%.1f  GH=%.8f  MC=%.8f  reldiff=%.2e\n", sig, gh, mc, abs(gh - mc)/mc)
end

# Emit the table as a Julia literal, rounded to 17 significant digits.
println("\nconst _GH_NODES = Float64[")
for i in 1:2:N
    i == N && (i = N)
    lo, hi = i, min(i + 1, N)
    parts = [repr(round(x[j], sigdigits=17)) for j in lo:hi]
    println("    ", join(parts, ", "), ",")
end
println("]")
println("const _GH_WEIGHTS = Float64[")
for i in 1:2:N
    lo, hi = i, min(i + 1, N)
    parts = [repr(round(w[j], sigdigits=17)) for j in lo:hi]
    println("    ", join(parts, ", "), ",")
end
println("]")
