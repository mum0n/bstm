using Pkg
Pkg.activate("."; io=devnull)
using bstm, LinearAlgebra, Statistics, Distributions, DataFrames

# Two claims the tracker records as never tested:
#   (A) AR(1) spectral eigenvalues -- "reported CRITICAL, then argued correct three times".
#   (B) reshard_spatial_field's `sqrt(P * (std.^2))` -- "retracted earlier as untested".
#
# Both are checked here against the OTHER two AR(1) code paths, which are independent
# constructions of the same model. If :spectral disagrees with :centered / :marginalized,
# then the default path is fitting a different model than the alternatives.

function main()
    println("="^78)
    println("  (A) AR(1) spectral vs centered vs marginalized")
    println("="^78)
    for (rho, sigma) in [(0.0, 1.0), (0.3, 1.0), (0.6, 1.0), (0.9, 1.0)]
        n = 12
        noise = 1e-9
        # Ground truth: the dense stationary AR(1) covariance used by the :centered path.
        K_centered = bstm._ar1_covariance_matrix(rho, sigma, n, noise)

        # What the :spectral path implies, rebuilt from its own formula. This must track
        # the library's `sigma_stable_scale` -- an earlier version of this script hardcoded
        # the pre-fix `sigma ./ sqrt(...)` and so could not detect the fix it was written
        # to check.
        tmpl = bstm.build_structure_template(:ar1, n)
        L_base = tmpl.L
        lambda_vals = (1.0 + rho^2) .+ rho .* L_base
        D_spectral = bstm.sigma_stable_scale(sigma, rho, Float64) ./ sqrt.(lambda_vals .+ noise)
        K_spectral = tmpl.U * Diagonal(D_spectral.^2) * tmpl.U'

        # What :marginalized implies. NOTE: `_ar1_precision_matrix` has NO CALLERS -- the
        # real `:marginalized` method uses `_ar1_log_marginal_likelihood`. So this row is a
        # check on a dead helper, not on a live path. An earlier version of this script
        # presented it as the live `:marginalized` model and made it look like the outlier.
        Q_marg = bstm._ar1_precision_matrix(rho, sigma, n, noise)
        K_marg = inv(Matrix(Q_marg))

        rel(c) = norm(c - K_centered) / norm(K_centered)
        relm(c) = norm(c - K_marg) / norm(K_marg)
        println("\n  rho=", rho, " sigma=", sigma)
        # The decisive check is the implied marginal variance: the stationary AR(1) prior
        # pins Var(phi_1) = sigma^2/(1-rho^2) exactly, so a spectral path that reproduces
        # that has the right scale factor, whatever its boundary treatment.
        println("    spectral  vs centered    rel err = ", round(rel(K_spectral), sigdigits=3))
        println("    implied Var(phi_1): centered  = ", round(K_centered[1,1], sigdigits=8),
                "   spectral = ", round(K_spectral[1,1], sigdigits=8),
                "   target sigma^2/(1-rho^2) = ", round(sigma^2/(1-rho^2), sigdigits=8))
        println("    -> spectral reproduces the stationary variance: ",
                isapprox(K_spectral[1,1], sigma^2/(1-rho^2); rtol=0.02))
        println("    (dead helper) _ar1_precision_matrix implied Var = ",
                round(K_marg[1,1], sigdigits=8), " -- no callers; not a live path")
    end

    println("\n" * "="^78)
    println("  (B) reshard_spatial_field summary-mode SD")
    println("="^78)

    # A fine grid of n_fine independent cells, aggregated into k coarse cells by a
    # row-stochastic P (each row = mean of its k members). This is the reshard case.
    k = 4
    n_fine = 4k
    n_coarse = 4
    P = zeros(n_coarse, n_fine)
    for c in 1:n_coarse
        P[c, (c-1)*k+1 : c*k] .= 1/k
    end
    sd_fine = fill(1.0, n_fine)          # every fine cell has SD 1
    fine = (mean=fill(1.0, n_fine), std=copy(sd_fine))

    got = bstm.reshard_spatial_field(P, fine)
    # Truth: the coarse value is the MEAN of k independent SD-1 cells, so its SD is 1/sqrt(k).
    truth = fill(1/sqrt(k), n_coarse)
    println("\n  k = ", k, " fine cells per coarse cell, each with SD 1")
    println("    code   sd_resharded = ", round.(got.std, sigdigits=6))
    println("    truth  (mean of k iid) = ", round.(truth, sigdigits=6))
    println("    ratio code/truth = ", round(got.std[1] / truth[1], sigdigits=6),
            "   (sqrt(k) = ", round(sqrt(k), sigdigits=4), ")")
    println("    overstates by ", round(got.std[1]/truth[1] - 1, sigdigits=4) * 100, "%")

    # And confirm the mathematically right formula.
    right = sqrt.(diag(P * Diagonal(sd_fine .^ 2) * P'))
    println("    P*diag(var)*P' diag = ", round.(right, sigdigits=6),
            "  matches truth: ", isapprox(right, truth; rtol=1e-12))

    # Correlated case: the diagonal-only formula ignores cross-covariances entirely.
    println("\n  With perfectly correlated fine cells, true coarse SD is 1.0, not 1/sqrt(k)")
    got2 = bstm.reshard_spatial_field(P, (mean=fill(1.0, n_fine), std=fill(1.0, n_fine)))
    println("    code gives ", round(got2.std[1], sigdigits=6),
            " -- cannot represent this case at all (no covariance is used)")
end

main()
