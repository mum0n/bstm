using Pkg
Pkg.activate("."; io=devnull)
using bstm, Distributions, Statistics, LogExpFunctions, Random, DataFrames
const logistic = LogExpFunctions.logistic

# Verify the new marginalization against Monte Carlo, using a SHARED field u (the case the
# conditional value gets wrong). Exposes the internals, so this checks the maths rather than
# just the plumbing.
function main()
    Random.seed!(7)
    beta, eta0, sigma = 0.4, -0.7, 0.9
    s2 = sigma^2

    println("  GH table sanity")
    println("    sum(w) - sqrt(pi)      = ", abs(sum(bstm._GH_WEIGHTS) - sqrt(pi)))
    println("    sum(w x^2) - sqrt(pi)/2= ", abs(sum(bstm._GH_WEIGHTS .* bstm._GH_NODES.^2) - sqrt(pi)/2))

    us = randn(MersenneTwister(99), 6_000_000) .* sigma

    # --- LOG link baseline: the 50% error that motivated the fix.
    mc_rate = mean(exp.(eta0 .+ us))
    gh_rate = bstm._logistic_normal_mean # unused for log
    pop_rate = exp(eta0 + s2/2)
    ref_rate = exp(eta0)
    println("\n  LOG link BASELINE (population-mean rate)")
    println("    reference individual  = ", round(ref_rate, sigdigits=6))
    println("    closed form exp(e+s2/2)= ", round(pop_rate, sigdigits=6))
    println("    Monte Carlo           = ", round(mc_rate, sigdigits=6))
    println("    GH/closed vs MC reldiff= ", round(abs(pop_rate - mc_rate)/mc_rate, sigdigits=3))
    println("    reference error       = ", round(abs(ref_rate - mc_rate)/mc_rate, sigdigits=3),
            "  (the bug)")

    # --- LOGIT link baseline via quadrature.
    mc_p0 = mean(logistic.(eta0 .+ us))
    gh_p0 = bstm._logistic_normal_mean(eta0, sigma)
    println("\n  LOGIT link BASELINE (population-mean p0)")
    println("    reference individual  = ", round(logistic(eta0), sigdigits=6))
    println("    Gauss-Hermite         = ", round(gh_p0, sigdigits=6))
    println("    Monte Carlo           = ", round(mc_p0, sigdigits=6))
    println("    GH vs MC reldiff      = ", round(abs(gh_p0 - mc_p0)/mc_p0, sigdigits=3))

    # --- LOGIT link marginal RR.
    mc_rr = mean(logistic.(eta0 .+ beta .+ us)) / mean(logistic.(eta0 .+ us))
    cond_rr = logistic(eta0 + beta) / logistic(eta0)
    marg_rr = bstm._marginal_logit_rr(beta, eta0, s2)
    println("\n  LOGIT link RR")
    println("    conditional (old)     = ", round(cond_rr, sigdigits=6))
    println("    marginal (new)        = ", round(marg_rr, sigdigits=6))
    println("    Monte Carlo           = ", round(mc_rr, sigdigits=6))
    println("    new vs MC reldiff     = ", round(abs(marg_rr - mc_rr)/mc_rr, sigdigits=3))
    println("    old vs MC reldiff     = ", round(abs(cond_rr - mc_rr)/mc_rr, sigdigits=3),
            "  (the bug)")

    # --- sigma -> 0 must recover the conditional value EXACTLY.
    println("\n  sigma -> 0 limit (must equal the conditional value)")
    lim = bstm._marginal_logit_rr(beta, eta0, 0.0)
    println("    marginal at var=0     = ", round(lim, sigdigits=10))
    println("    conditional           = ", round(cond_rr, sigdigits=10))
    println("    agrees                = ", isapprox(lim, cond_rr; rtol=1e-12))

    # --- End-to-end through the public API, with a synthetic chain carrying sigma_<key>.
    n = 500
    ch = Dict(
        :intercept     => reshape(fill(eta0, n), 1, n),
        :beta_exposure => reshape(fill(beta, n), 1, n),
        :sigma_region  => reshape(fill(sigma, n), 1, n),
    )
    df = DataFrame(exposed = repeat([0.0, 1.0], inner=2), y = [1, 1, 0, 0])

    r_cond = par_from_posterior(ch, "exposure"; family="binomial", exposure_var="exposed",
                                exposure_prevalence=0.5, data=df)
    r_pop  = par_from_posterior(ch, "exposure"; family="binomial", exposure_var="exposed",
                                exposure_prevalence=0.5, data=df, population_average=true)
    println("\n  END-TO-END (binomial, population_average)")
    println("    baseline conditional = ", round(r_cond.baseline_risk, sigdigits=6))
    println("    baseline population  = ", round(r_pop.baseline_risk, sigdigits=6))
    println("    MC population p0     = ", round(mc_p0, sigdigits=6))
    println("    RR conditional       = ", round(r_cond.rr_mean, sigdigits=6))
    println("    RR marginal          = ", round(r_pop.rr_mean, sigdigits=6))
    println("    MC marginal RR       = ", round(mc_rr, sigdigits=6))
    println("    default unchanged    = ", isapprox(r_cond.baseline_risk, logistic(eta0); rtol=1e-9))
    println("    PAR scales with I_0  = ", round(r_pop.par_mean / r_cond.par_mean, sigdigits=4),
            " (expect ratio of baselines)")
end

main()
