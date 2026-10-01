using Pkg
Pkg.activate("."; io=devnull)
using Distributions, Statistics, LogExpFunctions
const logistic = LogExpFunctions.logistic

# Does exp(beta) remain the MARGINAL risk ratio when a shared random effect is present?
# u ~ N(0, sigma^2) is SHARED by exposed and unexposed within a spatial unit.
function main()
    beta, eta0, sigma = 0.4, -0.7, 0.9
    us = rand(Normal(0.0, sigma), 4_000_000)

    # Log link: rate = exp(eta + u)
    r0 = exp.(eta0 .+ us)
    r1 = exp.(eta0 .+ beta .+ us)
    margRR_log = mean(r1) / mean(r0)
    println("  LOG link")
    println("    exp(beta)            = ", round(exp(beta), sigdigits=6))
    println("    marginal RR (MC)     = ", round(margRR_log, sigdigits=6))
    println("    exp(beta)*exp(s^2/2) = ", round(exp(beta + sigma^2/2), sigdigits=6))
    println("    -> exp(beta+sigma^2/2) is NOT the RR" * (isapprox(margRR_log, exp(beta); rtol=0.02) ? "  (cancels)" : ""))

    # Logit link: p = logistic(eta + u)
    p0 = logistic.(eta0 .+ us)
    p1 = logistic.(eta0 .+ beta .+ us)
    margRR_logit = mean(p1) / mean(p0)
    condRR_logit = logistic(eta0 + beta) / logistic(eta0)
    println("  LOGIT link")
    println("    conditional RR (code) = ", round(condRR_logit, sigdigits=6))
    println("    marginal RR (MC)      = ", round(margRR_logit, sigdigits=6))
    println("    relative error of code = ",
            round(abs(condRR_logit - margRR_logit) / margRR_logit, sigdigits=3))

    # And the baseline: reference-individual vs population-average risk.
    println("  BASELINE (logit link, reference_individual = logistic(eta0))")
    println("    logistic(eta0)        = ", round(logistic(eta0), sigdigits=6))
    println("    population mean p0    = ", round(mean(p0), sigdigits=6))
    println("    relative error        = ",
            round(abs(logistic(eta0) - mean(p0)) / mean(p0), sigdigits=3))
    println("  exp(beta+sigma^2/2) (log link, pop-mean RATE) = ",
            round(exp(eta0 + beta + sigma^2/2), sigdigits=6), " vs code exp(eta0+beta) = ",
            round(exp(eta0 + beta), sigdigits=6))
end

main()
