using Pkg
Pkg.activate("."; io=devnull)
using bstm, DataFrames, Distributions, Turing

# `excess_cases = PAF * total_observed`. With no outcome column the old code used
# total = 1.0, so `excess_cases_mean` came out numerically IDENTICAL to `paf_mean` -- a
# dimensionless fraction under a name that says "cases". Check both paths.

function main()
    # Minimal fake model: `par_counterfactual` only needs `M.outcomes` off `model.args`.
    n = 200
    df = DataFrame(
        exposed = repeat([0.0, 1.0], inner=n ÷ 2),
        y = repeat([2.0, 5.0], inner=n ÷ 2),      # 1000 total observed
    )
    M = (outcomes = ["y"],)
    m = bstm.bstm_core("likelihood(y, family=poisson) ~ intercept() + exposed", df;
                       verbose=false)
    chain = sample(MersenneTwister(3), m, bstm.Turing.MH(), 40;
                   progress=false, check_model=false)

    println("  WITH the outcome column present")
    r = bstm.par_counterfactual(m, chain; exposure_var = "exposed", data = df)
    println("     paf_mean          = ", round(r.paf_mean, digits = 6))
    println("     y_total           = ", sum(df.y))
    println("     excess_cases_mean = ", round(r.excess_cases_mean, digits = 4))
    want = r.paf_mean * sum(df.y)
    println("     == paf_mean*total? ", isapprox(r.excess_cases_mean, want; rtol = 1e-12))
    println("     is a real count (not the fraction): ",
            !isapprox(r.excess_cases_mean, r.paf_mean; rtol = 1e-6))

    println("\n  WITHOUT the outcome column (renamed away)")
    df2 = select(df, :exposed)          # no `y`
    r2 = bstm.par_counterfactual(m, chain; exposure_var = "exposed", data = df2)
    println("     paf_mean               = ", round(r2.paf_mean, digits = 6),
            "   <- still estimable")
    println("     excess_cases_mean      = ", repr(r2.excess_cases_mean))
    println("     excess_cases_ci_lower  = ", repr(r2.excess_cases_ci_lower))
    println("     excess_cases_ci_upper  = ", repr(r2.excess_cases_ci_upper))
    println("     all nothing: ",
            isnothing(r2.excess_cases_mean) && isnothing(r2.excess_cases_ci_lower) &&
            isnothing(r2.excess_cases_ci_upper))
    println("     paf is NOT nothing:    ", !isnothing(r2.paf_mean))
    println("     raw draws still there: ", length(r2.raw_paf_samples) == 40)

    println("\n  the old behaviour, for contrast:")
    println("     old excess_cases_mean would have been paf_mean = ", round(r2.paf_mean, digits = 6),
            " -- identical to the fraction, i.e. not a count")
end

main()
