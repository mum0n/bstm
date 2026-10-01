using Pkg
Pkg.activate("."; io=devnull)
using bstm, DataFrames, Turing, Distributions, ForwardDiff, LinearAlgebra, SpecialFunctions

# N1: "hurdle Poisson cannot be sampled with NUTS -- _gammalogcdf has no ForwardDiff method".
# The claim is plausible but the diagnosis was read off a stack trace, not executed. Verify
# (a) that the failure reproduces, (b) which function actually throws, and (c) whether a
# dual-number call to logccdf fails on its own.

function main()
    println("  Distributions version: ", pkgversion(Distributions))

    # 1. Does the advertised failure reproduce?
    s_N = 6
    df = DataFrame(y = [1, 3, 2, 0, 4, 1, 2, 5, 1, 0, 3, 2],
                   s_idx = repeat(1:s_N, 2))
    W = bstm.spatial_knn_graph([(Float64(s), 0.0) for s in 1:s_N], 2)[2]
    m = bstm.bstm_core("likelihood(y, family=poisson, hurdle=0.5) ~ 1 + random(s_idx, model=icar)",
                       df; s_N=s_N, W=W, verbose=false)
    println("\n  1. NUTS on the hurdle Poisson model")
    try
        sample(m, NUTS(), 30; progress=false, check_model=false)
        println("     NUTS SUCCEEDED -- N1 does not reproduce")
    catch e
        println("     NUTS FAILED: ", first(split(sprint(showerror, e), '\n')))
        st = catch_backtrace()
        frames = stacktrace(st)
        for fr in frames[1:min(6, length(frames))]
            f = string(fr.file)
            println("       at ", basename(f), ":", fr.line, "  ", fr.func)
        end
    end

    # 2. Is plain logccdf AD-capable on its own?
    println("\n  2. logccdf(Poisson, x) with a Dual argument")
    lam = 2.5
    d = ForwardDiff.Dual(lam, 1.0)
    try
        v = logccdf(Poisson(lam), 3)
        println("     Float64 arg  logccdf = ", v)
    catch e
        println("     Float64 arg  FAILED: ", first(split(sprint(showerror, e), '\n')))
    end
    try
        v = logccdf(Poisson(d), 3)
        g = ForwardDiff.derivative(x -> logccdf(Poisson(x), 3), lam)
        println("     Dual arg     logccdf = ", v, "  (numeric deriv = ", g, ")")
        println("     dual value matches Float64: ", isapprox(ForwardDiff.value(v), logccdf(Poisson(lam), 3)))
    catch e
        println("     Dual arg     FAILED: ", first(split(sprint(showerror, e), '\n')))
    end

    # 3. Is the *hurdle* the problem, or is Poisson logccdf generally not AD-capable?
    println("\n  3. Non-hurdle Poisson: can NUTS run?")
    m2 = bstm.bstm_core("likelihood(y, family=poisson) ~ 1 + random(s_idx, model=icar)",
                        df; s_N=s_N, W=W, verbose=false)
    try
        sample(m2, NUTS(), 30; progress=false, check_model=false)
        println("     NUTS on plain Poisson SUCCEEDED")
    catch e
        println("     NUTS on plain Poisson FAILED: ", first(split(sprint(showerror, e), '\n')))
    end

    # 4. Which pieces are AD-capable?
    println("\n  4. component AD-capability at x = ", lam)
    for (nm, f) in [("logccdf", () -> logccdf(Poisson(d), 3)),
                    ("logpdf",   () -> logpdf(Poisson(d), 3)),
                    ("logcdf",   () -> logcdf(Poisson(d), 3)),
                    ("logccdf(0)", () -> logccdf(Poisson(d), 0)),
                    ("logccdf(1)", () -> logccdf(Poisson(d), 1))]
        try
            r = f()
            println("     ", rpad(nm, 12), " ok  = ", r)
        catch e
            println("     ", rpad(nm, 12), " FAIL: ", first(split(sprint(showerror, e), '\n')))
        end
    end
end

main()
