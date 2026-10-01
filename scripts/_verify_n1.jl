using Pkg
Pkg.activate("."; io=devnull)
using bstm, DataFrames, Turing, Distributions, ForwardDiff, Statistics, Random, LinearAlgebra

# N1 verification. Two independent things must hold:
#   (1) NUTS now samples the hurdle Poisson model (it previously threw MethodError).
#   (2) the AD path agrees with the Float64 path, so the fix is not merely "no longer
#       throws" -- it must be numerically the same quantity.

function main()
    s_N = 6
    df = DataFrame(y = [1, 3, 2, 0, 4, 1, 2, 5, 1, 0, 3, 2], s_idx = repeat(1:s_N, 2))
    W = bstm.spatial_knn_graph([(Float64(s), 0.0) for s in 1:s_N], 2)[2]

    # 1. NUTS on the hurdle model.
    println("  1. NUTS on hurdle Poisson")
    for (nm, hurdle) in [("hurdle=0.5", 0.5), ("hurdle=1.0", 1.0), ("hurdle=2.0", 2.0)]
        f = "likelihood(y, family=poisson, hurdle=$hurdle) ~ 1 + random(s_idx, model=icar)"
        m = bstm.bstm_core(f, df; s_N=s_N, W=W, verbose=false)
        try
            ch = sample(MersenneTwister(5), m, NUTS(), 60; progress=false, check_model=false)
            println("     ", rpad(nm, 11), " OK  (", size(ch, 1), " draws)")
        catch e
            println("     ", rpad(nm, 11), " FAIL: ", first(split(sprint(showerror, e), '\n')))
        end
    end

    # 2. AD path vs Float64 path must agree.
    println("\n  2. AD path agrees with the Float64 path")
    for (fam, thr) in [("poisson", 0.5), ("poisson", 1.0), ("poisson", 2.0), ("negbin", 1.0)]
        d = bstm.bstm_Likelihood(fam, 1.3; phi_hurdle = 0.3, hurdle = thr)
        dist = bstm.get_dist_ref(bstm.get_model_family(fam), d, 1.3, 1.0, 1)
        float_way = bstm.logccdf(dist, thr)
        ad_way = bstm._logccdf_count_ad(dist, thr)
        lf = bstm.logcdf(dist, thr)
        la = bstm._logcdf_count_ad(dist, thr)
        println("     ", rpad(fam, 8), " thr=", thr,
                "  logccdf float=", round(float_way, sigdigits=10),
                " ad=", round(ad_way, sigdigits=10),
                "  match=", isapprox(float_way, ad_way; rtol=1e-12),
                " | logcdf match=", isapprox(lf, la; rtol=1e-12))
    end

    # 3. Gradients must be finite and match finite differences -- an AD path that returns
    #    a plausible value but a wrong derivative would break NUTS silently.
    println("\n  3. AD gradients vs finite differences (Poisson, hurdle branch)")
    f(lam) = bstm._logccdf_count_ad(bstm.get_dist_ref(bstm.get_model_family("poisson"),
                                                      bstm.bstm_Likelihood("poisson", lam), lam, 1.0, 1), 1.0)
    for lam in (0.5, 1.5, 4.0)
        g = ForwardDiff.derivative(f, lam)
        h = 1e-6
        fd = (f(lam + h) - f(lam - h)) / (2h)
        println("     lam=", lam, "  AD=", round(g, sigdigits=8), "  FD=", round(fd, sigdigits=8),
                "  match=", isapprox(g, fd; rtol=1e-5), "  finite=", isfinite(g))
    end

    # 4. Censored variants also route through cdf and must work under AD.
    println("\n  4. Censored variants under NUTS (all route through cdf)")
    for (nm, f) in [
        ("right-censored", "likelihood(y, family=poisson, hurdle=0.5, censor_lower=2) ~ 1 + random(s_idx, model=icar)"),
        ("left-censored",  "likelihood(y, family=poisson, hurdle=0.5, censor_upper=2) ~ 1 + random(s_idx, model=icar)"),
        ("interval",       "likelihood(y, family=poisson, hurdle=0.5, censor_lower=1, censor_upper=3) ~ 1 + random(s_idx, model=icar)"),
    ]
        m = bstm.bstm_core(f, df; s_N=s_N, W=W, verbose=false)
        try
            sample(MersenneTwister(5), m, NUTS(), 60; progress=false, check_model=false)
            println("     ", rpad(nm, 15), " OK")
        catch e
            println("     ", rpad(nm, 15), " FAIL: ", first(split(sprint(showerror, e), '\n')))
        end
    end

    # 5. The Float64 path must be UNCHANGED -- this is the guarantee that no reported
    #    number moves.
    println("\n  5. Float64 path unchanged (is_ad == false delegates to Distributions)")
    d5 = bstm.bstm_Likelihood("poisson", 1.3; phi_hurdle = 0.3, hurdle = 1.0)
    dist5 = bstm.get_dist_ref(bstm.get_model_family("poisson"), d5, 1.3, 1.0, 1)
    println("     logcdf  via dispatch  = ", bstm._logcdf_count(dist5, 1.0, false))
    println("     logcdf  direct        = ", logcdf(dist5, 1.0))
    println("     identical             = ", bstm._logcdf_count(dist5, 1.0, false) === logcdf(dist5, 1.0))
end

main()
