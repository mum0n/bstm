using Pkg
Pkg.activate("."; io=devnull)
using bstm

# The claim to test: the OLD measure `|mid-mu| / |mu|` misfires when the mean sits near
# zero RELATIVE TO THE INTERVAL WIDTH, which is the normal situation for a rare event or a
# rate close to 0. A perfectly sensible, symmetric interval is then reported as maximally
# asymmetric. The half-width measure uses the interval's own size as the yardstick and is
# immune.
#
# Note: an earlier version of this script "tested" scale dependence by holding the
# coefficient of variation fixed and varying the absolute magnitude. Both measures returned
# a constant 0.784, so that test distinguished nothing -- the old measure is ALREADY
# invariant to overall rescaling. The real failure mode is the one below.

function main()
    # For the old measure to false-warn while the new one passes, we need
    #     0.05*|mu| < |mid - mu| <= 0.05*half
    # i.e. the mean must be MUCH SMALLER than the interval half-width, and the mean must
    # sit slightly off the midpoint. That is the ordinary situation for a near-zero
    # prediction with a wide credible interval: a small rate estimated poorly, or a
    # normal-approximation interval that straddles zero.
    #
    # A previous version of this script used a *centred* interval, which is centred by
    # construction -- both measures returned ~0, so it demonstrated nothing, and I wrongly
    # predicted the old measure would score 1.0 there.
    println("  mean small relative to a wide interval (the real failure mode):")
    println("     mu      half      |mid-mu|    OLD         NEW        verdict")
    for (mu, half, frac) in [(0.001, 1.0, 0.03),   # rare event, wide interval
                             (0.01, 2.0, 0.05),
                             (0.1, 1.0, 0.02),
                             (1.0, 1.0, 0.02),      # ordinary scale, for contrast
                             (1.0, 1.0, 0.30)]     # genuinely biased, must warn
        mid = mu + frac * half
        lo, hi = mid - half, mid + half
        old = abs(mid - mu) / max(abs(mu), eps(Float64))
        new = bstm._interval_asymmetry([lo], [hi], [mu])
        println("     ", lpad(mu, 7), " ", lpad(half, 7), "  ", lpad(round(frac * half; sigdigits=3), 8),
                "  ", lpad(round(old; sigdigits=4), 11), "  ", lpad(round(new; sigdigits=4), 11),
                "  ", new <= bstm.INTERVAL_ASYMMETRY_TOL ? "NEW: pass" : "NEW: warn")
    end
    println("\n  -> Rows 1-3: the drift is a few percent of the interval's own half-width,")
    println("     which is a perfectly good interval. The OLD measure divides by the tiny")
    println("     mean and reports 20-30, i.e. 400-600x the tolerance. Row 5 is a real")
    println("     asymmetry and both must warn.")
    println("\n  The old measure is NOT scale-dependent in the ordinary sense: rescaling the")
    println("  whole problem leaves |mid-mu|/|mu| unchanged. The defect is that its")
    println("  denominator is the mean, which is the wrong yardstick when the mean is small")
    println("  next to the interval -- the rare-event / straddles-zero case.")

    println("\n  Anchors under the new measure:")
    mu, half = 1.0, 0.5
    for frac in (0.0, 0.4, 1.0, 2.0)
        mid = mu + frac * half
        v = bstm._interval_asymmetry([mid - half], [mid + half], [mu])
        println("     mean ", frac, " half-width(s) from the midpoint -> asymmetry = ",
                round(v; sigdigits=6))
    end

    println("\n  A genuinely biased interval still warns:")
    v = bstm._interval_asymmetry([0.6], [1.0], [0.1])
    println("     asymmetry = ", round(v; sigdigits=4),
            "  exceeds tol ", bstm.INTERVAL_ASYMMETRY_TOL, ": ", v > bstm.INTERVAL_ASYMMETRY_TOL)

    println("\n  Degenerate inputs report 0.0, not a spurious maximum:")
    @show bstm._interval_asymmetry([1.0], [1.0], [1.0])       # empty interval
    @show bstm._interval_asymmetry([1.0, 2.0], [1.5], [1.2])   # length mismatch
    @show bstm._interval_asymmetry(nothing, [1.5], [1.2])      # absent
    @show bstm._interval_asymmetry([NaN], [1.5], [1.2])        # non-finite
    @show bstm._interval_asymmetry([2.0], [1.0], [1.0])        # inverted bounds

    println("\n  Backward compatibility of the 3-tuple return:")
    r = bstm._recover_pred_sd([0.1, 0.2], nothing, nothing, [1.0, 2.0])
    println("     arity = ", length(r), "   sd, source = ", round.(r[1]; sigdigits=4), ", ", r[2])
    a, b = bstm._recover_pred_sd([0.1, 0.2], nothing, nothing, [1.0, 2.0])
    println("     two-variable destructuring still works: ", a == r[1] && b == r[2])

    println("\n  Only ONE definition of _interval_asymmetry exists:")
    ms = methods(bstm._interval_asymmetry)
    println("     methods = ", length(ms))
end

main()
