using Pkg
Pkg.activate("."; io=devnull)
using bstm, Statistics

# Post-stratification must satisfy, for every stratum h and every posterior draw s:
#     sum_{i in h} w[i,s] * yhat[i,s]  ==  T[h]
# That identity IS the definition. Check it, and check what the old formula did.

function main()
    # 3 strata, uneven sizes, uneven predictions, 4 draws.
    ids = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    ŷ = [10.0 12.0 11.0 15.0;
         1.0   1.2  0.9  1.1;
         100.0 90.0 110.0 105.0;
         200.0 190.0 210.0 205.0;
         50.0  52.0  48.0  51.0;
         5.0   6.0   4.0   5.5;
         7.0   7.5   6.5   7.2;
         9.0   9.4   8.6   9.1;
         2.0   2.2   1.8   2.1]
    T = Dict(1 => 33.0, 2 => 4.2, 3 => 512.0)   # externally known totals

    println("  1. THE IDENTITY: weighted stratum total == external total")
    w = bstm.post_stratification_weights(nothing, ids, ŷ, T; normalize=false)
    ok = true
    for h in sort(collect(keys(T)))
        idx = findall(==(h), ids)
        for s in 1:size(ŷ, 2)
            got = sum(w[idx, s] .* ŷ[idx, s])
            good = isapprox(got, T[h]; rtol=1e-12)
            ok &= good
            s == 1 && println("     stratum $h: model total=", round(sum(ŷ[idx, s]), digits=4),
                              "  weighted=", round(got, digits=10),
                              "  external=", T[h], "  ", good ? "OK" : "FAIL")
        end
    end
    println("     identity holds for all strata and draws: ", ok)

    println("\n  2. The old formula (stratum_mean / pred_i) against the same identity")
    for h in sort(collect(keys(T)))
        idx = findall(==(h), ids)
        mp = mean(ŷ[idx, 1])
        wold = mp ./ ŷ[idx, 1]
        got = sum(wold .* ŷ[idx, 1])
        println("     stratum $h: weighted total=", round(got, digits=4),
                "  external=", T[h], "  ratio=", round(got / T[h], digits=4))
    end
    println("     -> the old weights reconcile to nothing; they have no target.")

    println("\n  3. Normalisation preserves RELATIVE reweighting, rescales absolute totals")
    wn = bstm.post_stratification_weights(nothing, ids, ŷ, T; normalize=true)
    println("     mean weight per draw = ", round.(mean(wn, dims=1)[:], digits=10))
    for h in sort(collect(keys(T)))
        idx = findall(==(h), ids)
        s = 1
        base = sum(w[idx, s] .* ŷ[idx, s]); norm = sum(wn[idx, s] .* ŷ[idx, s])
        println("     stratum $h: unnormalised=", round(base, digits=8),
                "  normalised=", round(norm, digits=8),
                "  ratio=", round(norm / base, digits=6))
    end
    println("     -> one COMMON factor across strata: relative reweighting survives,")
    println("        absolute reconciliation to the external total does NOT.")

    println("\n  4. Weights are constant within a stratum (required to move a total)")
    for h in sort(collect(keys(T)))
        idx = findall(==(h), ids)
        col = w[idx, 1]
        println("     stratum $h: weights = ", round.(col, digits=6),
                "  constant: ", all(isapprox.(col, fill(col[1], length(col)); rtol=1e-12)))
    end

    println("\n  5. No external totals -> nothing, not a made-up number")
    r = bstm.post_stratification_weights(nothing, ids, ŷ, nothing)
    println("     returns nothing: ", isnothing(r))

    println("\n  6. A stratum with a zero predicted total is NaN, not fabricated")
    ŷ0 = copy(ŷ); ŷ0[4:5, :] .= 0.0   # rows 4-5 are stratum 2
    w0 = bstm.post_stratification_weights(nothing, ids, ŷ0, T; normalize=false)
    i2 = findall(==(2), ids)
    println("     stratum 2 weights = ", w0[i2, 1])
    println("     all NaN: ", all(isnan, w0[i2, 1]))
    println("     stratum 1 unaffected: ", all(isfinite, w0[findall(==(1), ids), 1]))

    println("\n  7. Vector form of totals, and the missing-stratum error")
    wv = bstm.post_stratification_weights(nothing, ids, ŷ, [33.0, 4.2, 512.0]; normalize=false)
    println("     vector totals agree with Dict: ", isapprox(wv, w; rtol=1e-12))
    try
        bstm.post_stratification_weights(nothing, ids, ŷ, Dict(1 => 33.0, 2 => 4.2))
        println("     missing stratum: NO ERROR (bad)")
    catch e
        println("     missing stratum errors: ", first(split(sprint(showerror, e), '\n')))
    end
end

main()
