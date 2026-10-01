using Pkg
Pkg.activate("."; io=devnull)
using bstm, DataFrames, Turing, Random, Statistics, LinearAlgebra, SparseArrays

# Every component that takes a scalar scale hyperparameter, so the same pin-and-check
# applies. The point is to find which ones still return a ZERO field when the scale is
# pinned to a constant.
const SCALAR_SCALE_COMPONENTS = [
    :icar, :besag, :bym2, :leroux, :localadaptive, :cyclic, :rw1, :rw2, :iid,
    :ar1, :ar2, :moran, :tar, :gp, :fitc, :sparsegp, :spectralgp, :eigen, :harmonic,
    :wavelet, :waveletgp, :graph_wavelet, :adaptivesmooth, :nonstationaryvariance,
    :bspline, :pspline, :tensorproductsmooth, :spde, :bcgn, :hyperbolic, :fft,
    :sciml, :nystrom, :svar, :dag, :networkflow, :barycentric, :tps, :pointprocess,
    :dynamics, :composed, :mixed, :sar,
]

function make_data()
    n = 12
    pts = DataFrame(idx = collect(1:n), x = collect(1.0:n), y = zeros(Float64, n))
    pts.depth = [10.0, 12.0, 9.0, 11.0, 10.5, 12.5, 9.5, 11.5, 10.2, 12.2, 9.2, 11.2]
    return pts
end

function try_component(model, pts, W)
    formula = "likelihood(depth, family=gaussian) ~ intercept() + " *
              "random(idx, model=$model, sigma=1.0)"
    m = try
        bstm.bstm_core(formula, pts; W=W, verbose=false)
    catch e
        return (:nomodel, sprint(showerror, e))
    end
    ch = try
        sample(MersenneTwister(3), m, MH(), 120; progress=false)
    catch e
        return (:nosample, sprint(showerror, e))
    end
    comp = m.args.M.components[1]
    eff = try
        bstm.get_effects(comp.component_obj, ch, comp, m.args.M, nothing)
    catch e
        return (:nogeteffects, sprint(showerror, e))
    end
    f = eff.structured[1]
    allzero = maximum(abs, f) <= 1e-12
    finite = all(isfinite, f)
    return (allzero ? :ZERO : :ok, allzero ? "all-zero field ($(size(f,1))x$(size(f,2)))" :
            "max|.|=$(round(maximum(abs, f); sigdigits=4)) finite=$finite")
end

function main()
    pts = make_data()
    W = spzeros(Int, 12, 12)
    for i in 1:11
        W[i, i+1] = 1
        W[i+1, i] = 1
    end
    W[1, 12] = 1; W[12, 1] = 1   # make it connected

    bad = Symbol[]
    errs = Pair{Symbol, String}[]
    for model in SCALAR_SCALE_COMPONENTS
        status, detail = try_component(model, pts, W)
        if status === :ZERO
            push!(bad, model)
            println("  ZERO      $model  $detail")
        elseif status !== :ok
            push!(errs, model => "$status: " * first(detail, 90))
            println("  $(rpad(uppercase(string(status)), 10)) $model  $(first(detail, 90))")
        else
            println("  ok        $model  $detail")
        end
    end
    println("\n===== components still returning a ZERO field with a pinned sigma =====")
    println("  count = ", length(bad))
    println("  ", join(bad, ", "))
    println("\n===== components that could not be exercised =====")
    println("  count = ", length(errs))
    for (k, v) in errs
        println("    $k: $v")
    end
end

main()