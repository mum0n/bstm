using Pkg
Pkg.activate("."; io=devnull)
using DataFrames, Distributions, Turing, Random, LinearAlgebra
using bstm

# Ground the sigma-extraction in a REAL chain rather than an inferred name pattern.
# The PAR fix depends on knowing (a) that component scales are `sigma_<key>`,
# (b) that `y_sigma` is a distinct observation-noise parameter, and (c) whether
# per-outcome suffixes appear in multivariate models.

function main()
    Random.seed!(11)
    n_reg, n_yr = 6, 5
    df = DataFrame(
        region = repeat(1:n_reg, inner=n_yr),
        year   = repeat(1:n_yr, outer=n_reg),
        smoke   = rand(MersenneTwister(1), 0:1, n_reg*n_yr),
        y       = rand(Poisson(3.0), n_reg*n_yr)
    )
    W = Array{Float64}(I, n_reg, n_reg)
    for i in 1:n_reg-1, j in i+1:n_reg
        (i % 2 == 0 && j % 2 == 0) && (W[i,j] = 1.0; W[j,i] = 1.0)
    end

    m = @bstm(likelihood(y, family=poisson) ~ intercept() + smoke +
              random(region, model=bym2) + random(year, model=ar1), df, W=W, verbose=false)
    chn = sample(m, MH(), 12; progress=false)

    keys_sorted = sort(string.(keys(chn)))
    println("  chain keys (", length(keys_sorted), "):")
    for k in keys_sorted
        println("    ", k)
    end

    println("\n  sigma-like keys:")
    sig = filter(k -> occursin("sigma", k), keys_sorted)
    for k in sig
        v = try
            collect(chn[!, Symbol(k)])
        catch
            Float64[]
        end
        println("    ", rpad(k, 34), " len=", length(v),
                isempty(v) ? "" : "  range=[$(round(minimum(v), sigdigits=3)), $(round(maximum(v), sigdigits=3))]")
    end

    # The key question for the fix: is `y_sigma` present and is it the ONLY
    # observation-noise param, i.e. separable from component scales?
    println("\n  y_sigma present: ", "y_sigma" in keys_sorted)
end

main()
