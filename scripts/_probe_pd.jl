using Pkg
Pkg.activate("."; io=devnull)
using bstm, Turing, DataFrames, Random

function main()
    s_N, t_N = 6, 2
    N = s_N * t_N
    df = DataFrame(y  = [1.0 + 0.1*s for t in 1:t_N for s in 1:s_N],
                   y2 = [2.0 - 0.1*s for t in 1:t_N for s in 1:s_N],
                   s_x = repeat(collect(1.0:s_N), t_N), s_y = zeros(N),
                   s_idx = repeat(1:s_N, t_N), t_idx = repeat(1:t_N, inner=s_N))
    W = bstm.spatial_knn_graph([(Float64(s), 0.0) for s in 1:s_N], 2)[2]
    m = bstm.bstm_core("likelihood(y + y2) ~ 1 + random(s_idx, model=icar)",
                       df; s_N=s_N, t_N=t_N, W=W, verbose=false)
    fails = String[]
    for seed in 1:30
        Random.seed!(seed)
        try
            sample(m, bstm.Turing.NUTS(), 40; progress=false, check_model=false)
        catch e
            push!(fails, "seed $seed -> " * sprint(showerror, e))
        end
    end
    if isempty(fails)
        println("RESULT: 30/30 seeds sampled, no PosDefException")
    else
        println("RESULT: $(length(fails))/30 seeds failed")
        foreach(f -> println("  ", f), fails)
    end
end

main()
