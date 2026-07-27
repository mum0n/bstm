using Test
using bstm
using Distributions
using LinearAlgebra
using DataFrames
using CategoricalArrays
using Turing
using Random
using SparseArrays
using MCMCChains
using Clustering
using LogExpFunctions
using Graphs

# Helper function for creating a simple adjacency matrix
function create_chain_adj_matrix(n)
    W = spzeros(Int, n, n)
    for i in 1:(n-1)
        W[i, i+1] = 1
        W[i+1, i] = 1
    end
    return W
end

# Helper function for stable log-difference of exponentials
stable_logdiffexp(a, b) = a + log1mexp(b - a)

@testset "BSTM Simple Unit Tests: Core Components & Interfaces" begin

    @testset "Core Utilities & Parser" begin
        @testset "split_terms_at_depth" begin
            @test bstm.split_terms_at_depth("a + b(c+d) + e", "+") == ["a", "b(c+d)", "e"]
            @test bstm.split_terms_at_depth("a", "+") == ["a"]
            @test bstm.split_terms_at_depth("a |> log", " |> ") == ["a", "log"]
        end

        @testset "Comprehensive Formula Parsing" begin
            formula = """
            y ~ 1 + cov1 + cov2 +
            spatial(s_idx, model='bym2') +
            temporal(year, model='ar1') +
            seasonal(u_idx, model='cyclic', period=12) +
            smooth(cov3, nbins=10, model='pspline') +
            nested(z ~ 1 + spatial(s_idx)) +
            eigen(cov1, cov2, n_factors=1) +
            mixed(1|f1) +
            (cov1 |> spatial(s_idx, model='icar')) +
            dynamics(dynamic_var, model='advection')
            """
            data = DataFrame(
                y = rand(100), cov1 = rand(100), cov2 = rand(100), cov3 = rand(100), cov4 = rand(100),
                s_idx = rand(1:10, 100), year = rand(2000:2005, 100), u_idx = rand(1:12, 100),
                f1 = categorical(rand(1:4, 100)), dynamic_var = rand(100), z = rand(100)
            )
            W = create_chain_adj_matrix(10)

            M_cfg = bstm_config(formula, data; W=W, s_N=size(W,1), t_N=length(unique(data.year)))
            
            @test M_cfg.add_intercept == true
            @test all(x -> x in string.(M_cfg.Xfixed_names), ["cov1", "cov2"])
            
            manifolds = M_cfg.manifolds
            @test any(spec -> spec.domain == :spatial && spec.manifold_obj isa bstm.BYM2, manifolds)
            @test any(spec -> spec.domain == :temporal && spec.manifold_obj isa bstm.AR1, manifolds)
            @test any(spec -> spec.domain == :seasonal && spec.manifold_obj isa bstm.Cyclic, manifolds)
            @test any(spec -> spec.domain == :smooth && spec.var == "cov3" && spec.manifold_obj isa bstm.PSpline, manifolds)
            @test any(spec -> spec.domain == :eigen, manifolds)
            @test any(spec -> spec.domain == :mixed && spec.var == "f1", manifolds)
            @test any(spec -> spec.domain == :interact && spec.manifold_obj.operator == :pipe, manifolds)
            @test any(spec -> spec.domain == :dynamics && spec.manifold_obj isa bstm.DynamicsManifold, manifolds)
            @test haskey(M_cfg, :nested_manifolds) && haskey(M_cfg.nested_manifolds, :z)
        end
    end

    @testset "Likelihood Engine Taxonomy" begin
        @testset "Discrete Families (ZI & Hurdle)" begin
            mu_p = exp(1.0)
            phi = 0.2
            d_pois = bstm.bstm_Likelihood("poisson", [0.0]; phi_zi=phi)
            ana_pois = LogExpFunctions.logsumexp(log(phi), log(1-phi) + logpdf(Poisson(mu_p), 0))
            @test isapprox(Distributions.logpdf(d_pois, 1.0), ana_pois)

            mu_nb = exp(1.5)
            r_val = 2.0
            d_nb_h = bstm.bstm_Likelihood("negbin", [2.0]; r_nb=r_val, hurdle=0.0)
            dist_nb = NegativeBinomial(r_val, r_val/(r_val + mu_nb))
            ana_nb_h = logpdf(dist_nb, 2.0) - logccdf(dist_nb, 0.0)
            @test isapprox(Distributions.logpdf(d_nb_h, 1.5), ana_nb_h)

            d_bin_int = bstm.bstm_Likelihood("binomial", [NaN]; trial=10, y_L=3.0, y_U=5.0)
            dist_bin = Binomial(10, LogExpFunctions.logistic(0.0))
            ana_bin_int = stable_logdiffexp(logcdf(dist_bin, 5.0), logcdf(dist_bin, 2.0))
            @test isapprox(Distributions.logpdf(d_bin_int, 0.0), ana_bin_int)
        end

        @testset "Continuous Families (Censoring)" begin
            d_gauss = bstm.bstm_Likelihood("gaussian", [NaN]; sigma_y=0.5, y_L=2.0)
            @test isapprox(Distributions.logpdf(d_gauss, 1.0), logccdf(Normal(1.0, 0.5), 2.0))

            d_beta = bstm.bstm_Likelihood("beta", [0.4]; extra_params=20.0)
            mu_b = LogExpFunctions.logistic(-0.5)
            dist_beta = Beta(mu_b * 20.0, (1-mu_b) * 20.0)
            @test isapprox(Distributions.logpdf(d_beta, -0.5), logpdf(dist_beta, 0.4))

            d_ln = bstm.bstm_Likelihood("lognormal", [NaN]; sigma_y=0.3, y_U=1.5)
            @test isapprox(Distributions.logpdf(d_ln, 0.5), logcdf(LogNormal(0.5, 0.3), 1.5))
        end
    end

    @testset "Core: Manifold & Model Construction" begin
        mock_inputs = Dict(:s_N => 10, :t_N => 20, :W => create_chain_adj_matrix(10))

        @testset "Temporal Manifolds" begin
            m_rw1 = bstm.RW1(Exponential(1.0))
            res_rw1 = bstm.build_model(m_rw1, mock_inputs)
            @test res_rw1.model_type == :rw1
            @test size(res_rw1.Q_template) == (20, 20)
        end

        @testset "Seasonal Manifolds" begin
            m_cyc = bstm.Cyclic(12, Exponential(1.0))
            res_cyc = bstm.build_model(m_cyc, Dict(:u_N => 12))
            @test res_cyc.model_type == :cyclic
            @test size(res_cyc.Q_template) == (12, 12)
        end

        @testset "Basis & Continuous Manifolds" begin
            m_bs = bstm.BSpline(15, 3, Exponential(1.0))
            res_bs = bstm.build_model(m_bs, mock_inputs)
            @test res_bs.model_type == :bspline
            @test size(res_bs.Q_template) == (15, 15)
        end
    end

    @testset "Spatial Partitioning Engine" begin
        s_N = 100
        t_N = 15
        coords = rand(s_N, 2) .* 100
        t_idx = repeat(1:t_N, inner=Int(s_N/t_N)+1)[1:s_N]

        partitioning_methods = [:cvt, :kvt, :qvt, :bvt, :avt, :hvt, :lattice]

        for method in partitioning_methods
            @testset "Method: $method" begin
                au = assign_spatial_units(
                    coords;
                    area_method = method,
                    t_idx = t_idx,
                    target_units = 20,
                    min_points = 1
                )
                @test au isa NamedTuple
                @test hasproperty(au, :centroids)
                @test hasproperty(au, :W)
                @test length(au.centroids) > 0
                @test size(au.W, 1) == size(au.W, 2)
                @test size(au.W, 1) == length(au.centroids)
                @test length(au.assignments) == s_N
            end
        end
    end

    @testset "Formulaic Interface & Model Instantiation" begin
        s_N_test = 16
        t_N_test = 5
        total_obs = s_N_test * t_N_test
        W_test = sparse(adjacency_matrix(Graphs.grid([Int(sqrt(s_N_test)), Int(sqrt(s_N_test))])))

        dummy_df = DataFrame(
            y = rand(total_obs), y1 = rand(total_obs), y2 = rand(total_obs),
            s_idx = repeat(1:s_N_test, inner=t_N_test), t_idx = repeat(1:t_N_test, outer=s_N_test),
            u_idx = repeat(1:min(12, t_N_test), inner=div(total_obs, min(12, t_N_test))),
            x_cont = rand(total_obs), x_cat1 = categorical(repeat(["A", "B"], outer=div(total_obs, 2))),
            Region = categorical(repeat(["East", "West"], outer=div(total_obs, 2))),
            lat = rand(total_obs), lon = rand(total_obs)
        )
        dummy_df.ycount = Int.(round.(dummy_df.y))

        test_cases = [
            ("ST_TypeIV_Poisson", "ycount ~ 1 + x_cont + (spatial(s_idx, model=besag) ⊗ temporal(t_idx, model=ar1))", "poisson", Dict(:W => W_test)),
            ("MixedEffects_Gaussian", "y ~ 1 + x_cont + mixed(x_cont | x_cat1) + spatial(s_idx, model=icar)", "gaussian", Dict(:W => W_test)),
            ("Multivariate_Gaussian", "y1 + y2 ~ 1 + x_cont + spatial(s_idx, model=bym2)", "gaussian", Dict(:W => W_test)),
            ("Seasonal_RW2_Poisson", "ycount ~ 1 + seasonal(u_idx, model=cyclic, period=5) + temporal(t_idx, model=rw2)", "poisson", Dict()),
            ("SVC_Gaussian", "y ~ 1 + (x_cont |> spatial(s_idx, model=icar))", "gaussian", Dict(:W => W_test)),
            ("Spatial_Smooth_RFF", "y ~ 1 + smooth(lat, lon, model=rff, n_features=15) + temporal(t_idx, model=ar1)", "gaussian", Dict())
        ]

        for (name, f_str, fam, extra_args) in test_cases
            @testset "$name" begin
                println("  Testing Instantiation: $name")
                model = bstm(f_str, dummy_df; model_family=fam, extra_args...)
                @test sample(model, Prior(), 1) isa Chains
            end
        end
    end

end