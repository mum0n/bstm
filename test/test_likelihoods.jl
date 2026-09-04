# ==============================================================================
# BSTM Test Suite: Likelihood Engine Taxonomy and Distribution Correctness
# ==============================================================================

if !@isdefined(bstm_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Likelihood Engine Taxonomy" begin
    @testset "Discrete Families (ZI & Hurdle)" begin
        mu_p = exp(1.0)
        phi = 0.2
        d_pois = bstm.bstm_Likelihood("poisson", [1.0]; phi_zi=phi)
        ana_pois = LogExpFunctions.logsumexp(log(phi), log(1 - phi) + logpdf(Poisson(mu_p), 0))
        @test isapprox(Distributions.logpdf(d_pois, 0.0), ana_pois)

        mu_nb = exp(1.5)
        r_val = 2.0
        d_nb_h = bstm.bstm_Likelihood("negbin", [1.5]; r_nb=r_val, phi_hurdle=0.9, hurdle=0.0)
        dist_nb = NegativeBinomial(r_val, r_val / (r_val + mu_nb))
        ana_nb_h = log(0.9) + logpdf(dist_nb, 2.0) - logccdf(dist_nb, 0.0)
        @test isapprox(Distributions.logpdf(d_nb_h, 2.0), ana_nb_h)

        d_bin_int = bstm.bstm_Likelihood(
            "binomial", [0.0]; trial=10, censor_lower=3.0, censor_upper=5.0
        )
        dist_bin = Binomial(10, LogExpFunctions.logistic(0.0))
        ana_bin_int = stable_logdiffexp(logcdf(dist_bin, 5.0), logcdf(dist_bin, 2.0))
        @test isapprox(Distributions.logpdf(d_bin_int, NaN), ana_bin_int)
    end

    @testset "Continuous Families (Censoring)" begin
        d_gauss = bstm.bstm_Likelihood("gaussian", [1.0]; sigma_y=0.5, censor_lower=2.0)
        @test isapprox(Distributions.logpdf(d_gauss, NaN), logccdf(Normal(1.0, 0.5), 2.0))

        d_beta = bstm.bstm_Likelihood("beta", [-0.5]; extra_params=20.0)
        mu_b = LogExpFunctions.logistic(-0.5)
        dist_beta = Beta(mu_b * 20.0, (1 - mu_b) * 20.0)
        @test isapprox(Distributions.logpdf(d_beta, 0.4), logpdf(dist_beta, 0.4))

        d_ln = bstm.bstm_Likelihood("lognormal", [0.5]; sigma_y=0.3, censor_upper=1.5)
        mu_ln = 0.5 - (0.3^2) / 2.0
        @test isapprox(Distributions.logpdf(d_ln, NaN), logcdf(LogNormal(mu_ln, 0.3), 1.5))
    end

    @testset "Likelihood Families & Distributions Correctness" begin
        nb_lik = bstm.bstm_Likelihood("negbin", [1.0]; r_nb=2.0)
        logp_nb = logpdf(nb_lik, 3)
        @test isfinite(logp_nb)
    end

    @testset "Vectorized Observation Parameters & Censoring" begin
        # 1. Vectorized weights
        eta_w = [0.1, 0.2, -0.1, 0.5]
        y_w = [1, 2, 0, 3]
        weights = [1.0, 2.0, 0.5, 1.5]
        lik_w = bstm.bstm_Likelihood(:poisson, eta_w; weight=weights)
        lp_w = logpdf(lik_w, y_w)
        expected_lp_w = sum(weights[i] * logpdf(Poisson(exp(eta_w[i])), y_w[i]) for i in 1:4)
        @test isapprox(lp_w, expected_lp_w; atol=1e-12)

        # 2. Vectorized trials for Binomial
        eta_bin = [0.0, 1.0, -1.0]
        trials = [10, 25, 50]
        y_bin = [5, 15, 10]
        lik_bin = bstm.bstm_Likelihood(:binomial, eta_bin; trial=trials)
        lp_bin = logpdf(lik_bin, y_bin)
        p_bin = [1.0 / (1.0 + exp(-e)) for e in eta_bin]
        expected_bin_lp = sum(logpdf(Binomial(trials[i], p_bin[i]), y_bin[i]) for i in 1:3)
        @test isapprox(lp_bin, expected_bin_lp; atol=1e-12)

        # 3. Vectorized censoring thresholds
        eta_censor = [1.0, 1.0]
        censor_limits = [10.0, 20.0]
        y_censor = [10.0, 20.0]
        lik_censor = bstm.bstm_Likelihood(:gaussian, eta_censor; sigma_y=2.0, censor_lower=censor_limits)
        lp_censor = logpdf(lik_censor, y_censor)
        expected_censor_lp = sum(logccdf(Normal(eta_censor[i], 2.0), censor_limits[i]) for i in 1:2)
        @test isapprox(lp_censor, expected_censor_lp; atol=1e-6)

        # 4. Empty array raises ArgumentError
        @test_throws ArgumentError bstm.bstm_Likelihood(:poisson, eta_w; trial=Int[])

        # 5. Length mismatch raises DimensionMismatch
        @test_throws DimensionMismatch bstm.bstm_Likelihood(:poisson, eta_w; weight=[1.0, 2.0])
    end

    @testset "Vectorized Censoring in Model Formula and Config" begin
        rng = MersenneTwister(42)
        N = 10
        df_censor = DataFrame(
            y = rand(rng, 1.0:10.0, N),
            x = randn(rng, N),
            det_limit = fill(2.0, N)
        )
        df_censor.det_limit[1:5] .= 1.0
        df_censor.det_limit[6:10] .= 3.0

        # Formula with column symbol censor_lower
        cfg1 = bstm.bstm_config(
            "likelihood(y, censor_lower=det_limit) ~ intercept() + fixed(x)",
            df_censor
        )
        @test cfg1[:user_provided_censor_lower] == true
        @test size(cfg1[:censor_lower]) == (N, 1)
        @test cfg1[:censor_lower][1:5, 1] == fill(1.0, 5)
        @test cfg1[:censor_lower][6:10, 1] == fill(3.0, 5)

        # Formula with direct vector censor_lower
        my_limits = [10.0, 20.0, 10.0, 15.0, 12.0, 18.0, 14.0, 16.0, 11.0, 13.0]
        cfg2 = bstm.bstm_config(
            "likelihood(y, censor_lower=my_limits) ~ intercept() + fixed(x)",
            df_censor;
            my_limits = my_limits
        )
        @test cfg2[:censor_lower][:, 1] == my_limits
    end

    @testset "Zero-Inflation and Hurdle Mutual Exclusivity" begin
        eta = 0.5

        # Individual models succeed
        d_zi = bstm.bstm_Likelihood(:poisson, eta; phi_zi=0.2)
        @test d_zi.zi_state isa bstm.ZeroInflated

        d_hurdle = bstm.bstm_Likelihood(:poisson, eta; phi_hurdle=0.2, hurdle=0.0)
        @test d_hurdle.phi_hurdle > -Inf
        @test d_hurdle.hurdle == 0.0

        # Conflicting combinations raise ArgumentError
        @test_throws ArgumentError bstm.bstm_Likelihood(:poisson, eta; phi_zi=0.3, hurdle=10.0)
        @test_throws ArgumentError bstm.bstm_Likelihood(:poisson, eta; phi_zi=0.3, phi_hurdle=0.5)

        # Formula parser raises ArgumentError when both specified
        df_conflict = DataFrame(y = [0, 1, 2], x = [0.1, 0.2, 0.3])
        @test_throws ArgumentError bstm.decompose_bstm_formula(
            "likelihood(y, family=poisson, zero_inflated=true, hurdle=0.0) ~ intercept() + fixed(x)",
            df_conflict
        )
    end

    @testset "Multivariate Normal Likelihood (MvNormalFamily)" begin
        lik_mvn = bstm.bstm_Likelihood(:mvnormal, [1.0, 2.0]; sigma_y = [0.5, 0.8])
        @test lik_mvn.family isa bstm.MvNormalFamily
        y_obs_vec = [1.2, 1.9]
        lp_vec = logpdf(lik_mvn, y_obs_vec)
        expected_lp = logpdf(MvNormal([1.0, 2.0], Diagonal([0.5^2, 0.8^2])), y_obs_vec)
        @test isapprox(lp_vec, expected_lp; atol=1e-8)

        # Matrix evaluation (batch of multivariate observations)
        Y_obs_mat = [1.2 1.9; 0.9 2.1]
        lp_mat = logpdf(lik_mvn, Y_obs_mat)
        expected_mat_lp = logpdf(MvNormal([1.0, 2.0], Diagonal([0.5^2, 0.8^2])), [1.2, 1.9]) +
                          logpdf(MvNormal([1.0, 2.0], Diagonal([0.5^2, 0.8^2])), [0.9, 2.1])
        @test isapprox(lp_mat, expected_mat_lp; atol=1e-8)

        # Sampling
        mvn_sample = rand(lik_mvn)
        @test length(mvn_sample) == 2
    end

    @testset "Censoring Boundary Validation (cl < cu)" begin
        # 1. Scalar bounds lower >= upper
        @test_throws ArgumentError bstm.bstm_Likelihood(
            :gaussian, [1.0]; sigma_y=1.0, censor_lower=100.0, censor_upper=10.0
        )
        @test_throws ArgumentError bstm.bstm_Likelihood(
            :gaussian, [1.0]; sigma_y=1.0, censor_lower=5.0, censor_upper=5.0
        )

        # 2. Vector bounds lower >= upper
        @test_throws ArgumentError bstm.bstm_Likelihood(
            :gaussian, [1.0, 2.0]; sigma_y=1.0, censor_lower=[1.0, 10.0], censor_upper=[5.0, 8.0]
        )

        # 3. Vector length mismatch
        @test_throws DimensionMismatch bstm.bstm_Likelihood(
            :gaussian, [1.0, 2.0]; sigma_y=1.0, censor_lower=[1.0, 2.0], censor_upper=[5.0, 6.0, 7.0]
        )

        # 4. Valid interval censoring
        lik_valid_censor = bstm.bstm_Likelihood(
            :gaussian, [1.0, 2.0]; sigma_y=1.0, censor_lower=[1.0, 2.0], censor_upper=[5.0, 8.0]
        )
        @test lik_valid_censor.censoring_state isa bstm.IntervalCensored
    end
end
