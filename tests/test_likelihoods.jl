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
end
