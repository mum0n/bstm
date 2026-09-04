# ==============================================================================
# BSTM Test Suite: Population Attributable Risk (PAR / PAF) Engine
# ==============================================================================

if !@isdefined(bstm_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Population Attributable Risk (PAR/PAF) Engine" begin
    # 1. Test Levin formula
    chain_mock = DataFrame(smoking = fill(log(2.0), 100))  # beta = log(2.0) -> RR = 2.0
    res_levin = par_from_posterior(
        chain_mock, "smoking";
        family = "poisson",
        exposure_prevalence = 0.4,
        method = :levin
    )
    expected_levin = (0.4 * (2.0 - 1.0)) / (1.0 + 0.4 * (2.0 - 1.0))
    @test isapprox(res_levin.paf_mean, expected_levin; atol=1e-6)
    @test isapprox(res_levin.rr_mean, 2.0; atol=1e-6)

    # 2. Test Miettinen formula
    res_miettinen = par_from_posterior(
        chain_mock, "smoking";
        family = "poisson",
        exposure_prevalence = 0.4,
        method = :miettinen
    )
    expected_miettinen = 0.4 * (2.0 - 1.0) / 2.0
    @test isapprox(res_miettinen.paf_mean, expected_miettinen; atol=1e-6)

    # 3. Test Binomial exact RR conversion with baseline risk
    res_binom = par_from_posterior(
        chain_mock, "smoking";
        family = "binomial",
        baseline_risk = 0.2,
        exposure_prevalence = 0.4
    )
    expected_rr_binom = 0.3333333333333333 / 0.2
    @test isapprox(res_binom.rr_mean, expected_rr_binom; atol=1e-5)

    # 4. Test Prevented Fraction for protective exposure
    chain_mock_prot = DataFrame(diet = fill(log(0.5), 100))  # beta = log(0.5) -> RR = 0.5
    res_prot = par_from_posterior(
        chain_mock_prot, "diet";
        family = "poisson",
        exposure_prevalence = 0.5
    )
    expected_pf = (0.5 * (1.0 - 0.5)) / (0.5 * (1.0 - 0.5) + 0.5)
    @test isapprox(res_prot.prevented_fraction_mean, expected_pf; atol=1e-5)
    @test res_prot.paf_mean < 0.0

    # 5. Test Continuous exposure validation and threshold dichotomization
    chain_mock[!, :pollution] = fill(log(1.5), 100)
    df_cont = DataFrame(
        pollution = [12.0, 15.0, 22.0, 28.0, 35.0],
        cases = [1, 2, 4, 5, 8]
    )
    # Continuous without threshold should throw ArgumentError (no silent fallback or clamping)
    @test_throws ArgumentError par_from_posterior(
        chain_mock, "pollution";
        family = "poisson",
        exposure_var = "pollution",
        data = df_cont
    )

    # Continuous with threshold=20.0 (3 of 5 exposed -> prevalence = 0.6)
    res_cont = par_from_posterior(
        chain_mock, "pollution";
        family = "poisson",
        exposure_var = "pollution",
        threshold = 20.0,
        data = df_cont,
        outcome_var = "cases"
    )
    @test isapprox(res_cont.exposure_prevalence, 0.6; atol=1e-6)
    @test res_cont.attributable_cases !== nothing

    # 6. Test summarize_par_effects and export_par_to_table
    chain_multi = DataFrame(
        smoking = fill(log(1.8), 50),
        pollution = fill(log(1.5), 50)
    )
    multi_res = summarize_par_effects(
        chain_multi;
        covariates = ["smoking", "pollution"],
        families = Dict("smoking" => "binomial", "pollution" => "poisson"),
        baseline_risks = Dict("smoking" => 0.1),
        exposure_prevalences = Dict("smoking" => 0.25, "pollution" => 0.40)
    )
    tbl = export_par_to_table(multi_res)
    @test nrow(tbl) == 2
    @test hasproperty(tbl, :paf_mean)
    @test hasproperty(tbl, :method)

    # 7. Test single result export_par_to_table
    tbl_single = export_par_to_table(res_levin)
    @test nrow(tbl_single) == 1
end
