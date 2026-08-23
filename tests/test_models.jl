# ==============================================================================
# BSTM Test Suite: Model Instantiation, Smoke Tests, and Complex Inference
# ==============================================================================

if !@isdefined(bstm_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Formulaic Interface & Model Instantiation" begin
    p_dummy, _ = bstm.bstm_data("scottish_lip")
    dummy_df = p_dummy.data
    W_test = p_dummy.au.W

    test_cases = [
        ("ST_TypeIV_Poisson", "likelihood(y_pois, family=poisson) ~ 1 + cov1 + (random(s_idx, model=besag) ⊗ random(t_idx, model=ar1))", "poisson", Dict(:W => W_test)),
        ("MixedEffects_Gaussian", "likelihood(y_gauss) ~ 1 + cov1 + mixed((1 + cov1) | region) + random(s_idx, model=icar)", "gaussian", Dict(:W => W_test)),
        ("Multivariate_Gaussian",
            "likelihood(species_1 + species_2) ~ 1 + cov1 + random(s_idx, model=bym2)",
            "gaussian", Dict(:W => W_test)),
        ("Seasonal_RW2_Poisson", "likelihood(y_pois, family=poisson) ~ 1 + random(month, model=harmonic, period=12) + random(t_idx, model=rw2)", "poisson", Dict()),
        ("SVC_Gaussian", "likelihood(y_gauss) ~ 1 + (cov1 |> random(s_idx, model=icar))",
            "gaussian", Dict(:W => W_test)),
        ("Spatial_Smooth_RFF", "likelihood(y_gauss) ~ 1 + random(s_y, s_x, model=rff, n_features=15) + random(t_idx, model=ar1)", "gaussian", Dict()),
        ("Leroux_Gaussian", "likelihood(y_gauss) ~ 1 + random(s_idx, model=leroux)",
            "gaussian", Dict(:W => W_test)),
        ("GP_Smooth_Gaussian",
            "likelihood(y_gauss) ~ 1 + random(cov1, model=gp, kernel=\"se\")", "gaussian",
            Dict())
    ]

    for (name, f_str, fam, extra_args) in test_cases
        @testset "$name" begin
            println("  Testing Instantiation: $name")
            model = bstm.bstm_core(f_str, dummy_df; model_family=fam, extra_args...)
            @test size(sample(model, Turing.Prior(), 1), 1) >= 1
        end
    end
end

@testset "Integration Tests: Smoke Tests" begin
    p_smoke, _ = bstm.bstm_data("scottish_lip")
    sim_df = p_smoke.data
    W_smoke = p_smoke.au.W

    @testset "IID Model Smoke Test" begin
        model_iid_sim = @bstm(likelihood(y_gauss) ~ intercept() + random(region,
            model=iid), sim_df)
        chain = sample(model_iid_sim, MH(), 100, progress=false)
        @test size(chain, 1) > 0
        @test mean(chain[:sigma_region]) > 0
    end

    @testset "AR1 Model Smoke Test" begin
        model_ar1_sim = @bstm(likelihood(y_gauss) ~ intercept() + random(year, model=ar1),
            sim_df)
        chain = sample(model_ar1_sim, NUTS(10, 0.65), 10, progress=false)
        @test size(chain, 1) > 0
        @test mean(chain[:sigma_year]) > 0
    end

    @testset "BYM2 ⊗ AR1 Interaction Smoke Test" begin
        model_int_sim = @bstm(
            likelihood(y_gauss) ~ intercept() + (random(s_idx, model=bym2) ⊗ random(t_idx,
                model=ar1)),
            sim_df, W=W_smoke
        )
        chain_int_sim = sample(model_int_sim, NUTS(10, 0.65), 10, progress=false)
        @test size(chain_int_sim, 1) > 0
        @test any(k -> occursin("sigma", string(k)) && (occursin("interaction",
            string(k)) || occursin("s_idx", string(k)) || occursin("t_idx", string(k))),
            keys(chain_int_sim))
    end
end

@testset "Complex Integration Tests" begin
    p_cplx, _ = bstm.bstm_data("scottish_lip")
    data = p_cplx.data
    W = p_cplx.au.W

    @testset "Hurdle Model" begin
        model = @bstm(likelihood(y_pois, family=poisson, hurdle=5) ~ 1 + random(s_idx,
            model=bym2), data, W=W)
        chain = sample(model, MH(), 100, progress=false)
        @test size(chain, 1) > 0
        @test any(k -> occursin("hurdle", string(k)), keys(chain))
    end

    @testset "Eigen Model" begin
        data_eigen = copy(data)
        data_eigen[!, :species_1] = randn(nrow(data_eigen))
        data_eigen[!, :species_2] = randn(nrow(data_eigen))
        data_eigen[!, :species_3] = randn(nrow(data_eigen))
        model = @bstm(likelihood(y_gauss) ~ 1 + eigen(species_1, species_2, species_3,
            n_factors=2), data_eigen)
        chain = sample(model, MH(), 10, progress=false)
        @test size(chain, 1) > 0
        @test any(k -> occursin("pca_sd", string(k)) || occursin("species", string(k)),
            keys(chain))
    end

    @testset "Multifidelity Signal Transfer" begin
        df_hi = copy(data)
        if !hasproperty(df_hi, :t_idx); df_hi[!, :t_idx] = df_hi[!, :year]; end
        W_mf = W
        df_lo = select(df_hi, :proxy_val => :y_low, :s_idx, :t_idx)

        model_mf = @bstm(
            likelihood(y_gauss) ~ 1 + random(s_idx, model=bym2) + random(t_idx,
                model=ar1) + nested(low_fi,
                formula="likelihood(y_low) ~ 1 + random(s_idx) + random(t_idx)",
                data_source=:low_quality_data),
            df_hi, W=W_mf, low_quality_data=df_lo
        )
        
        chain_mf = sample(model_mf, NUTS(10, 0.65), 10, progress=false, check_model=false)
        @test size(chain_mf, 1) > 0
        @test any(k -> occursin("nested_low_fi", string(k)), keys(chain_mf))
    end

    @testset "Prediction Engine" begin
        train_df = copy(data)
        if !hasproperty(train_df, :t_idx); train_df[!, :t_idx] = train_df[!, :year]; end
        W_train = W
        
        centroids_matrix = reduce(hcat, [[c[1], c[2]] for c in p_cplx.au.centroids])
        s_N_train = length(p_cplx.au.centroids)
        t_N_train = length(unique(train_df.t_idx))

        test_pts = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)]
        n_s_test = length(test_pts)
        test_df = DataFrame(s_x=repeat([p[1] for p in test_pts], inner=t_N_train), 
                            s_y=repeat([p[2] for p in test_pts], inner=t_N_train), 
                            s_idx=repeat(1:n_s_test, inner=t_N_train),
                            t_idx=repeat(1:t_N_train, outer=n_s_test))

        model_obj = bstm.bstm_core("likelihood(y_gauss) ~ 1 + random(s_idx, model=bym2) + random(t_idx, model=ar1)", 
                         train_df; s_N=s_N_train, t_N=t_N_train, W=W_train)
        
        chain_train = sample(model_obj, MH(), 500, progress=false)
        res_pred = bstm.predict(model_obj, chain_train, test_df; n_samples=100)
        
        @test res_pred.predictions_denoised isa NamedTuple
        @test length(res_pred.predictions_denoised.mean) == nrow(test_df)
    end

    @testset "Cross-Validation Orchestrator" begin
        cv_df = data
        cv_W = W
        cv_formula = "likelihood(y_gauss) ~ 1 + random(year, model=ar1)"

        @testset "k-fold CV" begin
            cv_results_kfold = bstm.bstm_cv_orchestrator(cv_formula, cv_df; method=:kfold,
                n_folds=3, n_samples=50, W=cv_W)
            @test length(cv_results_kfold.folds) == 3
            @test cv_results_kfold.mean_rmse isa Real
        end

        @testset "Temporal Forward-Chaining CV" begin
            cv_results_fchain = bstm.bstm_cv_orchestrator(cv_formula, cv_df;
                method=:temporal_forward_chain, cv_var=:year, n_folds=2, n_samples=50,
                W=cv_W)
            @test length(cv_results_fchain.folds) == 2
            @test cv_results_fchain.mean_r2 isa Real
        end
    end
end

@testset "Composite Gibbs Sampling & Optimal Sampler (Dual Promotion Verification)" begin
    W_test = [0 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 0]
    df_gibbs = DataFrame(
        s_idx = [1, 2, 3, 4, 1, 2, 3, 4],
        year = [1, 1, 1, 1, 2, 2, 2, 2],
        cov1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        log_offsets = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
        y = [2, 3, 1, 4, 3, 5, 2, 6]
    )
    m_gibbs = @bstm(
        likelihood(y, family=poisson, log_offsets=log_offsets) ~
            intercept() +
            fixed(cov1) +
            random(s_idx, model=icar) +
            random(year, model=ar1),
        df_gibbs,
        W = W_test,
        use_gpu = false,
        verbose = false
    )
    @test m_gibbs isa DynamicPPL.Model
    
    # Build optimal Gibbs sampler
    os = get_optimal_sampler(m_gibbs; adaptation_steps=10)
    @test os isa Turing.Gibbs || os isa AbstractMCMC.AbstractSampler

    chn_gibbs = sample(m_gibbs, os, 5; progress=false)
    @test size(chn_gibbs, 1) == 5

    # Test precompute_step_sizes with adaptation_steps
    step_sizes = precompute_step_sizes(m_gibbs; min_ϵ=0.01, max_ϵ=0.8, max_depth=8,
        adaptation_steps=:auto)
    @test haskey(step_sizes, :s_idx)
    @test haskey(step_sizes, :year)
    @test step_sizes[:s_idx].init_ϵ >= 0.01
    @test step_sizes[:s_idx].max_depth == 8
    @test step_sizes[:s_idx].adaptation_steps >= 100

    # Test get_optimal_sampler with :auto adaptation_steps
    os_auto = get_optimal_sampler(m_gibbs; adaptation_steps=:auto)
    @test os_auto isa Turing.Gibbs || os_auto isa AbstractMCMC.AbstractSampler

    # Test get_optimal_sampler with custom init_ϵ and max_depth
    os_custom = get_optimal_sampler(
        m_gibbs;
        init_ϵ = Dict(:s_idx => 0.03, :year => 0.08),
        max_depth = 6,
        adaptation_steps = 10
    )
    @test os_custom isa Turing.Gibbs || os_custom isa AbstractMCMC.AbstractSampler
    chn_custom = sample(m_gibbs, os_custom, 5; progress=false)
    @test size(chn_custom, 1) == 5

    # Test bstm_sample with MCMCThreads backend and deepcopy validation
    chn_bstm = bstm_sample(m_gibbs, os_custom, 5; progress=false)
    @test size(chn_bstm, 1) == 5

    # Test model_results_comprehensive on multi-chain / VNChain sample object
    res_gibbs = model_results_comprehensive(m_gibbs, chn_bstm)
    @test haskey(res_gibbs.metrics, :rmse)
    @test haskey(res_gibbs, :parameters)

    # Test parameter name resolution with Parameter(...) and parameters. wrappers
    raw_wrapped_names = ["Parameter(sigma_s_idx)", "Parameter(ure_s_idx[1])", "parameters.beta", "intercept"]
    @test bstm._find_parameter(raw_wrapped_names, "sigma_s_idx") == "sigma_s_idx"
    @test bstm._find_parameter(raw_wrapped_names, "ure_s_idx") == "ure_s_idx"
    @test bstm._find_parameter(raw_wrapped_names, "beta") == "beta"
    @test bstm._find_parameter(raw_wrapped_names, "intercept") == "intercept"

    # Test bstm_Likelihood random generation
    lik_poisson = bstm_Likelihood("poisson", 1.5)
    @test rand(lik_poisson) >= 0.0
    lik_gauss = bstm_Likelihood("gaussian", 0.0; sigma_y=1.0)
    @test isfinite(rand(lik_gauss))
    lik_vec = [lik_poisson for _ in 1:10]
    sampled_vec = rand.(lik_vec)
    @test length(sampled_vec) == 10

    # Test extract_param_matrix and extract_param_vector
    mock_chain_dict = Dict(:sigma_s_idx => [0.5, 0.6, 0.7], :ure_s_idx => [[0.1, 0.2], [0.3,
        0.4], [0.5, 0.6]])
    @test size(bstm.extract_param_matrix(mock_chain_dict, :sigma_s_idx), 1) == 3
    @test size(bstm.extract_param_matrix(mock_chain_dict, :ure_s_idx), 2) == 2
    @test length(bstm.extract_param_vector(mock_chain_dict, :sigma_s_idx)) == 3

    # Test multi-chain scalar parameter collapsing
    mock_multichain = (
        chains = [1, 2, 3],
        intercept = randn(10, 3),
        beta = randn(10, 3)
    )
    mat_mc = bstm.extract_param_matrix(mock_multichain, :intercept)
    @test size(mat_mc) == (30, 1)

    # Test direct parameter summary statistics
    df_direct_summary = bstm._compute_direct_parameter_summary(mock_chain_dict)
    @test df_direct_summary isa DataFrame
    @test "parameters" in names(df_direct_summary)
    @test "mean" in names(df_direct_summary)
    @test size(df_direct_summary, 1) >= 2

    # Test _generate_conditional_predictions type safety with integer columns
    cond_res = bstm._generate_conditional_predictions(m_gibbs, chn_bstm, m_gibbs.args.M, :cov1)
    @test !isnothing(cond_res)

    # Test temporal component get_effects with PS DataFrame
    ps_test = (data = DataFrame(s_idx = [1, 2], year = [1, 2], cov1 = [0.1, 0.2]), y_N = 2,
        y_obs = [0.0, 0.0])
    ar1_spec = filter(s -> s.structure == :temporal, m_gibbs.args.M.components)[1]
    ar1_effects = bstm.get_effects(ar1_spec.component_obj, chn_bstm, ar1_spec,
        m_gibbs.args.M, ps_test)
    @test haskey(ar1_effects, :structured)

    # Test predict with out-of-sample DataFrame
    df_new = DataFrame(s_idx = [1, 2, 3], year = [1, 2, 3], cov1 = [0.1, 0.2, 0.3])
    pred_out = bstm.predict(m_gibbs, chn_bstm, df_new)
    @test haskey(pred_out, :predictions_denoised)
    @test length(pred_out.predictions_denoised.mean) == 3

    # Test model with log_offsets end-to-end
    df_offset = DataFrame(
        y = [1.0, 2.0, 3.0, 4.0],
        s_idx = [1, 2, 1, 2],
        year = [1, 1, 2, 2],
        cov1 = [0.1, 0.2, 0.3, 0.4],
        log_e = [0.0, 0.1, 0.0, 0.1]
    )
    W_small = [0.0 1.0; 1.0 0.0]
    m_offset = bstm.bstm_core(
        :(likelihood(y, family=poisson,
            log_offsets=log_e) ~ intercept() + fixed(cov1) + random(s_idx,
            model=bym2) + random(year, model=ar1)),
        df_offset;
        W = W_small,
        verbose = false
    )
    chn_offset = bstm.bstm_sample(m_offset, 5; progress=false)
    au_test = (polygons = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0],
        [2.0, 0.0], [2.0, 1.0], [1.0, 0.0]]], centroids = [(0.5, 0.5), (1.5, 0.5)])
    res_offset = bstm.model_results_comprehensive(m_offset, chn_offset)
    @test haskey(res_offset.metrics, :rmse)
    @test haskey(res_offset.metrics, :waic)
    @test haskey(res_offset, :parameters)
    @test haskey(res_offset, :effects)
    @test haskey(res_offset, :predictions)
    @test haskey(res_offset, :draws)
    @test haskey(res_offset.draws, :weights)
    @test !haskey(res_offset, :model)
    @test !haskey(res_offset, :chain)

    # Test separate bstm_plots statement with data and spatial areal units (au)
    plot_dir = mktempdir()
    plots_res = bstm.bstm_plots(res_offset; data=df_offset, au=au_test, save_dir=plot_dir)
    @test haskey(plots_res.plots, :posterior_predictive_check)
    @test haskey(plots_res.plots, :fixed_effects)
    @test haskey(plots_res.plots, :spatial)
    @test haskey(plots_res.plots, :spatial_observed)
    @test haskey(plots_res.plots, :spatial_fitted)
    @test haskey(plots_res.plots, :temporal)
    @test haskey(plots_res.plots, :spacetime_predictions)
    @test haskey(plots_res.plots_data, :spacetime_predictions)
    @test isfile(joinpath(plot_dir, "posterior_predictive_check.png"))
    @test !isfile(joinpath(plot_dir, "posterior_predictive_check.png.png"))
    @test isfile(joinpath(plot_dir, "spacetime_predictions.png"))

    # Test positional data argument
    plots_res_pos = bstm.bstm_plots(res_offset, df_offset; au=au_test)
    @test haskey(plots_res_pos.plots, :posterior_predictive_check)
end
