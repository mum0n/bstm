using Test
using Distributions
using LinearAlgebra
using DataFrames
using CategoricalArrays
using Turing
using Random
using LogExpFunctions
using PDMats
using SparseArrays
using Graphs

@testset "BSTM Full Test Suite" begin

    @testset "Core Utilities" begin
        @testset "split_terms_at_depth" begin
            @test split_terms_at_depth("a + b(c+d) + e", "+") == ["a", "b(c+d)", "e"]
            @test split_terms_at_depth("a", "+") == ["a"]
            @test split_terms_at_depth("a |> log", "|>") == ["a", "log"]
            @test split_terms_at_depth("a+b+c", "-") == ["a+b+c"] # No separator
            @test split_terms_at_depth("", "+") == [""] # Empty string
            @test split_terms_at_depth("a + ", "+") == ["a", ""] # Trailing separator
        end

        @testset "parse_variable_and_transforms" begin
            var, transforms = parse_variable_and_transforms("x |> log |> zscore")
            @test var == "x"
            @test transforms == ["log", "zscore"]
            var, transforms = parse_variable_and_transforms("my_var")
            @test var == "my_var"
            @test isempty(transforms)
            var, transforms = parse_variable_and_transforms(" |> log")
            @test var == ""
            @test transforms == ["log"]
        end
    end

    @testset "Formula Parser (Exhaustive)" begin
        # Test LHS parsing
        @testset "decompose_bstm_formula_LHS" begin
            # Test basic LHS with likelihood
            formula1 = "likelihood(y, family=:gaussian) ~ 1 + x"
            decomposed1 = decompose_bstm_formula(formula1)
            @test length(decomposed1.outcomes) == 1
            @test decomposed1.outcomes[1][:var] == "y"
            @test decomposed1.outcomes[1][:params][:family] == :gaussian
            @test decomposed1.has_intercept == true
            @test "x" in decomposed1.fixed_effects

            # Test multivariate LHS
            formula2 = "likelihood(y1, family=:poisson, offsets=off1) + likelihood(y2) ~ 1"
            decomposed2 = decompose_bstm_formula(formula2)
            @test length(decomposed2.outcomes) == 2
            @test decomposed2.outcomes[1][:var] == "y1"
            @test decomposed2.outcomes[1][:params][:family] == :poisson
            @test decomposed2.outcomes[1][:params][:offsets] == :off1
            @test decomposed2.outcomes[2][:var] == "y2"
            @test isempty(decomposed2.outcomes[2][:params])
        end

        # Test RHS parsing
        @testset "decompose_bstm_formula_RHS" begin
            # Test modules and fixed effects
            formula3 = "likelihood(y) ~ 0 + x1 + spatial(s, model=\"bym2\") + temporal(t, model=\"ar1\")"
            decomposed3 = decompose_bstm_formula(formula3)
            @test decomposed3.has_intercept == false
            @test "x1" in decomposed3.fixed_effects
            @test length(decomposed3.modules) == 2
            @test any(m -> m.module_type == :spatial && m.args[:model] == "bym2", values(decomposed3.modules))
            @test any(m -> m.module_type == :temporal && m.args[:model] == "ar1", values(decomposed3.modules))

            # Test mixed effects
            formula4 = "likelihood(y) ~ mixed(1 | group_id)"
            decomposed4 = decompose_bstm_formula(formula4)
            @test any(m -> m.module_type == :mixed && m.args[:positional_args] == [1, :group_id], values(decomposed4.modules))

            # Test algebraic operators
            formula5 = "likelihood(y) ~ spatial(s_idx) ⊗ temporal(t_idx)"
            decomposed5 = decompose_bstm_formula(formula5)
            @test length(decomposed5.modules) == 1
            interaction_mod = first(values(decomposed5.modules))
            @test interaction_mod.module_type == :interaction
            @test interaction_mod.args[:operator] == :kronecker_product
        end
    end

    @testset "Likelihood Engine" begin
        # Test discrete families
        @testset "Discrete Families" begin
            mu_p = exp(1.0)
            phi = 0.2
            d_pois = bstm_Likelihood("poisson", [0.0]; phi_zi=phi)
            ana_pois = logsumexp(log(phi), log(1 - phi) + logpdf(Poisson(mu_p), 0))
            @test isapprox(Distributions.logpdf(d_pois, 1.0), ana_pois)
        end

        # Test continuous families
        @testset "Continuous Families" begin
            d_gauss = bstm_Likelihood("gaussian", [NaN]; sigma_y=0.5, y_L=2.0)
            @test isapprox(Distributions.logpdf(d_gauss, 1.0), logccdf(Normal(1.0, 0.5), 2.0))

            d_ln = bstm_Likelihood("lognormal", [NaN]; sigma_y=0.3, y_U=1.5)
            @test isapprox(Distributions.logpdf(d_ln, 0.5), logcdf(LogNormal(0.5, 0.3), 1.5))
        end
    end

    @testset "Manifold Template Factory" begin
        # Test basic template creation
        @testset "build_structure_template" begin
            W = sparse([0 1 0; 1 0 1; 0 1 0])
            template_icar = build_structure_template(:icar, 3; W=W)
            @test size(template_icar.matrix) == (3, 3)
            @test issparse(template_icar.matrix)

            # Edge case: n=1
            template_n1 = build_structure_template(:rw1, 1)
            @test size(template_n1.matrix) == (1, 1)
            @test template_n1.matrix[1, 1] == 1.0

            template_rw2 = build_structure_template(:rw2, 10)
            @test size(template_rw2.matrix) == (10, 10)
            @test isapprox(sum(template_rw2.matrix), 0.0, atol=1e-9)
        end
    end

    @testset "Basis Function Factory" begin
        # Test 1D basis functions
        @testset "bstm_smooth_basis_1D" begin
            vals = rand(100)
            B = bstm_smooth_basis_1D("pspline", vals, 20, 3)
            @test size(B) == (100, 20)
            B_wavelet = bstm_smooth_basis_1D("wavelet", vals, 16, 1)
            @test size(B_wavelet) == (100, 16)
            @test all(sum(B_wavelet, dims=2) .≈ 1.0) # Haar-like basis should partition the space
            # Edge case: nbins=1
            B_one_bin = bstm_smooth_basis_1D("pspline", vals, 1, 3)
            @test size(B_one_bin) == (100, 1)
            @test all(B_one_bin .== 1.0) # Should default to a constant basis
        end
        # Test 2D basis functions
        @testset "bstm_smooth_basis_2D" begin
            coords = rand(100, 2)
            B = bstm_smooth_basis_2D("tps", coords, 25)
            @test size(B) == (100, 25)
        end
    end

    @testset "Comprehensive Manifold Smoke Tests" begin
        # Setup dummy data for smoke tests
        s_N_test = 16
        t_N_test = 5
        total_obs = s_N_test * t_N_test
        W_test = sparse(adjacency_matrix(Graphs.grid([Int(sqrt(s_N_test)), Int(sqrt(s_N_test))])))
    
        dummy_df = DataFrame(
            y = rand(total_obs),
            s_idx = repeat(1:s_N_test, inner=t_N_test),
            t_idx = repeat(1:t_N_test, outer=s_N_test),
            x_cont = rand(total_obs),
            s_x = rand(total_obs),
            s_y = rand(total_obs),
            group_id = rand(1:3, total_obs)
        )
    
        @testset "Spatial Manifolds" begin
            for model_name in ["icar", "bym2", "leroux", "sar"]
                @testset "$model_name" begin
                    ex = :(likelihood(y) ~ 1 + spatial(s_idx, model=$model_name))
                    model = @eval @bstm($ex, $dummy_df, W=$W_test)
                    @test rand(model) isa Any
                end
            end
        end
    
        @testset "Temporal Manifolds" begin
            for model_name in ["ar1", "rw1", "rw2"]
                @testset "$model_name" begin
                    ex = :(likelihood(y) ~ 1 + temporal(t_idx, model=$model_name))
                    model = @eval @bstm($ex, $dummy_df)
                    @test rand(model) isa Any
                end
            end
        end
    
        @testset "Smooth Manifolds" begin
            for model_name in ["pspline", "tps", "rff"]
                @testset "$model_name" begin
                    ex = :(likelihood(y) ~ 1 + smooth(s_x, s_y, model=$model_name))
                    model = @eval @bstm($ex, $dummy_df)
                    @test rand(model) isa Any
                end
            end
        end
    
        @testset "Spacetime Interactions" begin
            @testset "Type IV" begin
                model = @bstm(likelihood(y) ~ 1 + spatial(s_idx, model="icar") ⊗ temporal(t_idx, model="ar1"), dummy_df, W=W_test)
                @test rand(model) isa Any
            end
        end
    
        @testset "Other Modules" begin
            @testset "mixed" begin
                model = @bstm(likelihood(y) ~ 1 + mixed(1|group_id), dummy_df)
                @test rand(model) isa Any
            end
            @testset "svc" begin
                model = @bstm(likelihood(y) ~ 1 + (x_cont |> spatial(s_idx, model="icar")), dummy_df, W=W_test)
                @test rand(model) isa Any
            end
        end
    end

    @testset "Integration: Signal Recovery" begin
        # Setup synthetic data for signal recovery
        Random.seed!(404)
        n_locations, n_time = 10, 24
        total_n = n_locations * n_time
        periodicity = 12.0

        s_idx = repeat(1:n_locations, inner=n_time)
        t_idx = repeat(1:n_time, outer=n_locations)
        u_idx_val = mod1.(1:total_n, Int(periodicity)) # Corrected u_idx generation

        t_angle_val = (collect(1:n_time) .* (2 * pi / periodicity))
        harmonic_truth = 1.8 .* sin.(t_angle_val[t_idx]) .- 0.5 .* cos.(t_angle_val[t_idx])

        basis_synth = zeros(total_n, 4)
        for k in 1:4
            basis_synth[:, k] = exp.(-((t_idx .- (k * 6)) .^ 2) ./ 15.0)
        end
        spline_truth = basis_synth * [1.2, -0.8, 0.4, 0.9]

        eta_truth = harmonic_truth .+ spline_truth
        y_obs_synth = eta_truth .+ randn(total_n) .* 0.1

        df_diag = DataFrame(y=y_obs_synth, s_idx=s_idx, t_idx=t_idx, u_idx=u_idx_val, spline_var=rand(total_n))

        # Test model with seasonal and smooth components
        diag_model = @bstm(
            likelihood(y, family=:gaussian) ~ 1 + seasonal(u_idx, model="harmonic") + smooth(spline_var, model="pspline"),
            df_diag;
            basis_matrices = Dict{Symbol, Any}(:spline_var => basis_synth),
            period=12.0
        )

        chain_diag = sample(diag_model, NUTS(200, 0.65), 500, progress=false)

        res_diag = model_results_comprehensive(diag_model, chain_diag)
        y_pred = res_diag.pstats.predictions_denoised.mean

        correlation = cor(y_pred, eta_truth)
        @test correlation > 0.7
    end

    @testset "Integration: Out-of-Sample Prediction" begin
        # Setup synthetic data for out-of-sample prediction
        Random.seed!(116)
        n_s_train = 16
        n_t = 12
        n_total_train = n_s_train * n_t

        x_range = collect(range(0, stop=100, length=4))
        y_range = collect(range(0, stop=100, length=4))
        train_coords = [(x, y) for x in x_range for y in y_range]

        s_idx_train = repeat(1:n_s_train, inner=n_t)
        t_idx_train = repeat(1:n_t, outer=n_s_train)

        s_clusters_train = repeat(1:4, inner=4)
        cluster_effects = [-2.0, 0.0, 1.0, 3.0]

        y_train_signal = cluster_effects[s_clusters_train[s_idx_train]] .+ 0.05 .* t_idx_train
        y_train_obs = y_train_signal .+ randn(n_total_train) .* 0.05

        train_df = DataFrame(
            y_obs = y_train_obs,
            s_idx = s_idx_train,
            t_idx = t_idx_train,
            s_x = [p[1] for p in train_coords[s_idx_train]],
            s_y = [p[2] for p in train_coords[s_idx_train]]
        )

        test_pts = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)] # One point in each quadrant
        n_s_test = length(test_pts)
        test_df = DataFrame(
            s_x = repeat([p[1] for p in test_pts], inner=n_t),
            s_y = repeat([p[2] for p in test_pts], inner=n_t),
            t_idx = repeat(1:n_t, outer=n_s_test)
        )

        # Test model with mosaic spatial component
        model_obj = @bstm(
            likelihood(y_obs, family=:gaussian) ~ 1 + spatial(s_idx, model="mosaic"),
            train_df;
            n_mosaics = 4,
            cluster_assignments = Dict(:s_idx => s_clusters_train),
            s_coord = reduce(hcat, collect.(train_coords))'
        )
        chain_train = sample(model_obj, MH(), 500, progress=false)

        res_pred = predict(model_obj, chain_train, test_df; n_samples=100)
        y_test_pred = res_pred.predictions_denoised.mean

        pred1 = mean(y_test_pred[1:n_t])
        pred4 = mean(y_test_pred[end-n_t+1:end])

        @test isapprox(pred1, -2.0, atol=0.5)
        @test isapprox(pred4, 3.0, atol=0.5)
    end

end