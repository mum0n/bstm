# ==============================================================================
# BSTM Test Suite: Core Formula Parsing, ParamRegistry, and Manifolds
# ==============================================================================

if !@isdefined(bstm_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Core Components & Interfaces" begin

    @testset "split_terms_at_depth" begin
        @test bstm.split_terms_at_depth("a + b(c+d) + e", "+") == ["a", "b(c+d)", "e"]
        @test bstm.split_terms_at_depth("a", "+") == ["a"]
        @test bstm.split_terms_at_depth("a |> log", " |> ") == ["a", "log"]
    end

    @testset "Comprehensive Formula Parsing" begin
        data = bstm.bstm_data("advanced", s_N=30, t_N=12)
        W = create_chain_adj_matrix(30)
        
        formula = """
        y_gauss ~ 1 + cov1 + cov2 +
        random(s_idx, model='bym2') +
        random(year, model='ar1') +
        random(month, model='cyclic', period=12) +
        random(cov3, nbins=10, model='pspline') +
        nested(proxy_val, formula="likelihood(proxy_val) ~ 1 + random(s_idx)") +
        eigen(cov1, cov2, n_factors=1) +
        mixed(1|region) +
        (cov1 |> random(s_idx, model='icar')) +
        dynamics(recruitment, model='advection')
        """

        M_cfg = bstm.bstm_config(formula, data; W=W, s_N=30, t_N=12)
        
        @test M_cfg.add_intercept == true
        @test all(x -> x in string.(M_cfg.Xfixed_names), ["cov1", "cov2"])
        
        components = M_cfg.components
        @test any(c -> c.component_obj isa bstm.BYM2, components)
        @test any(c -> c.component_obj isa bstm.AR1, components)
        @test any(c -> c.component_obj isa bstm.Cyclic || c.component_obj isa bstm.Harmonic,
            components)
        @test any(c -> c.component_obj isa bstm.PSpline, components)
        @test any(c -> c.component_obj isa bstm.Eigen, components)
        @test any(c -> c.component_obj isa bstm.Mixed, components)
        @test any(c -> c.component_obj isa bstm.SVC, components)
        @test any(c -> c.component_obj isa bstm.Dynamics, components)
        @test haskey(M_cfg.nested_components, :proxy_val)
    end

    @testset "ParamRegistry Architecture" begin
        # 1. Test ParamDescriptor creation
        desc = bstm.ParamDescriptor(
            :sigma_s_idx;
            component_key = :s_idx,
            role = :sigma,
            shape = (1,),
            prior = Exponential(1.0)
        )
        @test desc.symbol == :sigma_s_idx
        @test desc.component_key == :s_idx
        @test desc.role == :sigma

        # 2. Test empty registry & add_descriptor!
        reg = bstm.ParamRegistry()
        bstm.add_descriptor!(reg, desc)
        @test :sigma_s_idx in keys(reg.descriptors)
        @test "sigma_s_idx" in reg.names
        @test bstm.find_chain_param(reg, "sigma_s_idx") == "sigma_s_idx"
        @test haskey(reg.by_component, :s_idx)
        @test haskey(reg.by_component[:s_idx], :sigma)

        # 3. Test build_param_registry from M config
        p_data = bstm.bstm_data("scottish_lip")
        m_cfg = bstm.bstm_config("y ~ 1 + cov1 + random(s_idx, model=bym2) + random(year, model=ar1)", p_data.data; W=p_data.au.W)
        reg_m = bstm.build_param_registry(m_cfg)
        @test haskey(reg_m.by_component, :intercept)
        @test haskey(reg_m.by_component, :fixed)
        @test haskey(reg_m.by_component, :s_idx)
        @test haskey(reg_m.by_component, :year)

        # 4. Test calibrate_param_registry with prior sample NamedTuple and generic mapping
        s_N_lip = length(p_data.au.centroids)
        sample_nt = (
            intercept = 0.5,
            beta = [0.1, 0.2],
            sigma_s_idx = 1.2,
            ure_s_idx = randn(s_N_lip)
        )
        calibrated = bstm.calibrate_param_registry(reg_m, sample_nt)
        @test calibrated.descriptors[:ure_s_idx].shape == (s_N_lip,)

        # Mock VarNamedTuple mapping
        struct MockVarNamedTuple
            data::Dict{Symbol, Any}
        end
        Base.pairs(m::MockVarNamedTuple) = Base.pairs(m.data)
        t_N_lip = length(unique(p_data.data.year))
        mock_vnt = MockVarNamedTuple(Dict(:sigma_year => 0.8, :ure_year => randn(t_N_lip)))
        calibrated_vnt = bstm.calibrate_param_registry(calibrated, mock_vnt)
        @test calibrated_vnt.descriptors[:ure_year].shape == (t_N_lip,)

        # 5. Test get_samples with mock chain dictionary
        mock_ch = Dict(
            :sigma_s_idx => reshape([1.0, 1.1, 1.2, 1.3, 1.4], 1, 5),
            :ure_s_idx => randn(s_N_lip, 5)
        )
        sigma_s = bstm.get_samples(mock_ch, calibrated, :s_idx, :sigma)
        @test size(sigma_s, 1) == 5
        @test size(sigma_s, 2) == 1

        ure_s = bstm.get_samples(mock_ch, calibrated, :s_idx, :ure)
        @test size(ure_s, 1) == 5
        @test size(ure_s, 2) == s_N_lip

        # 6. Test canonical _find_parameter
        names_list = ["intercept", "sigma_s_idx_1", "ure_s_idx[1]"]
        @test bstm._find_parameter(names_list, "sigma_s_idx", 1, true) == "sigma_s_idx_1"
        @test bstm._find_parameter(names_list, :sigma_s_idx, 1, true) == "sigma_s_idx_1"
        @test bstm._find_parameter(names_list, "intercept", nothing, false) == "intercept"
        @test bstm._find_parameter(names_list, "ure_s_idx", 1, true) == "ure_s_idx[1]"

        # 7. Test Marginalized AR1 Likelihood & Latent Reconstruction
        ar1_m = bstm.AR1(Normal(0, 1), Exponential(1.0), :marginalized)
        y_res = [1.0, 1.2, 0.9, 1.5, 1.8]
        t_idx = [1, 2, 3, 4, 5]
        ll_m = bstm._ar1_log_marginal_likelihood(y_res, t_idx, 5, 0.7, 0.5, 0.2)
        @test isfinite(ll_m)
        @test ll_m < 0.0

        spec_ar1 = (key = :year, hyper = (n_latent = 5,), params = Dict())
        M_mock = (
            outcomes_N = 1,
            model_arch = "univariate",
            t_idx = t_idx,
            t_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        chain_ar1 = Dict(
            :rho_unconstrained_year => reshape([0.5, 0.6, 0.7], 1, 3),
            :sigma_year => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_ar1 = bstm.get_effects(ar1_m, chain_ar1, spec_ar1, M_mock, nothing)
        @test length(eff_ar1.structured) == 1
        @test size(eff_ar1.structured[1]) == (5, 3)
        @test !all(iszero, eff_ar1.structured[1])

        # 8. Test Marginalized AR2 Likelihood & Latent Reconstruction
        ar2_m = bstm.AR2(Normal(0, 1), Normal(0, 1), Exponential(1.0), :marginalized)
        ll_ar2 = bstm._ar2_log_marginal_likelihood(y_res, t_idx, 5, 0.4, 0.2, 0.5, 0.2)
        @test isfinite(ll_ar2)
        @test ll_ar2 < 0.0

        chain_ar2 = Dict(
            :rho1_unconstrained_year => reshape([0.3, 0.4, 0.5], 1, 3),
            :rho2_unconstrained_year => reshape([0.1, 0.2, 0.1], 1, 3),
            :sigma_year => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_ar2 = bstm.get_effects(ar2_m, chain_ar2, spec_ar1, M_mock, nothing)
        @test length(eff_ar2.structured) == 1
        @test size(eff_ar2.structured[1]) == (5, 3)
        @test !all(iszero, eff_ar2.structured[1])

        # 9. Test Marginalized RW1 Likelihood & Latent Reconstruction
        rw1_m = bstm.RW1(Exponential(1.0), :marginalized)
        rw1_template = bstm.build_structure_template(:rw1, 5)
        ll_rw1 = bstm._rw1_log_marginal_likelihood(y_res, t_idx, 5, rw1_template.matrix,
            0.5, 0.2)
        @test isfinite(ll_rw1)

        spec_rw1 = (key = :year, hyper = (n_latent = 5, Q_template = rw1_template.matrix,
            U = rw1_template.U, L = rw1_template.L), params = Dict())
        chain_rw1 = Dict(
            :sigma_year => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_rw1 = bstm.get_effects(rw1_m, chain_rw1, spec_rw1, M_mock, nothing)
        @test length(eff_rw1.structured) == 1
        @test size(eff_rw1.structured[1]) == (5, 3)
        @test !all(iszero, eff_rw1.structured[1])

        # 10. Test Marginalized RW2 Likelihood & Latent Reconstruction
        rw2_m = bstm.RW2(Exponential(1.0), :marginalized)
        rw2_template = bstm.build_structure_template(:rw2, 5)
        ll_rw2 = bstm._rw2_log_marginal_likelihood(y_res, t_idx, 5, rw2_template.matrix,
            0.5, 0.2)
        @test isfinite(ll_rw2)

        spec_rw2 = (key = :year, hyper = (n_latent = 5, Q_template = rw2_template.matrix,
            U = rw2_template.U, L = rw2_template.L), params = Dict())
        eff_rw2 = bstm.get_effects(rw2_m, chain_rw1, spec_rw2, M_mock, nothing)
        @test length(eff_rw2.structured) == 1
        @test size(eff_rw2.structured[1]) == (5, 3)
        @test !all(iszero, eff_rw2.structured[1])

        # 11. Test Marginalized ICAR Likelihood & Latent Reconstruction
        W_mock = [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0]
        icar_template = bstm.build_structure_template(:icar, 5; W=W_mock)
        icar_m = bstm.ICAR(Exponential(1.0), :marginalized)
        ll_icar = bstm._icar_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix,
            icar_template.L, 0.5, 0.2)
        @test isfinite(ll_icar)

        M_spatial = (
            outcomes_N = 1,
            model_arch = "univariate",
            s_idx = t_idx,
            s_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        spec_icar = (key = :region, hyper = (n_latent = 5,
            Q_template = icar_template.matrix, U = icar_template.U, L = icar_template.L),
            params = Dict())
        chain_icar = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_icar = bstm.get_effects(icar_m, chain_icar, spec_icar, M_spatial, nothing)
        @test length(eff_icar.structured) == 1
        @test size(eff_icar.structured[1]) == (5, 3)
        @test !all(iszero, eff_icar.structured[1])

        # 12. Test Marginalized Leroux Likelihood & Latent Reconstruction
        leroux_m = bstm.Leroux(Beta(1, 1), Exponential(1.0), :marginalized)
        ll_leroux = bstm._leroux_log_marginal_likelihood(y_res, t_idx, 5,
            icar_template.matrix, icar_template.L, 0.6, 0.5, 0.2)
        @test isfinite(ll_leroux)

        chain_leroux = Dict(
            :rho_region => reshape([0.5, 0.6, 0.7], 1, 3),
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_leroux = bstm.get_effects(leroux_m, chain_leroux, spec_icar, M_spatial, nothing)
        @test length(eff_leroux.structured) == 1
        @test size(eff_leroux.structured[1]) == (5, 3)
        @test !all(iszero, eff_leroux.structured[1])

        # 13. Test Marginalized PSpline Likelihood & Latent Reconstruction
        B_mock, _ = bstm.bstm_bspline_basis([1.0, 2.0, 3.0, 4.0, 5.0], 5, 3)
        pspline_m = bstm.PSpline(5, 3, 2, Exponential(1.0), :marginalized)
        ll_ps = bstm._pspline_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix,
            rw2_template.L, 2, 0.5, 0.2)
        @test isfinite(ll_ps)

        spec_ps = (key = :x, hyper = (n_latent = 5, basis_matrix = B_mock,
            Q_template = rw2_template.matrix, U = rw2_template.U, L = rw2_template.L),
            params = Dict(:positional_args => [:x]))
        chain_ps = Dict(
            :sigma_x => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_ps = bstm.get_effects(pspline_m, chain_ps, spec_ps, M_mock, nothing)
        @test length(eff_ps.structured) == 1
        @test size(eff_ps.structured[1]) == (5, 3)
        @test !all(iszero, eff_ps.structured[1])

        # 14. Test Marginalized BYM2 Likelihood & Latent Reconstruction
        bym2_m = bstm.BYM2(Normal(0, 0.5), Exponential(1.0), :marginalized)
        ll_bym2 = bstm._bym2_log_marginal_likelihood(y_res, t_idx, 5, icar_template.U,
            icar_template.L, 0.6, 0.5, 0.2)
        @test isfinite(ll_bym2)

        chain_bym2 = Dict(
            :rho_unconstrained_region => reshape([0.5, 0.6, 0.7], 1, 3),
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_bym2 = bstm.get_effects(bym2_m, chain_bym2, spec_icar, M_spatial, nothing)
        @test length(eff_bym2.structured) == 1
        @test length(eff_bym2.unstructured) == 1
        @test length(eff_bym2.noisy) == 1
        @test size(eff_bym2.structured[1]) == (5, 3)
        @test !all(iszero, eff_bym2.structured[1])
        @test !all(iszero, eff_bym2.noisy[1])

        # 15. Test Marginalized IID Likelihood & Latent Reconstruction
        iid_m = bstm.IID(Exponential(1.0), :marginalized)
        ll_iid = bstm._iid_log_marginal_likelihood(y_res, t_idx, 5, 0.5, 0.2)
        @test isfinite(ll_iid)

        spec_iid = (key = :group, structure = :mixed, var = "group", hyper = (n_latent = 5,
            ), params = Dict())
        M_iid = (
            outcomes_N = 1,
            model_arch = "univariate",
            mixed_idx_group = t_idx,
            y_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        chain_iid = Dict(
            :sigma_group => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_iid = bstm.get_effects(iid_m, chain_iid, spec_iid, M_iid, nothing)
        @test length(eff_iid.structured) == 1
        @test size(eff_iid.structured[1]) == (5, 3)
        @test !all(iszero, eff_iid.structured[1])

        # 16. Test Marginalized Cyclic Likelihood & Latent Reconstruction
        cyclic_m = bstm.Cyclic(5, Exponential(1.0), :marginalized)
        cyclic_template = bstm.build_structure_template(:cyclic, 5)
        ll_cyclic = bstm._cyclic_log_marginal_likelihood(y_res, t_idx, 5,
            cyclic_template.matrix, cyclic_template.L, 0.5, 0.2)
        @test isfinite(ll_cyclic)

        spec_cyclic = (key = :month, hyper = (n_latent = 5,
            Q_template = cyclic_template.matrix, U = cyclic_template.U,
            L = cyclic_template.L), params = Dict())
        M_cyclic = (
            outcomes_N = 1,
            model_arch = "univariate",
            u_idx = t_idx,
            u_N = 5,
            y_obs = y_res,
            noise = 1e-6
        )
        chain_cyclic = Dict(
            :sigma_month => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_cyclic = bstm.get_effects(cyclic_m, chain_cyclic, spec_cyclic, M_cyclic, nothing)
        @test length(eff_cyclic.structured) == 1
        @test size(eff_cyclic.structured[1]) == (5, 3)
        @test !all(iszero, eff_cyclic.structured[1])

        # 17. Test Marginalized BSpline Likelihood & Latent Reconstruction
        bspline_m = bstm.BSpline(5, 3, Exponential(1.0), :marginalized)
        ll_bs = bstm._bspline_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix,
            rw2_template.L, 0.5, 0.2)
        @test isfinite(ll_bs)

        eff_bs = bstm.get_effects(bspline_m, chain_ps, spec_ps, M_mock, nothing)
        @test length(eff_bs.structured) == 1
        @test size(eff_bs.structured[1]) == (5, 3)
        @test !all(iszero, eff_bs.structured[1])

        # 18. Test Marginalized Moran Likelihood & Latent Reconstruction
        moran_m = bstm.Moran(Exponential(1.0), :marginalized)
        moran_eigs = icar_template.U
        ll_moran = bstm._moran_log_marginal_likelihood(y_res, t_idx, 5, moran_eigs, 0.5, 0.2)
        @test isfinite(ll_moran)

        spec_moran = (key = :region, hyper = (n_latent = 5,
            moran_eigenvectors = moran_eigs), params = Dict())
        chain_moran = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_moran = bstm.get_effects(moran_m, chain_moran, spec_moran, M_spatial, nothing)
        @test length(eff_moran.structured) == 1
        @test size(eff_moran.structured[1]) == (5, 3)
        @test !all(iszero, eff_moran.structured[1])

        # 19. Test Marginalized SAR Likelihood & Latent Reconstruction
        sar_m = bstm.SAR(Normal(0, 0.5), Exponential(1.0), :marginalized)
        sar_W = [0.0 1.0 0.0 0.0 0.0; 0.5 0.0 0.5 0.0 0.0; 0.0 0.5 0.0 0.5 0.0; 0.0 0.0 0.5 0.0 0.5; 0.0 0.0 0.0 1.0 0.0]
        sar_eigs = eigvals(sar_W)
        ll_sar = bstm._sar_log_marginal_likelihood(y_res, t_idx, 5, sar_W, sar_eigs, 0.4,
            0.5, 0.2)
        @test isfinite(ll_sar)

        spec_sar = (key = :region, hyper = (n_latent = 5, Q_template = sar_W,
            eigenvalues = sar_eigs), params = Dict())
        chain_sar = Dict(
            :rho_region => reshape([0.3, 0.4, 0.5], 1, 3),
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_sar = bstm.get_effects(sar_m, chain_sar, spec_sar, M_spatial, nothing)
        @test length(eff_sar.structured) == 1
        @test size(eff_sar.structured[1]) == (5, 3)
        @test !all(iszero, eff_sar.structured[1])

        # 20. Test Marginalized TPS Likelihood & Latent Reconstruction
        tps_m = bstm.TPS(5, Exponential(1.0), :marginalized)
        ll_tps = bstm._tps_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix,
            rw2_template.L, 0.5, 0.2)
        @test isfinite(ll_tps)

        spec_tps = (key = :space, hyper = (n_latent = 5, basis_matrix = B_mock,
            Q_template = rw2_template.matrix, L = rw2_template.L, knots = randn(5, 2)),
            params = Dict())
        chain_tps = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_tps = bstm.get_effects(tps_m, chain_tps, spec_tps, M_mock, nothing)
        @test length(eff_tps.structured) == 1
        @test size(eff_tps.structured[1]) == (5, 3)
        @test !all(iszero, eff_tps.structured[1])

        # 21. Test Marginalized Barycentric Likelihood & Latent Reconstruction
        bary_m = bstm.Barycentric(Exponential(1.0), :marginalized)
        ll_bary = bstm._barycentric_log_marginal_likelihood(y_res, B_mock, nothing,
            nothing, 0.5, 0.2)
        @test isfinite(ll_bary)

        spec_bary = (key = :space, hyper = (n_knots = 5, B = B_mock,
            knots = [bstm.Point2D(0.0, 0.0) for _ in 1:5]), params = Dict())
        chain_bary = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_bary = bstm.get_effects(bary_m, chain_bary, spec_bary, M_mock, nothing)
        @test length(eff_bary.structured) == 1
        @test size(eff_bary.structured[1]) == (5, 3)
        @test !all(iszero, eff_bary.structured[1])

        # 22. Test Marginalized BCGN Likelihood & Latent Reconstruction
        bcgn_m = bstm.BCGN(Exponential(1.0), :marginalized)
        bcgn_map = Matrix{Float64}(I, 5, 5)
        ll_bcgn = bstm._bcgn_log_marginal_likelihood(y_res, bcgn_map, icar_template.matrix,
            icar_template.L, 0.5, 0.2)
        @test isfinite(ll_bcgn)

        spec_bcgn = (key = :space, hyper = (n_latent = 5, mapping_matrix = bcgn_map,
            Q_template = icar_template.matrix, L = icar_template.L, set1_indices = [1, 2,
            3, 4, 5]), params = Dict())
        chain_bcgn = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_bcgn = bstm.get_effects(bcgn_m, chain_bcgn, spec_bcgn, M_spatial, nothing)
        @test length(eff_bcgn.structured) == 1
        @test size(eff_bcgn.structured[1]) == (5, 3)
        @test !all(iszero, eff_bcgn.structured[1])

        # 23. Test Marginalized Besag Likelihood & Latent Reconstruction
        besag_m = bstm.Besag(Exponential(1.0), :marginalized)
        ll_besag = bstm._besag_log_marginal_likelihood(y_res, t_idx, 5,
            icar_template.matrix, icar_template.L, 0.5, 0.2)
        @test isfinite(ll_besag)

        spec_besag = (key = :region, hyper = (n_latent = 5,
            Q_template = icar_template.matrix, L = icar_template.L), params = Dict())
        chain_besag = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_besag = bstm.get_effects(besag_m, chain_besag, spec_besag, M_spatial, nothing)
        @test length(eff_besag.structured) == 1
        @test size(eff_besag.structured[1]) == (5, 3)
        @test !all(iszero, eff_besag.structured[1])

        # 24. Test Marginalized RFF Likelihood & Latent Reconstruction
        rff_m = bstm.RFF(Normal(1.0, 0.1), Exponential(1.0), 5, "se", :marginalized)
        ll_rff = bstm._rff_log_marginal_likelihood(y_res, B_mock, 0.5, 0.2)
        @test isfinite(ll_rff)

        coords_mock = randn(5, 2)
        spec_rff = (key = :space, hyper = (n_latent = 5, in_dims = 2, coords = coords_mock,
            W_fixed = randn(2, 5), b_fixed = rand(5)), params = Dict())
        chain_rff = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_rff = bstm.get_effects(rff_m, chain_rff, spec_rff, M_mock, nothing)
        @test length(eff_rff.structured) == 1
        @test size(eff_rff.structured[1]) == (5, 3)
        @test !all(iszero, eff_rff.structured[1])

        # 25. Test Marginalized SPDE Likelihood & Latent Reconstruction
        spde_m = bstm.SPDE(Exponential(1.0), LogNormal(0, 1), :marginalized)
        ll_spde = bstm._spde_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix,
            icar_template.L, 1.2, 0.5, 0.2)
        @test isfinite(ll_spde)

        spec_spde = (key = :region, hyper = (n_latent = 5,
            Q_template = icar_template.matrix, L = icar_template.L, U = icar_template.U),
            params = Dict())
        chain_spde = Dict(
            :sigma_region => reshape([0.4, 0.5, 0.6], 1, 3),
            :kappa_region => reshape([1.0, 1.2, 1.4], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_spde = bstm.get_effects(spde_m, chain_spde, spec_spde, M_spatial, nothing)
        @test length(eff_spde.structured) == 1
        @test size(eff_spde.structured[1]) == (5, 3)
        @test !all(iszero, eff_spde.structured[1])

        # 26. Test Marginalized GP Likelihood & Latent Reconstruction
        gp_m = bstm.GP(LogNormal(0, 1), Exponential(1.0), "se", :marginalized)
        ll_gp = bstm._gp_log_marginal_likelihood(y_res, coords_mock, 0.5, 1.0, :se, 0.2)
        @test isfinite(ll_gp)

        spec_gp = (key = :space, hyper = (n_latent = 5, coords = coords_mock), params = Dict())
        chain_gp = Dict(
            :sigma_space => reshape([0.4, 0.5, 0.6], 1, 3),
            :ls_space => reshape([1.0, 1.2, 1.4], 1, 3),
            :y_sigma => reshape([0.2, 0.2, 0.2], 1, 3)
        )
        eff_gp = bstm.get_effects(gp_m, chain_gp, spec_gp, M_mock, nothing)
        @test length(eff_gp.structured) == 1
        @test size(eff_gp.structured[1]) == (5, 3)
        @test !all(iszero, eff_gp.structured[1])
    end
end

@testset "Core: Manifold & Model Construction" begin
    @testset "Temporal Manifolds" begin
        m_rw1 = bstm.RW1(Distributions.Exponential(1.0), :statespace)
        res_rw1 = bstm.get_precomputes(m_rw1, (N_time=20,), Dict(:variables => :year))
        @test res_rw1.model_type == :rw1
        @test size(res_rw1.Q_template) == (20, 20)
    end

    @testset "Seasonal Manifolds" begin
        m_cyc = bstm.Harmonic(1, Distributions.Exponential(1.0), Distributions.Beta(1, 1),
            12.0, :twocoefficient)
        res_cyc = bstm.get_precomputes(m_cyc, (N_time=12,), Dict(:variables => :month))
        @test res_cyc.model_type == :harmonic
        @test res_cyc.period == 12.0
    end

    @testset "Basis & Continuous Manifolds" begin
        m_bs = bstm.BSpline(15, 3, Distributions.Exponential(1.0), :spectral)
        res_bs = bstm.get_precomputes(m_bs, (N_levels=15,), Dict(:variables => :cov3))
        @test res_bs.model_type == :bspline
        @test size(res_bs.B_matrix) == (15, 15)
    end
end

@testset "Formula Escapes, Operator Precedence & Tensor Registries" begin
    @testset "Escape Handling & Nested Parentheses" begin
        is_esc = bstm._is_escaped
        s1 = "a\\\"b"
        @test is_esc(s1, 3) == true
        s2 = "a\\\\\"b"
        @test is_esc(s2, 4) == false
        s3 = "a\\\\\\\"b"
        @test is_esc(s3, 5) == true

        # split_terms_at_depth with escaped quotes and braces
        s_split1 = "fixed(x, label=\"a + b\") + fixed(y)"
        parts1 = bstm.split_terms_at_depth(s_split1, "+")
        @test length(parts1) == 2
        @test parts1[1] == "fixed(x, label=\"a + b\")"
        @test parts1[2] == "fixed(y)"

        s_split2 = "fixed(x, label=\"foo \\\" + bar\") + fixed(z)"
        parts2 = bstm.split_terms_at_depth(s_split2, "+")
        @test length(parts2) == 2
        @test parts2[2] == "fixed(z)"

        s_split3 = "fixed(x, options={a=1 + 2, b=3}) + fixed(y)"
        parts3 = bstm.split_terms_at_depth(s_split3, "+")
        @test length(parts3) == 2

        # _is_outermost_grouping_parentheses
        is_outer = bstm._is_outermost_grouping_parentheses
        @test is_outer("(a + b)") == true
        @test is_outer("(a) + (b)") == false
        @test is_outer("((a + b))") == true
        @test is_outer("(a + (b * (c + d)))") == true
        @test is_outer("fixed(x)") == false
        @test is_outer("(a))") == false
        @test is_outer("((a)") == false
        @test is_outer("(fixed(x, label=\"test ) string\"))") == true
        @test is_outer("(fixed(x, label=\"test ( string\"))") == true
        @test is_outer("(fixed(x, label=\"test \\\" ) string\"))") == true
    end

    @testset "Operator Precedence" begin
        parse_expr = bstm._parse_rhs_expression

        # Pipe vs Kronecker product
        res1 = parse_expr("fixed(a) |> random(b) ⊗ random(c)")
        @test res1.type == :operator && res1.op == :pipe
        @test res1.children[2].op == :kronecker_product

        res2 = parse_expr("random(a) ⊗ random(b) |> fixed(c)")
        @test res2.type == :operator && res2.op == :pipe
        @test res2.children[1].op == :kronecker_product

        # Composition vs Kronecker product
        res3 = parse_expr("fixed(a) ∘ random(b) ⊗ random(c)")
        @test res3.type == :operator && res3.op == :composition
        @test res3.children[2].op == :kronecker_product

        res4 = parse_expr("random(a) ⊗ random(b) ∘ fixed(c)")
        @test res4.type == :operator && res4.op == :composition
        @test res4.children[1].op == :kronecker_product

        # Addition vs Pipe
        res5 = parse_expr("fixed(a) + fixed(b) |> fixed(c)")
        @test res5.type == :operator && res5.op == :add
        @test res5.children[2].op == :pipe

        res6 = parse_expr("(fixed(a) + fixed(b)) |> fixed(c)")
        @test res6.type == :operator && res6.op == :pipe
        @test res6.children[1].op == :add

        # Full formula parsing
        df_mock = DataFrame(y = [1, 2], cov = [0.1, 0.2], s_idx = [1, 2])
        decomp = bstm.decompose_bstm_formula(
            "y ~ intercept() + fixed(cov) |> random(s_idx, model=bym2)",
            df_mock
        )
        @test decomp.has_intercept == true
        @test length(decomp.modules) >= 1
    end

    @testset "Multi-dimensional Matrix/Tensor Parameters" begin
        in_dim = 3
        hidden_dim = 4
        nbins = 2

        comp_spec = (
            key = :nn_covar,
            component_obj = (hidden_dim = hidden_dim, nbins = nbins),
            hyper = (in_dim = in_dim, n_latent = 0),
            params = Dict{Symbol, Any}()
        )
        M_tensor = (
            model_arch = "univariate",
            outcomes_N = 1,
            add_intercept = true,
            Xfixed_N = 2,
            components = [comp_spec]
        )
        reg = bstm.build_param_registry(M_tensor)

        @test haskey(reg.descriptors, :W1_nn_covar)
        @test reg.descriptors[:W1_nn_covar].shape == (in_dim, hidden_dim)
        @test haskey(reg.descriptors, :W2_nn_covar)
        @test reg.descriptors[:W2_nn_covar].shape == (hidden_dim, nbins)
        @test haskey(reg.descriptors, :b1_nn_covar)
        @test reg.descriptors[:b1_nn_covar].shape == (hidden_dim,)

        # Sample extraction with tensor reshaping
        n_samples = 50
        chain_df = DataFrame()
        for j in 1:hidden_dim
            for i in 1:in_dim
                col_name = Symbol("W1_nn_covar[$i, $j]")
                chain_df[!, col_name] = [Float64(i * 10 + j + s * 100) for s in 1:n_samples]
            end
        end

        reg_chain = bstm.build_param_registry(chain_df)
        @test reg_chain.descriptors[:W1_nn_covar].shape == (in_dim, hidden_dim)

        samples_tensor = bstm.get_param_samples(chain_df, reg, :nn_covar, :W1)
        @test size(samples_tensor) == (n_samples, in_dim, hidden_dim)
        @test samples_tensor[1, 1, 1] == 111.0
        @test samples_tensor[50, 3, 4] == 5034.0

        samples_flat = bstm.get_param_samples(
            chain_df, reg, :nn_covar, :W1; reshape_to_shape = false
        )
        @test size(samples_flat) == (n_samples, in_dim * hidden_dim)
    end

    @testset "Log Transformations with Offsets" begin
        # Positive data
        x_pos = [1.0, 2.0, 10.0]
        @test bstm.apply_transformation(:log, x_pos) ≈ log.(x_pos)

        # Zero-containing data defaults to log1p
        x_zero = [0.0, 1.0, 9.0]
        @test bstm.apply_transformation(:log, x_zero) ≈ log1p.(x_zero)

        # Custom offset
        x_custom = [0.0, 2.0, 5.0]
        @test bstm.apply_transformation(:log, x_custom; offset=0.05) ≈ log.(x_custom .+ 0.05)

        # Negative data without offset throws ArgumentError
        x_neg = [-5.0, 0.0, 5.0]
        @test_throws ArgumentError bstm.apply_transformation(:log, x_neg)
        @test bstm.apply_transformation(:log, x_neg; offset=10.0) ≈ log.(x_neg .+ 10.0)

        # Formula pipeline transformation (non-mutating caller data by default)
        df_trans = DataFrame(val = [0.0, 1.0, 2.0])
        decomp = bstm.decompose_bstm_formula(
            "y ~ intercept() + log(val, offset=0.1) |> fixed()", df_trans
        )
        @test !hasproperty(df_trans, :val_log)
        @test hasproperty(decomp.data, :val_log)
        @test decomp.data.val_log ≈ log.(decomp.data.val .+ 0.1)

        decomp_mut = bstm.decompose_bstm_formula(
            "y ~ intercept() + log(val, offset=0.1) |> fixed()", df_trans; copy_data=false
        )
        @test hasproperty(df_trans, :val_log)
    end

    @testset "Transformation DataFrame Safety & Collision Avoidance" begin
        # Column collision handling
        df_collision = DataFrame(
            temp = [1.0, 2.0, 3.0, 4.0, 5.0],
            temp_zscore = [99.0, 99.0, 99.0, 99.0, 99.0],
            y = [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        decomp_col = bstm.decompose_bstm_formula("likelihood(y) ~ zscore(temp) |> fixed()", df_collision)
        @test hasproperty(decomp_col.data, :temp_zscore_2)
        @test decomp_col.data.temp_zscore == [99.0, 99.0, 99.0, 99.0, 99.0]
    end

    @testset "Multivariate Hyperprior Sharing (Item 8)" begin
        @test bstm.is_param_shared(true, :sigma) == true
        @test bstm.is_param_shared(false, :sigma) == false
        @test bstm.is_param_shared(:all, :sigma) == true
        @test bstm.is_param_shared(:sigma, :sigma) == true
        @test bstm.is_param_shared(:sigma, :rho) == false
        @test bstm.is_param_shared([:sigma, :range], :sigma) == true
        @test bstm.is_param_shared([:sigma, :range], :rho) == false

        # Shared intercept
        m_cfg_shared_int = (
            model_arch = "multivariate",
            outcomes_N = 3,
            add_intercept = true,
            shared_intercept = true,
            intercept_prior = Normal(0, 1),
            Xfixed_N = 0,
            components = []
        )
        reg_shared_int = bstm.build_param_registry(m_cfg_shared_int)
        @test haskey(reg_shared_int.descriptors, :intercept)
        @test reg_shared_int.descriptors[:intercept].is_shared == true
        @test !haskey(reg_shared_int.descriptors, :intercept_1)

        # Fine-grained component sharing
        m_cfg_shared_comp = (
            model_arch = "multivariate",
            outcomes_N = 2,
            add_intercept = false,
            Xfixed_N = 0,
            components = [
                (
                    key = :time,
                    params = Dict{Symbol, Any}(:shared => [:sigma]),
                    component_obj = bstm.AR1(Normal(0, 1), Exponential(1.0), :statespace),
                    hyper = (n_latent = 10,)
                )
            ]
        )
        reg_comp = bstm.build_param_registry(m_cfg_shared_comp)
        @test haskey(reg_comp.descriptors, :sigma_time)
        @test reg_comp.descriptors[:sigma_time].is_shared == true
        @test haskey(reg_comp.descriptors, :rho_unconstrained_time_1)
        @test haskey(reg_comp.descriptors, :rho_unconstrained_time_2)
        @test reg_comp.descriptors[:rho_unconstrained_time_1].is_shared == false
    end

    @testset "Parameter Extraction Disambiguation (Item 13)" begin
        reg_disambig = bstm.ParamRegistry()
        bstm.add_descriptor!(reg_disambig, bstm.ParamDescriptor(:Xfixed_beta; role=:fixed_coef))
        bstm.add_descriptor!(reg_disambig, bstm.ParamDescriptor(:beta_prop; role=:fixed_coef))

        @test bstm.find_chain_param(reg_disambig, "beta"; exact=true) == ""
        param_found = bstm.find_chain_param(reg_disambig, "beta"; exact=false)
        @test !isempty(param_found)
    end

    @testset "Kronecker Product Composition Validation (Item 14)" begin
        df_kronecker = DataFrame(y = rand(10), s = 1:10, t = 1:10, x = rand(10))
        @test_throws ArgumentError bstm.decompose_bstm_formula(
            "likelihood(y) ~ intercept() ⊗ fixed(x)", df_kronecker
        )
    end

    @testset "Bare Fixed Effect Term Parsing" begin
        p_scot = bstm.bstm_data("scottish_lip")
        df_bare = p_scot.data
        W_bare = p_scot.au.W

        m_bare = @bstm(
            likelihood(y_gauss) ~ 1 + cov1 + random(s_idx, model=bym2) + random(year, model=ar1),
            df_bare,
            W = W_bare,
            verbose = false
        )
        @test m_bare !== nothing
        @test hasproperty(m_bare, :args)
    end
end

