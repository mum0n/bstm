using Test
try
    using bstm
catch
    include(joinpath(@__DIR__, "..", "bstm.jl"))
    using .bstm
end
using DynamicPPL
using Plots
using Distributions
using LinearAlgebra
using DataFrames
using CategoricalArrays
using Turing
using Random
using SparseArrays
using Clustering
using LogExpFunctions
using Graphs
using StatsModels # For @formula, EffectsCoding

const bstm_Likelihood = bstm.bstm_Likelihood

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
stable_logdiffexp(a, b) = a + LogExpFunctions.log1mexp(b - a)

# --- New Helper Functions for Testing Components ---
# These will be needed to mock inputs for get_priors, get_updates, get_effects

# Mock M object (simplified for testing purposes)
function mock_M_config(N_obs, N_areas, N_time, model_arch="univariate")
    return (
        data = DataFrame(y=rand(N_obs), s_idx=repeat(1:N_areas, inner=N_time)[1:N_obs],
            t_idx=repeat(1:N_time, outer=N_areas)[1:N_obs], group_var=repeat(1:N_areas,
            inner=N_obs÷N_areas)[1:N_obs]),
        model_arch = model_arch,
        technical = Dict(
            :component_levels => Dict(), # To be filled by specific component tests
            :component_indices => Dict() # To be filled by specific component tests
        )
    )
end

function mock_spec(key, hyper_obj=NamedTuple(), params=Dict(), structure=:any)
    return (
        key = Symbol(key),
        structure = structure,
        var = string(key),
        hyper = hyper_obj,
        params = params
    )
end

# Mock chain object (simplified for testing parameter extraction)
# This mock directly returns values from a dictionary, bypassing MCMCChains.Chains object
#   complexity
function mock_chain(param_names_and_values::Dict, n_samples=10)
    mock_params = Dict{Symbol, Matrix{Float64}}()
    for (name, val) in param_names_and_values
        if val isa Vector # For scalar parameters, expand to (1, n_samples)
            mock_params[Symbol(name)] = reshape(val, 1, :)
        elseif val isa Matrix # For vector parameters, ensure correct shape
            mock_params[Symbol(name)] = val
        else # For single scalar values, expand to (1, n_samples)
            mock_params[Symbol(name)] = fill(Float64(val), 1, n_samples)
        end
    end
    return mock_params
end



# --- Test Suite ---

@testset "BSTM Comprehensive Test Suite" begin

    @testset "Core Components & Interfaces" begin
        @testset "split_terms_at_depth" begin
            @test bstm.split_terms_at_depth("a + b(c+d) + e", "+") == ["a", "b(c+d)", "e"]
            @test bstm.split_terms_at_depth("a", "+") == ["a"]
            @test bstm.split_terms_at_depth("a |> log", " |> ") == ["a", "log"]
        end

        @testset "Comprehensive Formula Parsing" begin
            # Use bstm_data to generate a rich dataset for parsing
            data = bstm.bstm_data("advanced", s_N=30, t_N=12)
            W = create_chain_adj_matrix(30)
            
            # This formula uses variable names present in the "advanced" dataset
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
            p_data, _ = bstm.bstm_data("scottish_lip")
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

            # Test get_effects with mock chain on marginalized AR1
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
            @test !all(iszero, eff_ar1.structured[1]) # Verified non-zero reconstruction

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

    @testset "Likelihood Engine Taxonomy" begin
        @testset "Discrete Families (ZI & Hurdle)" begin
            mu_p = exp(1.0)
            phi = 0.2
            d_pois = bstm.bstm_Likelihood("poisson", [1.0]; phi_zi=phi)
            ana_pois = LogExpFunctions.logsumexp(log(phi), log(1-phi) + logpdf(Poisson(mu_p), 0))
            @test isapprox(Distributions.logpdf(d_pois, 0.0), ana_pois)

            mu_nb = exp(1.5)
            r_val = 2.0
            d_nb_h = bstm.bstm_Likelihood("negbin", [1.5]; r_nb=r_val, phi_hurdle=0.9, hurdle=0.0)
            dist_nb = NegativeBinomial(r_val, r_val/(r_val + mu_nb))
            ana_nb_h = log(0.9) + logpdf(dist_nb, 2.0) - logccdf(dist_nb, 0.0)
            @test isapprox(Distributions.logpdf(d_nb_h, 2.0), ana_nb_h)

            d_bin_int = bstm.bstm_Likelihood("binomial", [0.0]; trial=10, censor_lower=3.0,
                censor_upper=5.0)
            dist_bin = Binomial(10, LogExpFunctions.logistic(0.0))
            ana_bin_int = stable_logdiffexp(logcdf(dist_bin, 5.0), logcdf(dist_bin, 2.0))
            @test isapprox(Distributions.logpdf(d_bin_int, NaN), ana_bin_int)
        end

        @testset "Continuous Families (Censoring)" begin
            d_gauss = bstm.bstm_Likelihood("gaussian", [1.0]; sigma_y=0.5, censor_lower=2.0)
            @test isapprox(Distributions.logpdf(d_gauss, NaN), logccdf(Normal(1.0, 0.5), 2.0))

            d_beta = bstm.bstm_Likelihood("beta", [-0.5]; extra_params=20.0)
            mu_b = LogExpFunctions.logistic(-0.5)
            dist_beta = Beta(mu_b * 20.0, (1-mu_b) * 20.0)
            @test isapprox(Distributions.logpdf(d_beta, 0.4), logpdf(dist_beta, 0.4))

            d_ln = bstm.bstm_Likelihood("lognormal", [0.5]; sigma_y=0.3, censor_upper=1.5)
            mu_ln = 0.5 - (0.3^2)/2.0
            @test isapprox(Distributions.logpdf(d_ln, NaN), logcdf(LogNormal(mu_ln, 0.3), 1.5))
        end
    end

    @testset "Core: Manifold & Model Construction" begin
        mock_inputs = Dict(:s_N => 10, :t_N => 20, :W => create_chain_adj_matrix(10))

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

    @testset "Spatial Partitioning Engine" begin
        s_N = 100
        t_N = 15
        coords = rand(s_N, 2) .* 100
        coords_tuples = [(coords[i, 1], coords[i, 2]) for i in 1:s_N]
        t_idx = repeat(1:t_N, inner=cld(s_N, t_N))[1:s_N]

        partitioning_methods = [:cvt, :kvt, :qvt, :bvt, :avt, :hvt, :lattice]

        for method in partitioning_methods
            @testset "Method: $method" begin
                au = bstm.assign_spatial_units(
                    coords_tuples;
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
                @test length(au.s_idx) == s_N
            end
        end
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

    @testset "ComponentModel Interface Unit Tests" begin
        # Test IID Component
        @testset "IID Component" begin
            N_obs, N_levels = 100, 10
            m_iid = bstm.IID(Distributions.Exponential(1.0), :noncentered)
            
            mock_M_ds = Dict(:data => DataFrame(group_var=repeat(1:N_levels,
                inner=N_obs÷N_levels)[1:N_obs]))
            mock_mod_data = Dict(:variables => :group_var)
            
            mock_M_pc = (data=mock_M_ds[:data],)
            res_pc = bstm.get_precomputes(m_iid, mock_M_pc, mock_mod_data)
            @test res_pc == NamedTuple() || res_pc == (n_latent=0,)

            mock_M_priors = (technical=(component_levels=Dict(:group_var => N_levels),),)
            mock_spec_priors = mock_spec(:group_var)
            priors_str = bstm.get_priors(m_iid, mock_spec_priors, "univariate", nothing,
                mock_M_priors)
            @test contains(priors_str, "sigma_group_var ~ Exponential(1.0)")
            @test contains(priors_str, "ure_group_var ~ MvNormal(zeros(T,")

            mock_M_updates = (technical=(component_indices=Dict(:group_var => mock_M_ds[:data].group_var),), model_arch="univariate")
            mock_spec_updates = mock_spec(:group_var)
            updates_str = bstm.get_updates(m_iid, mock_spec_updates, "univariate", nothing,
                mock_M_updates)
            @test contains(updates_str, "sre_group_var = ure_group_var .* sigma_group_var")
            @test contains(updates_str, "view(sre_group_var,")

            mock_chain_effects = mock_chain(Dict(:sigma_group_var => 0.5,
                :ure_group_var => randn(N_levels, 10)), 10)
            mock_M_effects = (technical=(component_indices=Dict(:group_var => mock_M_ds[:data].group_var),), model_arch="univariate")
            mock_spec_effects = mock_spec(:group_var)
            p_names_effects = collect(keys(mock_chain_effects))
            
            effects_result = bstm.get_effects(m_iid, mock_chain_effects, mock_spec_effects,
                (outcomes_N=1, model_arch="univariate", technical=mock_M_effects.technical),
                nothing)
            @test size(effects_result.structured[1]) == (N_obs, 10)
        end

        # Test Leroux Component
        @testset "Leroux Component" begin
            N_areas, N_obs = 10, 100
            W_leroux = create_chain_adj_matrix(N_areas)
            m_leroux = bstm.Leroux(Distributions.Beta(1, 1), Distributions.Exponential(1.0),
                :spectral)

            mock_M_ds = Dict(:data => DataFrame(s_idx=repeat(1:N_areas,
                inner=N_obs÷N_areas)[1:N_obs]), :W => W_leroux)
            mock_mod_data = Dict(:variables => :s_idx)

            mock_M_pc = (data=mock_M_ds[:data], W=W_leroux, s_N=N_areas)
            precomputes = bstm.get_precomputes(m_leroux, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :Q_template)
            @test size(precomputes.Q_template) == (N_areas, N_areas)

            mock_M_priors = (technical=(component_levels=Dict(:s_idx => N_areas),),)
            mock_spec_priors = mock_spec(:s_idx, precomputes)
            priors_str = bstm.get_priors(m_leroux, mock_spec_priors, "univariate", nothing,
                mock_M_priors)
            @test contains(priors_str, "sigma_s_idx ~ Exponential(1.0)")
            @test contains(priors_str, "rho_s_idx ~ Beta(1.0, 1.0)")
            @test contains(priors_str, "ure_s_idx ~ MvNormal(zeros(T, 10), I)")

            mock_M_updates = (technical=(component_indices=Dict(:s_idx => repeat(1:N_areas,
                inner=N_obs÷N_areas)[1:N_obs]), ), model_arch="univariate")
            updates_str = bstm.get_updates(m_leroux, mock_spec_priors, "univariate", nothing,
                mock_M_updates)
            @test contains(updates_str, "sre_s_idx = hyper.U * (diag_D .* ure_s_idx)")
            @test contains(updates_str, "eta = eta .+ view(sre_s_idx, M.s_idx)")
        end

        # Test GP Component
        @testset "GP Component" begin
            N_obs, N_dims = 100, 2
            m_gp = bstm.GP(Distributions.Gamma(2, 0.5), Distributions.Exponential(1.0), "se",
                :noncentered)
            
            mock_M_ds = Dict(:data => DataFrame(x=rand(N_obs), y=rand(N_obs)))
            mock_mod_data = Dict(:variables => [:x, :y], :params => Dict(:coords => rand(N_obs,
                N_dims)))

            mock_M_pc = (data=mock_M_ds[:data],)
            precomputes = bstm.get_precomputes(m_gp, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :n_latent)
            @test precomputes.n_latent == N_obs

            mock_M_priors = (technical=(component_levels=Dict(),),)
            mock_spec_priors = mock_spec(:x_y, precomputes)
            priors_str = bstm.get_priors(m_gp, mock_spec_priors, "univariate", nothing,
                mock_M_priors)
            @test contains(priors_str, "sigma_x_y ~ Exponential(1.0)")
            @test contains(priors_str, "ls_x_y ~ Gamma(2.0, 0.5)")
            @test contains(priors_str, "ure_x_y ~ MvNormal(zeros(T, 100), I)")

            mock_M_updates = (technical=(component_indices=Dict(),), model_arch="univariate")
            updates_str = bstm.get_updates(m_gp, mock_spec_priors, "univariate", nothing,
                mock_M_updates)
            @test contains(updates_str, "sre_x_y = F_gp.L * ure_x_y")
            @test contains(updates_str, "eta = eta .+ sre_x_y")
        end

        # Test Harmonic Component
        @testset "Harmonic Component" begin
            N_obs, N_time, n_harm, period = 120, 12, 2, 12.0
            m_harm = bstm.Harmonic(n_harm, Distributions.Exponential(1.0),
                Distributions.Beta(1, 1), period, :twocoefficient)
            
            mock_M_ds = Dict(:data => DataFrame(month=rand(1:N_time, N_obs)))
            mock_mod_data = Dict(:variables => :month)

            mock_M_pc = (data=mock_M_ds[:data], N_time=N_time)
            precomputes = bstm.get_precomputes(m_harm, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :u_N)
            @test precomputes.u_N == N_time

            mock_M_priors = (technical=(component_levels=Dict(:month => N_time),),)
            mock_spec_priors = mock_spec(:month, precomputes)
            priors_str = bstm.get_priors(m_harm, mock_spec_priors, "univariate", nothing,
                mock_M_priors)
            @test contains(priors_str, "beta_cos_month ~ filldist(Normal(0.0, 1.0), 2)")
            @test contains(priors_str, "beta_sin_month ~ filldist(Normal(0.0, 1.0), 2)")

            mock_M_updates = (technical=(component_indices=Dict(:month => mock_M_ds[:data].month),), model_arch="univariate")
            updates_str = bstm.get_updates(m_harm, mock_spec_priors, "univariate", nothing,
                mock_M_updates)
            @test contains(updates_str, "sre_month = zeros(T_num, u_N_val)")
            @test contains(updates_str, "eta = eta .+ view(sre_month, u_idx_val)")
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

    @testset "bstm_data Monolithic Synthetic Data Generator" begin
        @testset "Default scottish_lip Dataset & Covariate Completeness" begin
            p_out, n_out = bstm.bstm_data() # Default "scottish_lip"
            df = p_out.data
            au = p_out.au

            @test df isa DataFrame
            @test au.W isa AbstractMatrix
            @test size(df, 1) > 0

            required_cols = [
                :y, :y_rate, :y_bin, :y_gauss, :y_pois, :ordinal_y,
                :y_cat1, :y_cat2, :y_cat3, :counts, :t_idx, :year, :month, :day,
                :region, :district, :group, :group_id, :group_var, :cell_area,
                :effort, :removal, :removal_total, :proxy_val, :predator_pop,
                :recruitment, :habitat, :species_1, :species_2, :species_3,
                :age_1, :age_2, :age_3, :class_1, :class_2, :class_3, :class_4,
                :s_idx, :s_x, :s_y
            ]
            for col in required_cols
                @test col in propertynames(df)
            end
        end

        @testset "All Supported Synthetic Dataset Types" begin
            # This test remains the same as it's testing the data generator itself.
            df_ord = bstm.bstm_data("ordinal"; n_obs=100)
            @test df_ord isa DataFrame
            @test :ordinal_y in propertynames(df_ord)
        end
    end

    @testset "Likelihood Families & Distributions Correctness" begin
        # This test remains the same as it tests the likelihood struct directly.
        nb_lik = bstm.bstm_Likelihood("negbin", [1.0]; r_nb=2.0)
        logp_nb = logpdf(nb_lik, 3)
        @test isfinite(logp_nb)
    end

    @testset "Plotting Subsystem" begin
        # 1. Theme generator
        thm = bstm.create_theme(fontsize=11)
        @test haskey(thm, :titlefontsize)
        @test thm.size == (900, 600)

        # 2. Choropleth plotting
        polys = [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)]
        ]
        vals = [1.5, 2.5]
        p_choro = bstm.choropleth(polys, vals; title="Test Choro")
        @test p_choro isa Plots.Plot

        # Test flipped argument order
        p_choro2 = bstm.choropleth(vals, polys)
        @test p_choro2 isa Plots.Plot

        # 3. Timeseries with Credible Interval
        ts_p = bstm.timeseries_ci(1:5, [1.0, 2.0, 3.0, 2.5, 2.0], [0.8, 1.7, 2.5, 2.0, 1.5],
            [1.2, 2.3, 3.5, 3.0, 2.5]; title="Trend")
        @test ts_p isa Plots.Plot

        # 4. Spatial Graph Plot
        cents = [(0.5, 0.5), (1.5, 0.5)]
        g = Graphs.SimpleGraph(2)
        Graphs.add_edge!(g, 1, 2)
        sg_p = bstm.spatial_graph_plot(cents, g; polygons=polys)
        @test sg_p isa Plots.Plot

        # Test keyword-based au dispatch
        au_mock = (polygons=polys, centroids=cents, graph=g, hull_coords=polys[1])
        sg_p2 = bstm.spatial_graph_plot(au=au_mock)
        @test sg_p2 isa Plots.Plot

        # 5. Render Paths & Occupancy
        path_mock = [[(0.5, 0.5), (1.5, 0.5)]]
        p_paths = bstm.render_paths!(deepcopy(sg_p), path_mock)
        @test p_paths isa Plots.Plot

        occ_p = bstm.map_point_occupancy(polys, cents, 1, 1)
        @test occ_p isa Plots.Plot

        # 6. Save Plot
        tmp_file = joinpath(tempdir(), "test_bstm_plot.png")
        saved_path = bstm.save_plot(p_choro, tmp_file)
        @test isfile(saved_path)
        try; rm(saved_path; force=true); catch; end

        # 7. Test Spatiotemporal Time Coordinates in bstm_plots
        # When data has 56 spatial units and 10 time units (560 rows),
        # temporal trend plot x-axis must have length 10 (1:10), NOT 560.
        df_st_mock = DataFrame(
            s_idx = repeat(1:56, inner=10),
            year = repeat(1:10, outer=56),
            y_gauss = randn(560)
        )
        res_mock = (
            effects = Dict(
                :year => (
                    structured = (
                        mean = repeat(sin.(1:10), outer=56),
                        lower = repeat(sin.(1:10) .- 0.2, outer=56),
                        upper = repeat(sin.(1:10) .+ 0.2, outer=56)
                    ),
                )
            ),
            predictions = (
                denoised = (
                    mean = randn(560),
                    lower = randn(560) .- 0.5,
                    upper = randn(560) .+ 0.5,
                    observed = randn(560)
                ),
            )
        )
        M_mock_st = (
            s_N = 56,
            t_N = 10,
            t_idx_var = :year,
            t_idx = df_st_mock.year,
            s_idx = df_st_mock.s_idx,
            data = df_st_mock
        )
        plots_out = bstm._bstm_plots_impl(nothing, nothing, res_mock, M_mock_st; data=df_st_mock)
        @test haskey(plots_out.plots, :temporal)
        @test length(plots_out.plots_data[:temporal].time) == 10
        @test collect(plots_out.plots_data[:temporal].time) == 1:10
    end

    @testset "Partitioning Subsystem Enhancements" begin
        # Synthetic coordinates
        rng = MersenneTwister(123)
        sx = rand(rng, 50) .* 10.0
        sy = rand(rng, 50) .* 10.0
        df_geo = DataFrame(s_x=sx, s_y=sy, t_idx=rand(rng, 1:4, 50))

        # 1. Hexagonal partitioning
        au_hex = bstm.assign_spatial_units(sx, sy; area_method=:hexagonal, target_units=8)
        @test length(au_hex.centroids) > 0
        @test size(au_hex.W, 1) == length(au_hex.centroids)
        @test length(au_hex.s_idx) == 50

        # 2. Fast Lattice partitioning
        au_lat = bstm.assign_spatial_units(sx, sy; area_method=:lattice, target_units=9)
        @test length(au_lat.centroids) > 0
        @test size(au_lat.W, 1) == length(au_lat.centroids)

        # 3. Spatial Weights Matrix (Row standardization)
        W_row = bstm.spatial_weights_matrix(au_hex.W; style=:row_standardized)
        @test all(r -> isapprox(r, 1.0; atol=1e-5) || isapprox(r, 0.0; atol=1e-5), sum(W_row,
            dims=2))

        # 4. Spatial KNN and Radius Graphs
        coords_tuple = tuple.(sx, sy)
        g_knn, W_knn = bstm.spatial_knn_graph(coords_tuple, 4)
        @test Graphs.nv(g_knn) == 50
        @test size(W_knn) == (50, 50)

        g_rad, W_rad = bstm.spatial_radius_graph(coords_tuple, 3.0)
        @test Graphs.nv(g_rad) == 50

        # 5. Spatial Block Cross-Validation
        folds_km = bstm.spatial_block_cv(sx, sy; n_folds=5, method=:kmeans)
        @test length(folds_km) == 50
        @test length(unique(folds_km)) <= 5

        folds_grid = bstm.spatial_block_cv(sx, sy; n_folds=4, method=:grid)
        @test length(folds_grid) == 50

        # 6. Spatiotemporal Units Assignment
        st_res = bstm.assign_spatiotemporal_units(df_geo; target_units=6, area_method=:hexagonal)
        @test length(st_res.st_idx) == 50
        @test st_res.S == length(st_res.au_spatial.centroids)
        @test st_res.T == 4
        @test st_res.scaling_factor_spatial > 0.0

        # 7. Granular Polygon Sizing & Exact Count Control
        # Exact units test
        au_exact = bstm.assign_spatial_units(sx, sy; area_method=:cvt, target_units=7,
            exact_units=true)
        @test au_exact.n_units == 7
        @test length(au_exact.centroids) == 7
        @test length(au_exact.areas) == 7
        @test haskey(au_exact.metrics, :mean_area)
        @test haskey(au_exact.metrics, :total_area)

        # Target area test
        au_area = bstm.assign_spatial_units(sx, sy; area_method=:lattice, target_area=20.0)
        @test au_area.n_units > 0
        @test all(a -> a > 0.0, au_area.areas)

        # Grid resolution (rows, cols)
        au_grid_res = bstm.assign_spatial_units(sx, sy; area_method=:lattice,
            grid_resolution=(4, 5))
        @test au_grid_res.n_units > 0

        # Minimum area & merging small slivers
        au_min_area = bstm.assign_spatial_units(sx, sy; area_method=:hexagonal,
            target_units=10, min_area=2.0, merge_small_polygons=true)
        @test all(a -> a >= 2.0 || isapprox(a, 2.0; atol=1e-3), au_min_area.areas)
    end

    @testset "Model State & Results Persistence (JLD2 & DuckDB)" begin
        # 1. Setup simple model & sampling
        rng = MersenneTwister(123)
        df_test = DataFrame(
            y = rand(rng, 1:10, 25),
            x = randn(rng, 25),
            group = rand(rng, 1:5, 25)
        )
        m_test = @bstm(likelihood(y, family=poisson) ~ intercept() + fixed(x) + random(group,
            model=iid), df_test, verbose=false)
        chn_test = sample(m_test, MH(), 30; progress=false)
        res_test = model_results_comprehensive(m_test, chn_test)

        temp_dir = mktempdir()
        jld2_path = joinpath(temp_dir, "test_model.jld2")
        duckdb_path = joinpath(temp_dir, "test_results.duckdb")
        bundle_base = joinpath(temp_dir, "test_bundle")

        # 2. Test JLD2 Model State Save & Load
        saved_file = save_bstm_model(jld2_path, m_test; chain=chn_test,
            metadata=Dict("test_run"=>"v1"))
        @test isfile(saved_file)

        loaded_bundle = load_bstm_model(saved_file)
        @test hasproperty(loaded_bundle, :model)
        @test loaded_bundle.model isa DynamicPPL.Model
        @test hasproperty(loaded_bundle, :chain)
        @test !isnothing(loaded_bundle.chain)
        @test loaded_bundle.metadata["test_run"] == "v1"

        # Test sampling on reloaded model
        chn_reloaded = sample(loaded_bundle.model, MH(), 10; progress=false)
        @test size(chn_reloaded, 1) == 10

        # 3. Test Chain Appending & Extension
        chn_extended = append_posterior_samples(chn_test, chn_reloaded)
        @test size(chn_extended, 1) == size(chn_test, 1) + size(chn_reloaded, 1)

        chn_ext_sampled = extend_sampling(m_test, chn_test, 10; sampler=MH(), progress=false)
        @test size(chn_ext_sampled, 1) == size(chn_test, 1) + 10

        # 4. Test DuckDB Results Save & Query
        saved_db = save_bstm_results(duckdb_path, res_test; model=m_test, chain=chn_test)
        @test isfile(saved_db)

        # SQL Queries via DuckDB
        df_metrics = query_duckdb(duckdb_path, "SELECT * FROM metrics")
        @test df_metrics isa DataFrame
        @test "metric" in names(df_metrics)

        df_params = query_duckdb(duckdb_path, "SELECT * FROM parameter_stats")
        @test df_params isa DataFrame

        df_preds = query_duckdb(duckdb_path, "SELECT * FROM predictions")
        @test df_preds isa DataFrame
        @test nrow(df_preds) == 25

        # Reload DuckDB results
        res_reloaded = load_bstm_results(duckdb_path)
        @test hasproperty(res_reloaded, :metrics)
        @test hasproperty(res_reloaded, :parameters)
        @test hasproperty(res_reloaded, :predictions)
        @test hasproperty(res_reloaded, :plots_data)

        # 5. Test Unified Bundle Save & Load
        bundle_paths = save_bstm_bundle(bundle_base, m_test, chn_test, res_test;
            metadata=Dict("experiment"=>"bundle_test"))
        @test isfile(bundle_paths.model_file)
        @test isfile(bundle_paths.results_file)

        loaded_all = load_bstm_bundle(bundle_base)
        @test loaded_all.model isa DynamicPPL.Model
        @test !isnothing(loaded_all.chain)
        @test hasproperty(loaded_all.results, :metrics)
        @test loaded_all.metadata["experiment"] == "bundle_test"

        # 6. Test Sequential Prior Extraction
        extracted_priors = extract_posterior_priors(res_test)
        @test haskey(extracted_priors, :intercept)
        @test extracted_priors[:intercept] isa Normal

        # 7. Test Multi-Model Ensembling & BMA
        ens_db = joinpath(temp_dir, "ensemble.duckdb")
        m2_test = @bstm(likelihood(y, family=poisson) ~ intercept() + fixed(x), df_test,
            verbose=false)
        c2_test = sample(m2_test, MH(), 30; progress=false)
        r2_test = model_results_comprehensive(m2_test, c2_test)

        ensemble_dict = Dict(
            :full => (model=m_test, chain=chn_test, results=res_test),
            :reduced => (model=m2_test, chain=c2_test, results=r2_test)
        )
        df_registry = save_model_ensemble(ens_db, ensemble_dict)
        @test nrow(df_registry) == 2
        @test "bma_weight" in names(df_registry)

        df_bma = bma_weighted_predictions(ens_db)
        @test nrow(df_bma) == 25
        @test "bma_pred_mean" in names(df_bma)

        # 8. Test Parquet & CSV Export
        pq_path = joinpath(temp_dir, "predictions.parquet")
        csv_path = joinpath(temp_dir, "predictions.csv")
        export_results_to_parquet(duckdb_path, "predictions", pq_path)
        export_results_to_csv(duckdb_path, "predictions", csv_path)
        @test isfile(pq_path)
        @test isfile(csv_path)

        # 9. Test DuckDB Compaction
        compact_duckdb(duckdb_path)

        # 10. Test GeoJSON Export
        au_dummy = (
            polygons = Vector{Tuple{Float64, Float64}}[
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
                [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
            ],
            centroids = [(0.5, 0.5), (1.5, 0.5)]
        )
        geojson_path = joinpath(temp_dir, "spatial_units.geojson")
        export_spatial_results_to_geojson(geojson_path, res_test, au_dummy)
        @test isfile(geojson_path)
        @test filesize(geojson_path) > 0

        # Cleanup
        rm(temp_dir; recursive=true, force=true)
    end

    @testset "Movement & ADR Telemetry Engine" begin
        # 1. Test simulation bundle generation
        sim = generate_ADR_simulation_bundle(100.0, 9, 3, 10; area_method=:grid,
            rng=MersenneTwister(123))
        @test nrow(sim.data) == sim.n_spatial * 3
        @test nrow(sim.telemetry_data) == 20 # 10 release + 10 recapture
        @test hasproperty(sim.au, :W)
        @test hasproperty(sim.au, :centroids)

        # 2. Test velocity field computation
        prob_vec = fill(0.5, 9)
        vel = compute_velocity_field(prob_vec, 3, 1.0)
        @test length(vel.vx) == 9
        @test length(vel.vy) == 9

        # 3. Test multi-step transition & trajectory simulations
        Gamma_base = [0.6 0.4; 0.3 0.7]
        Gamma_2 = calculate_multistep_transition(Gamma_base, 2)
        @test size(Gamma_2) == (2, 2)
        @test isapprox(sum(Gamma_2[1, :]), 1.0; atol=1e-5)

        au_simple = (
            centroids = [(0.0, 0.0), (1.0, 1.0)],
            polygons = Vector{Tuple{Float64, Float64}}[
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
                [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]
            ],
            W = [0 1; 1 0]
        )
        paths = simulate_posterior_trajectories(Gamma_base, [1, 2], 5, au_simple;
            rho_persistence=1.0)
        @test size(paths) == (2, 6)
        @test all(1 .<= paths .<= 2)

        # 4. Test regional connectivity and A/D ratio
        C = calculate_regional_connectivity(Gamma_base, ["A", "B"])
        @test size(C) == (2, 2)

        p_ad = plot_ad_ratio_distribution([1.0, 2.0], [0.5, 0.5])
        @test p_ad isa Plots.Plot
    end

    @testset "NNGP and MCAR Spatial Components" begin
        # Test 1: NNGP model definition
        df_nngp = DataFrame(
            s_x = [0.1, 0.4, 0.7, 0.2, 0.8, 0.5, 0.9, 0.3],
            s_y = [0.2, 0.5, 0.1, 0.8, 0.9, 0.3, 0.6, 0.7],
            y = randn(MersenneTwister(42), 8)
        )
        m_nngp = @bstm(likelihood(y, family=gaussian) ~ intercept() + random(s_x, model=nngp,
            m=3, kernel=:exponential), df_nngp, verbose=false)
        @test m_nngp isa DynamicPPL.Model
        chn_nngp = sample(m_nngp, MH(), 20; progress=false)
        @test size(chn_nngp, 1) == 20

        # Test 2: MCAR model definition
        df_mcar = DataFrame(
            s_idx = [1, 2, 3, 4, 1, 2, 3, 4],
            y = rand(MersenneTwister(42), 0:10, 8)
        )
        W_4 = [0 1 1 0; 1 0 0 1; 1 0 0 1; 0 1 1 0]
        m_mcar = @bstm(likelihood(y, family=poisson) ~ intercept() + random(s_idx, model=mcar,
            K=2), df_mcar, W=W_4, verbose=false)
        @test m_mcar isa DynamicPPL.Model
        chn_mcar = sample(m_mcar, MH(), 20; progress=false)
        @test size(chn_mcar, 1) == 20
    end

    @testset "Composite Gibbs Sampling & Optimal Sampler (Dual Promotion Verification)" begin
        # Test composite Gibbs sampling with ICAR + AR1 and Fixed Effects + Offsets
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

        # Sample with Gibbs sampler - verify no MethodError: no method matching
        #   Float64(::ForwardDiff.Dual)
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

        # Test bstm_Likelihood random generation (prevent recursion / StackOverflowError)
        lik_poisson = bstm_Likelihood("poisson", 1.5)
        @test rand(lik_poisson) >= 0.0
        lik_gauss = bstm_Likelihood("gaussian", 0.0; sigma_y=1.0)
        @test isfinite(rand(lik_gauss))
        lik_vec = [lik_poisson for _ in 1:10]
        sampled_vec = rand.(lik_vec)
        @test length(sampled_vec) == 10

        # Test extract_param_matrix and extract_param_vector
        mock_chain = Dict(:sigma_s_idx => [0.5, 0.6, 0.7], :ure_s_idx => [[0.1, 0.2], [0.3,
            0.4], [0.5, 0.6]])
        @test size(bstm.extract_param_matrix(mock_chain, :sigma_s_idx), 1) == 3
        @test size(bstm.extract_param_matrix(mock_chain, :ure_s_idx), 2) == 2
        @test length(bstm.extract_param_vector(mock_chain, :sigma_s_idx)) == 3

        # Test multi-chain scalar parameter collapsing (e.g. 10 iters x 3 chains)
        mock_multichain = (
            chains = [1, 2, 3],
            intercept = randn(10, 3),
            beta = randn(10, 3)
        )
        mat_mc = bstm.extract_param_matrix(mock_multichain, :intercept)
        @test size(mat_mc) == (30, 1)

        # Test direct parameter summary statistics
        df_direct_summary = bstm._compute_direct_parameter_summary(mock_chain)
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

    @testset "Prediction & Observation Magnitude and PPC Validation" begin
        # 1. Intercept-Only Model Magnitude Check
        @testset "Intercept-Only Model Magnitude" begin
            n_obs = 60
            true_mu = 4.5
            y_sim = true_mu .+ randn(n_obs) .* 0.3
            df_intercept = DataFrame(y = y_sim)
            
            m_intercept = @bstm(likelihood(y) ~ 1, df_intercept, verbose=false)
            chn_intercept = sample(m_intercept, MH(), 150, progress=false)
            res_intercept = bstm.model_results_comprehensive(m_intercept, chn_intercept)
            
            y_obs_vec = res_intercept.predictions.observed
            pred_denoised = res_intercept.predictions.denoised.mean
            pred_noisy = res_intercept.predictions.noisy.mean
            
            @test length(pred_denoised) == n_obs
            @test length(pred_noisy) == n_obs
            # Magnitude check: mean of predictions should be within tight tolerance of
            #   observed mean
            @test abs(mean(pred_denoised) - mean(y_obs_vec)) < 0.5
            @test abs(mean(pred_noisy) - mean(y_obs_vec)) < 0.5
            # Predictions should span within reasonable bounds of the observed data
            @test minimum(pred_denoised) >= minimum(y_obs_vec) - 1.5
            @test maximum(pred_denoised) <= maximum(y_obs_vec) + 1.5
            @test res_intercept.metrics.rmse < 1.0
        end

        # 2. Complex Model (Fixed + Spatiotemporal) Magnitude and PPC Pearson Correlation Check
        @testset "Complex Model Magnitude and PPC Pearson Correlation" begin
            p_data, _ = bstm.bstm_data("scottish_lip")
            df_cplx = p_data.data
            W_cplx = p_data.au.W
            
            m_cplx = @bstm(likelihood(y_gauss) ~ 1 + cov1 + random(s_idx,
                model=bym2) + random(year, model=ar1), df_cplx, W=W_cplx, verbose=false)
            chn_cplx = sample(m_cplx, MH(), 150, progress=false)
            res_cplx = bstm.model_results_comprehensive(m_cplx, chn_cplx)
            
            y_obs_vec = res_cplx.predictions.observed
            pred_denoised = res_cplx.predictions.denoised.mean
            pred_noisy = res_cplx.predictions.noisy.mean
            
            # Magnitude check: mean and scale consistency
            @test abs(mean(pred_denoised) - mean(y_obs_vec)) < 1.0
            @test abs(mean(pred_noisy) - mean(y_obs_vec)) < 1.0
            @test abs(std(pred_noisy) - std(y_obs_vec)) < 1.5
            
            # PPC Pearson Correlation Check
            @test haskey(res_cplx.metrics, :r_pearson)
            r_val = res_cplx.metrics.r_pearson
            @test r_val isa Real
            @test !isnan(r_val)
            @test !isinf(r_val)
            @test -1.0 <= r_val <= 1.0
            # Direct correlation computation on observed vs predicted
            calc_r = cor(y_obs_vec, pred_denoised)
            @test isapprox(r_val, calc_r, atol=1e-6)
            @test r_val > 0.0 # Should exhibit positive correlation with signal
            
            # PPC Plotting generation and r_pearson validation
            plots_res = bstm.bstm_plots(res_cplx, df_cplx; au=p_data.au)
            @test haskey(plots_res.plots, :posterior_predictive_check)
            @test plots_res.metrics.r_pearson == r_val
        end

        # 3. Poisson with Log Offsets Count-Scale Magnitude Check
        @testset "Poisson with Log Offsets Magnitude" begin
            p_data, _ = bstm.bstm_data("scottish_lip")
            df_lip = p_data.data
            W_lip = p_data.au.W
            
            m_pois = @bstm(likelihood(y_pois, family=poisson,
                log_offsets=log_offsets) ~ 1 + cov1 + random(s_idx, model=bym2), df_lip,
                W=W_lip, verbose=false)
            chn_pois = sample(m_pois, MH(), 100, progress=false)
            res_pois = bstm.model_results_comprehensive(m_pois, chn_pois)
            
            preds_p = res_pois.predictions.denoised.mean
            # Expected counts scale should match observed counts
            @test median(preds_p) > 0.0
            @test median(preds_p) < 2.0 * maximum(df_lip.y_pois)
            @test abs(median(preds_p) - median(df_lip.y_pois)) < 10.0
        end

        # 4. Binomial with Trials Scale Check
        @testset "Binomial with Trials Magnitude" begin
            n_obs = 50
            n_trials = 40
            true_prob = 0.35
            y_sim = rand(Distributions.Binomial(n_trials, true_prob), n_obs)
            df_bin = DataFrame(y = y_sim, trials = fill(n_trials, n_obs))
            
            m_bin = @bstm(likelihood(y, family=binomial, trials=trials) ~ 1, df_bin, verbose=false)
            chn_bin = sample(m_bin, MH(), 100, progress=false)
            res_bin = bstm.model_results_comprehensive(m_bin, chn_bin)
            
            preds_bin = res_bin.predictions.denoised.mean
            # Denoised predictions should be on the [0, n_trials] count scale, not [0, 1]
            @test abs(mean(preds_bin) - mean(y_sim)) < 4.0
            @test all(v -> 0.0 <= v <= n_trials, preds_bin)
        end

        # 5. LogNormal Natural Scale Check
        @testset "LogNormal Natural Scale Magnitude" begin
            n_obs = 50
            true_mu = 1.5
            y_sim = exp.(true_mu .+ randn(n_obs) .* 0.2)
            df_ln = DataFrame(y = y_sim)
            
            m_ln = @bstm(likelihood(y, family=lognormal) ~ 1, df_ln, verbose=false)
            chn_ln = sample(m_ln, MH(), 100, progress=false)
            res_ln = bstm.model_results_comprehensive(m_ln, chn_ln)
            
            preds_ln = res_ln.predictions.denoised.mean
            # Denoised predictions should be on natural scale (~exp(1.5)=4.48), not log scale (1.5)
            @test abs(mean(preds_ln) - mean(y_sim)) < 1.5
            @test minimum(preds_ln) > 1.0
        end

        # 6. Smooth Model (P-Spline) Magnitude Check
        @testset "Smooth Model (P-Spline) Magnitude" begin
            p_data, _ = bstm.bstm_data("scottish_lip")
            df_lip = p_data.data
            
            m_smooth = @bstm(likelihood(y_gauss) ~ 1 + random(cov1, model=pspline), df_lip,
                verbose=false)
            chn_smooth = sample(m_smooth, MH(), 100, progress=false)
            res_smooth = bstm.model_results_comprehensive(m_smooth, chn_smooth)
            
            preds_smooth = res_smooth.predictions.denoised.mean
            @test abs(mean(preds_smooth) - mean(df_lip.y_gauss)) < 1.0
            @test haskey(res_smooth.effects, :cov1)
        end
    end

end
