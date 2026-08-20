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
using StatsModels # For @formula, EffectsCoding

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
        data = DataFrame(y=rand(N_obs), s_idx=repeat(1:N_areas, inner=N_time)[1:N_obs], t_idx=repeat(1:N_time, outer=N_areas)[1:N_obs], group_var=repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]),
        model_arch = model_arch,
        technical = Dict(
            :component_levels => Dict(), # To be filled by specific component tests
            :component_indices => Dict() # To be filled by specific component tests
        )
    )
end

# Mock spec object (simplified for testing purposes)
function mock_spec(key, hyper_obj=NamedTuple())
    return (
        key = Symbol(key),
        hyper = hyper_obj
    )
end

# Mock chain object (simplified for testing parameter extraction)
# This mock directly returns values from a dictionary, bypassing MCMCChains.Chains object complexity
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

# Simplified _find_parameter for mock_chain
function bstm._find_parameter(p_names, target_name_base, outcome_idx, is_multivariate_model)
    target_name = is_multivariate_model ? Symbol("$(target_name_base)_$(outcome_idx)") : Symbol(target_name_base)
    if target_name in p_names
        return target_name
    end
    # Fallback for univariate models where outcome_idx might be passed but not used in name
    if !is_multivariate_model && Symbol(target_name_base) in p_names
        return Symbol(target_name_base)
    end
    error("Parameter $(target_name) not found in mock chain.")
end

# Simplified get_params_vector for mock_chain
function bstm.get_params_vector(chain_mock::Dict, param_name::Symbol, dim_idx::Int=1)
    return vec(chain_mock[param_name][dim_idx, :])
end

# Simplified get_params_matrix for mock_chain
function bstm.get_params_matrix(chain_mock::Dict, param_name::Symbol)
    return chain_mock[param_name]
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
            p, n = bstm.bstm_data("advanced", s_N=10, t_N=10)
            data = p.data
            W = p.au.W
            
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

            M_cfg = bstm.bstm_config(formula, data; W=W, s_N=p.s_N, t_N=p.t_N)
            
            @test M_cfg.add_intercept == true
            @test all(x -> x in string.(M_cfg.fixed_effects_names), ["cov1", "cov2"])
            
            components = M_cfg.components
            @test any(c -> c.component_obj isa bstm.BYM2, components)
            @test any(c -> c.component_obj isa bstm.AR1, components)
            @test any(c -> c.component_obj isa bstm.Harmonic, components)
            @test any(c -> c.component_obj isa bstm.PSpline, components)
            @test any(c -> c.component_obj isa bstm.Eigen, components)
            @test any(c -> c.component_obj isa bstm.Mixed, components)
            @test any(c -> c.component_obj isa bstm.SVC, components)
            @test any(c -> c.component_obj isa bstm.Dynamics, components)
            @test haskey(M_cfg.nested_models, :proxy_val)
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
            sample_nt = (
                intercept = 0.5,
                beta = [0.1, 0.2],
                sigma_s_idx = 1.2,
                ure_s_idx = randn(p_data.s_N)
            )
            calibrated = bstm.calibrate_param_registry(reg_m, sample_nt)
            @test calibrated.descriptors[:ure_s_idx].shape == (p_data.s_N,)

            # Mock VarNamedTuple mapping
            struct MockVarNamedTuple
                data::Dict{Symbol, Any}
            end
            Base.pairs(m::MockVarNamedTuple) = Base.pairs(m.data)
            mock_vnt = MockVarNamedTuple(Dict(:sigma_year => 0.8, :ure_year => randn(p_data.t_N)))
            calibrated_vnt = bstm.calibrate_param_registry(calibrated, mock_vnt)
            @test calibrated_vnt.descriptors[:ure_year].shape == (p_data.t_N,)

            # 5. Test get_samples with mock chain dictionary
            mock_ch = Dict(
                :sigma_s_idx => reshape([1.0, 1.1, 1.2, 1.3, 1.4], 1, 5),
                :ure_s_idx => randn(p_data.s_N, 5)
            )
            sigma_s = bstm.get_samples(mock_ch, calibrated, :s_idx, :sigma)
            @test size(sigma_s, 1) == 5
            @test size(sigma_s, 2) == 1

            ure_s = bstm.get_samples(mock_ch, calibrated, :s_idx, :ure)
            @test size(ure_s, 1) == 5
            @test size(ure_s, 2) == p_data.s_N

            # Test alias lookup
            innov_s = bstm.get_samples(mock_ch, calibrated, :s_idx, :innovations)
            @test size(innov_s, 1) == 5
            @test size(innov_s, 2) == p_data.s_N

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
                :unconstrained_rho_year => reshape([0.5, 0.6, 0.7], 1, 3),
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
                :unconstrained_rho1_year => reshape([0.3, 0.4, 0.5], 1, 3),
                :unconstrained_rho2_year => reshape([0.1, 0.2, 0.1], 1, 3),
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
            ll_rw1 = bstm._rw1_log_marginal_likelihood(y_res, t_idx, 5, rw1_template.matrix, 0.5, 0.2)
            @test isfinite(ll_rw1)
            @test ll_rw1 < 0.0

            spec_rw1 = (key = :year, hyper = (n_latent = 5, Q_template = rw1_template.matrix, U = rw1_template.U, L = rw1_template.L), params = Dict())
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
            ll_rw2 = bstm._rw2_log_marginal_likelihood(y_res, t_idx, 5, rw2_template.matrix, 0.5, 0.2)
            @test isfinite(ll_rw2)
            @test ll_rw2 < 0.0

            spec_rw2 = (key = :year, hyper = (n_latent = 5, Q_template = rw2_template.matrix, U = rw2_template.U, L = rw2_template.L), params = Dict())
            eff_rw2 = bstm.get_effects(rw2_m, chain_rw1, spec_rw2, M_mock, nothing)
            @test length(eff_rw2.structured) == 1
            @test size(eff_rw2.structured[1]) == (5, 3)
            @test !all(iszero, eff_rw2.structured[1])

            # 11. Test Marginalized ICAR Likelihood & Latent Reconstruction
            W_mock = [0 1 0 0 0; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0]
            icar_template = bstm.build_structure_template(:icar, 5; W=W_mock)
            icar_m = bstm.ICAR(Exponential(1.0), :marginalized)
            ll_icar = bstm._icar_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix, icar_template.L, 0.5, 0.2)
            @test isfinite(ll_icar)
            @test ll_icar < 0.0

            M_spatial = (
                outcomes_N = 1,
                model_arch = "univariate",
                s_idx = t_idx,
                s_N = 5,
                y_obs = y_res,
                noise = 1e-6
            )
            spec_icar = (key = :region, hyper = (n_latent = 5, Q_template = icar_template.matrix, U = icar_template.U, L = icar_template.L), params = Dict())
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
            ll_leroux = bstm._leroux_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix, icar_template.L, 0.6, 0.5, 0.2)
            @test isfinite(ll_leroux)
            @test ll_leroux < 0.0

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
            ll_ps = bstm._pspline_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix, rw2_template.L, 2, 0.5, 0.2)
            @test isfinite(ll_ps)
            @test ll_ps < 0.0

            spec_ps = (key = :x, hyper = (n_latent = 5, basis_matrix = B_mock, Q_template = rw2_template.matrix, U = rw2_template.U, L = rw2_template.L), params = Dict(:positional_args => [:x]))
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
            ll_bym2 = bstm._bym2_log_marginal_likelihood(y_res, t_idx, 5, icar_template.U, icar_template.L, 0.6, 0.5, 0.2)
            @test isfinite(ll_bym2)
            @test ll_bym2 < 0.0

            chain_bym2 = Dict(
                :unconstrained_rho_region => reshape([0.5, 0.6, 0.7], 1, 3),
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
            @test ll_iid < 0.0

            spec_iid = (key = :group, structure = :mixed, var = "group", hyper = (n_latent = 5,), params = Dict())
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
            ll_cyclic = bstm._cyclic_log_marginal_likelihood(y_res, t_idx, 5, cyclic_template.matrix, cyclic_template.L, 0.5, 0.2)
            @test isfinite(ll_cyclic)
            @test ll_cyclic < 0.0

            spec_cyclic = (key = :month, hyper = (n_latent = 5, Q_template = cyclic_template.matrix, U = cyclic_template.U, L = cyclic_template.L), params = Dict())
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
            ll_bs = bstm._bspline_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix, rw2_template.L, 0.5, 0.2)
            @test isfinite(ll_bs)
            @test ll_bs < 0.0

            eff_bs = bstm.get_effects(bspline_m, chain_ps, spec_ps, M_mock, nothing)
            @test length(eff_bs.structured) == 1
            @test size(eff_bs.structured[1]) == (5, 3)
            @test !all(iszero, eff_bs.structured[1])

            # 18. Test Marginalized Moran Likelihood & Latent Reconstruction
            moran_m = bstm.Moran(Exponential(1.0), :marginalized)
            moran_eigs = icar_template.U
            ll_moran = bstm._moran_log_marginal_likelihood(y_res, t_idx, 5, moran_eigs, 0.5, 0.2)
            @test isfinite(ll_moran)
            @test ll_moran < 0.0

            spec_moran = (key = :region, hyper = (n_latent = 5, moran_eigenvectors = moran_eigs), params = Dict())
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
            ll_sar = bstm._sar_log_marginal_likelihood(y_res, t_idx, 5, sar_W, sar_eigs, 0.4, 0.5, 0.2)
            @test isfinite(ll_sar)
            @test ll_sar < 0.0

            spec_sar = (key = :region, hyper = (n_latent = 5, Q_template = sar_W, eigenvalues = sar_eigs), params = Dict())
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
            ll_tps = bstm._tps_log_marginal_likelihood(y_res, B_mock, rw2_template.matrix, rw2_template.L, 0.5, 0.2)
            @test isfinite(ll_tps)
            @test ll_tps < 0.0

            spec_tps = (key = :space, hyper = (n_latent = 5, basis_matrix = B_mock, Q_template = rw2_template.matrix, L = rw2_template.L, knots = randn(5, 2)), params = Dict())
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
            ll_bary = bstm._barycentric_log_marginal_likelihood(y_res, B_mock, nothing, nothing, 0.5, 0.2)
            @test isfinite(ll_bary)
            @test ll_bary < 0.0

            spec_bary = (key = :space, hyper = (n_knots = 5, B = B_mock, knots = [bstm.Point2D(0.0, 0.0) for _ in 1:5]), params = Dict())
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
            ll_bcgn = bstm._bcgn_log_marginal_likelihood(y_res, bcgn_map, icar_template.matrix, icar_template.L, 0.5, 0.2)
            @test isfinite(ll_bcgn)
            @test ll_bcgn < 0.0

            spec_bcgn = (key = :space, hyper = (n_latent = 5, mapping_matrix = bcgn_map, Q_template = icar_template.matrix, L = icar_template.L, set1_indices = [1, 2, 3, 4, 5]), params = Dict())
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
            ll_besag = bstm._besag_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix, icar_template.L, 0.5, 0.2)
            @test isfinite(ll_besag)
            @test ll_besag < 0.0

            spec_besag = (key = :region, hyper = (n_latent = 5, Q_template = icar_template.matrix, L = icar_template.L), params = Dict())
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
            @test ll_rff < 0.0

            coords_mock = randn(5, 2)
            spec_rff = (key = :space, hyper = (n_latent = 5, in_dims = 2, coords = coords_mock, W_fixed = randn(2, 5), b_fixed = rand(5)), params = Dict())
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
            ll_spde = bstm._spde_log_marginal_likelihood(y_res, t_idx, 5, icar_template.matrix, icar_template.L, 1.2, 0.5, 0.2)
            @test isfinite(ll_spde)
            @test ll_spde < 0.0

            spec_spde = (key = :region, hyper = (n_latent = 5, Q_template = icar_template.matrix, L = icar_template.L, U = icar_template.U), params = Dict())
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
            @test ll_gp < 0.0

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
            m_rw1 = bstm.RW1(Distributions.Exponential(1.0))
            res_rw1 = bstm.get_precomputes(m_rw1, (N_time=20,), Dict())
            @test res_rw1.model_type == :rw1
            @test size(res_rw1.Q_template) == (20, 20)
        end

        @testset "Seasonal Manifolds" begin
            m_cyc = bstm.Harmonic(1, 12.0, Distributions.Exponential(1.0), Distributions.Beta(1,1), :twocoefficient)
            res_cyc = bstm.get_precomputes(m_cyc, (N_time=12,), Dict())
            @test res_cyc.model_type == :harmonic
            @test res_cyc.period == 12.0
        end

        @testset "Basis & Continuous Manifolds" begin
            m_bs = bstm.BSpline(15, 3, Distributions.Exponential(1.0))
            res_bs = bstm.get_precomputes(m_bs, (N_levels=15,), Dict())
            @test res_bs.model_type == :bspline
            @test size(res_bs.B_matrix) == (15, 15) # Assuming N_levels is the number of bins
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
                au = bstm.assign_spatial_units(
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
        p, n = bstm.bstm_data("sim", s_N=16, t_N=5)
        dummy_df = p.data
        W_test = p.au.W

        test_cases = [
            ("ST_TypeIV_Poisson", "likelihood(y_pois) ~ 1 + cov1 + (random(s_idx, model=besag) ⊗ random(t_idx, model=ar1))", "poisson", Dict(:W => W_test)),
            ("MixedEffects_Gaussian", "likelihood(y_gauss) ~ 1 + cov1 + mixed(cov1 | region) + random(s_idx, model=icar)", "gaussian", Dict(:W => W_test)),
            ("Multivariate_Gaussian", "likelihood(species_1 + species_2) ~ 1 + cov1 + random(s_idx, model=bym2)", "gaussian", Dict(:W => W_test)),
            ("Seasonal_RW2_Poisson", "likelihood(y_pois) ~ 1 + random(month, model=harmonic, period=12) + random(t_idx, model=rw2)", "poisson", Dict()),
            ("SVC_Gaussian", "likelihood(y_gauss) ~ 1 + (cov1 |> random(s_idx, model=icar))", "gaussian", Dict(:W => W_test)),
            ("Spatial_Smooth_RFF", "likelihood(y_gauss) ~ 1 + random(s_y, s_x, model=rff, n_features=15) + random(t_idx, model=ar1)", "gaussian", Dict()),
            ("Leroux_Gaussian", "likelihood(y_gauss) ~ 1 + random(s_idx, model=leroux)", "gaussian", Dict(:W => W_test)),
            ("GP_Smooth_Gaussian", "likelihood(y_gauss) ~ 1 + random(cov1, model=gp, kernel=\"se\")", "gaussian", Dict())
        ]

        for (name, f_str, fam, extra_args) in test_cases
            @testset "$name" begin
                println("  Testing Instantiation: $name")
                model = bstm.bstm(f_str, dummy_df; model_family=fam, extra_args...)
                @test sample(model, Turing.Prior(), 1) isa Chains
            end
        end
    end

    @testset "ComponentModel Interface Unit Tests" begin
        # Test IID Component
        @testset "IID Component" begin
            N_obs, N_levels = 100, 10
            m_iid = bstm.IID(Distributions.Exponential(1.0))
            
            mock_M_ds = Dict(:data => DataFrame(group_var=repeat(1:N_levels, inner=N_obs÷N_levels)[1:N_obs]))
            mock_mod_data = Dict(:variables => :group_var)
            
            @test bstm.get_datastructures!(typeof(m_iid), mock_M_ds, mock_mod_data) == true
            
            mock_M_pc = (data=mock_M_ds[:data],)
            mock_mod_data_pc = Dict()
            @test bstm.get_precomputes(m_iid, mock_M_pc, mock_mod_data_pc) == NamedTuple()

            mock_M_priors = (technical=(component_levels=Dict(:group_var => N_levels),),)
            mock_spec_priors = mock_spec(:group_var)
            priors_str = bstm.get_priors(m_iid, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_group_var ~ Exponential(1.0)")
            @test contains(priors_str, "ure_group_var ~ MvNormal(zeros(T, spec_registry[:group_var].hyper.n_latent), I)")

            mock_M_updates = (technical=(component_indices=Dict(:group_var => mock_M_ds[:data].group_var),), model_arch="univariate")
            mock_spec_updates = mock_spec(:group_var)
            updates_str = bstm.get_updates(m_iid, mock_spec_updates, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "sre_group_var = sigma_group_var .* ure_group_var")
            @test contains(updates_str, "eta .+= view(sre_group_var, M.data[!, :group_var])")

            mock_chain_effects = mock_chain(Dict(:sigma_group_var => 0.5, :ure_group_var => randn(N_levels, 10)), 10)
            mock_M_effects = (technical=(component_indices=Dict(:group_var => mock_M_ds[:data].group_var),), model_arch="univariate")
            mock_spec_effects = mock_spec(:group_var)
            p_names_effects = collect(keys(mock_chain_effects))
            
            effects_result = bstm.get_effects(m_iid, mock_chain_effects, mock_spec_effects, (outcomes_N=1, model_arch="univariate", technical=mock_M_effects.technical), nothing)
            @test size(effects_result.structured[1]) == (N_obs, 10)
        end

        # Test Leroux Component
        @testset "Leroux Component" begin
            N_areas, N_obs = 10, 100
            W_leroux = create_chain_adj_matrix(N_areas)
            m_leroux = bstm.Leroux(Distributions.Exponential(1.0), Distributions.Beta(1,1), :spectral)

            mock_M_ds = Dict(:data => DataFrame(s_idx=repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]), :W => W_leroux)
            mock_mod_data = Dict(:variables => :s_idx)
            @test bstm.get_datastructures!(typeof(m_leroux), mock_M_ds, mock_mod_data) == true

            mock_M_pc = (data=mock_M_ds[:data], W=W_leroux, N_areas=N_areas)
            precomputes = bstm.get_precomputes(m_leroux, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :Q_template)
            @test size(precomputes.Q_template) == (N_areas, N_areas)

            mock_M_priors = (technical=(component_levels=Dict(:s_idx => N_areas),),)
            mock_spec_priors = mock_spec(:s_idx, precomputes)
            priors_str = bstm.get_priors(m_leroux, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_s_idx ~ Exponential(1.0)")
            @test contains(priors_str, "rho_s_idx ~ Beta(1,1)")
            @test contains(priors_str, "ure_s_idx ~ MvNormal(zeros(T, 10), I)")

            mock_M_updates = (technical=(component_indices=Dict(:s_idx => repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]),), model_arch="univariate")
            updates_str = bstm.get_updates(m_leroux, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "sre_s_idx = hyper.U * (diag_D .* ure_s_idx)")
            @test contains(updates_str, "eta .+= view(sre_s_idx, M.s_idx)")
        end

        # Test GP Component
        @testset "GP Component" begin
            N_obs, N_dims = 100, 2
            m_gp = bstm.GP(Distributions.Exponential(1.0), Distributions.Gamma(2, 0.5), "se", false, :noncentered)
            
            mock_M_ds = Dict(:data => DataFrame(x=rand(N_obs), y=rand(N_obs)))
            mock_mod_data = Dict(:variables => [:x, :y], :params => Dict(:coords => rand(N_obs, N_dims)))
            @test bstm.get_datastructures!(typeof(m_gp), mock_M_ds, mock_mod_data) == true

            mock_M_pc = (data=mock_M_ds[:data],)
            precomputes = bstm.get_precomputes(m_gp, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :K)
            @test size(precomputes.K) == (N_obs, N_obs)

            mock_M_priors = (technical=(component_levels=Dict(),),)
            mock_spec_priors = mock_spec(:x_y, precomputes)
            priors_str = bstm.get_priors(m_gp, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_x_y ~ Exponential(1.0)")
            @test contains(priors_str, "ls_x_y ~ Gamma(2.0, 0.5)")
            @test contains(priors_str, "ure_x_y ~ MvNormal(zeros(T, 100), I)")

            mock_M_updates = (technical=(component_indices=Dict(),), model_arch="univariate")
            updates_str = bstm.get_updates(m_gp, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "sre_x_y = F_gp.L * ure_x_y")
            @test contains(updates_str, "eta .+= sre_x_y")
        end

        # Test Harmonic Component
        @testset "Harmonic Component" begin
            N_obs, N_time, n_harm, period = 120, 12, 2, 12.0
            m_harm = bstm.Harmonic(n_harm, period, Distributions.Exponential(1.0), Distributions.Beta(1,1), :twocoefficient)
            
            mock_M_ds = Dict(:data => DataFrame(month=rand(1:N_time, N_obs)))
            mock_mod_data = Dict(:variables => :month)
            @test bstm.get_datastructures!(typeof(m_harm), mock_M_ds, mock_mod_data) == true

            mock_M_pc = (data=mock_M_ds[:data], N_time=N_time)
            precomputes = bstm.get_precomputes(m_harm, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :H)
            @test size(precomputes.H) == (N_time, 2 * n_harm)

            mock_M_priors = (technical=(component_levels=Dict(:month => N_time),),)
            mock_spec_priors = mock_spec(:month, precomputes)
            priors_str = bstm.get_priors(m_harm, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "beta_cos_month ~ MvNormal(zeros(2), 1.0)")
            @test contains(priors_str, "beta_sin_month ~ MvNormal(zeros(2), 1.0)")

            mock_M_updates = (technical=(component_indices=Dict(:month => mock_M_ds[:data].month),), model_arch="univariate")
            updates_str = bstm.get_updates(m_harm, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_month = H_month * vcat(beta_cos_month, beta_sin_month)")
            @test contains(updates_str, "eta .+= latent_month[idx_var]")
        end
    end

    @testset "Integration Tests: Smoke Tests" begin
        @testset "IID Model Smoke Test" begin
            p, n = bstm.bstm_data("sim", s_N=10, t_N=10)
            sim_df = p.data
            model_iid_sim = @bstm(likelihood(y_gauss) ~ intercept() + random(region, model=iid), sim_df)
            chain = sample(model_iid_sim, MH(), 100, progress=false)
            @test chain isa Chains
            @test mean(chain[:sigma_region]) > 0
        end

        @testset "AR1 Model Smoke Test" begin
            p, n = bstm.bstm_data("sim", s_N=10, t_N=10)
            sim_df = p.data
            model_ar1_sim = @bstm(likelihood(y_gauss) ~ intercept() + random(year, model=ar1), sim_df)
            chain = sample(model_ar1_sim, NUTS(100, 0.65), 200, progress=false)
            @test chain isa Chains
            @test mean(chain[:sigma_year]) > 0
        end

        @testset "BYM2 ⊗ AR1 Interaction Smoke Test" begin
            p, n = bstm.bstm_data("sim", s_N=5, t_N=10)
            sim_df = p.data
            W_sim = p.au.W
            model_int_sim = @bstm(
                likelihood(y_gauss) ~ intercept() + (random(s_idx, model=bym2) ⊗ random(t_idx, model=ar1)),
                sim_df, W=W_sim
            )
            chain_int_sim = sample(model_int_sim, NUTS(100, 0.65), 200, progress=false)
            @test chain_int_sim isa Chains
            @test mean(chain_int_sim[:sigma_s_idx_t_idx]) > 0.0
        end
    end

    @testset "Complex Integration Tests" begin
        @testset "Hurdle Model" begin
            p, n = bstm.bstm_data("sim", s_N=10, t_N=10)
            data = p.data
            W = p.au.W
            model = @bstm(likelihood(y_pois, family=poisson, hurdle=5) ~ 1 + random(s_idx, model=bym2), data, W=W)
            chain = sample(model, MH(), 100, progress=false)
            @test chain isa Chains
            @test haskey(chain, :lik_phi_hurdle)
        end

        @testset "Eigen Model" begin
            p, n = bstm.bstm_data("sim", s_N=10, t_N=10)
            data = p.data
            model = @bstm(likelihood(y_gauss) ~ 1 + eigen(species_1, species_2, species_3, n_factors=2), data)
            chain = sample(model, NUTS(100, 0.65), 200, progress=false)
            @test chain isa Chains
            @test haskey(chain, :pca_sd_species_1_species_2_species_3)
        end

        @testset "Multifidelity Signal Transfer" begin
            p, n = bstm.bstm_data("advanced", s_N=12, t_N=18)
            df_hi = p.data
            W_mf = p.au.W
            df_lo = select(df_hi, :proxy_val, :s_idx, :t_idx)
            rename!(df_lo, :proxy_val => :y_low)

            model_mf = @bstm(
                "likelihood(y_gauss) ~ 1 + random(s_idx, model='bym2') + random(t_idx, model='ar1') + nested(low_fi, formula=\"likelihood(y_low) ~ 1 + random(s_idx) + random(t_idx)\", data_source=:low_quality_data)",
                df_hi, W=W_mf, low_quality_data=df_lo
            )
            
            chain_mf = sample(model_mf, NUTS(100, 0.65), 200, progress=false, check_model=false)
            @test chain_mf isa Chains
            @test haskey(chain_mf, :rho_nested_low_fi)
        end

        @testset "Prediction Engine" begin
            p_train, n_train = bstm.bstm_data("sim", s_N=16, t_N=12)
            train_df = p_train.data
            W_train = p_train.au.W
            
            centroids_matrix = hcat(p_train.au.centroids.s_x, p_train.au.centroids.s_y)'
            kmeans_res = kmeans(centroids_matrix, 4)
            assignments = kmeans_res.assignments

            test_pts = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)]
            n_s_test = length(test_pts)
            test_df = DataFrame(s_x=repeat([p[1] for p in test_pts], inner=n_train.t_N), 
                                s_y=repeat([p[2] for p in test_pts], inner=n_train.t_N), 
                                t_idx=repeat(1:n_train.t_N, outer=n_s_test))

            model_obj = bstm.bstm("likelihood(y_gauss) ~ 1 + random(s_idx, model=mosaic, n_regions=4) + random(t_idx, model=ar1)", 
                             train_df; s_N=n_train.s_N, t_N=n_train.t_N, W=W_train, cluster_assignments=assignments)
            
            chain_train = sample(model_obj, MH(), 500, progress=false)
            res_pred = bstm.predict(model_obj, chain_train, test_df; n_samples=100)
            
            @test res_pred.predictions_denoised isa NamedTuple
            @test length(res_pred.predictions_denoised.mean) == nrow(test_df)
        end

        @testset "Cross-Validation Orchestrator" begin
            p_cv, n_cv = bstm.bstm_data("sim", s_N=20, t_N=10)
            cv_df = p_cv.data
            cv_W = p_cv.au.W
            cv_formula = "likelihood(y_gauss) ~ 1 + random(t_idx, model=ar1)"

            @testset "k-fold CV" begin
                cv_results_kfold = bstm.bstm_cv_orchestrator(cv_formula, cv_df; method=:kfold, n_folds=3, n_samples=50, W=cv_W)
                @test length(cv_results_kfold.folds) == 3
                @test cv_results_kfold.mean_rmse isa Real
            end

            @testset "Temporal Forward-Chaining CV" begin
                cv_results_fchain = bstm.bstm_cv_orchestrator(cv_formula, cv_df; method=:temporal_forward_chain, cv_var=:t_idx, n_folds=2, n_samples=50, W=cv_W)
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

end
