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
stable_logdiffexp(a, b) = a + log1mexp(b - a)

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
            formula = """
            y ~ 1 + cov1 + cov2 +
            random(s_idx, model='bym2') +
            random(year, model='ar1') +
            random(u_idx, model='cyclic', period=12) +
            random(cov3, nbins=10, model='pspline') +
            nested(z ~ 1 + random(s_idx)) +
            eigen(cov1, cov2, n_factors=1) +
            mixed(1|f1) +
            (cov1 |> random(s_idx, model='icar')) +
            dynamics(dynamic_var, model='advection')
            """
            data = DataFrame(
                y = rand(100), cov1 = rand(100), cov2 = rand(100), cov3 = rand(100), cov4 = rand(100),
                s_idx = rand(1:10, 100), year = rand(2000:2005, 100), u_idx = rand(1:12, 100),
                f1 = categorical(rand(1:4, 100)), dynamic_var = rand(100), z = rand(100)
            )
            W = create_chain_adj_matrix(10)

            M_cfg = bstm.bstm_config(formula, data; W=W, s_N=size(W,1), t_N=length(unique(data.year)))
            
            @test M_cfg.add_intercept == true
            @test all(x -> x in string.(M_cfg.fixed_effects_names), ["cov1", "cov2"])
            
            components = M_cfg.components
            @test any(c -> c.model_obj isa bstm.BYM2, components)
            @test any(c -> c.model_obj isa bstm.AR1, components)
            @test any(c -> c.model_obj isa bstm.Harmonic, components) # Cyclic maps to Harmonic
            @test any(c -> c.model_obj isa bstm.PSpline, components)
            @test any(c -> c.model_obj isa bstm.Eigen, components)
            @test any(c -> c.model_obj isa bstm.Mixed, components)
            @test any(c -> c.model_obj isa bstm.SVC, components)
            @test any(c -> c.model_obj isa bstm.Dynamics, components)
            @test haskey(M_cfg.nested_models, :z)
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
            ("ST_TypeIV_Poisson", "likelihood(ycount) ~ 1 + x_cont + (random(s_idx, model=besag) ⊗ random(t_idx, model=ar1))", "poisson", Dict(:W => W_test)),
            ("MixedEffects_Gaussian", "likelihood(y) ~ 1 + x_cont + mixed(x_cont | x_cat1) + random(s_idx, model=icar)", "gaussian", Dict(:W => W_test)),
            ("Multivariate_Gaussian", "likelihood(y1 + y2) ~ 1 + x_cont + random(s_idx, model=bym2)", "gaussian", Dict(:W => W_test)),
            ("Seasonal_RW2_Poisson", "likelihood(ycount) ~ 1 + random(u_idx, model=harmonic, period=5) + random(t_idx, model=rw2)", "poisson", Dict()),
            ("SVC_Gaussian", "likelihood(y) ~ 1 + (x_cont |> random(s_idx, model=icar))", "gaussian", Dict(:W => W_test)),
            ("Spatial_Smooth_RFF", "likelihood(y) ~ 1 + random(lat, lon, model=rff, n_features=15) + random(t_idx, model=ar1)", "gaussian", Dict())
        ]

        for (name, f_str, fam, extra_args) in test_cases
            @testset "$name" begin
                println("  Testing Instantiation: $name")
                model = bstm.bstm(f_str, dummy_df; model_family=fam, extra_args...)
                @test sample(model, Turing.Prior(), 1) isa Chains
            end
        end
    end

    # --- New Unit Tests for ComponentModel Interface ---
    @testset "ComponentModel Interface Unit Tests" begin
        # Test IID Component
        @testset "IID Component" begin
            N_obs, N_levels = 100, 10
            m_iid = bstm.IID(Distributions.Exponential(1.0))
            
            # Mock M and mod_data for get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(group_var=repeat(1:N_levels, inner=N_obs÷N_levels)[1:N_obs]))
            mock_mod_data = Dict(:variables => :group_var)
            
            @test bstm.get_datastructures!(typeof(m_iid), mock_M_ds, mock_mod_data) == true
            
            # Mock M and spec for get_precomputes
            mock_M_pc = (data=mock_M_ds[:data],) # Read-only NamedTuple
            mock_mod_data_pc = Dict()
            @test bstm.get_precomputes(m_iid, mock_M_pc, mock_mod_data_pc) == NamedTuple()

            # Mock M and spec for get_priors
            mock_M_priors = (technical=(component_levels=Dict(:group_var => N_levels),),)
            mock_spec_priors = mock_spec(:group_var)
            priors_str = bstm.get_priors(m_iid, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_group_var ~ Exponential(1.0)")
            @test contains(priors_str, "innovations_group_var ~ MvNormal(zeros(10), 1.0)")

            # Mock M and spec for get_updates
            mock_M_updates = (technical=(component_indices=Dict(:group_var => repeat(1:N_levels, inner=N_obs÷N_levels)[1:N_obs]),), model_arch="univariate")
            mock_spec_updates = mock_spec(:group_var)
            updates_str = bstm.get_updates(m_iid, mock_spec_updates, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_group_var = sigma_group_var .* innovations_group_var")
            @test contains(updates_str, "eta .+= latent_group_var[idx_var]")

            # Mock chain and M for get_effects
            mock_chain_effects = mock_chain(Dict(
                :sigma_group_var => 0.5,
                :innovations_group_var => randn(N_levels, 10)
            ), 10)
            mock_M_effects = (technical=(component_indices=Dict(:group_var => repeat(1:N_levels, inner=N_obs÷N_levels)[1:N_obs]),), model_arch="univariate")
            mock_spec_effects = mock_spec(:group_var)
            p_names_effects = collect(keys(mock_chain_effects))
            
            effects_result = bstm.get_effects(m_iid, mock_chain_effects, mock_M_effects, 10, 1, p_names_effects, mock_spec_effects, nothing, N_obs)
            @test size(effects_result.structured[1]) == (N_obs, 10)
            @test size(effects_result.noisy[1]) == (N_obs, 10)
        end

        # Test BYM2 Component
        @testset "BYM2 Component" begin
            N_areas, N_obs = 10, 100
            W_bym2 = create_chain_adj_matrix(N_areas)
            m_bym2 = bstm.BYM2(Distributions.Exponential(1.0), Distributions.Beta(1,1), :spectral)

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(s_idx=repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]), :W => W_bym2)
            mock_mod_data = Dict(:variables => :s_idx)
            @test bstm.get_datastructures!(typeof(m_bym2), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            mock_M_pc = (data=mock_M_ds[:data], W=W_bym2, N_areas=N_areas)
            mock_mod_data_pc = Dict(:variables => :s_idx)
            precomputes = bstm.get_precomputes(m_bym2, mock_M_pc, mock_mod_data_pc)
            @test hasproperty(precomputes, :Q_template)
            @test hasproperty(precomputes, :eigen_values)
            @test size(precomputes.Q_template) == (N_areas, N_areas)

            # get_priors
            mock_M_priors = (technical=(component_levels=Dict(:s_idx => N_areas),),)
            mock_spec_priors = mock_spec(:s_idx, precomputes)
            priors_str = bstm.get_priors(m_bym2, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_s_idx ~ Exponential(1.0)")
            @test contains(priors_str, "rho_s_idx ~ Beta(1,1)")
            @test contains(priors_str, "innovations_structured_s_idx ~ MvNormal(zeros(10), 1.0)")
            @test contains(priors_str, "innovations_unstructured_s_idx ~ MvNormal(zeros(10), 1.0)")

            # get_updates
            mock_M_updates = (technical=(component_indices=Dict(:s_idx => repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]),), model_arch="univariate")
            mock_spec_updates = mock_spec(:s_idx, precomputes)
            updates_str = bstm.get_updates(m_bym2, mock_spec_updates, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_s_idx = sigma_s_idx .* (sqrt.(rho_s_idx) .* (U_s_idx * innovations_structured_s_idx) .+ sqrt.(1.0 .- rho_s_idx) .* innovations_unstructured_s_idx)")
            @test contains(updates_str, "eta .+= latent_s_idx[idx_var]")

            # get_effects (simplified check)
            mock_chain_effects = mock_chain(Dict(
                :sigma_s_idx => 0.5,
                :rho_s_idx => 0.8,
                :innovations_structured_s_idx => randn(N_areas, 10),
                :innovations_unstructured_s_idx => randn(N_areas, 10)
            ), 10)
            mock_M_effects = (technical=(component_indices=Dict(:s_idx => repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]),), model_arch="univariate")
            mock_spec_effects = mock_spec(:s_idx, precomputes)
            p_names_effects = collect(keys(mock_chain_effects))
            
            effects_result = bstm.get_effects(m_bym2, mock_chain_effects, mock_M_effects, 10, 1, p_names_effects, mock_spec_effects, nothing, N_obs)
            @test size(effects_result.structured[1]) == (N_obs, 10)
            @test size(effects_result.noisy[1]) == (N_obs, 10)
        end

        # Test AR1 Component
        @testset "AR1 Component" begin
            N_time, N_obs = 20, 100
            m_ar1 = bstm.AR1(Distributions.Exponential(1.0), Distributions.Beta(1,1))

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(t_idx=repeat(1:N_time, outer=N_obs÷N_time)[1:N_obs]))
            mock_mod_data = Dict(:variables => :t_idx)
            @test bstm.get_datastructures!(typeof(m_ar1), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            mock_M_pc = (data=mock_M_ds[:data], N_time=N_time)
            mock_mod_data_pc = Dict(:variables => :t_idx)
            precomputes = bstm.get_precomputes(m_ar1, mock_M_pc, mock_mod_data_pc)
            @test hasproperty(precomputes, :Q_template)
            @test size(precomputes.Q_template) == (N_time, N_time)

            # get_priors
            mock_M_priors = (technical=(component_levels=Dict(:t_idx => N_time),),)
            mock_spec_priors = mock_spec(:t_idx, precomputes)
            priors_str = bstm.get_priors(m_ar1, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_t_idx ~ Exponential(1.0)")
            @test contains(priors_str, "rho_t_idx ~ Beta(1,1)")
            @test contains(priors_str, "innovations_t_idx ~ MvNormal(zeros(20), 1.0)")

            # get_updates
            mock_M_updates = (technical=(component_indices=Dict(:t_idx => repeat(1:N_time, outer=N_obs÷N_time)[1:N_obs]),), model_arch="univariate")
            mock_spec_updates = mock_spec(:t_idx, precomputes)
            updates_str = bstm.get_updates(m_ar1, mock_spec_updates, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_t_idx = sigma_t_idx .* (L_t_idx * innovations_t_idx)")
            @test contains(updates_str, "eta .+= latent_t_idx[idx_var]")

            # get_effects (simplified check)
            mock_chain_effects = mock_chain(Dict(
                :sigma_t_idx => 0.5,
                :rho_t_idx => 0.8,
                :innovations_t_idx => randn(N_time, 10)
            ), 10)
            mock_M_effects = (technical=(component_indices=Dict(:t_idx => repeat(1:N_time, outer=N_obs÷N_time)[1:N_obs]),), model_arch="univariate")
            mock_spec_effects = mock_spec(:t_idx, precomputes)
            p_names_effects = collect(keys(mock_chain_effects))
            
            effects_result = bstm.get_effects(m_ar1, mock_chain_effects, mock_M_effects, 10, 1, p_names_effects, mock_spec_effects, nothing, N_obs)
            @test size(effects_result.structured[1]) == (N_obs, 10)
            @test size(effects_result.noisy[1]) == (N_obs, 10)
        end

        # Test RW2 Component
        @testset "RW2 Component" begin
            N_time, N_obs = 20, 100
            m_rw2 = bstm.RW2(Distributions.Exponential(1.0))

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(t_idx=repeat(1:N_time, outer=N_obs÷N_time)[1:N_obs]))
            mock_mod_data = Dict(:variables => :t_idx)
            @test bstm.get_datastructures!(typeof(m_rw2), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            mock_M_pc = (data=mock_M_ds[:data], N_time=N_time)
            precomputes = bstm.get_precomputes(m_rw2, mock_M_pc, Dict(:variables => :t_idx))
            @test hasproperty(precomputes, :Q_template)
            @test size(precomputes.Q_template) == (N_time, N_time)

            # get_priors
            mock_M_priors = (technical=(component_levels=Dict(:t_idx => N_time),),)
            mock_spec_priors = mock_spec(:t_idx, precomputes)
            priors_str = bstm.get_priors(m_rw2, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_t_idx ~ Exponential(1.0)")
            @test contains(priors_str, "innovations_t_idx ~ MvNormal(zeros(18), 1.0)") # Rank deficient

            # get_updates
            mock_M_updates = (technical=(component_indices=Dict(:t_idx => repeat(1:N_time, outer=N_obs÷N_time)[1:N_obs]),), model_arch="univariate")
            updates_str = bstm.get_updates(m_rw2, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_t_idx = U_t_idx * (D_inv_t_idx .* innovations_t_idx)")
            @test contains(updates_str, "eta .+= (sigma_t_idx .* latent_t_idx)[idx_var]")
        end

        # Test PSpline Component
        @testset "PSpline Component" begin
            N_obs, n_bins = 100, 15
            m_ps = bstm.PSpline(n_bins, 2, Distributions.Exponential(1.0))
            
            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(cov=rand(N_obs)))
            mock_mod_data = Dict(:variables => :cov)
            @test bstm.get_datastructures!(typeof(m_ps), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            mock_M_pc = (data=mock_M_ds[:data],)
            precomputes = bstm.get_precomputes(m_ps, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :B_matrix)
            @test hasproperty(precomputes, :Q_template)
            @test size(precomputes.B_matrix) == (N_obs, n_bins)

            # get_priors
            mock_M_priors = (technical=(component_levels=Dict(:cov => n_bins),),)
            mock_spec_priors = mock_spec(:cov, precomputes)
            priors_str = bstm.get_priors(m_ps, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_cov ~ Exponential(1.0)")
            @test contains(priors_str, "innovations_cov ~ MvNormal(zeros(13), 1.0)")

            # get_updates
            mock_M_updates = (technical=Dict(), model_arch="univariate")
            updates_str = bstm.get_updates(m_ps, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "beta_cov = U_cov * (D_inv_cov .* innovations_cov)")
            @test contains(updates_str, "eta .+= B_matrix_cov * (sigma_cov .* beta_cov)")
        end

        # Test Mixed Component (Random Intercept)
        @testset "Mixed Component" begin
            N_obs, N_levels = 100, 10
            m_mixed = bstm.Mixed("1", bstm.IID(Distributions.Exponential(1.0)))

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(group=rand(1:N_levels, N_obs)))
            mock_mod_data = Dict(:variables => :group)
            @test bstm.get_datastructures!(typeof(m_mixed), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            mock_M_pc = (data=mock_M_ds[:data],)
            precomputes = bstm.get_precomputes(m_mixed, mock_M_pc, mock_mod_data)
            @test precomputes == NamedTuple()

            # get_priors
            mock_M_priors = (technical=(component_levels=Dict(:group => N_levels),),)
            mock_spec_priors = mock_spec(:group)
            priors_str = bstm.get_priors(m_mixed, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_group ~ Exponential(1.0)")
            @test contains(priors_str, "innovations_group ~ MvNormal(zeros(10), 1.0)")
        end

        # Test SVC Component
        @testset "SVC Component" begin
            N_obs, N_areas = 100, 10
            W_svc = create_chain_adj_matrix(N_areas)
            m_svc = bstm.SVC("cov", bstm.ICAR(Distributions.Exponential(1.0)))

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(cov=rand(N_obs), s_idx=rand(1:N_areas, N_obs)), :W => W_svc)
            mock_mod_data = Dict(:variables => [:cov, :s_idx])
            @test bstm.get_datastructures!(typeof(m_svc), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            mock_M_pc = (data=mock_M_ds[:data], W=W_svc, N_areas=N_areas)
            precomputes = bstm.get_precomputes(m_svc, mock_M_pc, mock_mod_data)
            @test hasproperty(precomputes, :Q_template)
            @test size(precomputes.Q_template) == (N_areas, N_areas)

            # get_priors
            mock_M_priors = (technical=(component_levels=Dict(:cov_s_idx => N_areas),),)
            mock_spec_priors = mock_spec(:cov_s_idx, precomputes)
            priors_str = bstm.get_priors(m_svc, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "sigma_cov_s_idx ~ Exponential(1.0)")
            @test contains(priors_str, "innovations_cov_s_idx ~ MvNormal(zeros(9), 1.0)")

            # get_updates
            mock_M_updates = (technical=(component_indices=Dict(:cov_s_idx => mock_M_ds[:data].s_idx),), model_arch="univariate", data=mock_M_ds[:data])
            updates_str = bstm.get_updates(m_svc, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_cov_s_idx = U_cov_s_idx * (D_inv_cov_s_idx .* innovations_cov_s_idx)")
            @test contains(updates_str, "eta .+= (sigma_cov_s_idx .* latent_cov_s_idx)[idx_var] .* M.data.cov")
        end

        # Test Dynamics Component
        @testset "Dynamics Component" begin
            N_obs, N_time = 100, 20
            m_dyn = bstm.Dynamics(:logistic, Distributions.LogNormal(0,1), Distributions.LogNormal(5,1))

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(time=rand(1:N_time, N_obs)))
            mock_mod_data = Dict(:variables => :time)
            @test bstm.get_datastructures!(typeof(m_dyn), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            @test bstm.get_precomputes(m_dyn, (N_time=N_time,), mock_mod_data) == NamedTuple()

            # get_priors
            mock_M_priors = (N_time=N_time,)
            mock_spec_priors = mock_spec(:time)
            priors_str = bstm.get_priors(m_dyn, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "r_time ~ LogNormal{T}(0.0, 1.0)")
            @test contains(priors_str, "K_time ~ LogNormal{T}(5.0, 1.0)")

            # get_updates
            mock_M_updates = (technical=(component_indices=Dict(:time => mock_M_ds[:data].time),), model_arch="univariate", N_time=N_time)
            updates_str = bstm.get_updates(m_dyn, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_time[t] = latent_time[t-1] + r_time * latent_time[t-1] * (1.0 - latent_time[t-1] / K_time)")
        end

        # Test Eigen Component
        @testset "Eigen Component" begin
            N_obs, n_vars = 100, 3
            m_eigen = bstm.Eigen(1, Distributions.Exponential(1.0), Distributions.Exponential(1.0))

            # get_datastructures!
            mock_M_ds = Dict(:data => DataFrame(y1=rand(N_obs), y2=rand(N_obs), y3=rand(N_obs)))
            mock_mod_data = Dict(:variables => [:y1, :y2, :y3])
            @test bstm.get_datastructures!(typeof(m_eigen), mock_M_ds, mock_mod_data) == true

            # get_precomputes
            precomputes = bstm.get_precomputes(m_eigen, (data=mock_M_ds[:data],), mock_mod_data)
            @test hasproperty(precomputes, :eigen_data)
            @test size(precomputes.eigen_data) == (N_obs, n_vars)

            # get_priors
            mock_M_priors = (technical=Dict(),)
            mock_spec_priors = mock_spec(:y1_y2_y3, precomputes)
            priors_str = bstm.get_priors(m_eigen, mock_spec_priors, "univariate", nothing, mock_M_priors)
            @test contains(priors_str, "pca_sd_y1_y2_y3 ~ Exponential(1.0)")
            @test contains(priors_str, "pdef_sd_y1_y2_y3 ~ Exponential(1.0)")
            @test contains(priors_str, "v_raw_y1_y2_y3 ~ MvNormal(zeros(3), 1.0)")
        end
    end

    # --- New Integration Tests ---
    @testset "Integration Tests: Signal Recovery" begin
        # Test 1: Simple IID model - parameter recovery
        @testset "IID Parameter Recovery" begin
            N_obs_sim = 200
            N_groups_sim = 5
            true_sigma_sim = 0.7
            true_effect_sim = randn(N_groups_sim) .* true_sigma_sim
            
            sim_group_idx = repeat(1:N_groups_sim, inner=N_obs_sim÷N_groups_sim)[1:N_obs_sim]
            sim_y = true_effect_sim[sim_group_idx] .+ randn(N_obs_sim) .* 0.1 # Add observation noise
            
            sim_df = DataFrame(y=sim_y, group_idx=sim_group_idx)
            
            model_iid_sim = @bstm(
                likelihood(y, family=gaussian) ~ intercept() + random(group_idx, model=iid),
                sim_df
            )
            
            # Use NUTS for better parameter recovery
            chain_iid_sim = sample(model_iid_sim, NUTS(100, 0.65), 500, progress=false)
            
            # Check if posterior mean is close to true sigma
            posterior_sigma = mean(chain_iid_sim[:sigma_group_idx])
            @test isapprox(posterior_sigma, true_sigma_sim, atol=0.2) # Allow some tolerance
        end

        # Test 2: Simple AR1 model - parameter recovery
        @testset "AR1 Parameter Recovery" begin
            N_time_sim = 50
            true_sigma_sim = 0.5
            true_rho_sim = 0.8
            
            # Simulate AR1 process
            ar1_process = zeros(N_time_sim)
            ar1_process[1] = randn() * true_sigma_sim
            for t in 2:N_time_sim
                ar1_process[t] = true_rho_sim * ar1_process[t-1] + randn() * true_sigma_sim
            end
            
            sim_y = ar1_process .+ randn(N_time_sim) .* 0.1 # Add observation noise
            sim_df = DataFrame(y=sim_y, t_idx=1:N_time_sim)
            
            model_ar1_sim = @bstm(
                likelihood(y, family=gaussian) ~ intercept() + random(t_idx, model=ar1),
                sim_df
            )
            
            chain_ar1_sim = sample(model_ar1_sim, NUTS(100, 0.65), 500, progress=false)
            
            posterior_sigma = mean(chain_ar1_sim[:sigma_t_idx])
            posterior_rho = mean(chain_ar1_sim[:rho_t_idx])
            
            @test isapprox(posterior_sigma, true_sigma_sim, atol=0.2)
            @test isapprox(posterior_rho, true_rho_sim, atol=0.2)
        end

        # Test 3: Complex model - BYM2 ⊗ AR1 interaction
        @testset "BYM2 ⊗ AR1 Interaction" begin
            N_areas_sim, N_time_sim = 5, 10
            total_obs_sim = N_areas_sim * N_time_sim
            W_sim = create_chain_adj_matrix(N_areas_sim)
            
            # Simulate data with a known interaction pattern
            true_int_pattern = randn(N_areas_sim, N_time_sim)
            sim_y = vec(true_int_pattern) .+ randn(total_obs_sim) .* 0.1
            
            sim_df = DataFrame(
                y=sim_y,
                s_idx=repeat(1:N_areas_sim, inner=N_time_sim),
                t_idx=repeat(1:N_time_sim, outer=N_areas_sim)
            )
            
            model_int_sim = @bstm(
                likelihood(y, family=gaussian) ~ intercept() + 
                    random(s_idx, model=bym2) ⊗ random(t_idx, model=ar1),
                sim_df, W=W_sim
            )
            
            chain_int_sim = sample(model_int_sim, NUTS(100, 0.65), 200, progress=false)
            
            # Check if the interaction parameters are sampled reasonably
            @test mean(chain_int_sim[:sigma_s_idx_t_idx]) > 0.01 # Should be non-zero
            @test mean(chain_int_sim[:rho_s_idx_t_idx]) > 0.01 # Should be non-zero
        end
    end

    @testset "Complex Integration Tests" begin
        @testset "Advanced Features: Signal Recovery" begin
            # Rationale: Verify that complex model features (Hurdle, Eigen, Multifidelity) can
            # correctly recover a known ground-truth signal from simulated data.
            
            @testset "Hurdle-Eigen Integration" begin
                n_s, n_t, n_factors = 15, 12, 2
                total_n = n_s * n_t
                t_basis_synth = hcat(sin.((1:n_t) .* (2π/n_t)), cos.((1:n_t) .* (2π/n_t)))
                s_idx = repeat(1:n_s, inner=n_t)
                t_idx = repeat(1:n_t, outer=n_s)
                
                beta_eigen = [2.5, -1.2]
                shared_signal = (t_basis_synth * beta_eigen)[t_idx]
                
                s_clusters = repeat(1:5, inner=3)
                cluster_vals = [-1.5, -0.5, 0.0, 1.0, 2.0]
                hierarchy_signal = cluster_vals[s_clusters[s_idx]]
                
                eta_h = -0.8 .+ shared_signal .+ hierarchy_signal
                y_p = [rand() < LogExpFunctions.logistic(val) for val in eta_h]
                
                eta_i = 1.2 .+ shared_signal .+ hierarchy_signal
                y_counts = [y_p[idx] ? Float64(rand(Poisson(exp(eta_i[idx])))) : missing for idx in 1:total_n]
                
                y_obs_hurdle = hcat(y_p, y_counts)

                M_audit = bstm.bstm_options(
                    y_obs = y_obs_hurdle, s_idx = s_idx, t_idx = t_idx,
                    model_family = "hurdle_poisson", model_arch = "multivariate",
                    t_basis = t_basis_synth, t_n_dims = 2, t_ltri_indices = [1, 2, 3],
                    model_time = "eigen",
                    spatial_hierarchy = Dict(:cluster => (n_units=5, indices=s_clusters[s_idx], model="iid", template=(matrix=sparse(I(5)),))),
                    noise = 1e-4
                )

                model_audit = bstm.bstm_multivariate(M_audit)
                chain_audit = sample(model_audit, MH(), 200, progress=false)
                res_audit = bstm._reconstruct(bstm.MultivariateArchitecture(), "audit", chain_audit, M_audit, nothing, 0.05)
                
                y_pred = res_audit.predictions_denoised[1].mean # Participation probability
                truth_total = LogExpFunctions.logistic.(eta_h)
                
                corr_val = cor(y_pred, truth_total)
                @test corr_val > 0.5
            end

            @testset "Multifidelity Signal Transfer" begin
                n_units_mf, n_time_mf = 12, 18
                total_mf = n_units_mf * n_time_mf
                s_idx_mf = repeat(1:n_units_mf, inner=n_time_mf)
                t_idx_mf = repeat(1:n_time_mf, outer=n_units_mf)
                W_mf = create_chain_adj_matrix(n_units_mf)

                s_lat_lo = cumsum(randn(n_units_mf)) .* 0.5
                t_lat_lo = sin.((1:n_time_mf) .* (2π / 12)) .* 0.8
                signal_low_latent = s_lat_lo[s_idx_mf] + t_lat_lo[t_idx_mf]

                rho_mf_true = 1.4
                innovation_hi = randn(total_mf) .* 0.3
                signal_high_latent = (rho_mf_true .* signal_low_latent) + innovation_hi

                y_lo = signal_low_latent .+ randn(total_mf) .* 0.1
                y_hi = signal_high_latent .+ randn(total_mf) .* 0.1
                
                df_mf = DataFrame(y_high=y_hi, y_low=y_lo, s_idx=s_idx_mf, t_idx=t_idx_mf)

                model_mf = bstm.bstm(
                    "likelihood(y_high) ~ 1 + random(s_idx, model='bym2') + random(t_idx, model='ar1') + nested(low_fi, formula=\"likelihood(y_low) ~ 1 + random(s_idx) + random(t_idx)\")",
                    df_mf, model_family="gaussian", W=W_mf
                )
                
                chain_mf = sample(model_mf, NUTS(100, 0.65), 200, progress=false, check_model=false)
                rho_est = mean(chain_mf[:rho_nested_low_fi])
                @test abs(rho_est - rho_mf_true) < 0.5
            end
        end

        @testset "Prediction Engine" begin
            # Rationale: Verify out-of-sample prediction, especially for complex manifolds like mosaics.
            n_s_train, n_t = 16, 12
            total_train = n_s_train * n_t
            x_range = range(0, stop=100, length=4)
            y_range = range(0, stop=100, length=4)
            train_coords = [(x, y) for x in x_range, y in y_range][:]
            s_idx_train = repeat(1:n_s_train, inner=n_t)
            t_idx_train = repeat(1:n_t, outer=n_s_train)
            
            s_clusters_train = repeat(1:4, inner=4)
            cluster_effects = [-2.0, 0.0, 1.0, 3.0]
            y_train_signal = cluster_effects[s_clusters_train[s_idx_train]] .+ 0.05 .* t_idx_train
            y_train_obs = y_train_signal .+ randn(total_train) .* 0.05

            train_df = DataFrame(y=y_train_obs, s_idx=s_idx_train, t_idx=t_idx_train, 
                                 s_x=[p[1] for p in train_coords[s_idx_train]], s_y=[p[2] for p in train_coords[s_idx_train]])

            test_pts = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)]
            n_s_test = length(test_pts)
            test_df = DataFrame(s_x=repeat([p[1] for p in test_pts], inner=n_t), 
                                s_y=repeat([p[2] for p in test_pts], inner=n_t), 
                                t_idx=repeat(1:n_t, outer=n_s_test))

            model_obj = bstm.bstm("likelihood(y) ~ 1 + random(s_idx, model=mosaic, n_regions=4) + random(t_idx, model=ar1)", 
                             train_df; s_N=n_s_train, t_N=n_t, cluster_assignments=s_clusters_train)
            
            chain_train = sample(model_obj, MH(), 500, progress=false)
            res_pred = bstm.predict(model_obj, chain_train, test_df; n_samples=100)
            
            y_test_pred = res_pred.predictions_denoised.mean
            
            # Check if predictions align with cluster means
            @test isapprox(mean(y_test_pred[1:n_t]), mean(cluster_effects[1] .+ 0.05 .* (1:n_t)), atol=1.0)
            @test isapprox(mean(y_test_pred[end-n_t+1:end]), mean(cluster_effects[4] .+ 0.05 .* (1:n_t)), atol=1.0)
        end

        @testset "Cross-Validation Orchestrator" begin
            # Rationale: Verify that the CV orchestrator can correctly partition data
            # and run the train-predict loop for different CV methods.
            
            # Create a small, simple dataset for CV testing
            cv_s = 20
            cv_t = 10
            cv_total = cv_s * cv_t
            cv_df = DataFrame(
                y = randn(cv_total),
                s_idx = repeat(1:cv_s, inner=cv_t),
                t_idx = repeat(1:cv_t, outer=cv_s),
                s_x = rand(cv_total),
                s_y = rand(cv_total)
            )
            cv_W = create_chain_adj_matrix(cv_s)
            cv_formula = "likelihood(y) ~ 1 + random(t_idx, model=ar1)"

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

end