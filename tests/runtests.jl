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
            @test contains(priors_str, "innovations_group_var ~ MvNormal(zeros(10), 1.0)")

            mock_M_updates = (technical=(component_indices=Dict(:group_var => mock_M_ds[:data].group_var),), model_arch="univariate")
            mock_spec_updates = mock_spec(:group_var)
            updates_str = bstm.get_updates(m_iid, mock_spec_updates, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_group_var = sigma_group_var .* innovations_group_var")
            @test contains(updates_str, "eta .+= latent_group_var[$(mock_M_updates.technical[:component_indices][:group_var])]")

            mock_chain_effects = mock_chain(Dict(:sigma_group_var => 0.5, :innovations_group_var => randn(N_levels, 10)), 10)
            mock_M_effects = (technical=(component_indices=Dict(:group_var => mock_M_ds[:data].group_var),), model_arch="univariate")
            mock_spec_effects = mock_spec(:group_var)
            p_names_effects = collect(keys(mock_chain_effects))
            
            effects_result = bstm.get_effects(m_iid, mock_chain_effects, mock_M_effects, 10, 1, p_names_effects, mock_spec_effects, nothing, N_obs)
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
            @test contains(priors_str, "innovations_s_idx ~ MvNormal(zeros(10), 1.0)")

            mock_M_updates = (technical=(component_indices=Dict(:s_idx => repeat(1:N_areas, inner=N_obs÷N_areas)[1:N_obs]),), model_arch="univariate")
            updates_str = bstm.get_updates(m_leroux, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_s_idx = U_s_idx * (D_inv_s_idx .* innovations_s_idx)")
            @test contains(updates_str, "eta .+= (sigma_s_idx .* latent_s_idx)[idx_var]")
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
            @test contains(priors_str, "innovations_x_y ~ MvNormal(zeros(100), 1.0)")

            mock_M_updates = (technical=(component_indices=Dict(),), model_arch="univariate")
            updates_str = bstm.get_updates(m_gp, mock_spec_priors, "univariate", nothing, mock_M_updates)
            @test contains(updates_str, "latent_x_y = L_x_y * innovations_x_y")
            @test contains(updates_str, "eta .+= latent_x_y")
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
