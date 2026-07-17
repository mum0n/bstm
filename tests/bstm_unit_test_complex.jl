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

# Helper function for creating a simple adjacency matrix
function create_chain_adj_matrix(n)
    W = spzeros(Int, n, n)
    for i in 1:(n-1)
        W[i, i+1] = 1
        W[i+1, i] = 1
    end
    return W
end

@testset "BSTM Complex Integration Tests" begin

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
            y_p = [rand() < logistic(val) for val in eta_h]
            
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
            truth_total = logistic.(eta_h)
            
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

            model_mf = bstm(
                "y_high ~ 1 + spatial(s_idx, model='bym2') + temporal(t_idx, model='ar1') + nested(low_fi, formula=\"y_low ~ 1 + spatial(s_idx) + temporal(t_idx)\")",
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

        model_obj = bstm("y ~ 1 + spatial(s_idx, model=mosaic, n_regions=4) + temporal(t_idx, model=ar1)", 
                         train_df; s_N=n_s_train, t_N=n_t, cluster_assignments=s_clusters_train)
        
        chain_train = sample(model_obj, MH(), 500, progress=false)
        res_pred = predict(model_obj, chain_train, test_df; n_samples=100)
        
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
        cv_formula = "y ~ 1 + temporal(t_idx, model=ar1)"

        @testset "k-fold CV" begin
            cv_results_kfold = bstm_cv_orchestrator(cv_formula, cv_df; method=:kfold, n_folds=3, n_samples=50, W=cv_W)
            @test length(cv_results_kfold.folds) == 3
            @test cv_results_kfold.mean_rmse isa Real
        end

        @testset "Temporal Forward-Chaining CV" begin
            cv_results_fchain = bstm_cv_orchestrator(cv_formula, cv_df; method=:temporal_forward_chain, cv_var=:t_idx, n_folds=2, n_samples=50, W=cv_W)
            @test length(cv_results_fchain.folds) == 2
            @test cv_results_fchain.mean_r2 isa Real
        end
    end
end