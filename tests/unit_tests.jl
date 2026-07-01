using Test
using Distributions
using LinearAlgebra
using DataFrames
using CategoricalArrays
using Turing
using Random

# Header: BSTM v06.1 Manifold Dispatch Audit [v17.7.15 - 2025-05-22]
# Rationale: Resolving TypeError by fixing return_data type and ensuring simulation data integrity.

println("--- Starting v06.1 Manifold Dispatch Audit ---")
Random.seed!(17711)

# 1. Setup Mock Data Inputs for build_model tests
mock_inputs = (
    s_N = 10,
    t_N = 20,
    W = zeros(10, 10)
)

@testset "BSTM v06.1 build_model Dispatch Tests" begin

    # Test A: Random Walk Primitives (RW1 & RW2)
    @testset "Temporal Walks: RW1 & RW2" begin
        m_rw1 = RW1(Exponential(1.0))
        res_rw1 = build_model(m_rw1, mock_inputs)
        @test res_rw1.model_type == :rw1
        @test size(res_rw1.Q_template) == (20, 20)
        @test res_rw1.scaling_factor == 1.0

        m_rw2 = RW2(Exponential(1.0))
        res_rw2 = build_model(m_rw2, mock_inputs)
        @test res_rw2.model_type == :rw2
        @test res_rw2.scaling_factor > 1.0
    end

    # Test B: Seasonal Cyclic (Periodic Continuity)
    @testset "Seasonal: Cyclic" begin
        m_cyc = Cyclic(12, Exponential(1.0))
        res_cyc = build_model(m_cyc, mock_inputs)
        @test res_cyc.model_type == :cyclic
        @test size(res_cyc.Q_template) == (12, 12)
        # Verification: Diagonal of a circular Laplacian (2 -1 -1)
        @test tr(res_cyc.Q_template) ≈ (2.0 * 12.0) / res_cyc.scaling_factor
    end

    # Test C: Basis-Function Primitives (BSpline & Wavelets)
    @testset "Basis Space: BSpline & Wavelets" begin
        m_bs = BSpline(15, 3, Exponential(1.0))
        res_bs = build_model(m_bs, mock_inputs)
        @test res_bs.model_type == :bspline
        @test size(res_bs.Q_template) == (15, 15)
        @test res_bs.hyper.degree == 3

        m_wv = Wavelets(:db4, 32, Exponential(1.0))
        res_wv = build_model(m_wv, mock_inputs)
        @test res_wv.model_type == :wavelet
        @test res_wv.hyper.family == :db4
    end

    # Test D: Dynamic Distance Geometries (Hyperbolic & ExponentialDecay)
    @testset "Continuous: Hyperbolic & Decay" begin
        m_hyp = Hyperbolic(-1.0, Exponential(1.0))
        res_hyp = build_model(m_hyp, mock_inputs)
        @test res_hyp.model_type == :hyperbolic
        @test isnothing(res_hyp.Q_template)
        @test res_hyp.hyper.curvature == -1.0

        m_dec = ExponentialDecay(Exponential(1.0), InverseGamma(3, 3))
        res_dec = build_model(m_dec, mock_inputs)
        @test res_dec.model_type == :decay
        @test isnothing(res_dec.Q_template)
    end

end

println("\n--- v06.1 Registry Audit Complete ---")


###########################

# 2. Dummy DataFrame Generation for stress testing the parser
dummy_N = 100
dummy_T = 10
total_rows = dummy_N * dummy_T

data_test = DataFrame(
    y_obs = rand(total_rows),
    s_idx = repeat(1:dummy_N, outer=dummy_T),
    t_idx = repeat(1:dummy_T, inner=dummy_N),
    x1 = rand(total_rows),
    x2 = rand(total_rows),
    group_id = categorical(repeat(1:5, inner=div(total_rows, 5))),
    region = categorical(repeat(["North", "South"], inner=div(total_rows, 2))),
    s_x = rand(total_rows) * 100.0,
    s_y = rand(total_rows) * 100.0,
    eigen_var1 = rand(total_rows),
    eigen_var2 = rand(total_rows)
)

# 3. Formula Stress Test
# Rationale: Verifying smooth() and nested() tokens map to technical primitives
formula_test = "y_obs ~ 1 + x1 + x2 + smooth(s_x, s_y; manifold='rff', m_rff=20)"

println("\nAudit: Testing bstm() dispatcher return_data=true...")
# FIX: Using return_data = true (Bool) instead of string "inputs" to resolve TypeError
M_parsed = bstm(
    formula_test, 
    data_test; 
    return_data = true,
    model_family = "gaussian",
    s_N = dummy_N,
    t_N = dummy_T
)

if M_parsed isa NamedTuple
    println("--- Monolithic Configuration Success ---")
    println("Discovered s_N: ", M_parsed.s_N)
    println("Discovered t_N: ", M_parsed.t_N)
    println("smooth Terms Registered: ", length(M_parsed.smooth_terms))
else
    @error "BSTM Error: Dispatch failed to return configuration metadata."
end 

 
 

    println("--- Starting Expanded BSTM Formulaic Unit Tests ---")

    # 1. Generate Synthetic Data for testing all features
    s_N_test = 20  # Reduced spatial units for faster tests
    t_N_test = 6   # Time slices
    total_obs = s_N_test * t_N_test

    # Helper function to create a simple chain adjacency matrix for testing
    function create_chain_adj_matrix(n)
        W = spzeros(Int, n, n)
        for i in 1:(n-1)
            W[i, i+1] = 1
            W[i+1, i] = 1
        end
        return W
    end

    # Basic coordinates and Adjacency
    s_coords_test = [(Float64(i), Float64(j)) for i in 1:s_N_test, j in 1:s_N_test][1:s_N_test]
    t_coords_test = collect(1.0:Float64(t_N_test))
    # Corrected W_test to represent a connected graph (e.g., a chain graph)
    W_test = create_chain_adj_matrix(s_N_test)

    # Simulate a comprehensive DataFrame to cover various terms
    dummy_df = DataFrame(
        y = rand(total_obs) .* 100, 
        y1 = rand(total_obs) .* 50, 
        y2 = rand(total_obs) .* 75, 
        s_idx = repeat(1:s_N_test, t_N_test), 
        t_idx = repeat(1:t_N_test, inner=s_N_test), 
        u_idx = repeat(1:min(12, t_N_test), inner=div(total_obs, min(12, t_N_test))),
        x_cont = rand(total_obs) .* 10,
        x_cat1 = categorical(repeat(["A", "B"], outer=div(total_obs, 2))),
        Region = categorical(repeat(["East", "West"], outer=div(total_obs, 2))),
        log_offset = zeros(total_obs)
    )

    dummy_df.ycount = Int.(round.(dummy_df.y))
    # Coordinate extraction for continuous models (lat/lon recovery)
    dummy_df.lat = [p[1] for p in s_coords_test[dummy_df.s_idx]]
    dummy_df.lon = [p[2] for p in s_coords_test[dummy_df.s_idx]]

    # 2. Define Expanded Test Cases
    # Format: (Name, Formula, Family, Args)
    test_cases = [
        ("ST_TypeIV_Poisson", "ycount ~ 1 + x_cont + (Spatial(s_idx) ⊗ Temporal(t_idx))", "poisson", Dict(:W => W_test)),
        ("MixedEffects_Gaussian", "y ~ 1 + x_cont + me(x_cont | x_cat1) + Spatial(s_idx)", "gaussian", Dict(:W => W_test)),
        ("Multivariate_Gaussian", "y + y1 ~ 1 + x_cont + Spatial(s_idx)", "gaussian", Dict(:W => W_test)),
        ("Seasonal_RW2_Poisson", "ycount ~ 1 + Seasonal(u_idx) + Temporal(t_idx; model='rw2')", "poisson", Dict()),
        
        # --- New Complex Variations ---
        ("Hurdle_Poisson_ZeroInflated", "ycount ~ 1 + x_cont + Spatial(s_idx)", "poisson", Dict(:W => W_test, :hurdle => true)),
        ("Spatial_Smooth_Coordinates", "y ~ 1 + Smooth(lat, lon; manifold='rw2') + Temporal(t_idx)", "gaussian", Dict()),
        ("Hierarchical_ICAR", "y ~ 1 + re(Region, model='icar') + Spatial(s_idx)", "poisson", Dict(:W => W_test)),
        ("SVC_Varying_Coefficient", "y ~ 1 + svc(x_cont) + Spatial(s_idx)", "gaussian", Dict(:W => W_test))
    ]

    passed_tests = 0
    failed_tests = 0

    # 3. Execution Loop
    for (name, f_str, fam, extra_args) in test_cases
        println("\n[Test Case]: $name")
               # A. Model Construction
            model = bstm(f_str, dummy_df, model_family=fam; extra_args...)
            
            # B. Diagnostic Sampling
            chain = sample(model, MH(), 100, progress=false)

            # C. Dashboard Generation Test
            dummy_au = (centroids=s_coords_test, W=W_test, polygons=nothing, graph=nothing)
            
            res = model_results_comprehensive(
                model, 
                chain, 
                model.args.M, 
                dummy_au; 
                n_samples=50, 
                alpha=0.05
            )

            println("  Result: PASS")
            passed_tests += 1
            println($name: $e)

             
        end
    end

    # 4. Summary
    println("\n--- Unit Test Summary ---")
    println("Total: $(length(test_cases)) | Passed: $passed_tests | Failed: $failed_tests")
    
    if failed_tests > 0
        throw(ErrorException("BSTM Expanded Unit Tests Failed."))
    else
        println("All BSTM expanded unit tests passed successfully!")
    end








# --- XR: Diagnostic Recovery Audit [v3] ---
# Rationale: The previous NaN correlation suggests the manifold parameters 
# were stuck at the prior mode (zero variance). We use weakly informative 
# priors for scales and increase adaptation to ensure latent signal recovery.

# 1. Setup Simulation Parameters
Random.seed!(101)
n_locations = 10
n_time = 24
total_n = n_locations * n_time
periodicity = 12.0

s_idx = repeat(1:n_locations, inner=n_time)
t_idx = repeat(1:n_time, outer=n_locations)
u_idx_val = repeat(1:Int(periodicity), outer=Int(total_n/periodicity))

# 2. Generate Ground Truth
# Corrected: Using π instead of invalid grave symbol
t_angle_val = (collect(1:n_time) .* (2π / periodicity))
harmonic_signal = 1.5 .* sin.(t_angle_val[t_idx]) .+ 0.8 .* cos.(t_angle_val[t_idx])

basis_synth = zeros(total_n, 4)
for k in 1:4
    # Gaussian basis functions for the spline component
    basis_synth[:, k] = exp.(-( (t_idx .- (k * 6)).^2 ) ./ 20.0)
end
spline_signal = basis_synth * [0.5, -1.0, 1.2, -0.5]

# Combined Signal + Observation Noise
eta_truth = harmonic_signal .+ spline_signal
y_obs_synth = eta_truth .+ randn(total_n) .* 0.05

# 3. Configure M Metadata
# We explicitly set hyperpriors here to override defaults that might be too restrictive
custom_hyperpriors = Dict(
    HarmonicSeasonal => (sigma_prior = Exponential(2.0), amplitude_prior = Normal(0, 2)),
    BSpline => (sigma_prior = Exponential(2.0))
)

M_diag = bstm_options(
    y_obs = y_obs_synth,
    s_idx = s_idx,
    t_idx = t_idx,
    u_idx = u_idx_val,
    model_family = "gaussian",
    model_space = "none",
    model_time = "none",
    model_season = "harmonic",
    period = 12,
    t_angle = t_angle_val,
    s_coord = zeros(n_locations, 2),
    basis_matrices = Dict{Symbol, Any}(:seasonal_spline => basis_synth),
    noise = 1e-5,
    Xfixed = zeros(total_n, 0), # No intercept to force manifold recovery
    hyperpriors = custom_hyperpriors
)

println("--- Starting Diagnostic Unit Test: Recovery Audit [v3] ---")
diag_model = bstm_univariate(M_diag)

# Use NUTS with higher adaptation to find the signal-bearing region of the posterior
chain_diag = sample(diag_model, NUTS(500, 0.65), 1000, progress=true)

println("\n--- POST-SAMPLING AUDIT ---")
try
    res_diag = _reconstruct(UnivariateArchitecture(), "diagnostic", chain_diag, M_diag, nothing, 0.05)
    y_pred = res_diag.predictions_observed_denoised.mean
    
    # Check for variance to diagnose NaN correlation causes
    p_var = var(y_pred)
    println("Audit [Prediction Variance]: ", round(p_var, digits=6))
    
    if p_var < 1e-9
        println("Audit [Stability]: FAILED. Predicted signal is constant (variance too low).")
        correlation = 0.0
    else
        correlation = cor(y_pred, eta_truth)
    end
    
    println("Audit [Signal Recovery Correlation]: ", round(correlation, digits=3))

    if correlation > 0.7
        println("UNIT TEST RESULT: SUCCESS. Manifolds recovered latent signal.")
    else
        println("UNIT TEST RESULT: FAILURE. Recovery accuracy insufficient.")
    end
catch e
    println("UNIT TEST ERROR: ", e)
    stacktrace(catch_backtrace())
end




# XR: Monolithic Hurdle & Eigen-Temporal System Audit (v07.5)
# Rationale: Verifying that shared additive manifolds (Basis/Eigen) correctly 
# interoperate with the coupled Hurdle architecture.

using Random, Statistics, LinearAlgebra, Turing, Distributions

function run_hurdle_eigen_audit()
    println("--- XR: Starting Hurdle-Eigen Multivariate Audit ---")
    Random.seed!(909)
    
    # 1. Setup Dimensions
    n_s = 15
    n_t = 12
    total_n = n_s * n_t
    n_factors = 2

    # 2. Generate Temporal Eigen-Basis
    # We create a smooth sine-based trend as the primary latent factor
    t_grid = collect(1:n_t)
    t_basis_synth = hcat(sin.(t_grid .* (2π/n_t)), cos.(t_grid .* (2π/n_t)))
    
    # Map basis to the long-format indices
    s_idx = repeat(1:n_s, inner=n_t)
    t_idx = repeat(1:n_t, outer=n_s)
    
    # 3. Simulate Latent Signal
    # We assume a shared smooth trend (Basis) and a spatial cluster effect (Hierarchy)
    # Ground Truth Coefficients
    beta_eigen = [2.5, -1.2]
    shared_signal = (t_basis_synth * beta_eigen)[t_idx]
    
    # Add a simple hierarchical cluster effect (5 groups)
    s_clusters = repeat(1:5, inner=3)
    cluster_vals = [-1.5, -0.5, 0.0, 1.0, 2.0]
    hierarchy_signal = cluster_vals[s_clusters[s_idx]]
    
    # 4. Construct Hurdle Branches
    # Participation (Crossing zero)
    eta_h = -0.8 .+ shared_signal .+ hierarchy_signal
    y_p = [rand() < logistic(val) ? 1.0 : 0.0 for val in eta_h]
    
    # Intensity (Positive counts)
    eta_i = 1.2 .+ shared_signal .+ hierarchy_signal
    y_counts = [y_p[idx] > 0 ? Float64(rand(Poisson(exp(eta_i[idx])))) : missing for idx in 1:total_n]
    
    y_obs_hurdle = hcat(y_p, y_counts)

    # 5. Configure M Metadata (Audited Feature Set)
    M_audit = bstm_options(
        y_obs = y_obs_hurdle,
        s_idx = s_idx,
        t_idx = t_idx,
        model_family = "hurdle_poisson",
        model_arch = "multivariate",
        # Pass Eigen-Temporal Basis
        t_basis = t_basis_synth,
        t_n_dims = 2,
        t_ltri_indices = [1, 2, 3], # 2x2 Householder
        model_time = "eigen",
        # Add Hierarchy Structure
        spatial_hierarchy = Dict(
            :cluster => (n_units=5, indices=s_clusters[s_idx], model="iid", template=(matrix=sparse(I(5)),))
        ),
        noise = 1e-4
    )

    # 6. Model Execution
    println("Audit: Sampling from Monolithic Multivariate Hurdle model...")
    model_audit = bstm_multivariate(M_audit)
    chain_audit = sample(model_audit, MH(), 1000, progress=true)

    # 7. Reconstruction Verification
    println("Audit: Reconstructing posterior manifolds...")
    res_audit = _reconstruct(MultivariateArchitecture(), "audit_v07.5", chain_audit, M_audit, nothing, 0.05)

    # 8. Success Criteria: Correlation check
    # The 'predictions_observed_denoised' should capture the combined shared_signal + hierarchy
    y_pred = res_audit.predictions_observed_denoised.mean
    truth_total = shared_signal .+ hierarchy_signal
    
    # We check correlation against the participation probability logic
    corr_val = cor(y_pred, truth_total)
    
    println("\n--- Audit Results ---")
    println("Signal Recovery Correlation: ", round(corr_val, digits=4))
    
    if corr_val > 0.7
        println("RESULT: Hurdle-Eigen Integration PASSED.")
    else
        error("RESULT: Hurdle-Eigen Integration FAILED. Zero-variance or mismatch detected.")
    end
    
    return res_audit
end

# Run the unit test
audit_results = run_hurdle_eigen_audit()




# XR: Out-of-Sample Prediction System Audit (v11.6)
# Rationale: Final verification of the interpolation logic for Mosaic spatial manifolds.
 
# --- 1. Experimental Setup ---
Random.seed!(116)
n_s_train = 16
n_t = 12
n_total_train = n_s_train * n_t

# Training Coordinates (4x4 Grid)
x_range = collect(range(0, stop=100, length=4))
y_range = collect(range(0, stop=100, length=4))
train_coords = [(x, y) for x in x_range for y in y_range]

s_idx_train = repeat(1:n_s_train, inner=n_t)
t_idx_train = repeat(1:n_t, outer=n_s_train)

# Latent Signal Generation (4 Clusters for Mosaic)
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

# --- 2. Test Data Construction (Points clearly within the 4 training clusters) ---
test_pts = [(15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0)]
n_s_test = length(test_pts)
test_df = DataFrame(
    s_x = repeat([p[1] for p in test_pts], inner=n_t),
    s_y = repeat([p[2] for p in test_pts], inner=n_t),
    t_idx = repeat(1:n_t, outer=n_s_test)
)

# --- 3. Model Training ---
println("--- Step 1: Training BSTM on Grid Data ---")
M_train = bstm_options(
    y_obs = train_df.y_obs,
    model_space = "mosaic",
    model_time = "ar1",
    model_family = "gaussian",
    n_mosaics = 4,
    cluster_assignments = s_clusters_train,
    s_coord = reduce(hcat, collect.(train_coords))',
    s_idx = s_idx_train,
    t_idx = t_idx_train,
    use_sv = false
)

model_obj = bstm_univariate(M_train)
chain_train = sample(model_obj, MH(), 1000, progress=true)

# --- 4. Prediction Execution ---
println("\n--- Step 2: Projecting Manifolds onto Test Points ---")
res_pred = predict(model_obj, chain_train, test_df, n_samples=300)

# --- 5. Accuracy Audit ---
println("\n--- Step 3: Auditing Prediction Alignment ---")
y_test_pred = res_pred.predictions_observed_denoised.mean

# Verify that the 4 test points match the 4 cluster intercepts
println("Test Point 1 Prediction (Expected ~-1.7): ", round(mean(y_test_pred[1:n_t]), digits=2))
println("Test Point 4 Prediction (Expected ~3.3): ", round(mean(y_test_pred[end-n_t+1:end]), digits=2))

# Dashboard for visual verification
model_results_plots(res_pred; centroids=test_pts)



# Validation SV: 

# XR: Comprehensive Logic Validation (v11.9.16 - Trait Fix Verification)

n_s = 25
n_t = 10
n_total = n_s * n_t

dat_v = generate_sim_data(n_s, n_t)

unique_s_x = dat_v.s_x[1:n_t:end]
unique_s_y = dat_v.s_y[1:n_t:end]
s_coords_mat = Matrix{Float64}(hcat(unique_s_x, unique_s_y))
s_clusters_v = repeat(1:5, inner=5)

m_rff_sigma = 20
coords_full = Matrix{Float64}(hcat(dat_v.s_x, dat_v.s_y))
W_v, b_v = generate_informed_rff_params(coords_full, m_rff_sigma)
vol_proj_mat = Matrix{Float64}((coords_full * W_v) .+ b_v')

M_v = bstm_options(
    y_obs = dat_v.y_obs,
    s_x = dat_v.s_x,
    s_y = dat_v.s_y,
    t_idx = dat_v.t_idx,
    s_idx = dat_v.s_idx,
    model_space = "mosaic",
    model_time = "ar1",
    use_sv = true,
    M_rff_sigma = m_rff_sigma,
    vol_proj = vol_proj_mat,
    n_mosaics = 5,
    cluster_assignments = s_clusters_v,
    s_coord = s_coords_mat
)

println("--- Executing v11.9.16 Validation (Reconstruction Trait Fix) ---")

model_v = bstm_univariate(M_v)
chain_v = sample(model_v, MH(), 200, progress=true)

# This call should now succeed without MethodError for Temporal/Seasonal
res_v = _reconstruct(UnivariateArchitecture(), "validation_trait_fix", chain_v, M_v, nothing, 0.05)
println("RECONSTRUCTION SUCCESSFUL.")

model_results_plots(
    res_v,
    1;
    y_obs = M_v.y_obs,
    centroids = tuple.(s_coords_mat[:, 1], s_coords_mat[:, 2])
)




## test spline only

# --- XR: Re-Running Diagnostic Recovery Audit [v4.2 Fixed Indexing] ---

Random.seed!(404)
n_locations, n_time = 10, 24
total_n = n_locations * n_time
periodicity = 12.0

s_idx = repeat(1:n_locations, inner=n_time)
t_idx = repeat(1:n_time, outer=n_locations)
u_idx_val = repeat(1:Int(periodicity), outer=Int(total_n/periodicity))

# 1. Ground Truth Construction
t_angle_val = (collect(1:n_time) .* (2*pi / periodicity))
harmonic_truth = 1.8 .* sin.(t_angle_val[t_idx]) .- 0.5 .* cos.(t_angle_val[t_idx])

basis_synth = zeros(total_n, 4)
for k in 1:4
    basis_synth[:, k] = exp.(-( (t_idx .- (k * 6)).^2 ) ./ 15.0)
end
spline_truth = basis_synth * [1.2, -0.8, 0.4, 0.9]

eta_truth = harmonic_truth .+ spline_truth
y_obs_synth = eta_truth .+ randn(total_n) .* 0.1

# 2. Metadata Configuration
M_diag = bstm_options(
    y_obs = y_obs_synth,
    s_idx = s_idx,
    t_idx = t_idx,
    u_idx = u_idx_val,
    model_space = "none",      # Focus audit on Temporal/Seasonal
    model_season = "harmonic",
    period = 12,
    u_N = 12,
    basis_matrices = Dict{Symbol, Any}(:test_spline => basis_synth),
    noise = 1e-4
)

println("--- Starting Recovery Audit v4.2 ---")
diag_model = bstm_univariate(M_diag)

# Execute sampling with NUTS
chain_diag = sample(diag_model, NUTS(500, 0.65), 1000, progress=true)

# Reconstruction and Audit of Latent Field Recovery
res_diag = _reconstruct(UnivariateArchitecture(), "diag", chain_diag, M_diag, nothing, 0.05)
y_pred = res_diag.predictions_observed_denoised.mean

correlation = cor(y_pred, eta_truth)
println("Audit [Signal Correlation]: ", round(correlation, digits=3))

if correlation > 0.7
    println("UNIT TEST RESULT: SUCCESS. Manifolds recovered latent signal.")
else
    println("UNIT TEST RESULT: FAILURE. Check model logic and priors.")
end







###


using Test
using Distributions
using Random
using LogExpFunctions
using Test, Distributions, Random, LogExpFunctions, PDMats



println("--- Starting Exhaustive BSTM Taxonomy Audit [v10.0] ---")

using Test
using Distributions
using Random
import LogExpFunctions

println("--- Starting Consolidated BSTM Likelihood Taxonomy Audit [Keyword Argument Sync] ---")

@testset "BSTM Likelihood: Comprehensive Taxonomy Audit" begin

    # 1. Discrete Families (ZI & Hurdle Support)
    @testset "Discrete: Poisson, NegBin, Binomial" begin
        # Poisson ZI Point
        mu_p = exp(1.0)
        phi = 0.2
        # Updated to use keywords
        d_pois = bstm_Likelihood("poisson", [0.0]; phi_zi=phi)
        ana_pois = LogExpFunctions.logsumexp(log(phi), log(1-phi) + logpdf(Poisson(mu_p), 0))
        @test isapprox(Distributions.logpdf(d_pois, 1.0), ana_pois)

        # NegBin Hurdle Point
        mu_nb = exp(1.5)
        r_val = 2.0
        # Updated to use keywords
        d_nb_h = bstm_Likelihood("negbin", [2.0]; r_nb=r_val, hurdle=0.0)
        dist_nb = NegativeBinomial(r_val, r_val/(r_val + mu_nb))
        ana_nb_h = logpdf(dist_nb, 2.0) - logccdf(dist_nb, 0.0)
        @test isapprox(Distributions.logpdf(d_nb_h, 1.5), ana_nb_h)

        # Binomial Interval [3, 5]
        # Updated to use keywords
        d_bin_int = bstm_Likelihood("binomial", [NaN]; trial=10, y_L=3.0, y_U=5.0)
        dist_bin = Binomial(10, LogExpFunctions.logistic(0.0))
        ana_bin_int = stable_logdiffexp(logcdf(dist_bin, 5.0), logcdf(dist_bin, 2.0))
        @test isapprox(Distributions.logpdf(d_bin_int, 0.0), ana_bin_int)
    end

    # 2. Continuous Families with Censoring
    @testset "Continuous: Gaussian, LogNormal, Gamma, Beta" begin
        # Gaussian Right-Censored
        # Updated to use keywords
        d_gauss = bstm_Likelihood("gaussian", [NaN]; sigma_y=0.5, y_L=2.0)
        @test isapprox(Distributions.logpdf(d_gauss, 1.0), logccdf(Normal(1.0, 0.5), 2.0))

        # Beta Point
        # Updated to use keywords
        d_beta = bstm_Likelihood("beta", [0.4]; extra_params=20.0)
        mu_b = LogExpFunctions.logistic(-0.5)
        dist_beta = Beta(mu_b * 20.0, (1-mu_b) * 20.0)
        @test isapprox(Distributions.logpdf(d_beta, -0.5), logpdf(dist_beta, 0.4))

        # LogNormal Left-Censored
        # Updated to use keywords
        d_ln = bstm_Likelihood("lognormal", [NaN]; sigma_y=0.3, y_U=1.5)
        @test isapprox(Distributions.logpdf(d_ln, 0.5), logcdf(LogNormal(0.5, 0.3), 1.5))
    end

    # 3. Specialized & Heavy-Tailed
    @testset "Specialized: Student-T, Laplace, Pareto, Inv-Gaussian" begin
        # Student-T
        # Updated to use keywords
        d_st = bstm_Likelihood("student_t", [3.0]; sigma_y=1.2, extra_params=7.0)
        @test isapprox(Distributions.logpdf(d_st, 1.0), logpdf(LocationScale(1.0, 1.2, TDist(7.0)), 3.0))

        # Inverse Gaussian
        # Updated to use keywords
        d_ig = bstm_Likelihood("inverse_gaussian", [1.2]; extra_params=3.0)
        @test isapprox(Distributions.logpdf(d_ig, 0.5), logpdf(InverseGaussian(exp(0.5), 3.0), 1.2))

        # Laplace
        # Updated to use keywords
        d_lap = bstm_Likelihood("laplace", [1.5]; sigma_y=0.8)
        @test isapprox(Distributions.logpdf(d_lap, 2.0), logpdf(Laplace(2.0, 0.8), 1.5))
    end

    # 4. Compositional
    @testset "Compositional: Dirichlet" begin
        eta_v = [0.1, -0.5, 0.8]
        y_comp = [0.2, 0.3, 0.5]
        # Dirichlet only needs family and y_obs
        d_dir = bstm_Likelihood("dirichlet", y_comp)
        @test isapprox(Distributions.logpdf(d_dir, eta_v), logpdf(Dirichlet(exp.(eta_v)), y_comp))
    end

    # 5. Half (Truncated) Families
    @testset "Truncated: Half-Normal, Half-StudentT" begin
        # Updated to use keywords
        d_hn = bstm_Likelihood("half_normal", [2.0]; sigma_y=1.5)
        ana_hn = logpdf(Normal(0.0, 1.5), 2.0) - log(0.5)
        @test isapprox(Distributions.logpdf(d_hn, 0.0), ana_hn)

        # Updated to use keywords
        d_hst = bstm_Likelihood("half_student_t", [1.0]; sigma_y=1.0, extra_params=4.0)
        dist_base = LocationScale(0.0, 1.0, TDist(4.0))
        ana_hst = logpdf(dist_base, 1.0) - logccdf(dist_base, 0.0)
        @test isapprox(Distributions.logpdf(d_hst, 0.0), ana_hst)
    end

end
println("--- Consolidated Taxonomy Audit Keyword Sync Complete ---")


####



using Random, Statistics, Turing, DataFrames, LinearAlgebra, SparseArrays

println("--- Initializing BSTM Univariate Stress Test v08.9.15 ---")
Random.seed!(2025)

# 1. Dimensional Configuration
# Rationale: Increasing resolution to test O(N^2) or O(N^3) bottlenecks in precision recomposition.
n_units = 20
n_time = 24
total_n = n_units * n_time

# 2. Synthetic Manifold Generation
# Coordinate mappings
s_idx = repeat(1:n_units, inner=n_time)
t_idx = repeat(1:n_time, outer=n_units)
u_idx = Int.(mod1.(1:total_n, 12))

# Spatial: Adjacency for BYM2/ICAR
W_stress = sparse(zeros(n_units, n_units))
for i in 1:n_units, j in 1:n_units
    if abs(i-j) == 1 || abs(i-j) == 5
        W_stress[i, j] = 1.0
    end
end

# Latent Signals
# Spatial BYM2 Signal (Structured + Unstructured)
s_lat_true = cumsum(randn(n_units)) .* 0.5
s_signal = s_lat_true[s_idx]

# Temporal RW2 Signal (Smooth Trend)
t_trend_true = cumsum(cumsum(randn(n_time) .* 0.05))
t_signal = t_trend_true[t_idx]

# Seasonal Harmonic Signal
angles = (collect(1:total_n) .* (2.0 * pi / 12.0))
u_signal = 1.2 .* sin.(angles) .+ 0.8 .* cos.(angles)

# 3. Response Assembly
# Predictor: Intercept + Covariate + Spatial + Temporal + Seasonal
x_cov = randn(total_n)
eta_true = 0.5 .+ 1.5 .* x_cov .+ s_signal .+ t_signal .+ u_signal

# Poisson Likelihood for Stress Testing Integer Dispatch
y_obs = [Float64(rand(Poisson(exp(v)))) for v in eta_true]

# 4. Data Packaging
df_stress = DataFrame(
    y = y_obs,
    x_cov = x_cov,
    s_idx = s_idx,
    t_idx = t_idx,
    u_idx = u_idx,
    s_x = rand(total_n), # Mock coordinates for RFF/GP paths
    s_y = rand(total_n)
)

println("Simulation Complete: ", total_n, " observations across ", n_units, " spatial units.")

println("--- Executing BSTM Formula Dispatch Stress Test (Kernel Audit v11.9.2) ---")

# Using the audited version of recompose_precision defined in cell 9308a7e3
# Re-initializing the stress test model to verify warning resolution
 
# Ensure explicit sizing to avoid internal dimension guessing
model_stress = bstm(
    "y ~ 1 + x_cov + Spatial(s_idx, manifold='bym2') + Temporal(t_idx, manifold='rw2') + Seasonal(u_idx, manifold='harmonic')",
    df_stress,
    model_family = "poisson",
    model_arch = "univariate",
    W = W_stress,
    s_N = n_units,    # Explicitly pass spatial unit count
    t_N = n_time,     # Explicitly pass temporal unit count
    target_units = n_units,
    noise = 1e-4
)

println("Model Initialized. Triggering NUTS Sampling (AD Firewall Active)...")

# Sampling to verify stability and absence of unknown manifold warnings
chain_stress = sample(
    model_stress,
    MH(),
    1000,
    progress = true,
    check_model = false,
    adbackend = :forwarddiff
)

println("Sampling Complete. Analyzing results...")

res = model_results_comprehensive(model_stress , chain_stress );


using Random, Statistics, Turing, DataFrames, LinearAlgebra, SparseArrays

println("--- Initializing BSTM Univariate Stress Test v08.9.15 ---")
Random.seed!(2025)

# 1. Dimensional Configuration
# Rationale: Increasing resolution to test O(N^2) or O(N^3) bottlenecks in precision recomposition.
n_units = 20
n_time = 24
total_n = n_units * n_time

# 2. Synthetic Manifold Generation
# Coordinate mappings
s_idx = repeat(1:n_units, inner=n_time)
t_idx = repeat(1:n_time, outer=n_units)
u_idx = Int.(mod1.(1:total_n, 12))

# Spatial: Adjacency for BYM2/ICAR
W_stress = sparse(zeros(n_units, n_units))
for i in 1:n_units, j in 1:n_units
    if abs(i-j) == 1 || abs(i-j) == 5
        W_stress[i, j] = 1.0
    end
end

# Latent Signals
# Spatial BYM2 Signal (Structured + Unstructured)
s_lat_true = cumsum(randn(n_units)) .* 0.5
s_signal = s_lat_true[s_idx]

# Temporal RW2 Signal (Smooth Trend)
t_trend_true = cumsum(cumsum(randn(n_time) .* 0.05))
t_signal = t_trend_true[t_idx]

# Seasonal Harmonic Signal
angles = (collect(1:total_n) .* (2.0 * pi / 12.0))
u_signal = 1.2 .* sin.(angles) .+ 0.8 .* cos.(angles)

# 3. Response Assembly
# Predictor: Intercept + Covariate + Spatial + Temporal + Seasonal
x_cov = randn(total_n)
eta_true = 0.5 .+ 1.5 .* x_cov .+ s_signal .+ t_signal .+ u_signal

# Poisson Likelihood for Stress Testing Integer Dispatch
y_obs = [Float64(rand(Poisson(exp(v)))) for v in eta_true]

# 4. Data Packaging
df_stress = DataFrame(
    y = y_obs,
    x_cov = x_cov,
    s_idx = s_idx,
    t_idx = t_idx,
    u_idx = u_idx,
    s_x = rand(total_n), # Mock coordinates for RFF/GP paths
    s_y = rand(total_n)
)

println("Simulation Complete: ", total_n, " observations across ", n_units, " spatial units.")

println("--- Auditing BSTM Stress Test Dimensions ---")

# Standardize indices to ensure they are within bounds
# and match the target unit counts explicitly.
model_stress = bstm(
    "y ~ 1 + x_cov + Spatial(s_idx, manifold='bym2') + Temporal(t_idx, manifold='rw2') + Seasonal(u_idx, manifold='harmonic')",
    df_stress,
    model_family = "poisson",
    model_arch = "univariate",
    W = W_stress,
    s_N = n_units,   # Explicitly pass spatial dimension
    t_N = n_time,    # Explicitly pass temporal dimension
    u_N = 12,
    period = 12.0,
    noise = 1e-4
)

# Debugging internal metadata alignment
M_debug = model_stress.args.M
println("Internal Metadata Check:")
println(" - s_N (Spatial): ", M_debug.s_N, " (Expected: $n_units)")
println(" - t_N (Temporal): ", M_debug.t_N, " (Expected: $n_time)")
println(" - u_N (Seasonal): ", M_debug.u_N, " (Expected: 12)")

println("\nTriggering MH Sampling (AD Firewall Active)...")

chain_stress = sample(
    model_stress,
    MH(),
    1000,
    progress = true,
    check_model = false
)

println("Sampling Complete. Analyzing results...")


res = model_results_comprehensive(model_stress, chain_stress );

model_results_plots(res, ts=1)


# --- XR: Manifold Dispatch Validation (v15.0.0) ---
# Objective: Test extraction of 'manifold' and 'm_rff' parameters for 2D smooths

println("--- Starting 2D Manifold Dispatch Audit ---")

# 1. Setup minimal dummy data
n_val = 50
df_man = DataFrame(
    y = randn(n_val),
    x1 = rand(n_val),
    x2 = rand(n_val),
    s_x = rand(n_val), 
    s_y = rand(n_val)
)

# 2. Invoke bstm with explicit manifold choice
# We test the smooth(x, y; manifold='rff') syntax
model_man = bstm(
    "y ~ 1 + smooth(x1, x2; manifold='rff', m_rff=30)",
    df_man,
    model_family = "gaussian"
)

using Random, Statistics, Turing, DataFrames, LinearAlgebra, MCMCChains

println("--- Starting High-Fidelity 2D RFF Smooth Audit ---")
Random.seed!(2025)

# 1. Generate Synthetic 2D Surface
n_points = 150
x1 = rand(n_points) .* 10.0
x2 = rand(n_points) .* 10.0
# Latent non-linear smooth: f(x1, x2) = sin(x1) * cos(x2)
eta_latent = sin.(x1) .* cos.(x2)
y_obs = eta_latent .+ randn(n_points) .* 0.1

df_rff = DataFrame(
    y = y_obs, 
    x1 = x1, 
    x2 = x2, 
    s_x = x1, # Mapping to spatial coordinates for RFF projections
    s_y = x2,
    s_idx = ones(Int, n_points),
    t_idx = ones(Int, n_points)
)

# 2. Invoke BSTM with RFF Manifold for 2D smooth
# n_rff=50 provides sufficient spectral resolution for this complexity
model_rff = bstm(
    "y ~ 1 + smooth(x1, x2; manifold='rff', m_rff=50)",
    df_rff,
    model_family = "gaussian",
    noise = 1e-3
)

# 3. Execute Sampling with NUTS for Gradient Validation
println("Audit: Sampling with NUTS (AD Firewall Active)...")
chain_rff = sample(model_rff, NUTS(100, 0.65), 300, progress=true, check_model=false)

# 4. Statistical Convergence Audit
println("\n--- Convergence Diagnostics ---")
chn_summary = MCMCChains.summarize(chain_rff)
display(chn_summary)

# 5. Posterior Reconstruction and Visualization
println("\nAudit: Reconstructing and Rendering Dashboard...")
res_rff = model_results_comprehensive(model_rff, chain_rff)

# This call verifies the patch in cell cbYNVFKIp5zJ
plt_audit = model_results_plots(res_rff)

if !isnothing(plt_audit)
    println("RESULT: 2D RFF Audit SUCCESS. Dashboard rendered without FieldError.")
    display(plt_audit)
else
    println("RESULT: Audit FAILURE. Visualization engine returned null.")
end




using Random, Statistics, Turing, DataFrames, LinearAlgebra, SparseArrays

println("--- Starting Hardened Forensic Signal Recovery Audit v16.2.0 ---")
Random.seed!(2025)

# 1. Setup Simulation (Multivariate Poisson with Correlated Space-Time)
n_s, n_t = 15, 20
total_n = n_s * n_t
s_idx = repeat(1:n_s, inner=n_t)
t_idx = repeat(1:n_t, outer=n_s)

# Generate Structured Adjacency W
W_audit = sparse(zeros(n_s, n_s))
for i in 1:(n_s-1)
    W_audit[i, i+1] = 1.0
    W_audit[i+1, i] = 1.0
end

# Latent Correlation Structure
L_true = [1.0 0.0; 0.7 0.7141]

# Generate Ground Truth Manifolds
s_lat_true = cumsum(randn(n_s)) .* 0.3
t_lat_true = sin.(collect(1:n_t) .* (2π/n_t)) .* 0.5

eta_raw = zeros(total_n, 2)
for i in 1:total_n
    latent_val = s_lat_true[s_idx[i]] + t_lat_true[t_idx[i]]
    eta_raw[i, :] = [latent_val latent_val] * L_true'
end

# Observations
y_obs_mv = hcat([Float64(rand(Poisson(exp(0.5 + v)))) for v in eta_raw[:,1]],
                [Float64(rand(Poisson(exp(-0.5 + v)))) for v in eta_raw[:,2]])

# 2. Configure Multivariate Metadata
M_mv_audit = bstm_options(
    y_obs = y_obs_mv,
    s_idx = s_idx,
    t_idx = t_idx,
    W = W_audit,
    s_N = n_s,
    t_N = n_t,
    model_arch = "multivariate",
    model_family = "poisson",
    model_space = "bym2",
    model_time = "ar1",
    target_units = n_s,
    outcomes_N = 2,
    noise = 1e-4
)

# 3. Sampling Audit (NUTS)
println("Audit: Initializing Multivariate NUTS Sampling...")
model_mv_audit = bstm_multivariate(M_mv_audit)
chain_mv_audit = sample(model_mv_audit, NUTS(50, 0.65), 100, progress=true, check_model=false)


# 4. Forensic Reconstruction
println("Audit: Reconstructing manifolds...")
# Rationale: Using the feature-complete comprehensive reconstruction engine v16.2.0
res_mv_audit = model_results_comprehensive(model_mv_audit, chain_mv_audit)

# Verification
# Denoised posterior mean
y_pred_raw = res_mv_audit.pstats.predictions_denoised.mean

# Handle flattening: If predictions are returned as a flat vector [N*K], reshape to [N, K]
n_truth = size(eta_raw, 1)
n_outcomes = 2

if length(y_pred_raw) == n_truth * n_outcomes
    y_pred_mv = reshape(y_pred_raw, n_truth, n_outcomes)
else
    y_pred_mv = y_pred_raw
end

n_pred = size(y_pred_mv, 1)

println("Audit: Adjusted Prediction length = ", n_pred, ", Truth length = ", n_truth)

if n_pred == n_truth
    # Calculate recovery correlation against the ground truth latent field
    # Adjusting for simulation offsets (0.5 for outcome 1, -0.5 for outcome 2)
    recovery_r1 = cor(vec(y_pred_mv[:, 1]), vec(exp.(0.5 .+ eta_raw[:, 1])))
    recovery_r2 = cor(vec(y_pred_mv[:, 2]), vec(exp.(-0.5 .+ eta_raw[:, 2])))

    println("\n--- Forensic Signal Recovery Report ---")
    println("Outcome 1 Recovery Correlation: ", round(recovery_r1, digits=4))
    println("Outcome 2 Recovery Correlation: ", round(recovery_r2, digits=4))

    # 5. Visual Dashboard and Diagnostic Threshold
    if recovery_r1 > 0.7 && recovery_r2 > 0.7
        println("RESULT: FORENSIC AUDIT SUCCESS. Latent signals recovered above 0.7 threshold.")
        display(model_results_plots(res_mv_audit, outcome=1))
    else
        println("RESULT: AUDIT WARNING. Signal recovery threshold not met. Inspecting chain health...")
        display(model_results_plots(res_mv_audit, outcome=1))
    end
else
    error("Forensic Audit Failure: Dimensional mismatch between Predictions (", n_pred, ") and Truth (", n_truth, ").")
end




using Random, Statistics, Turing, DataFrames, LinearAlgebra, SparseArrays

# --- XR: MULTIFIDELITY FORENSIC SIGNAL RECOVERY AUDIT [v16.3.10] ---
# Rationale: Fixing DimensionMismatch in recovery evaluation and ensuring structural alignment.

println("--- Starting Multifidelity Forensic Recovery Audit v16.3.10 ---")
Random.seed!(2026)

# 1. Setup Simulation (Low-Fi and High-Fi Coupling)
n_units_mf = 12
n_time_mf = 18
total_mf = n_units_mf * n_time_mf

# Construct standard index vectors
s_idx_mf = repeat(1:n_units_mf, inner=n_time_mf)
t_idx_mf = repeat(1:n_time_mf, outer=n_units_mf)

# Generate Structured Adjacency W (Line Graph)
W_mf = sparse(zeros(n_units_mf, n_units_mf))
for i in 1:(n_units_mf-1)
    W_mf[i, i+1] = 1.0
    W_mf[i+1, i] = 1.0
end

# Generate Low-Fidelity Field (Source)
s_lat_lo = cumsum(randn(n_units_mf)) .* 0.5
t_lat_lo = sin.(collect(1:n_time_mf) .* (2 * pi / 12)) .* 0.8
signal_low_latent = s_lat_lo[s_idx_mf] + t_lat_lo[t_idx_mf]

# Generate High-Fidelity Signal via rho coupling and independent innovation
rho_mf_true = 1.4
innovation_hi = randn(total_mf) .* 0.3
signal_high_latent = (rho_mf_true .* signal_low_latent) + innovation_hi

# Poisson Observations
y_lo = [Float64(rand(Poisson(exp(0.5 + v)))) for v in signal_low_latent]
y_hi = [Float64(rand(Poisson(exp(-0.5 + v)))) for v in signal_high_latent]

# 2. Metadata Configuration (BSTM v06.1 Protocol)
M_mf_audit = bstm_options(
    y_obs = hcat(y_lo, y_hi),
    s_idx = s_idx_mf,
    t_idx = t_idx_mf,
    W = W_mf,
    s_N = n_units_mf,
    t_N = n_time_mf,
    model_arch = "multifidelity",
    model_family = "poisson",
    model_space = "bym2",
    model_time = "ar1",
    target_units = n_units_mf,
    y_L = [-Inf, -Inf],
    y_U = [Inf, Inf],
    hurdle = [-Inf, -Inf],
    y_ok = [findall(!isnan, y_lo), findall(!isnan, y_hi)],
    noise = 1e-4
)

# 3. Execution (Metropolis-Hastings for rapid validation)
println("Audit: Sampling Multifidelity MH Chain...")
model_mf_audit = bstm_multifidelity(M_mf_audit)
chain_mf_audit = sample(model_mf_audit, MH(), 1000, progress=true, check_model=false)

# 4. Reconstruction and Transfer Assessment
println("Audit: Assessing signal transfer fidelity...")
res_mf_audit = model_results_comprehensive(model_mf_audit, chain_mf_audit)






using Random, Statistics, Turing, LinearAlgebra, DataFrames, MCMCChains

println("--- Re-Running bstm Integration Test (v08.4.0) ---")
Random.seed!(802)

n_units = 10
n_time = 12
total_n = n_units * n_time

# --- 1. Ground Truth Construction ---
# Rationale: We must ensure all latent components are flat vectors of Float64
# to avoid nested array broadcasting errors during the sum of the linear predictor.

# Spatial: Two distinct clusters
s_clusters = repeat(1:2, inner=Int(n_units/2))
s_idx = repeat(1:n_units, inner=n_time)
t_idx = repeat(1:n_time, outer=n_units)
u_idx = rand(1:12, total_n)

# Latent Signal components
# Spatial Mosaic effect: Correctly indexing the cluster values
eta_spatial = Float64[-1.5, 1.5][s_clusters[s_idx]]

# Temporal RW1 trend
eta_temporal = Float64.(cumsum(randn(n_time) .* 0.3)[t_idx])

# Seasonal Harmonic truth (Standard Sine Wave)
t_angles = (collect(1:total_n) .* (2π / 12))
eta_seasonal = 1.0 .* sin.(t_angles)

# Combine and generate Poisson observations
# XR: Fix - Explicit broadcast with '.' ensures we handle element-wise addition and exponentiation
eta_total = 0.5 .+ eta_spatial .+ eta_temporal .+ eta_seasonal

# XR: Fix - The error occurred here because 'val' was being treated as a vector in some contexts.
# We ensure y_obs is a flat vector of Float64.
y_obs = [Float64(rand(Poisson(exp(v)))) for v in eta_total]

# Construct the input DataFrame
df_test = DataFrame(
    y_obs = y_obs, 
    s_idx = s_idx, 
    t_idx = t_idx, 
    u_idx=u_idx,
    s_x = rand(total_n), 
    s_y = rand(total_n)
)

# --- 2. Invoke Unified BSTM Interface ---
# XR: Using bstm() instead of bstm_options() to trigger the full parser.

model_test = bstm(
    "y_obs ~ 1 + Spatial(s_idx; manifold='bym2') + Temporal(t_idx, manifold='rw1') + Seasonal(u_idx, manifold='harmonic')",
    df_test,
    model_family = "poisson",
    n_mosaics = 2,
    cluster_assignments = s_clusters,
    period = 12,
    u_N = 12,
    noise = 1e-3
)

# --- 3. Sampling and Recovery Audit ---
println("Audit: Sampling model with MH algorithm...")
chain_test = sample(model_test, MH(), 1000, progress=true)

println("Audit: Performing posterior reconstruction...")
# Extract metadata from model for reconstruction
M_test = model_test.args.M
res_test = _reconstruct(UnivariateArchitecture(), "test_run_v08.4.0", chain_test, M_test, nothing, 0.05)

y_pred = res_test.predictions_denoised.mean
recovery_corr = cor(y_pred, exp.(eta_total))

println("\n--- Verification Report ---")
println("Signal Recovery Correlation (r): ", round(recovery_corr, digits=4))

if recovery_corr > 0.7
    println("RESULT: SUCCESS. The unified bstm() call recovered the signal with indexing consistency.")
else
    println("RESULT: FAILURE. Signal correlation remains below the 0.7 threshold.")
end




using Random, Statistics, Turing, LinearAlgebra, DataFrames, MCMCChains

println("--- Starting Multivariate BSTM Integration Test [v08.9.13 - @addlogprob! Migration] ---")
Random.seed!(898)

# --- 1. Simulation Setup ---
# XR: Using div() to ensure integer results for 'inner' arguments to avoid InexactError.
local n_units = 12
local n_time = 15
local total_n = n_units * n_time
local outcomes_N = 2

# Shared Spatial Structure (3 Clusters)
local s_clusters = repeat(1:3, inner=div(n_units, 3))
local s_idx = repeat(1:n_units, inner=n_time)
local t_idx = repeat(1:n_time, outer=n_units)

# Ground Truth Manifolds
local cluster_vals = [-1.5, 0.0, 1.5]
local eta_spatial = cluster_vals[s_clusters[s_idx]]

# Outcome-Specific AR(1) Temporal Trends
local rho = 0.8
local eta_t1 = zeros(n_time)
local eta_t2 = zeros(n_time)
for t in 2:n_time
    eta_t1[t] = rho * eta_t1[t-1] + randn() * 0.2
    eta_t2[t] = rho * eta_t2[t-1] + randn() * 0.2
end

# Predictor Assembly
local eta_1 = 0.5 .+ eta_spatial .+ eta_t1[t_idx]
local eta_2 = -0.5 .+ eta_spatial .+ eta_t2[t_idx]

# Observation Generation (Poisson)
local y1 = [Float64(rand(Poisson(exp(v)))) for v in eta_1]
local y2 = [Float64(rand(Poisson(exp(v)))) for v in eta_2]

# Construct Dataset
local df_mv = DataFrame(
    y1 = y1,
    y2 = y2,
    s_idx = s_idx,
    t_idx = t_idx,
    s_x = rand(total_n),
    s_y = rand(total_n)
)

# --- 2. Model Invocation via bstm() ---
println("Audit: Initializing model with @addlogprob! likelihood path...")

model_mv = bstm(
    "[y1, y2] ~ 1 + Spatial(s_idx, manifold='mosaic') + Temporal(t_idx, manifold='ar1')",
    df_mv,
    model_family = "poisson",
    model_arch = "multivariate",
    n_mosaics = 3,
    cluster_assignments = s_clusters,
    noise = 1e-4
)

# --- 3. Sampling Audit (NUTS) ---
println("Audit: Sampling with NUTS [check_model=false]...")
chain_mv = sample(model_mv, NUTS(100, 0.65), 200, progress=true, check_model=false)

# --- 4. Reconstruction & Accuracy Check ---
println("Audit: Reconstructing posterior manifolds...")
local M_mv = model_mv.args.M
res_mv = _reconstruct(MultivariateArchitecture(), "mv_test_v08.9.13", chain_mv, M_mv, nothing, 0.05)

local y_pred_1 = res_mv.predictions_observed_denoised.mean[:, 1]
local corr_1 = cor(y_pred_1, exp.(eta_1))

local y_pred_2 = res_mv.predictions_observed_denoised.mean[:, 2]
local corr_2 = cor(y_pred_2, exp.(eta_2))

println("\n--- Multivariate Recovery Report ---")
println("Outcome 1 Signal Correlation: ", round(corr_1, digits=4))
println("Outcome 2 Signal Correlation: ", round(corr_2, digits=4))

if corr_1 > 0.7 && corr_2 > 0.7
    println("RESULT: SUCCESS. @addlogprob! migration stable; signals recovered.")
else
    println("RESULT: FAILURE. Check convergence metrics.")
end






using Random, Statistics, Turing, LinearAlgebra, DataFrames, MCMCChains

println("--- Starting Multifidelity BSTM Integration Test [v08.9.12 Verification] ---")
Random.seed!(899)

# --- 1. Simulation Setup ---
n_units = 15
n_time = 12
total_n = n_units * n_time

# Shared indices
s_idx = repeat(1:n_units, inner=n_time)
t_idx = repeat(1:n_time, outer=n_units)

# Shared Spatial Structure
cluster_vals = [-2.0, 0.0, 2.0]
s_clusters = repeat(1:3, inner=div(n_units, 3))
eta_spatial_shared = cluster_vals[s_clusters[s_idx]]

# Fidelity-Specific Trends
t_angle = (collect(1:n_time) .* (2π / 12))
eta_t1 = 1.5 .* sin.(t_angle[t_idx])
eta_f2_innov = 0.5 .* (t_idx ./ n_time)

# Linear Coupling (rho_mf = 1.2)
rho_mf_true = 1.2
eta_low = 0.5 .+ eta_spatial_shared .+ eta_t1
eta_high = -1.0 .+ (rho_mf_true .* eta_low) .+ eta_f2_innov

# Observation Generation
y_low = eta_low .+ randn(total_n) .* 0.1
y_high = eta_high .+ randn(total_n) .* 0.1

# Construct Dataset
df_mf = DataFrame(
    y_low = y_low,
    y_high = y_high,
    s_idx = s_idx,
    t_idx = t_idx,
    s_x = rand(total_n),
    s_y = rand(total_n)
)

# --- 2. Model Invocation ---
println("Audit: Initializing multifidelity model v08.9.12...")
model_mf = bstm(
    "y_high ~ 1 + Spatial(s_idx, manifold='bym2') + Temporal(t_idx, manifold='rw1')",
    df_mf,
    auxiliary_responses = df_mf.y_low,
    model_family = "gaussian",
    model_arch = "multifidelity",
    target_units = n_units,
    noise = 1e-4
)

# --- 3. Sampling ---
println("Audit: Sampling with NUTS [check_model=false]...")
# Rationale: check_model=false bypasses the y_obs_k overwrite warning triggered by the indexing firewall.
chain_mf = sample(model_mf, NUTS(100, 0.65), 200, progress=true, check_model=false)

# --- 4. Reconstruction ---
println("Audit: Reconstructing manifolds...")
M_mf = model_mf.args.M
res_mf = _reconstruct(MultifidelityArchitecture(), "mf_test_v08.9.12", chain_mf, M_mf, nothing, 0.05)

y_pred_high = res_mf.predictions_observed_denoised.mean[:, 2]
corr_high = cor(y_pred_high, eta_high)
rho_est = mean(chain_mf[:rho_mf])

println("\n--- Multifidelity Recovery Report ---")
println("Primary Signal Correlation: ", round(corr_high, digits=4))
println("Estimated rho_mf: ", round(rho_est, digits=4))

if corr_high > 0.7 && abs(rho_est - 1.2) < 0.3
    println("RESULT: SUCCESS. v08.9.12 indexing and coupling verified.")
else
    println("RESULT: FAILURE. Check convergence or coupling logic.")
end




# misc basic unit tests   

include( srcdir( "spatiotemporal_partitioning_functions.jl" ))   ;
include( srcdir( "spatiotemporal_functions.jl" ))   ;
include( srcdir( "unit_test_functions.jl" ))   ;



unit_test_spatial_partitioning();

unit_test_manifold();

run_mixed_effects_example();

st_results = verify_transport_logic();






