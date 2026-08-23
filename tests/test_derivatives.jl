# ==============================================================================
# BSTM Test Suite: Analytical Surface Derivatives & Topographic Metrics
# ==============================================================================

if !@isdefined(bstm_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Surface Derivatives & Differential Geometry Tests" begin

    @testset "Topographic Metrics Differential Geometry Formulation" begin
        # Test flat horizontal plane: z(x, y) = 10
        zx = [0.0]
        zy = [0.0]
        zxx = [0.0]
        zyy = [0.0]
        zxy = [0.0]
        topo_flat = bstm.compute_topographic_metrics(zx, zy, zxx, zyy, zxy)
        @test topo_flat.slope_gradient[1] ≈ 0.0 atol=1e-6
        @test topo_flat.slope_degrees[1] ≈ 0.0 atol=1e-6
        @test topo_flat.aspect_degrees[1] == -1.0 # Flat surface aspect
        @test topo_flat.laplacian[1] ≈ 0.0 atol=1e-6

        # Test tilted plane: z(x, y) = 3x + 4y (gradient = (3, 4), slope = 5.0)
        zx_tilt = [3.0]
        zy_tilt = [4.0]
        topo_tilt = bstm.compute_topographic_metrics(zx_tilt, zy_tilt, zxx, zyy, zxy)
        @test topo_tilt.slope_gradient[1] ≈ 5.0 atol=1e-6
        @test topo_tilt.slope_degrees[1] ≈ atan(5.0) * (180.0 / pi) atol=1e-4
        @test 0.0 <= topo_tilt.aspect_degrees[1] <= 360.0

        # Test parabolic bowl: z(x, y) = x^2 + y^2 at (1, 1)
        # zx = 2, zy = 2, zxx = 2, zyy = 2, zxy = 0 -> laplacian = 4.0
        topo_bowl = bstm.compute_topographic_metrics([2.0], [2.0], [2.0], [2.0], [0.0])
        @test topo_bowl.slope_gradient[1] ≈ sqrt(8.0) atol=1e-6
        @test topo_bowl.laplacian[1] ≈ 4.0 atol=1e-6
        @test topo_bowl.profile_curvature[1] < 0.0 # Curving upward in direction of slope
    end

    @testset "RFF Analytical Surface Derivatives & Bessel BPI" begin
        Random.seed!(123)
        N = 60
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        # True bathymetric surface
        z_true = [100.0 + 20.0 * sin(x / 15.0) + 15.0 * cos(y / 15.0) + randn() * 0.5 for (x, y) in zip(x_pts, y_pts)]

        df_bathy = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true)

        m_rff = @bstm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=rff, n_features=25),
            df_bathy, verbose=false
        )
        chn_rff = sample(m_rff, NUTS(20, 0.65), 40; progress=false)

        # Query points on a test grid
        query_coords = DataFrame(
            s_x = [30.0, 50.0, 70.0],
            s_y = [30.0, 50.0, 70.0]
        )

        deriv_res = bstm_surface_derivatives(
            m_rff, chn_rff, query_coords;
            metrics=[:slope, :curvature, :bpi],
            radii=[5.0, 15.0],
            return_samples=true
        )

        @test haskey(deriv_res, :summary)
        @test haskey(deriv_res, :metrics)
        @test haskey(deriv_res, :samples)

        sum_df = deriv_res.summary
        @test nrow(sum_df) == 3
        @test hasproperty(sum_df, :z_mean)
        @test hasproperty(sum_df, :slope_mean)
        @test hasproperty(sum_df, :slope_deg_mean)
        @test hasproperty(sum_df, :aspect_deg_mean)
        @test hasproperty(sum_df, :laplacian_mean)
        @test hasproperty(sum_df, :profile_curv_mean)
        @test hasproperty(sum_df, :planform_curv_mean)
        @test hasproperty(sum_df, :bpi_r5_0_mean)
        @test hasproperty(sum_df, :bpi_r15_0_mean)

        # Sanity checks on physical realism
        @test all(sum_df.slope_mean .>= 0.0)
        @test all(0.0 .<= sum_df.slope_deg_mean .<= 90.0)
        @test all(sum_df.z_mean .> 40.0)

        # Test sample matrices shape
        @test size(deriv_res.samples.elevation) == (3, 40)
        @test size(deriv_res.samples.slope_gradient) == (3, 40)
        @test size(deriv_res.samples.laplacian) == (3, 40)
        @test size(deriv_res.samples.bpi[5.0]) == (3, 40)
    end

    @testset "SpectralGP Exact Fourier Domain Derivatives" begin
        Random.seed!(456)
        N = 50
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        z_true = [150.0 - 0.5 * x + 0.3 * y + randn() * 0.5 for (x, y) in zip(x_pts, y_pts)]
        df_gp = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true)

        m_sgp = @bstm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=spectral_gp, resolution=16),
            df_gp, verbose=false
        )
        chn_sgp = sample(m_sgp, NUTS(20, 0.65), 40; progress=false)

        query_coords = Matrix{Float64}([40.0 40.0; 60.0 60.0])
        sgp_derivs = bstm_surface_derivatives(
            m_sgp, chn_sgp, query_coords;
            radii=[10.0],
            return_samples=true
        )

        @test nrow(sgp_derivs.summary) == 2
        @test all(sgp_derivs.summary.slope_mean .>= 0.0)
        @test haskey(sgp_derivs.metrics, :slope)
        @test haskey(sgp_derivs.metrics, :laplacian)
        @test haskey(sgp_derivs.metrics, :bpi)
        @test size(sgp_derivs.samples.elevation) == (2, 40)
    end

    @testset "WaveletGP Multi-Scale Wavelet Derivatives & BPI" begin
        Random.seed!(789)
        N = 50
        x_pts = rand(Uniform(10.0, 90.0), N)
        y_pts = rand(Uniform(10.0, 90.0), N)
        z_true = [120.0 + 10.0 * sin(x / 10.0) + randn() * 0.5 for (x, y) in zip(x_pts, y_pts)]
        df_wgp = DataFrame(s_x = x_pts, s_y = y_pts, depth = z_true)

        m_wgp = @bstm(
            likelihood(depth) ~ intercept() + random(s_x, s_y, model=waveletgp, resolution=16, wavelet=:db4),
            df_wgp, verbose=false
        )
        chn_wgp = sample(m_wgp, NUTS(20, 0.65), 40; progress=false)

        query_coords = Matrix{Float64}([35.0 35.0; 65.0 65.0])
        wgp_derivs = bstm_surface_derivatives(
            m_wgp, chn_wgp, query_coords;
            radii=[12.0],
            return_samples=true
        )

        @test nrow(wgp_derivs.summary) == 2
        @test all(wgp_derivs.summary.slope_mean .>= 0.0)
        @test haskey(wgp_derivs.metrics, :slope)
        @test haskey(wgp_derivs.metrics, :laplacian)
        @test haskey(wgp_derivs.metrics, :bpi)
        @test size(wgp_derivs.samples.elevation) == (2, 40)
    end

    @testset "SPDE Triangulation & Graph Laplacian Surface Derivatives" begin
        Random.seed!(321)
        N_units = 25
        # Create 2D grid centroids
        grid_1d = range(10.0, stop=90.0, length=5)
        cx = repeat(collect(grid_1d), inner=5)
        cy = repeat(collect(grid_1d), outer=5)
        
        # Build spatial adjacency matrix W
        W_spde = spzeros(Int, N_units, N_units)
        for i in 1:N_units
            for j in 1:N_units
                if i != j && sqrt((cx[i] - cx[j])^2 + (cy[i] - cy[j])^2) <= 25.0
                    W_spde[i, j] = 1
                end
            end
        end

        N_obs = 60
        s_indices = rand(1:N_units, N_obs)
        z_obs = [100.0 + 0.5 * cx[s] - 0.2 * cy[s] + randn() * 0.5 for s in s_indices]
        df_spde = DataFrame(s_idx = s_indices, depth = z_obs)

        m_spde = @bstm(
            likelihood(depth) ~ intercept() + random(s_idx, model=spde),
            df_spde, W=W_spde, verbose=false
        )
        chn_spde = sample(m_spde, NUTS(20, 0.65), 40; progress=false)

        unit_coords = [cx cy]
        spde_derivs = bstm_surface_derivatives(
            m_spde, chn_spde, unit_coords;
            radii=[20.0],
            return_samples=true
        )

        @test nrow(spde_derivs.summary) == N_units
        @test all(spde_derivs.summary.slope_mean .>= 0.0)
        @test haskey(spde_derivs.metrics, :slope)
        @test haskey(spde_derivs.metrics, :laplacian)
        @test haskey(spde_derivs.metrics, :bpi)
        @test size(spde_derivs.samples.elevation) == (N_units, 40)
    end

end
