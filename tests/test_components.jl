# ==============================================================================
# BSTM Test Suite: ComponentModel Interface, NNGP, MCAR, and Movement Engine
# ==============================================================================

if !@isdefined(bstm_Likelihood)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "ComponentModel Interface Unit Tests" begin
    # Test IID Component
    @testset "IID Component" begin
        N_obs, N_levels = 100, 10
        m_iid = bstm.IID(Distributions.Exponential(1.0), :noncentered)
        
        mock_M_ds = Dict(:data => DataFrame(group_var=repeat(1:N_levels,
            inner=N_obs ÷ N_levels)[1:N_obs]))
        mock_mod_data = Dict(:variables => :group_var)
        
        mock_M_pc = (data=mock_M_ds[:data],)
        res_pc = bstm.get_precomputes(m_iid, mock_M_pc, mock_mod_data)
        @test hasproperty(res_pc, :n_latent) || res_pc == NamedTuple()

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
            inner=N_obs ÷ N_areas)[1:N_obs]), :W => W_leroux)
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
            inner=N_obs ÷ N_areas)[1:N_obs]), ), model_arch="univariate")
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

@testset "Movement & ADR Telemetry Engine" begin
    # 1. Test simulation bundle generation
    sim = generate_ADR_simulation_bundle(100.0, 9, 3, 10; area_method=:grid,
        rng=MersenneTwister(123))
    @test nrow(sim.data) == sim.n_spatial * 3
    @test nrow(sim.telemetry_data) == 20
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
