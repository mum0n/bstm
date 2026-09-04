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

    # 5. Test _process_telemetry_data with :time and :timestamp columns
    df_telem_time = DataFrame(
        tagid = [1, 1, 2, 2],
        s_idx = [1, 2, 2, 1],
        time = [2020.0, 2021.0, 2020.0, 2022.0],
        tag = [0, 1, 0, 1],
        individual_covariate = [0.5, 0.5, -0.2, -0.2]
    )
    mat_time = bstm._process_telemetry_data(df_telem_time)
    @test size(mat_time) == (2, 4)
    @test mat_time[1, 1] == 1.0 && mat_time[1, 2] == 2.0 && mat_time[1, 3] == 1.0

    df_telem_timestamp = DataFrame(
        tagid = [1, 1],
        s_idx = [1, 3],
        timestamp = [10.0, 12.0],
        tag = [0, 1]
    )
    mat_timestamp = bstm._process_telemetry_data(df_telem_timestamp)
    @test size(mat_timestamp) == (1, 4)
    @test mat_timestamp[1, 3] == 2.0

    # 6. Test movement component formula compilation and hyperprior resolution
    df_mov = DataFrame(
        y = [1.0, 2.0],
        s_idx = [1, 2],
        t_idx = [1, 1],
        s_x = [0.0, 1.0],
        s_y = [0.0, 1.0]
    )
    W_mov = [0 1; 1 0]
    m_mov_test = @bstm(
        likelihood(y, family=gaussian) ~ intercept() +
            movement(s_idx, t_idx,
                velocity  = truncated(Normal(0.0, 1.0), lower=0.0),
                diffusion = truncated(Normal(0.0, 1.0), lower=0.0),
                sigma     = truncated(Normal(0.0, 0.5), lower=0.0),
                beta_het  = Normal(0.0, 1.0)),
        df_mov,
        W = W_mov,
        mark_recapture_data = df_telem_time,
        verbose = false
    )
    @test m_mov_test isa DynamicPPL.Model
    spec_mov = m_mov_test.args.M.components[1]
    @test spec_mov.component_obj isa bstm.Movement
    @test !isnothing(spec_mov.component_obj.beta_het)

    # 7. Test haversine_distance calculation
    d_m = bstm.haversine_distance(-63.57, 44.65, -60.18, 46.14)
    @test isapprox(d_m / 1000.0, 311.0; atol=5.0)
    @test bstm.haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0
    @test isnan(bstm.haversine_distance(NaN, 0.0, 1.0, 1.0))

    # 8. Test tag_to_study_id lookup mappings
    @test bstm.tag_to_study_id(100) == 1
    @test bstm.tag_to_study_id("G1234") == 6
    @test bstm.tag_to_study_id("t1605") == 49
    @test bstm.tag_to_study_id("s99105") == 57
    @test bstm.tag_to_study_id(999999) === nothing

    # 9. Test filter_dead_tags terminal stationary period detection
    df_dead_test = DataFrame(
        tagid = fill("tag_A", 4),
        lon = [-63.0, -62.5, -62.0, -62.0001],
        lat = [44.0, 44.5, 45.0, 45.0001],
        timestamp = [Date(2021, 1, 1), Date(2021, 2, 1), Date(2021, 3, 1), Date(2021, 5, 1)]
    )
    df_filtered = bstm.filter_dead_tags(df_dead_test; time_threshold_days=30.0, dist_threshold_meters=50.0)
    @test hasproperty(df_filtered, :is_dead)
    @test df_filtered.is_dead[1] == false
    @test df_filtered.is_dead[2] == false
    @test df_filtered.is_dead[3] == true
    @test df_filtered.is_dead[4] == true

    # 10. Test summarize_tag_activity calculation
    df_act = DataFrame(
        tagid = ["tag_1", "tag_1", "tag_2"],
        lon = [-63.0, -63.1, -60.0],
        lat = [44.0, 44.1, 45.0],
        timestamp = [Date(2021, 1, 1), Date(2021, 1, 11), Date(2021, 1, 1)],
        cw = [100.0, 105.0, 90.0],
        cc = ["2", "3", "1"]
    )
    act_summary = bstm.summarize_tag_activity(df_act)
    @test nrow(act_summary) == 2
    row1 = filter(r -> r.tagid == "tag_1", act_summary)[1, :]
    @test row1.duration_days == 10.0
    @test row1.cw_change == 5.0
    @test row1.n_points == 2
    @test row1.total_dist_m > 0.0

    # 11. Test validate_telemetry
    df_valid = DataFrame(
        tagid = [1, 1],
        lon = [-63.0, -62.5],
        lat = [44.0, 44.5],
        time = [2021.0, 2022.0],
        tag = [0, 1]
    )
    @test bstm.validate_telemetry(df_valid) === nothing
    @test_throws ArgumentError bstm.validate_telemetry(DataFrame(tagid=[1], lon=[0.0]))

    # 12. Test map_telemetry_to_units & time_steps_between
    df_mapped = bstm.map_telemetry_to_units(df_valid, au_simple)
    @test hasproperty(df_mapped, :s_idx)
    @test all(1 .<= df_mapped.s_idx .<= 2)
    @test bstm.time_steps_between(2020.0, 2022.2) == 2
    @test bstm.time_steps_between(2020.0, 2020.1) == 1

    # 13. Test reconstruct_posterior_kernel with mock chain
    mock_chain = (
        velocity = [0.8, 1.0],
        diffusion = [0.2, 0.3]
    )
    W_k = [0 1; 1 0]
    Gamma_rec = bstm.reconstruct_posterior_kernel(mock_chain, W_k)
    @test size(Gamma_rec) == (2, 2)
    @test isapprox(sum(Gamma_rec[1, :]), 1.0; atol=1e-5)
    @test isapprox(sum(Gamma_rec[2, :]), 1.0; atol=1e-5)

    # 14. Test reshard_hsi_field
    hsi_source = [0.2, 0.8, 0.5, 0.9]
    hsi_resharded = bstm.reshard_hsi_field(hsi_source, au_simple)
    @test length(hsi_resharded) == length(au_simple.centroids)
    @test all(0.0 .<= hsi_resharded .<= 1.0)

    # 15. Test construct_stochastic_transition_kernel & vector overloads
    S_t = 5
    W_dense = zeros(Float64, S_t, S_t)
    for i in 1:(S_t - 1)
        W_dense[i, i + 1] = 1.0
        W_dense[i + 1, i] = 1.0
    end
    W_sp_t = sparse(W_dense)
    hsi_t = [0.1, 0.3, 0.7, 0.9, 0.5]

    # Scalar parameters -> Matrix{Float64}
    P_sc = bstm.construct_stochastic_transition_kernel(
        W_sp_t, hsi_t; gamma=1.0, residence=0.2, advection=0.5
    )
    @test P_sc isa Matrix{Float64}
    @test size(P_sc) == (S_t, S_t)
    for i in 1:S_t
        @test isapprox(sum(P_sc[i, :]), 1.0; atol=1e-12)
    end

    # Group vector mode (G=3) -> Vector{Matrix{Float64}}
    g_vec_t = [0.5, 1.0, 2.0]
    rho_vec_t = [0.1, 0.2, 0.3]
    adv_vec_t = [0.3, 0.5, 0.8]
    P_grp = bstm.construct_stochastic_transition_kernel(
        W_sp_t, hsi_t; gamma=g_vec_t, residence=rho_vec_t, advection=adv_vec_t
    )
    @test P_grp isa Vector{Matrix{Float64}}
    @test length(P_grp) == 3
    for g in 1:3
        @test size(P_grp[g]) == (S_t, S_t)
        for i in 1:S_t
            @test isapprox(sum(P_grp[g][i, :]), 1.0; atol=1e-12)
        end
    end

    # Mixed scalar and vector (scalar broadcasts)
    P_mix = bstm.construct_stochastic_transition_kernel(
        W_sp_t, hsi_t; gamma=1.0, residence=rho_vec_t, advection=0.5
    )
    @test P_mix isa Vector{Matrix{Float64}}
    @test length(P_mix) == 3

    # Spatial vector mode (spatial=true) -> Matrix{Float64}
    rho_sp = [0.1, 0.2, 0.3, 0.2, 0.1]
    adv_sp = [0.4, 0.5, 0.6, 0.5, 0.4]
    P_sp = bstm.construct_stochastic_transition_kernel(
        W_sp_t, hsi_t; gamma=1.0, residence=rho_sp, advection=adv_sp, spatial=true
    )
    @test P_sp isa Matrix{Float64}
    @test size(P_sp) == (S_t, S_t)
    for i in 1:S_t
        @test isapprox(sum(P_sp[i, :]), 1.0; atol=1e-12)
    end

    # predict_path & predict_corridor with Vector of kernels
    path_g1 = bstm.predict_path(P_grp, 1, 5, 4; group=1)
    path_g3 = bstm.predict_path(P_grp, 1, 5, 4; group=3)
    @test path_g1 isa Vector{Int}
    @test path_g3 isa Vector{Int}
    @test length(path_g1) == 5
    @test first(path_g1) == 1 && last(path_g1) == 5

    corr_g1 = bstm.predict_corridor(P_grp, 1, 5, 4; group=1)
    @test size(corr_g1) == (S_t, 5)
    for t in 1:5
        @test isapprox(sum(corr_g1[:, t]), 1.0; atol=1e-12)
    end
end


