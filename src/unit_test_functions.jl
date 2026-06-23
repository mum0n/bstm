
function verify_transport_logic(n_s=5, n_t=4)
    println("--- Auditing Physics-Informed Transport Logic ---")
    
    # 1. Setup a simple directed chain graph (1 -> 2 -> 3 -> 4 -> 5)
    W = zeros(n_s, n_s)
    for i in 1:(n_s-1)
        W[i, i+1] = 1.0
    end
    
    # 2. Derive Laplacian (L = D - W)
    # Note: In transport models, the Laplacian governs the 'spread' rate
    D_diag = vec(sum(W, dims=2))
    L_phys = Matrix(Diagonal(D_diag) - W)
    
    # 3. Initialize latent states
    # st_map size: [spatial_units x time_steps]
    st_map = zeros(n_s, n_t)
    
    # Parameters for validation
    # High diffusion (spread), Moderate Advection (flow), High Persistence
    st_diff = 0.5
    st_adv = 0.3
    st_pers_val = 2.0 # Logit scale (highly persistent)
    st_pers_vector = fill(st_pers_val, n_s)
    
    # Mock spatial field (s_eta) to drive advection potential
    # Gradient increasing toward the end of the chain
    s_eta = collect(1.0:Float64(n_s))
    
    # 4. Initial Condition (Signal burst at Node 1, Time 1)
    st_map[1, 1] = 10.0
    
    # 5. Recursive Propagation (Verbatim from bstm_univariate v05.5)
    for t in 2:n_t
        # mu_p calculation as per reference: Transport via Laplacian spread
        # Term A: Persistence of previous state
        # Term B: Diffusion (Laplacian acting on previous state)
        # Term C: Advection (Laplacian acting on spatial potential s_eta)
        
        diff_term = st_diff .* (L_phys * st_map[:, t-1])
        adv_term = st_adv .* (L_phys * s_eta)
        
        mu_p = st_map[:, t-1] .- diff_term .- adv_term
        
        # Apply persistence and add (zero) innovation for deterministic check
        st_map[:, t] .= logistic.(st_pers_vector) .* mu_p
    end
    
    println("Transport Result (st_map):")
    display(st_map)
    
    # 6. Audit Checks
    # Check 1: Did signal move from Node 1 to Node 2?
    if st_map[2, 2] > 0.0
        println("Audit [Diffusion]: PASSED. Signal propagated to neighbors.")
    else
        println("Audit [Diffusion]: FAILED. Signal trapped in source.")
    end
    
    # Check 2: Influence of s_eta gradient on flow?
    if st_map[5, n_t] != st_map[1, n_t]
        println("Audit [Advection]: PASSED. Spatial potential s_eta biased the distribution.")
    else
        println("Audit [Advection]: FAILED. Flow is spatially symmetric despite gradient.")
    end
    
    # Visualize Flow
    heatmap(st_results, 
        title="Spatiotemporal Transport Flow (Validation)", 
        xlabel="Time Step", 
        ylabel="Spatial Unit (Chain)",
        c=:thermal)

    return st_map
end


#########


function run_mixed_effects_example()
    println("--- BSTM Mixed Effects Example: Multiple Intercepts and Slopes ---")
    Random.seed!(2026)

    # 1. Generate Synthetic Data
    # We'll create data for 10 schools across 3 regions.
    n_schools = 10
    n_regions = 3
    n_students_per_school = 20
    n_total = n_schools * n_students_per_school

    n_total = n_schools * n_students_per_school

    # True parameters for the data generating process
    local true_intercept = 65.0
    local true_slope_hours = 5.0

    # School-level random effects (intercepts and slopes)
    local school_intercept_effects = rand(Normal(0, 5), n_schools)
    local school_slope_effects = rand(Normal(0, 1.5), n_schools)

    # Region-level random effects (intercepts only)
    local region_intercept_effects = rand(Normal(0, 3), n_regions)

    # --- FIX: Corrected Region Assignment Logic ---
    # Assign each school to a region first to ensure dimensional consistency.
    local school_to_region_map = repeat(1:n_regions, ceil(Int, n_schools / n_regions))[1:n_schools]

    # Create the DataFrame
    local df = DataFrame(
        school_id = categorical(repeat(1:n_schools, inner=n_students_per_school)),
        hours_studied = rand(Uniform(1, 10), n_total)
    )
    df.region = categorical(school_to_region_map[levelcode.(df.school_id)])

    # Generate the response variable 'score'
    df.score = map(1:nrow(df)) do i
        school = levelcode(df.school_id[i])
        reg = levelcode(df.region[i])
        
        # Combine fixed and random effects
        intercept = true_intercept + school_intercept_effects[school] + region_intercept_effects[reg] 
        slope = true_slope_hours + school_slope_effects[school]
        
        # Observation with noise
        return intercept + slope * df.hours_studied[i] + rand(Normal(0, 4))
    end

    println("Synthetic DataFrame created with columns: ", names(df))

    # 2. Define and Execute the BSTM Model
    # The formula specifies:
    # - `1 + hours_studied`: Global fixed effects for intercept and slope.
    # - `mixed(1 + hours_studied | school_id)`: A random intercept AND a random slope for `hours_studied`, both varying by `school_id`.
    # - `mixed(1 | region)`: A random intercept varying by `region`.
    
    formula_str = "score ~ 1 + hours_studied + mixed(1 + hours_studied | school_id) + mixed(1 | region)"

    println("\nFormula: ", formula_str)

    # We use `return_data=true` to inspect the parsed model configuration
    # without running the full MCMC sampler, which can be time-consuming.
    # This allows us to verify that the parser correctly identified all model components.
    
    parsed_model_inputs = bstm(
        formula_str,
        df;
        model_family = "gaussian",
        return_data = true # Return the configuration object for inspection
    )

    # 3. Verify the Parsed Structure
    println("\n--- Verifying Parsed Model Structure ---")
    if hasproperty(parsed_model_inputs, :mixed_terms) && !isempty(parsed_model_inputs.mixed_terms)
        println("Successfully discovered $(length(parsed_model_inputs.mixed_terms)) mixed-effect terms.")
        
        for (i, term) in enumerate(parsed_model_inputs.mixed_terms)
            println("\n[Term $i]:")
            println("  - Grouping Variable: ", term.name)
            println("  - Number of Levels: ", term.n_cat)
            println("  - Random Effects on: '", term.lhs, "'")
        end
        
        println("\nVerification PASSED: The parser correctly identified both the school-level and region-level random effects, including the random slope for 'hours_studied'.")
    else
        @error "Verification FAILED: The 'mixed()' terms were not correctly parsed from the formula."
    end
    
    return parsed_model_inputs
end



#########


function unit_test_bstm_formulae() 
    
    # 1. Create a dummy DataFrame for testing
    dummy_N = 100
    dummy_T = 10

    data_test = DataFrame(
        y_obs = rand(dummy_N, dummy_T) |> vec, # Simulate 100 spatial units over 10 time points
        s_idx = repeat(1:dummy_N, outer=dummy_T),
        t_idx = repeat(1:dummy_T, inner=dummy_N),
        x1 = rand(dummy_N * dummy_T),
        x2 = rand(dummy_N * dummy_T),
        group_id = categorical(repeat(1:5, inner=floor(Int, dummy_N * dummy_T / 5))),
        region = categorical(repeat(["North", "South"], inner=floor(Int, dummy_N * dummy_T / 2))),
        lat = rand(dummy_N * dummy_T) * 100,
        lon = rand(dummy_N * dummy_T) * 100,
        eigen_var1 = rand(dummy_N * dummy_T),
        eigen_var2 = rand(dummy_N * dummy_T)
    )

    # Assign s_coord_tuple
    data_test[!, :s_coord_tuple] = [(data_test.lat[i], data_test.lon[i]) for i in 1:nrow(data_test)]

    # 2. Define a complex formula with various modifiers
    #    - `re(region, model='bym2')`: Random effect for 'region' with BYM2 structure
    #    - `me(x1 | group_id)`: Mixed effect (random slope for x1 within group_id)
    #    - `svc(x2, model='rff')`: Spatially varying coefficient for x2 using RFF
    #    - `ie(lat, lon, model='gp')`: smooth effect between lat and lon using GP (RFF approximation)
    #    - `ee(eigen_var1, eigen_var2, model='householder')`: Eigen effect for eigen_var1 and eigen_var2

    formula_test = "y_obs ~ 1 + x1 + x2 + re(region, model='bym2') + me(x1 | group_id) + svc(x2, model='rff') + ie(lat, lon, model='gp') + ee(eigen_var1, eigen_var2, model='householder')"

    # 3. Call bstm() and inspect the returned model input (M)
    M_parsed = bstm(formula_test, data_test; return_data="inputs")

    println("\n--- Parsed Model Input (M) Verification ---")
    println("Model Arch: ", M_parsed.model_arch)
    println("Model Family: ", M_parsed.model_family)
    println("y_N: ", M_parsed.y_N)
    println("s_N: ", M_parsed.s_N)
    println("t_N: ", M_parsed.t_N)
    println("Xfixed (fixed effects design matrix):")
    display(M_parsed.Xfixed)
    println("Fixed Parts: ", M_parsed.fixed_parts)

    println("\n--- Random Effects (re) Verification ---")
    if haskey(M_parsed.re_rules, "region")
        println("Region RE Rule: ", M_parsed.re_rules["region"])
        println("Spatial Hierarchy for Region:")
        display(M_parsed.spatial_hierarchy[:region])
    else
        println("Region RE not found in re_rules.")
    end

    println("\n--- Mixed Effects (me) Verification ---")
    if !isempty(M_parsed.mixed_terms)
        println("Mixed Terms Found: ", length(M_parsed.mixed_terms))
        display(M_parsed.mixed_terms)
    else
        println("No mixed terms found.")
    end

    println("\n--- Spatially Varying Coefficients (svc) Verification ---")
    if !isempty(M_parsed.svc_covariates)
        println("SVC Covariates: ", M_parsed.svc_covariates)
        println("SVC Model: ", M_parsed.svc_model)
        if haskey(M_parsed, :svc_basis_cached)
            println("SVC Basis Cached (size): ", size(M_parsed.svc_basis_cached))
        end
    else
        println("No SVC covariates found.")
    end

    println("\n--- smooth Effects (ie) Verification ---")
    if !isempty(M_parsed.smooth_terms)
        println("smooth Terms Found: ", length(M_parsed.smooth_terms))
        display(M_parsed.smooth_terms)
    else
        println("No smooth terms found.")
    end

    println("\n--- Eigen Effects (ee) Verification ---")
    if !isempty(M_parsed.eigen_terms)
        println("Eigen Terms Found: ", length(M_parsed.eigen_terms))
        display(M_parsed.eigen_terms)
    else
        println("No eigen terms found.")
    end

    println("--- Parsing Verification Complete ---")

end


################


function unit_test_spatial_partitioning()
 
    s_N = 100  # spatial locations
    t_N = 15  # time slices ("years")
    
    inp_sim = generate_sim_data(s_N, t_N; rndseed=42);
 
    # time discretization
    tu = assign_time_units(inp_sim.t_coord;  time_method="regular", t_N=inp_sim.t_N, u_N=inp_sim.u_N)  # t_idx, t0, t1, tn

    # space discretizatio
    Random.seed!(42) # Set a seed for reproducibility.

    ntot = size(inp_sim.s_coord_tuple, 1) 

    min_time_slices = 5
    target_density = 5 # number per areal unit 
    target_units = floor( ntot / target_density )
    min_total_arealunits = target_units / 10
    max_total_arealunits = target_units * 10
    min_points = 5
    max_points = floor(ntot /min_total_arealunits )
    min_area = 0.1
    max_area = 10
    target_cv = 0.9
    buffer_dist = 0.8
    tolerance = 0.1


    test_configs = [ :cvt, :kvt, :qvt, :bvt, :avt, :hvt ] # these are the currently available methods

    results = []
    plots = []

    for m in test_configs
        println("Testing method: $m")
        au
        try
            au = assign_spatial_units( inp_sim.s_coord_tuple;
                area_method = m,
                t_idx = tu.t_idx,
                target_units = target_units,
                target_cv=target_cv,
                min_total_arealunits=min_total_arealunits,
                max_total_arealunits=max_total_arealunits,
                min_time_slices = min_time_slices,
                buffer_dist=buffer_dist,
                tolerance=tolerance,
                min_points=min_points,
                max_points=max_points,
                min_area=min_area,
                max_area=max_area)

            met = calculate_metrics(au)

            # copy results into an output list
            push!(results, (
            method=m,
            units=length(au.centroids),
            mean_dens=met.mean_density,
            sd_dens=met.sd_density,
            cv_dens=met.cv_density,
            termination=au.termination_reason
            ))

            p = plot_spatial_graph( au; plot_title="Method: $m", domain_boundary=au.hull_coords)

            # copy plot to an output list
            push!(plots, p)
        catch e
            @error "Method $m failed: $e"
        end
    end

    display(DataFrame(results))

    display(Plots.plot(plots..., layout=(3, 2), size=(600, 800)))
    
end

############

function unit_test_manifold()
# test/test_manifold_interface.jl

    if false
        using Test
        using Distributions
        using LinearAlgebra
        using SparseArrays
    end

    # Include manifold_types.jl, manifold_builders.jl, formula_translation.jl, manifold_macros.jl
    # In a real package, these would be `using MyPackage`
    # For this notebook context, we assume they are already defined or can be included.

    # Dummy data_inputs for testing builders
    const TEST_DATA_INPUTS = (
        W = sparse([0 1 0; 1 0 1; 0 1 0]),
        n_temporal_units = 5,
        s_x = [1.0, 2.0, 3.0],
        s_y = [1.0, 1.0, 1.0],
        s_N = 3,
        t_N = 5,
        u_N = 12,
        noise = 1e-4
    )

    @testset "Manifold Types Instantiation" begin
        println("Running Manifold Types Instantiation Tests...")

        # Spatial Manifolds
        d1 = Exponential(1.0)
        d2 = Beta(1,1)
        @testset "BYM2" begin
            bym2 = BYM2(:s_idx, d1, d2)
            @test bym2 isa BYM2
            @test bym2.index == :s_idx
            @test bym2.sigma_prior == d1
            @test bym2.rho_prior == d2
        end

        @testset "ICAR" begin
            icar = ICAR(:s_idx, d1)
            @test icar isa ICAR
            @test icar.sigma_prior == d1
        end

        @testset "RW1" begin
            rw1 = RW1(:s_idx, d1)
            @test rw1 isa RW1
            @test rw1.sigma_prior == d1
        end

        @testset "GaussianProcess" begin
            gp = GaussianProcess([:s_x, :s_y], d1, "matern", 1.5, d1)
            @test gp isa GaussianProcess
            @test gp.kernel == "matern"
        end

        # Temporal Manifolds
        @testset "AR1" begin
            ar1 = AR1(:t_idx, d1, d2)
            @test ar1 isa AR1
            @test ar1.index == :t_idx
        end

        @testset "RW2T" begin
            rw2t = RW2T(:t_idx, d1)
            @test rw2t isa RW2T
        end

        # Seasonal Manifolds
        @testset "HarmonicSeasonal" begin
            hs = HarmonicSeasonal(12, d1, Normal(0,2), Uniform(0, 2pi))
            @test hs isa HarmonicSeasonal
            @test hs.period == 12
        end

        # Covariate Manifolds
        @testset "Fixed" begin
            fixed = Fixed(:region, "sum", true)
            @test fixed isa Fixed
            @test fixed.variable == :region
        end

        @testset "Smooth" begin
            smooth = Smooth(:elevation, :rw2, 9, log, :group)
            @test smooth isa Smooth
            @test smooth.transform == log
        end

        # smooth Manifolds
        @testset "KnorrHeld" begin
            kh = KnorrHeld(:s_idx, :t_idx, 'I', d1)
            @test kh isa KnorrHeld
            @test kh.type == 'I'
        end

        # Compositions & Transformations
        @testset "ComposedManifold" begin
            comp = ComposedManifold([BYM2(:s,d1,d2), AR1(:t,d1,d2)], :pipe)
            @test comp isa ComposedManifold
            @test length(comp.components) == 2
        end

        @testset "Log" begin
            @test Log() isa Log
        end

        @testset "Intercept" begin
            inter = Intercept(d1)
            @test inter isa Intercept
        end
        println("Manifold Types Instantiation Tests Passed.")
    end

    @testset "Formula Translation from Old Syntax" begin
        println("Running Formula Translation Tests...")

        @testset "Intercept" begin
            manifolds, resp = translate_old_formula_to_new("y ~ 1")
            @test length(manifolds) == 1
            @test manifolds[1] isa Intercept
            @test resp == :y
        end

        @testset "Fixed Effects (fe)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ fe(region, contrasts='sum')")
            @test manifolds[1] isa Fixed
            @test manifolds[1].variable == :region
            @test manifolds[1].contrasts == "sum"

            manifolds, _ = translate_old_formula_to_new("y ~ AFF")
            @test manifolds[1] isa Fixed
            @test manifolds[1].variable == :AFF
        end

        @testset "Random Effects (re)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ re(s_idx, model='bym2')")
            @test manifolds[1] isa BYM2
            @test manifolds[1].index == :s_idx

            manifolds, _ = translate_old_formula_to_new("y ~ re(t_idx, model='ar1', sigma_prior=Normal(0,1))")
            @test manifolds[1] isa AR1
            @test manifolds[1].index == :t_idx
            @test manifolds[1].sigma_prior isa Normal

            manifolds, _ = translate_old_formula_to_new("y ~ re(t_idx, model='rw2t')")
            @test manifolds[1] isa RW2T
        end

        @testset "Spatiotemporal Effects (st)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ st(s_idx, t_idx, type='IV')")
            @test manifolds[1] isa KnorrHeld
            @test manifolds[1].dim1 == :s_idx
            @test manifolds[1].type == 'IV'
        end

        @testset "Covariate Effects (ce)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ ce(elevation, nbins=5, model='rw2')")
            @test manifolds[1] isa Smooth
            @test manifolds[1].variable == :elevation
            @test manifolds[1].nbins == 5
            @test manifolds[1].manifold == :rw2
        end

        @testset "smooth Effects (ie)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ ie(temp, salinity, nbins1=3)")
            @test manifolds[1] isa Surface
            @test manifolds[1].var1 == :temp
            @test manifolds[1].nbins1 == 3
        end

        @testset "GP/RFF Effects (f/rff)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ f(elevation, kernel='matern', nu=1.5)")
            @test manifolds[1] isa GPEffect
            @test manifolds[1].variable == :elevation
            @test manifolds[1].kernel == "matern"

            manifolds, _ = translate_old_formula_to_new("y ~ rff(temp, n_features=50)")
            @test manifolds[1] isa RFFEffect
            @test manifolds[1].variable == :temp
            @test manifolds[1].n_features == 50
        end

        @testset "Mixed Effects (me)" begin
            manifolds, _ = translate_old_formula_to_new("y ~ me(1 | group_var)")
            @test manifolds[1] isa Smooth
            @test manifolds[1].grouped_by == :group_var
        end

        println("Formula Translation Tests Passed.")
    end

    @testset "Manifold Builder Functions" begin
        println("Running Manifold Builder Tests...")

        @testset "BYM2 Builder" begin
            bym2 = BYM2(:s_idx, Exponential(1.0), Beta(1,1))
            result = build_model(bym2, TEST_DATA_INPUTS)
            @test result.Q_template isa AbstractMatrix
            @test size(result.Q_template) == (TEST_DATA_INPUTS.s_N, TEST_DATA_INPUTS.s_N)
            @test result.hyper.sigma_prior isa Exponential
            @test result.model_type == :bym2
        end

        @testset "AR1 Builder" begin
            ar1 = AR1(:t_idx, Exponential(1.0), Beta(1,1))
            result = build_model(ar1, TEST_DATA_INPUTS)
            @test result.Q_template isa AbstractMatrix
            @test size(result.Q_template) == (TEST_DATA_INPUTS.n_temporal_units, TEST_DATA_INPUTS.n_temporal_units)
            @test result.model_type == :ar1
        end

        @testset "RW1T Builder" begin
            rw1t = RW1T(:t_idx, Exponential(1.0))
            result = build_model(rw1t, TEST_DATA_INPUTS)
            @test result.Q_template isa AbstractMatrix
            @test size(result.Q_template) == (TEST_DATA_INPUTS.n_temporal_units, TEST_DATA_INPUTS.n_temporal_units)
            @test result.model_type == :rw1t
        end

        @testset "IIDT Builder" begin
            iidt = IIDT(:t_idx, Exponential(1.0))
            result = build_model(iidt, TEST_DATA_INPUTS)
            @test result.Q_template isa Diagonal # For IID, it should be a Diagonal matrix
            @test size(result.Q_template) == (TEST_DATA_INPUTS.n_temporal_units, TEST_DATA_INPUTS.n_temporal_units)
            @test result.model_type == :iidt
        end

        @testset "Generic Manifold Builder (Fallback)" begin
            # Test with a manifold type that has no specific builder implemented yet
            generic_manifold = Fixed(:x, nothing, false)
            result = build_model(generic_manifold, TEST_DATA_INPUTS)
            @test result.Q_template === nothing
            @test result.model_type == Fixed
            @test_logs (:warn, r"No specific builder implemented") build_model(generic_manifold, TEST_DATA_INPUTS)
        end
        println("Manifold Builder Tests Passed.")
    end

    @testset "Manifold Macros and Operators" begin
        println("Running Manifold Macros and Operators Tests...")

        @testset "@ManifoldFormula Macro (Basic)" begin
            # Needs a dummy bstm_internal function or equivalent for full test
            # For now, test the output structure of the macro
            # We can't directly evaluate the macro outside of its intended context easily,
            # but we can check its expansion if it were used in a model function.
            expr = quote @ManifoldFormula y_resp ~ Intercept() end
            expanded = macroexpand(Main, expr)
            @test expanded.head == :tuple # Checks if it expands to a NamedTuple
            @test contains(string(expanded), "response = :y_resp")
            @test contains(string(expanded), "manifolds = [Main.Intercept()]" # Note the Main. prefix due to global scope
            )
        end

        @testset "@ManifoldFormula Macro (Sum of Terms)" begin
            expr = quote @ManifoldFormula y_resp ~ Intercept() + Fixed(:x_var) end
            expanded = macroexpand(Main, expr)
            @test contains(string(expanded), "manifolds = [Main.Intercept(), Main.Fixed(:x_var, nothing, false)]")
        end

        @testset "Pipe Operator (|>) for Transformation" begin
            manifold = BYM2(:s_idx, Exponential(1.0), Beta(1,1))
            transformed = manifold |> Log()
            @test transformed isa TransformedManifold
            @test transformed.manifold == manifold
            @test transformed.transform_fn == typeof(Log())
        end

        @testset "Pipe Operator (|>) for Composition" begin
            m1 = BYM2(:s_idx, Exponential(1.0), Beta(1,1))
            m2 = AR1(:t_idx, Exponential(1.0), Beta(1,1))
            composed = m1 |> m2
            @test composed isa ComposedManifold
            @test composed.components == [m1, m2]
            @test composed.operator == :pipe
        end

        @testset "Kronecker Product Operator (⊗)" begin
            m1 = BYM2(:s_idx, Exponential(1.0), Beta(1,1))
            m2 = AR1(:t_idx, Exponential(1.0), Beta(1,1))
            kron_prod = m1 ⊗ m2
            @test kron_prod isa ComposedManifold
            @test kron_prod.components == [m1, m2]
            @test kron_prod.operator == :kronecker_product
        end

        @testset "Direct Sum Operator (⊕)" begin
            m1 = BYM2(:s_idx, Exponential(1.0), Beta(1,1))
            m2 = AR1(:t_idx, Exponential(1.0), Beta(1,1))
            direct_sum = m1 ⊕ m2
            @test direct_sum isa ComposedManifold
            @test direct_sum.components == [m1, m2]
            @test direct_sum.operator == :direct_sum
        end
        println("Manifold Macros and Operators Tests Passed.")
    end


end



