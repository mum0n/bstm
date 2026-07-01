using Test
# using DataFrames
# using Turing
# using bstm
# using SparseArrays
# using Distributions

@testset "BSTM Comprehensive Tests" begin

    # --- 1. Data Setup ---
    (primary_data, nested_data) = scottish_lip_cancer_data_spacetime()
    data = primary_data.data
    W = primary_data.au.W

    # Add synthetic columns for testing all formula features
    data.dynamic_var = randn(nrow(data))
    data.u_idx = mod1.(1:nrow(data), 12)
    data.weights_col = ones(nrow(data))

    @testset "Formula Parsing & Config Engine" begin
        # 2. Define a comprehensive formula that tests every major keyword
        formula = """
        y ~ 1 + cov1 + cov2 +
        bias(hurdle=0.0) +
        spatial(s_idx, manifold='bym2') +
        temporal(year, manifold='ar1') +
        seasonal(u_idx, manifold='cyclic', period=12) +
        smooth(cov3, nbins=10, manifold='pspline') +
        smooth(cov4 |> log, nbins=15, manifold='rff') +
        nested(z ~ 1) +
        eigen(cov1, cov2, n_factors=1) +
        mixed(1|f1) +
        svc(cov1, manifold='icar') +
        volatility(s_idx) +
        network(s_idx) +
        spacetime(manifold='I') +
        hyperbolic(s_idx) +
        dynamics(dynamic_var)
        """

        # 3. Run the config engine
        M = bstm_config(formula, data; W=W)

        # 4. Verify that all keywords resulted in a manifold object
        manifolds = M.manifolds
        @test M.add_intercept == true
        @test "cov1" in string.(M.fixed_parts) && "cov2" in string.(M.fixed_parts)
        @test haskey(M, :hurdle)

        # Check for correct manifold object instantiation
        @test any(spec -> spec.domain == :spatial && spec.manifold_obj isa BYM2, manifolds)
        @test any(spec -> spec.domain == :temporal && spec.manifold_obj isa AR1, manifolds)
        @test any(spec -> spec.domain == :seasonal && spec.manifold_obj isa Cyclic, manifolds)
        @test any(spec -> spec.domain == :smooth && spec.var == :cov3 && spec.manifold_obj isa PSpline, manifolds)
        @test any(spec -> spec.domain == :smooth && spec.var == :cov4 && spec.manifold_obj isa RFF, manifolds)
        @test any(spec -> spec.domain == :eigen, manifolds)
        @test any(spec -> spec.domain == :mixed && spec.var == :f1, manifolds)
        @test any(spec -> spec.domain == :svc && spec.var == :cov1 && spec.manifold_obj isa ICAR, manifolds)
        @test any(spec -> spec.domain == :network, manifolds)
        @test any(spec -> spec.domain == :spacetime && spec.manifold_obj isa ST_I, manifolds)
        @test any(spec -> spec.domain == :hyperbolic, manifolds)

        # Check for correct flag setting
        @test M.use_sv == true
        @test M.use_dynamics == true
        @test !isempty(M.nested_manifolds)
    end

    @testset "Manifold Structs and Methods" begin
        # Test if all major manifold types can be resolved
        dummy_M_config = (s_N=10, t_N=10, u_N=12, period=12, nbins=10, W=sprand(10,10,0.2))
        dummy_priors = Dict()

        # Test resolve_technical_primitive for all known manifolds
        manifold_mappings = Dict(
            "iid" => IID, "icar" => ICAR, "besag" => ICAR, "bym2" => BYM2, "leroux" => Leroux,
            "sar" => SAR, "rw1" => RW1, "rw2" => RW2, "ar1" => AR1, "gp" => GP,
            "fitc" => FITC, "rff" => RFF, "fft" => FFT, "spde" => SPDE, "dag" => DAG,
            "svgp" => SVGP, "warp" => Warp, "hyperbolic" => Hyperbolic,
            "advection" => Advection, "diffusion" => Diffusion, "advection_diffusion" => AdvectionDiffusion,
            "I" => ST_I, "II" => ST_II, "III" => ST_III, "IV" => ST_IV,
            "tps" => TPS, "bspline" => BSpline, "pspline" => PSpline, "wavelet" => Wavelets,
            "harmonic" => Harmonic, "cyclic" => Cyclic
        )

        for (name, type) in manifold_mappings
            @test resolve_technical_primitive(name, dummy_M_config, dummy_priors, :pcpriors) isa type
        end

        # Test build_model for a representative subset
        @test build_model(ICAR(Exponential(1.0)), dummy_M_config).model_type == :icar
        @test build_model(BYM2(Beta(1,1), Exponential(1.0)), dummy_M_config).model_type == :bym2
        @test build_model(AR1(Beta(2,2), Exponential(1.0)), dummy_M_config).model_type == :ar1
        @test build_model(PSpline(10,3,2,Exponential(1.0)), dummy_M_config).model_type == :smooth
        @test build_model(Cyclic(12, Exponential(1.0)), dummy_M_config).model_type == :cyclic
    end

    @testset "Model Execution" begin
        # A simpler formula that is faster to run for an execution test
        formula_run = "y ~ 1 + cov1 + spatial(s_idx, manifold='icar')"
        M_run = bstm_config(formula_run, data; W=W)
        model = bstm_univariate(M_run)

        # Test that the model can be instantiated and a random sample can be drawn
        @test rand(model) isa NamedTuple
    end

end