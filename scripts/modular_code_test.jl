 
project_directory = joinpath( "C:\\", "home", "jae", "projects", "bstm")  
 
include( joinpath( project_directory, "startup.jl" ) ) # might need to run this a few times if there are stragglers   
 
# load_project_functions( srcdir() )

data_scot, _ = scottish_lip_cancer_data_spacetime(); # additional noise and "fake" time slices added

m = @bstm(likelihood(y, family=:poisson, offsets=:log_offset) ~
    intercept() +
    spatial(s_idx, model=:besag) +
    temporal(year, model=:ar1),
    data_scot[:data],
    W=data_scot[:au][:W]
)

spec_registry = compile_generated_model(m.args.M)
compiled_model_func = Main.bstm_generated_univariate
turing_model_inst = Base.invokelatest(compiled_model_func, m.args.M, spec_registry)
chn_generated = sample(turing_model_inst, MH(), 100)

println("--- Code Generation Pipeline Successful ---")
display(summarize(chn_generated))




if @isdefined(data_scot)
    println("--- Verifying Modular Assembler v9 (Dynamic Generation) ---")
    
    # 1. Define Formula for Univariate Model
    formula_uv = "likelihood(y, family=poisson, offsets=log_offset) ~ intercept() + spatial(s_idx, model=besag) + temporal(year, model=ar1)"

    # 2. Build Technical Configuration
    config_uv = bstm_config(
        formula_uv,
        data_scot[:data],
        W = data_scot[:au][:W]
    )

    # 3. Dynamic Model Instantiation
    # This calls the modular assembler, parses the code string, and returns a Turing model object.
    println("Assembler: Generating and compiling @model function...")
    m_modular = bstm_dynamic_univariate(config_uv)

    # 4. Structural Sampling Check
    println("Sampling: Executing MH check (100 samples) for graph verification...")
    # Using MH for speed during structural verification
    chn_modular = sample(m_modular, MH(), 100, progress=false)

    println("\nSUCCESS: Modularly generated model assembled and sampled correctly.")

    # 5. Symbolic Discovery Audit
    p_names = string.(FlexiChains.parameters(chn_modular))
    println("Discovered Clean Symbols:")
    display(filter(n -> !occursin("beta", n), p_names[1:min(10, end)]))

    # 6. Parity Check (RMSE Assessment)
    # Recovering results for a quick diagnostic summary
    results_mod = model_results_comprehensive(
        m_modular,
        chn_modular,
        au = data_scot[:au],
        data = data_scot[:data]
    )
else
    println("Error: Scottish Lip Cancer dataset not found. Please re-run the data factory cell.")
end




### BSTM Multivariate Verification Suite v1.1.7
# Rationale: Re-testing the full pipeline after refactoring latent updates to use `view` for indexing.

if @isdefined(data_scot)
    println("--- Final Multivariate Structural Time Series Test (View-Indexing Optimization) ---")

    # 1. Define Multivariate Formula & Config
    formula_mv = "likelihood(y, family=poisson) + likelihood(y_bin, family=binomial) ~ intercept() + spatial(s_idx, model=besag)"

    config_mv = bstm_config(
        formula_mv,
        data_scot[:data],
        W=data_scot[:au][:W]
    )

    # 2. Re-assemble with Assembler v7.2.5 logic and latest v7.2.0 fragments
    expr_mv, reg_mv = bstm_text_assembler_v7_2(config_mv)
    eval(expr_mv)

    println("Starting Sampling Verification...")
    model_mv = Base.invokelatest(getfield(Main, :bstm_text_generated_multivariate), config_mv, reg_mv)

    # 3. Structural Validation with MH Sampler
    # We sample a small number of points to verify the mathematical graph and indexing logic
    chn_mv = sample(model_mv, MH(), 100, progress=false)

    println("\nSUCCESS: Multivariate model with view-optimized indexing executed correctly.")

    # Explicitly convert to MCMCChains.Chains for robust summarization
    println("Summarizing results...")
    final_chains = MCMCChains.Chains(chn_mv)
    display(MCMCChains.summarize(final_chains))
end







### BSTM Multifidelity Verification Suite v1.1.4
# Rationale: Re-verifying Multifidelity Pipeline after forcing v2.1.4 logic re-definition.

if @isdefined(scottish_lip_cancer_data_spacetime)
    println("--- Verifying Multifidelity Assembly & Sampling (v8.0.6 + v2.1.4 Forced) ---")

    try
        # 1. Load data partitions
        p_set, n_set = scottish_lip_cancer_data_spacetime(10, 1.2, 1.2; recreate=false)

        # 2. Define formula with a nested supervisor (RHS-only formula)
        formula_mf = "likelihood(y, family=poisson, offsets=log_offset) ~ intercept() + spatial(s_idx, model=besag) + nested(expanded_field, formula=\"spatial(district, model=besag)\", data_source=nested_data)"

        # 3. Build Configuration
        println("Config: Building Model Registry with Nested Primitive...")
        config_mf = bstm_config(
            formula_mf,
            p_set.data,
            W=p_set.au.W,
            nested_data=n_set.data,
            model_arch="multifidelity"
        )

        # 4. Assemble and Parse Model
        println("Assembler: Generating and parsing model string...")
        expr_mf, reg_mf = bstm_text_assembler_multifidelity(config_mf)

        # Evaluate in current world age
        Base.invokelatest(eval, expr_mf)
        println("Compilation: Success.")

        # 5. Execute Metropolis-Hastings Sampling Check
        println("Sampling: Initializing MH sampler (100 samples)... verification of nested resolution...")
        model_func = getfield(Main, :bstm_text_generated_multifidelity)
        model_inst = Base.invokelatest(model_func, config_mf, reg_mf)

        # Verify that the sampler can initialize and explore
        chn_mf = sample(model_inst, MH(), 100, progress=false)

        println("\nSUCCESS: Multi-fidelity model v8.0.6 assembled and sampled correctly.")

        # Display linking and noise parameters
        p_names = string.(FlexiChains.parameters(chn_mf))
        println("\nDiscovered Nested Parameters:")
        display(filter(n -> occursin("nested", n) || occursin("y_sigma", n), p_names))

    catch e
        @error "Verification Failed for Multifidelity Pipeline."
        if hasproperty(e, :msg) println("Error Message: ", e.msg) end
        rethrow(e)
    end
else
    println("Error: Reference data factory not found in workspace.")
end


