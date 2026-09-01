"""
    par.jl

Population Attributable Risk (PAR) computation and summarization for disease 
epidemiological models fitted with BSTM.

The module provides post-estimation extraction of covariate effects, relative risk 
computation, and PAR estimation with credible intervals for univariate and multivariate 
disease models across multiple likelihood families (Poisson, Binomial, Negative Binomial, etc.).

Accounts for reference population baseline risk, exposure prevalence, and family-specific
link functions (log-linear vs logit).

# Mathematical Background

## Poisson and Negative Binomial Models
For count outcomes with a log-linear predictor (canonical link):
    η = log(μ)  →  RR = exp(β)
    
where β is the fitted log-relative-risk coefficient.
The reference population baseline does NOT affect RR (multiplicative model).

## Binomial Models  
For binary/proportion outcomes with a logit link:
    η = logit(p)  →  OR = exp(β)  (odds ratio, not relative risk)
    
The conversion from OR to RR depends on baseline risk p₀:
    RR = (p₁) / (p₀) = logistic(η₀ + β) / logistic(η₀)

where η₀ is the baseline linear predictor (baseline_eta) or p₀ is specified directly.

## Population Attributable Risk (General Form)
    PAR = p_exp * (RR - 1) / RR
    
where:
- p_exp is the population prevalence/frequency of the exposure
- RR is the relative risk (or odds ratio for binomial with rare outcome)
- Both depend critically on the reference population specification

Version: v1.0.0

Key references:
 
Levin, M. L. (1953). "The occurrence of lung cancer in man." Acta Unio Internationalis Contra Cancrum, 9(3), 531–541.

Rockhill, B., Newman, B., & Weinberg, C. (1998). "Use and misuse of population attributable fractions." American Journal of Public Health, 88(1), 15–19.

Greenland, S., & Drescher, K. (1993). "Maximum likelihood estimation of the attributable fraction from logistic models." Biometrics, 49(3), 865–872.

Bruzzi, P., Green, S. B., Byar, D. P., Brinton, L. A., & Schairer, C. (1985). "Estimating the population attributable risk for multiple risk factors using case-control data." American Journal of Epidemiology, 122(6), 904–914.

Zeileis, A., Hothorn, T., & Hornik, K. (2008). "Model-based recursive partitioning." Journal of Computational and Graphical Statistics, 17(2), 492–514.

Greenland, S. (2004). "Model-based estimation of relative risks and other epidemiologic measures." Journal of Epidemiology and Community Health, 58(7), 575–581.
 
Clayton, D., & Hills, M. (1993). Statistical Models in Epidemiology. Oxford University Press.
 
"""


"""
    extract_scalar_par_effect(chain, param_name::String)::Vector{Float64}

Extracts a scalar fixed effect parameter from the MCMC chain.

# Arguments
- `chain`: The MCMC chain object (supports VNChain, FlexiChain, DataFrame, Dict).
- `param_name::String`: The name of the parameter to extract.

# Returns
- A vector of posterior samples of length n_samples.
"""
function extract_scalar_par_effect(chain, param_name::String)::Vector{Float64}
    samples = try
        get_params_vector(chain, param_name, 1)[:, 1]
    catch
        try
            chain_df = DataFrame(chain)
            if hasproperty(chain_df, Symbol(param_name))
                collect(chain_df[!, Symbol(param_name)])
            else
                Float64[]
            end
        catch
            @warn "Could not extract parameter '$param_name' from chain."
            Float64[]
        end
    end
    return samples
end


"""
    extract_intercept_from_chain(chain)::Vector{Float64}

Attempts to extract the intercept parameter from the MCMC chain.
Searches for common naming patterns: "intercept", "alpha", "β0", etc.

# Returns
- A vector of posterior intercept samples, or empty vector if not found.
"""
function extract_intercept_from_chain(chain)::Vector{Float64}
    intercept_names = ["intercept", "alpha", "β0", "b0", "Intercept", "INTERCEPT"]
    for name in intercept_names
        try
            samples = extract_scalar_par_effect(chain, name)
            if !isempty(samples)
                return samples
            end
        catch
        end
    end
    return Float64[]
end


"""
    _compute_baseline_risk_from_intercept(intercept_samples::Vector{Float64}, 
        family::String)::Vector{Float64}

Converts posterior intercept samples to baseline risk samples, conditional on family.

# Arguments
- `intercept_samples::Vector{Float64}`: Posterior samples of the intercept parameter.
- `family::String`: Likelihood family ("binomial", "poisson", etc.).

# Returns
- Vector of baseline risk/rate samples.
"""
function _compute_baseline_risk_from_intercept(intercept_samples::Vector{Float64}, 
    family::String)::Vector{Float64}
    
    family_lower = lowercase(strip(family))
    
    if family_lower == "binomial"
        # p₀ = logistic(intercept)
        return LogExpFunctions.logistic.(intercept_samples)
    elseif family_lower in ["poisson", "negbin", "negative_binomial"]
        # μ₀ = exp(intercept) [rate, not probability]
        return exp.(intercept_samples)
    else
        # Generic: assume log-linear
        return exp.(intercept_samples)
    end
end


"""
    _compute_rr_for_family(family::String, log_coef::Real, 
        baseline_risk::Union{Real, Nothing}=nothing)::Real

Computes the relative/odds ratio for a covariate effect conditional on the likelihood family 
and link function.

# Arguments
- `family::String`: The likelihood family (e.g., "poisson", "negbin", "binomial", "gaussian").
- `log_coef::Real`: The coefficient/effect size (typically on log or logit scale).
- `baseline_risk::Union{Real, Nothing}`: Baseline risk p₀ (only for binomial to compute exact RR).
  If `nothing`, uses rare outcome approximation (OR ≈ RR).

# Returns
- The relative risk (RR) or odds ratio (OR) as a scalar.

# Notes
- **Poisson & Negative Binomial**: RR = exp(β) [independent of baseline]
- **Binomial**: OR = exp(β). If baseline_risk is provided:
    - Exact RR = logistic(logit(p₀) + β) / p₀
    - Otherwise uses rare outcome approximation: RR ≈ OR 

Currently handles: poisson, negbin, binomial, gamma, exponential, 
inverse_gaussian, lognormal, beta.
"""
function _compute_rr_for_family(family::String, log_coef::Real, 
    baseline_risk::Union{Real, Nothing}=nothing)::Real
    
    family_lower = lowercase(strip(family))
    
    # Log-linear families (all use canonical log link)
    if family_lower in ["poisson", "negbin", "negative_binomial",
                        "gamma", "exponential", "inverse_gaussian", 
                        "lognormal", "pareto"]
        return exp(log_coef)
    
    # Logit-link families (binomial, beta)
    elseif family_lower in ["binomial", "beta"]
        or_value = exp(log_coef)
        
        if isnothing(baseline_risk)
            return or_value  # Rare outcome approximation
        else
            if baseline_risk > 1e-10 && baseline_risk < 1.0 - 1e-10
                logit_p0 = LogExpFunctions.logit(baseline_risk)
                exposed_prob = LogExpFunctions.logistic(logit_p0 + log_coef)
                return exposed_prob / baseline_risk
            else
                return or_value
            end
        end
    
    # Identity-link families (NOT suitable for PAR)
    elseif family_lower in ["gaussian", "studentt", "laplace"]
        @warn "Likelihood family '$family' uses identity link (additive effects). " *
              "PAR interpretation is not meaningful. Consider effect modification instead."
        return NaN
    
    # Zero-inflated models (requires special handling)
    elseif family_lower in ["zipoisson", "zinegbin"]
        @warn "Zero-inflated likelihoods require special PAR handling (mixture model). " *
              "Current implementation does not support this family. " *
              "See documentation for manual computation."
        return NaN
    
    # Compositional / multivariate (not applicable)
    elseif family_lower in ["dirichlet_multinomial", "inverse_wishart", "ordinal"]
        @warn "Likelihood family '$family' is not supported for PAR computation. " *
              "Use covariate-specific effect measures instead."
        return NaN
    
    else
        @warn "Likelihood family '$family' not recognized. Attempting log-linear assumption."
        return exp(log_coef)
    end
end


"""
    _infer_baseline_risk(chain, family::String, baseline_eta::Union{Real, Nothing},
        baseline_risk::Union{Real, Nothing}, data::Union{DataFrame, Nothing},
        outcome_var::Union{String, Nothing}, reference_population::String)
        ::Union{Float64, Vector{Float64}, Nothing}

Infers baseline risk (p₀) for a binomial model using a priority hierarchy.

# Priority Order
1. User-specified `baseline_risk` (explicit scalar)
2. User-specified `baseline_eta` converted via link function
3. Extract intercept from MCMC chain → convert to p₀
4. Compute from observed outcome in data (if reference_population="sample")
5. None/default (for Poisson, Negbin models where not needed)

# Returns
- Scalar baseline risk, or Nothing if not determinable (e.g., Poisson family)
"""
function _infer_baseline_risk(
    chain,
    family::String,
    baseline_eta::Union{Real, Nothing},
    baseline_risk::Union{Real, Nothing},
    data::Union{DataFrame, Nothing},
    outcome_var::Union{String, Nothing},
    reference_population::String
)::Union{Float64, Vector{Float64}, Nothing}

    family_lower = lowercase(Base.strip(family))
    
    # For Poisson/Negbin, baseline risk doesn't affect RR
    if family_lower in ["poisson", "negbin", "negative_binomial"]
        return nothing
    end
    
    # For binomial: attempt to infer p₀ in priority order
    if !isnothing(baseline_risk)
        return Float64(baseline_risk)
    end
    
    if !isnothing(baseline_eta)
        p0 = LogExpFunctions.logistic(Float64(baseline_eta))
        return p0
    end
    
    # Try to extract intercept from chain and convert
    try
        intercept_samples = extract_intercept_from_chain(chain)
        if !isempty(intercept_samples)
            p0_samples = _compute_baseline_risk_from_intercept(intercept_samples, family)
            return mean(p0_samples)  # Return posterior mean baseline risk
        end
    catch
    end
    
    # Try to compute from observed data
    if reference_population == "sample" && !isnothing(data) && !isnothing(outcome_var)
        try
            if hasproperty(data, Symbol(outcome_var))
                y_obs = data[!, Symbol(outcome_var)]
                if eltype(y_obs) <: Integer || eltype(y_obs) <: Bool
                    return mean(Float64.(y_obs))  # Proportion with outcome
                end
            end
        catch
        end
    end
    
    # Could not infer baseline risk
    @warn "Could not infer baseline risk for $family model. PAR will use OR approximation " *
          "(valid only for rare outcomes). Specify baseline_risk or baseline_eta explicitly."
    return nothing
end


"""
    _infer_exposure_prevalence(exposure_var::Union{String, Nothing}, 
        exposure_prevalence::Union{Real, Nothing}, data::Union{DataFrame, Nothing})::Float64

Infers exposure prevalence (p_exp) with priority hierarchy.

# Priority Order
1. User-specified `exposure_prevalence`
2. Computed from `exposure_var` in data (mean for binary/continuous)
3. Default to 0.5 (50% exposed)

# Returns
- Exposure prevalence as scalar in [0, 1]
"""
function _infer_exposure_prevalence(exposure_var::Union{String, Nothing}, 
    exposure_prevalence::Union{Real, Nothing}, data::Union{DataFrame, Nothing})::Float64
    
    if !isnothing(exposure_prevalence)
        return clamp(Float64(exposure_prevalence), 0.0, 1.0)
    end
    
    if !isnothing(exposure_var) && !isnothing(data)
        try
            if hasproperty(data, Symbol(exposure_var))
                exp_col = data[!, Symbol(exposure_var)]
                p_exp = mean(Float64.(exp_col))
                return clamp(p_exp, 0.0, 1.0)
            end
        catch
        end
    end
    
    return 0.5  # Default: equal exposed/unexposed
end


"""
    par_from_posterior(chain, covariate::String; 
        family="poisson", baseline_eta=nothing, baseline_risk=nothing,
        exposure_var=nothing, exposure_prevalence=nothing,
        data=nothing, outcome_var=nothing, reference_population="sample", 
        alpha=0.05)::NamedTuple

Compute population attributable risk (PAR) for a covariate from posterior samples.

This function extracts fixed covariate effects from the MCMC chain and computes PAR 
with credible intervals, accounting for the likelihood family's link function and 
reference population baseline risk.

# Arguments
- `chain`: The MCMC chain object from `sample()`.
- `covariate::String`: The name of the fixed effect covariate (must be fitted as `fixed(...)`).
- `family::String`: The likelihood family. Supported: "poisson", "negbin", "binomial", "gaussian".
  Default: "poisson".
- `baseline_eta::Union{Real, Nothing}`: Baseline linear predictor for binomial models. 
  Used to compute exact RR = logistic(baseline_eta + β) / logistic(baseline_eta).
  Default: `nothing`.
- `baseline_risk::Union{Real, Nothing}`: Explicit baseline disease/outcome risk (0 to 1).
  Only used for binomial family. Overrides baseline_eta if both provided.
  Default: `nothing`.
- `exposure_var::Union{String, Nothing}`: Column name in data to auto-compute exposure prevalence.
  Default: `nothing`.
- `exposure_prevalence::Union{Real, Nothing}`: Population exposure prevalence (0 to 1).
  Overrides automatic computation from exposure_var. Default: `nothing`.
- `data::Union{DataFrame, Nothing}`: Training data (used to infer exposure_prev and baseline_risk).
  Default: `nothing`.
- `outcome_var::Union{String, Nothing}`: Column name of outcome (for inferring baseline risk).
  Default: `nothing`.
- `reference_population::String`: Either "sample" (infer from data) or "external" (use explicit values).
  Default: "sample".
- `alpha::Float64`: Significance level for credible interval (default: 0.05 → 95% CI).

# Returns
- A `NamedTuple` containing:
  - `par_mean::Float64`: Posterior mean PAR.
  - `par_median::Float64`: Posterior median PAR.
  - `par_std::Float64`: Posterior standard deviation of PAR.
  - `par_lower::Float64`: Lower credible interval bound (alpha/2).
  - `par_upper::Float64`: Upper credible interval bound (1 - alpha/2).
  - `rr_mean::Float64`: Posterior mean relative/odds risk.
  - `rr_median::Float64`: Posterior median relative/odds risk.
  - `rr_ci_lower::Float64`: Lower credible interval for RR/OR.
  - `rr_ci_upper::Float64`: Upper credible interval for RR/OR.
  - `exposure_prevalence::Float64`: Exposure prevalence used in computation.
  - `baseline_risk::Union{Float64, Nothing}`: Baseline risk (p₀) for binomial.
  - `reference_population::String`: "sample" or "external".
  - `covariate::String`: The covariate name.
  - `family::String`: The likelihood family.
  - `n_samples::Int`: Number of posterior samples.
  - `raw_par_samples::Vector{Float64}`: Full posterior PAR samples.
  - `raw_rr_samples::Vector{Float64}`: Full posterior RR/OR samples.

# Example: Poisson Model (Disease Count Data)

```julia
m = @bstm(
    likelihood(cases, family=poisson, log_offsets=log_pop_at_risk) ~
        intercept() +
        fixed(log_RR_occupation) +
        random(s_idx, model=bym2, W=W) +
        random(year, model=ar1),
    df
)
chn = sample(m, NUTS(), 1000; progress=false)

par_results = par_from_posterior(
    chn, 
    covariate="log_RR_occupation",
    family="poisson",
    exposure_var="occupational_exposure",
    data=df
)
Example: Binomial Model (Disease Prevalence)
Julia
m = @bstm(
    likelihood(disease, family=binomial, trials=n_individuals) ~
        intercept() +
        fixed(smoking_status) +
        random(region_idx, model=bym2, W=W),
    df
)
chn = sample(m, NUTS(), 1000; progress=false)

par_results = par_from_posterior(
    chn,
    covariate="smoking_status",
    family="binomial",
    baseline_risk=0.12,
    exposure_var="smoking_status",
    data=df
)
Example: Binomial, Inferred from Sample
Julia
par_results = par_from_posterior(
    chn,
    covariate="smoking_status",
    family="binomial",
    exposure_var="smoking_status",
    outcome_var="disease",
    reference_population="sample",
    data=df
)
""" 
function par_from_posterior( 
    chain, 
    covariate::String; 
    family::String="poisson", 
    baseline_eta::Union{Real, Nothing}=nothing, 
    baseline_risk::Union{Real, Nothing}=nothing, 
    exposure_var::Union{String, Nothing}=nothing, 
    exposure_prevalence::Union{Real, Nothing}=nothing, 
    data::Union{DataFrame, Nothing}=nothing, 
    outcome_var::Union{String, Nothing}=nothing, 
    reference_population::String="sample", 
    alpha::Float64=0.05 
    )::NamedTuple

    # Extract coefficient samples
    coef_samples = extract_scalar_par_effect(chain, covariate)

    if isempty(coef_samples)
        @warn "No posterior samples found for covariate '$covariate'. " *
            "Check that the covariate is fitted as a fixed effect."
        return (
            par_mean=NaN, par_median=NaN, par_std=NaN, par_lower=NaN, par_upper=NaN,
            rr_mean=NaN, rr_median=NaN, rr_ci_lower=NaN, rr_ci_upper=NaN,
            exposure_prevalence=NaN, baseline_risk=nothing, reference_population=reference_population,
            covariate=covariate, family=family, n_samples=0,
            raw_par_samples=Float64[], raw_rr_samples=Float64[]
        )
    end

    # Infer baseline risk (binomial-specific)
    inferred_baseline_risk = _infer_baseline_risk(
        chain, family, baseline_eta, baseline_risk, data, outcome_var, reference_population
    )

    # Infer exposure prevalence
    p_exposed = _infer_exposure_prevalence(exposure_var, exposure_prevalence, data)

    # Compute relative/odds risk samples
    rr_samples = if inferred_baseline_risk isa Vector
        # Baseline risk is stochastic (from intercept posterior)
        [_compute_rr_for_family(family, coef_samples[i], inferred_baseline_risk[i]) 
        for i in eachindex(coef_samples)]
    else
        # Baseline risk is scalar or nothing
        [_compute_rr_for_family(family, coef_samples[i], inferred_baseline_risk) 
        for i in eachindex(coef_samples)]
    end

    # Compute PAR samples
    par_samples = @. p_exposed * (rr_samples - 1.0) / rr_samples

    # Compute summary statistics
    low_prob = alpha / 2.0
    high_prob = 1.0 - low_prob

    par_mean = mean(par_samples)
    par_median = median(par_samples)
    par_std = std(par_samples)
    par_lower = quantile(par_samples, low_prob)
    par_upper = quantile(par_samples, high_prob)

    rr_mean = mean(rr_samples)
    rr_median = median(rr_samples)
    rr_ci_lower = quantile(rr_samples, low_prob)
    rr_ci_upper = quantile(rr_samples, high_prob)

    # Return scalar baseline_risk (mean if stochastic)
    baseline_risk_return = if inferred_baseline_risk isa Vector
        mean(inferred_baseline_risk)
    else
        inferred_baseline_risk
    end

    return (
        par_mean=par_mean,
        par_median=par_median,
        par_std=par_std,
        par_lower=par_lower,
        par_upper=par_upper,
        rr_mean=rr_mean,
        rr_median=rr_median,
        rr_ci_lower=rr_ci_lower,
        rr_ci_upper=rr_ci_upper,
        exposure_prevalence=p_exposed,
        baseline_risk=baseline_risk_return,
        reference_population=reference_population,
        covariate=covariate,
        family=family,
        n_samples=length(par_samples),
        raw_par_samples=par_samples,
        raw_rr_samples=rr_samples
    )
end

""" summarize_par_effects(chain; covariates, families, baseline_etas, baseline_risks, exposure_vars, exposure_prevalences, data, outcome_var, reference_population, alpha)::NamedTuple

Compute PAR for multiple covariates simultaneously, with family-specific configurations.

Arguments
chain: The MCMC chain object.
covariates::Vector{String}: Vector of covariate names to analyze.
families::Union{String, Dict}: Either a single family string applied to all covariates, or a Dict mapping covariate → family. Default: "poisson".
baseline_etas::Union{Dict, Nothing}: Dict mapping covariate → baseline_eta (binomial only). Default: nothing.
baseline_risks::Union{Dict, Nothing}: Dict mapping covariate → baseline_risk (binomial only). Default: nothing.
exposure_vars::Union{Dict, Vector, Nothing}: Dict mapping covariate → exposure_var column, or Vector parallel to covariates. Default: nothing.
exposure_prevalences::Union{Dict, Vector, Real, Nothing}: Dict mapping covariate → prevalence, or single scalar for all. Default: nothing.
data::Union{DataFrame, Nothing}: Training data. Default: nothing.
outcome_var::Union{String, Nothing}: Outcome column name. Default: nothing.
reference_population::String: "sample" or "external". Default: "sample".
alpha::Float64: Significance level. Default: 0.05.
Returns
A NamedTuple where each key is a covariate name, value is result from par_from_posterior().
Example
Julia
results = summarize_par_effects(
    chn,
    covariates=["smoking", "occupation", "diet"],
    families=Dict(
        "smoking" => "binomial",
        "occupation" => "poisson",
        "diet" => "negbin"
    ),
    baseline_risks=Dict("smoking" => 0.12),
    exposure_vars=Dict(
        "smoking" => "smoking_status",
        "occupation" => "occupational_exposure",
        "diet" => "high_fat_diet"
    ),
    exposure_prevalences=Dict(
        "smoking" => 0.25,
        "occupation" => 0.30,
        "diet" => 0.50
    ),
    data=df,
    outcome_var="disease"
)
""" 
function summarize_par_effects( 
    chain; covariates::Vector{String}, 
    families::Union{String, Dict}="poisson", 
    baseline_etas::Union{Dict, Nothing}=nothing, 
    baseline_risks::Union{Dict, Nothing}=nothing, 
    exposure_vars::Union{Dict, Vector, Nothing}=nothing, 
    exposure_prevalences::Union{Dict, Vector, Real, Nothing}=nothing, 
    data::Union{DataFrame, Nothing}=nothing, 
    outcome_var::Union{String, Nothing}=nothing, 
    reference_population::String="sample", alpha::Float64=0.05 )::NamedTuple


    results = Dict{Symbol, Any}()

    for (idx, cov) in enumerate(covariates)
        # Lookup family for this covariate
        fam = if families isa String
            families
        elseif families isa Dict
            get(families, cov, "poisson")
        else
            "poisson"
        end
        
        # Lookup baseline_eta
        eta_0 = if isnothing(baseline_etas)
            nothing
        elseif baseline_etas isa Dict
            get(baseline_etas, cov, nothing)
        else
            nothing
        end
        
        # Lookup baseline_risk
        risk_0 = if isnothing(baseline_risks)
            nothing
        elseif baseline_risks isa Dict
            get(baseline_risks, cov, nothing)
        else
            nothing
        end
        
        # Lookup exposure_var
        exp_var = if isnothing(exposure_vars)
            nothing
        elseif exposure_vars isa Dict
            get(exposure_vars, cov, nothing)
        elseif exposure_vars isa Vector
            if length(exposure_vars) == length(covariates)
                exposure_vars[idx]
            else
                nothing
            end
        else
            nothing
        end
        
        # Lookup exposure_prevalence
        p_exp = if isnothing(exposure_prevalences)
            nothing
        elseif exposure_prevalences isa Dict
            get(exposure_prevalences, cov, nothing)
        elseif exposure_prevalences isa Vector
            if length(exposure_prevalences) == length(covariates)
                exposure_prevalences[idx]
            else
                nothing
            end
        else  # Single scalar
            exposure_prevalences
        end
        
        par_res = par_from_posterior(
            chain, cov; 
            family=fam,
            baseline_eta=eta_0,
            baseline_risk=risk_0,
            exposure_var=exp_var,
            exposure_prevalence=p_exp,
            data=data,
            outcome_var=outcome_var,
            reference_population=reference_population,
            alpha=alpha
        )
        results[Symbol(cov)] = par_res
    end

    return NamedTuple(results)
end

""" 
    export_par_to_table(par_results::Union{Dict, NamedTuple}; include_raw_samples::Bool=false)::DataFrame

Export PAR results to a DataFrame for downstream analysis or saving.

Arguments
par_results::Union{Dict, NamedTuple}: Single PAR result or dict/NamedTuple of results.
include_raw_samples::Bool: If true, includes full posterior samples in output.
Returns
A DataFrame with one row per covariate containing PAR and RR/OR summaries.
Example
Julia
par_table = export_par_to_table(
    summarize_par_effects(chn, covariates=["smoking", "occupation"]),
    include_raw_samples=false
)

CSV.write("disease_par_results.csv", par_table)
display(par_table)
""" 
function export_par_to_table( par_results::Union{Dict, NamedTuple}; include_raw_samples::Bool=false )::DataFrame

    # Normalize input
    if par_results isa NamedTuple && !haskey(par_results, :covariate)
        results_dict = Dict("covariate" => par_results)
    elseif par_results isa NamedTuple
        results_dict = Dict(String(k) => v for (k, v) in pairs(par_results))
    else
        results_dict = par_results
    end

    rows = []
    for (k, res) in pairs(results_dict)
        if res isa NamedTuple && haskey(res, :par_mean)
            row = (
                covariate = res.covariate,
                family = res.family,
                reference_population = res.reference_population,
                par_mean = res.par_mean,
                par_median = res.par_median,
                par_std = res.par_std,
                par_lower = res.par_lower,
                par_upper = res.par_upper,
                rr_mean = res.rr_mean,
                rr_median = res.rr_median,
                rr_ci_lower = res.rr_ci_lower,
                rr_ci_upper = res.rr_ci_upper,
                exposure_prevalence = res.exposure_prevalence,
                baseline_risk = res.baseline_risk,
                n_samples = res.n_samples
            )
            
            if include_raw_samples
                push!(rows, merge(row, (
                    raw_par_samples=res.raw_par_samples,
                    raw_rr_samples=res.raw_rr_samples
                )))
            else
                push!(rows, row)
            end
        end
    end

    return if !isempty(rows)
        DataFrame(rows)
    else
        DataFrame()
    end
end



""" par_credible_interval_plot(par_result::NamedTuple; title="PAR Estimate")

Generate a simple visualization of PAR credible intervals (requires Plots.jl).

Arguments
par_result::NamedTuple: Result from par_from_posterior().
title::String: Plot title.
Returns
A Plots figure or nothing if Plots not available. 
""" 
function par_credible_interval_plot(par_result::NamedTuple; title="PAR Estimate") 
    try 
        
        y_pos = 1
        label_text = "$(par_result.covariate) ($(par_result.family), ref=$(par_result.reference_population))"
        
        p = plot(
            title=title,
            xlabel="Population Attributable Risk (PAR)",
            ylabel="",
            legend=:topright,
            size=(700, 300)
        )
        
        plot!(
            [par_result.par_lower, par_result.par_upper],
            [y_pos, y_pos],
            linewidth=3,
            color=:steelblue,
            label="95% CI"
        )
        scatter!(
            [par_result.par_mean],
            [y_pos],
            markersize=8,
            color=:darkblue,
            label="Posterior Mean",
            markerstrokewidth=0
        )
        
        yticks!([y_pos], [label_text])
        
            return p
        catch e @warn "Could not generate plot. Ensure Plots.jl is loaded. Error: $e" 
            return nothing 
        end 
end



""" 
    par_forest_plot(par_results::Union{Dict, NamedTuple}; title="Forest Plot: PAR by Covariate")

Generate a forest plot comparing PAR estimates across multiple covariates.

Arguments
par_results::Union{Dict, NamedTuple}: Dictionary or NamedTuple of covariate → PAR results.
title::String: Plot title.
Returns
A Plots figure or nothing if Plots not available.
Example
Julia
forest_plot = par_forest_plot(
    summarize_par_effects(chn, covariates=["smoking", "occupation"]),
    title="Disease PAR by Risk Factor"
)
""" 
function par_forest_plot(par_results::Union{Dict, NamedTuple}; title="Forest Plot: PAR by Covariate") 
    
    try 
        
 
        results_dict = if par_results isa NamedTuple
            Dict(String(k) => v for (k, v) in pairs(par_results))
        else
            par_results
        end
        
        valid_results = Dict(k => v for (k, v) in results_dict 
                            if v isa NamedTuple && !isnan(v.par_mean))
        
        if isempty(valid_results)
            @warn "No valid PAR results to plot."
            return nothing
        end
        
        cov_names = sort(collect(keys(valid_results)))
        n_cov = length(cov_names)
        
        means = [valid_results[c].par_mean for c in cov_names]
        lowers = [valid_results[c].par_lower for c in cov_names]
        uppers = [valid_results[c].par_upper for c in cov_names]
        
        p = plot(
            title=title,
            xlabel="Population Attributable Risk (PAR)",
            ylabel="",
            legend=false,
            size=(700, 300 + 50*n_cov)
        )
        
        for (i, cov) in enumerate(reverse(cov_names))
            y = n_cov + 1 - i
            plot!(
                [lowers[n_cov + 1 - i], uppers[n_cov + 1 - i]],
                [y, y],
                linewidth=2,
                color=:steelblue,
                label=""
            )
            scatter!(
                [means[n_cov + 1 - i]],
                [y],
                markersize=7,
                color=:darkblue,
                label="",
                markerstrokewidth=0
            )
        end
        
        yticks!(1:n_cov, reverse(cov_names))
        
        return p
    catch e
        @warn "Could not generate forest plot. Ensure Plots.jl is loaded. Error: $e"
        return nothing
    end
end