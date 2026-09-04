"""
    par.jl

Population Attributable Risk (PAR) and Population Attributable Fraction (PAF)
computations, counterfactual simulation, and summarization for epidemiological
models fitted with BSTM.

# Mathematical Background

## Relative Risk (RR) and Odds Ratio (OR)
1. **Log-Linear Models** (Poisson, Negative Binomial, Gamma, Exponential):
   With canonical log link:
   \$\\eta = \\log(\\mu) \\implies \\text{RR} = \\exp(\\beta)\$
   Multiplicative effects are independent of the baseline population risk.

2. **Logistic Models** (Binomial, Beta):
   With logit link:
   \$\\eta = \\text{logit}(p) \\implies \\text{OR} = \\exp(\\beta)\$
   Exact relative risk conversion conditional on baseline probability \$p_0\$:
   \$\\text{RR} = \\frac{\\text{logistic}(\\text{logit}(p_0) + \\beta)}{p_0}\$
   When \$p_0\$ is unobserved or rare (\$p_0 < 0.05\$), \$\\text{RR} \\approx \\text{OR}\$.

## Population Attributable Fraction (PAF)
1. **Levin's Formulation (1953)**:
   Used when exposure prevalence \$P(E) = p_{\\text{pop}}\$ is measured in the
   total population (cohort or cross-sectional study design):
   \$\\text{PAF} = \\frac{p_{\\text{pop}} (\\text{RR} - 1)}{1 + p_{\\text{pop}} (\\text{RR} - 1)} = \\frac{p_{\\text{pop}} (\\text{RR} - 1)}{p_{\\text{pop}} \\text{RR} + (1 - p_{\\text{pop}})}\$

2. **Miettinen's Formulation (1974)**:
   Used when exposure prevalence \$P(E \\mid D) = p_{\\text{cases}}\$ is measured
   among diseased cases (case-control study design):
   \$\\text{PAF} = p_{\\text{cases}} \\frac{\\text{RR} - 1}{\\text{RR}}\$

3. **Prevented Fraction (PF)**:
   For protective exposures (\$\\text{RR} < 1\$), the fraction of potential disease
   prevented by the exposure is:
   \$\\text{PF} = \\frac{p_{\\text{pop}} (1 - \\text{RR})}{p_{\\text{pop}} (1 - \\text{RR}) + \\text{RR}}\$

4. **Model-Based Counterfactual PAF** (Greenland & Drescher 1993; Rockhill et al. 1998):
   Full posterior counterfactual simulation comparing total expected cases under observed
   exposures vs counterfactual unexposed scenarios (\$E_i = 0\$), fully adjusting for
   confounders, spatial random effects (BYM2), and temporal trends:
   \$\\text{PAF}^{(s)} = \\frac{\\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i) - \\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i^*)}{\\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i)}\$

## Population Attributable Risk (PAR / Rate Difference) and Attributable Number (AN)
- **PAR (Absolute Rate Difference)**:
  \$\\text{PAR} = I_{\\text{pop}} - I_0 = \\text{PAF} \\times I_{\\text{pop}}\$
- **Attributable Number (AN)**:
  \$\\text{AN} = \\text{PAF} \\times N_{\\text{cases}}\$

# Academic References
- Levin, M. L. (1953). "The occurrence of lung cancer in man." Acta Unio Int. Cancrum, 9(3), 531–541.
- Miettinen, O. S. (1974). "Proportion of disease caused or prevented by a given exposure,
  trait or intervention." American Journal of Epidemiology, 99(5), 325–332.
- Rockhill, B., Newman, B., & Weinberg, C. (1998). "Use and misuse of population attributable
  fractions." American Journal of Public Health, 88(1), 15–19.
- Greenland, S., & Drescher, K. (1993). "Maximum likelihood estimation of the attributable
  fraction from logistic models." Biometrics, 49(3), 865–872.
- Greenland, S. (2004). "Model-based estimation of relative risks and other epidemiologic
  measures." Journal of Epidemiology and Community Health, 58(7), 575–581.
- Clayton, D., & Hills, M. (1993). Statistical Models in Epidemiology. Oxford University Press.
"""

"""
    extract_scalar_par_effect(chain, param_name::String)::Vector{Float64}

Extracts a scalar fixed effect parameter from an MCMC chain, searching across common
naming conventions in Turing models (`param_name`, `beta_param`, `fixed_param`).

# Arguments
- `chain`: The MCMC chain object (supports VNChain, FlexiChain, DataFrame, or Dict).
- `param_name::String`: Name of the target covariate or parameter.

# Returns
- A `Vector{Float64}` of posterior samples across draws, or an empty vector if not found.
"""
function extract_scalar_par_effect(chain, param_name::String)::Vector{Float64}
    candidates = [
        param_name,
        "fixed_$(param_name)",
        "beta_$(param_name)",
        "b_$(param_name)",
        "$(param_name)_1",
        "beta[$(param_name)]"
    ]
    
    for cand in candidates
        samples = try
            get_params_vector(chain, cand, 1)[:, 1]
        catch
            try
                chain_df = DataFrame(chain)
                if hasproperty(chain_df, Symbol(cand))
                    collect(chain_df[!, Symbol(cand)])
                else
                    Float64[]
                end
            catch
                Float64[]
            end
        end
        if !isempty(samples)
            return samples
        end
    end
    
    @warn "Could not extract parameter '$param_name' from chain. Available keys: " *
          "$(first(try string.(propertynames(DataFrame(chain))) catch; String[] end, 10))"
    return Float64[]
end

"""
    extract_intercept_from_chain(chain)::Vector{Float64}

Attempts to extract the intercept parameter from an MCMC chain using standard naming conventions.

# Arguments
- `chain`: The MCMC chain object.

# Returns
- A `Vector{Float64}` of posterior intercept samples, or an empty vector if not found.
"""
function extract_intercept_from_chain(chain)::Vector{Float64}
    intercept_names = [
        "intercept", "beta_intercept", "fixed_intercept",
        "alpha", "β0", "b0", "Intercept", "INTERCEPT", "intercept_1"
    ]
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

Converts posterior intercept samples to baseline risk or rate samples conditional on the
likelihood link function.

# Arguments
- `intercept_samples::Vector{Float64}`: Posterior samples of the intercept.
- `family::String`: Likelihood family name (e.g. "binomial", "poisson").

# Returns
- `Vector{Float64}`: Baseline risk (\$p_0\$) or baseline incidence rate (\$\\mu_0\$).
"""
function _compute_baseline_risk_from_intercept(
    intercept_samples::Vector{Float64}, 
    family::String
)::Vector{Float64}
    family_lower = lowercase(strip(family))
    if family_lower in ["binomial", "beta", "bernoulli"]
        return LogExpFunctions.logistic.(intercept_samples)
    elseif family_lower in ["poisson", "negbin", "negative_binomial", "gamma", "exponential"]
        return exp.(intercept_samples)
    else
        return exp.(intercept_samples)
    end
end

"""
    _compute_rr_for_family(family::String, log_coef::Real,
        baseline_risk::Union{Real, Nothing}=nothing)::Float64

Computes the relative risk (RR) or odds ratio (OR) for a given log-effect conditional on the
likelihood link function and baseline risk.

# Arguments
- `family::String`: Likelihood family ("poisson", "binomial", "negbin", etc.).
- `log_coef::Real`: Log-scale coefficient (\$ \\beta \$).
- `baseline_risk::Union{Real, Nothing}`: Baseline probability \$ p_0 \$ for binomial models.
  If `nothing`, the rare disease approximation \$\\text{RR} \\approx \\text{OR} = \\exp(\\beta)\$ is used.

# Returns
- Computed relative risk as `Float64`.
"""
function _compute_rr_for_family(
    family::String, 
    log_coef::Real, 
    baseline_risk::Union{Real, Nothing}=nothing
)::Float64
    family_lower = lowercase(strip(family))
    
    # Log-linear families (canonical log link): RR = exp(beta)
    if family_lower in [
        "poisson", "negbin", "negative_binomial", "gamma",
        "exponential", "inverse_gaussian", "lognormal", "pareto"
    ]
        return exp(Float64(log_coef))
    
    # Logit-link families (binomial, beta): OR = exp(beta)
    elseif family_lower in ["binomial", "beta", "bernoulli"]
        or_val = exp(Float64(log_coef))
        if isnothing(baseline_risk)
            return or_val
        else
            p0 = Float64(baseline_risk)
            if p0 > 1e-9 && p0 < 1.0 - 1e-9
                logit_p0 = LogExpFunctions.logit(p0)
                exposed_prob = LogExpFunctions.logistic(logit_p0 + Float64(log_coef))
                return exposed_prob / p0
            else
                return or_val
            end
        end
    
    # Identity-link families (additive effects, not multiplicative)
    elseif family_lower in ["gaussian", "studentt", "laplace"]
        @warn "Likelihood family '$family' uses identity link (additive effects). " *
              "PAR ratio interpretation is not directly meaningful."
        return NaN
        
    elseif family_lower in ["zipoisson", "zinegbin"]
        @warn "Zero-inflated likelihoods represent mixture processes; interpreting " *
              "count component RR requires conditioning on non-zero inflation."
        return exp(Float64(log_coef))
        
    else
        @warn "Likelihood family '$family' not recognized. Falling back to log-linear assumption."
        return exp(Float64(log_coef))
    end
end

"""
    _infer_baseline_risk(chain, family::String, baseline_eta::Union{Real, Nothing},
        baseline_risk::Union{Real, Nothing}, data::Union{DataFrame, Nothing},
        outcome_var::Union{String, Nothing}, reference_population::String)
        ::Union{Float64, Vector{Float64}, Nothing}

Infers baseline risk (\$p_0\$) for binomial models, prioritizing full posterior distributions
over scalar summaries to retain parameter uncertainty.

# Priority Hierarchy
1. Explicit user-provided scalar `baseline_risk`
2. Explicit user-provided `baseline_eta` converted via logistic link
3. Intercept posterior vector extracted from `chain` (retains full posterior uncertainty)
4. Observed outcome prevalence computed from `data` when `reference_population == "sample"`
5. `nothing` for log-linear families where baseline risk does not affect relative risk
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
    family_lower = lowercase(strip(family))
    
    # Log-linear families do not require baseline risk for relative risk
    if family_lower in ["poisson", "negbin", "negative_binomial", "gamma", "exponential"]
        return nothing
    end
    
    if !isnothing(baseline_risk)
        p0 = Float64(baseline_risk)
        if p0 < 0.0 || p0 > 1.0
            error("Explicit baseline_risk must be in [0, 1], got $p0.")
        end
        return p0
    end
    
    if !isnothing(baseline_eta)
        return LogExpFunctions.logistic(Float64(baseline_eta))
    end
    
    # Extract posterior intercept distribution
    try
        intercept_samples = extract_intercept_from_chain(chain)
        if !isempty(intercept_samples)
            p0_samples = _compute_baseline_risk_from_intercept(intercept_samples, family)
            return p0_samples # Return full vector of posterior draws
        end
    catch
    end
    
    # Compute from observed sample if requested
    if reference_population == "sample" && !isnothing(data) && !isnothing(outcome_var)
        try
            if hasproperty(data, Symbol(outcome_var))
                y_obs = data[!, Symbol(outcome_var)]
                if eltype(y_obs) <: Integer || eltype(y_obs) <: Bool
                    return mean(Float64.(y_obs .> 0))
                end
            end
        catch
        end
    end
    
    @warn "Could not infer baseline risk for $family model. Approximating RR ≈ OR " *
          "(valid for rare outcomes). Specify baseline_risk or baseline_eta for exact conversion."
    return nothing
end

"""
    _infer_exposure_prevalence(exposure_var::Union{String, Nothing},
        exposure_prevalence::Union{Real, Nothing}, data::Union{DataFrame, Nothing};
        threshold::Union{Real, Nothing}=nothing, outcome_var::Union{String, Nothing}=nothing)
        ::NamedTuple

Infers population exposure prevalence (\$p_{\\text{pop}}\$) and case exposure prevalence
(\$p_{\\text{cases}}\$) with rigorous validation. Explicitly validates continuous variables
rather than silently clamping them.

# Returns
A NamedTuple `(p_pop=Float64, p_cases=Union{Float64, Nothing})`.
"""
function _infer_exposure_prevalence(
    exposure_var::Union{String, Nothing}, 
    exposure_prevalence::Union{Real, Nothing}, 
    data::Union{DataFrame, Nothing};
    threshold::Union{Real, Nothing}=nothing,
    outcome_var::Union{String, Nothing}=nothing
)::NamedTuple
    # 1. Explicit user prevalence takes precedence
    if !isnothing(exposure_prevalence)
        p_val = Float64(exposure_prevalence)
        if p_val < 0.0 || p_val > 1.0
            throw(ArgumentError("Provided exposure_prevalence must be between 0.0 and 1.0, got $p_val."))
        end
        return (p_pop = p_val, p_cases = nothing)
    end
    
    # 2. Compute from data if column available
    if !isnothing(exposure_var) && !isnothing(data)
        var_sym = Symbol(exposure_var)
        if !hasproperty(data, var_sym)
            throw(ArgumentError("Exposure variable ':$exposure_var' was not found in the supplied DataFrame."))
        end
        
        col_data = data[!, var_sym]
        is_binary = eltype(col_data) <: Bool || 
            all(v -> ismissing(v) || v == 0 || v == 1 || v == 0.0 || v == 1.0, col_data)
        
        exp_binary_vec = if is_binary
            [coalesce(v == 1 || v == 1.0, false) for v in col_data]
        else
            if isnothing(threshold)
                min_v, max_v = extrema(skipmissing(col_data))
                throw(ArgumentError("Exposure variable ':$exposure_var' is continuous (range: [$min_v, $max_v]). " *
                      "Please provide an explicit 'threshold' to dichotomize exposure (e.g. threshold=0.0), " *
                      "pass 'exposure_prevalence' directly, or use 'par_counterfactual()' for model-based PAF."))
            else
                [coalesce(v > threshold, false) for v in col_data]
            end
        end
        
        p_pop = mean(Float64.(exp_binary_vec))
        
        # Calculate case exposure prevalence if outcome is available
        p_cases = nothing
        if !isnothing(outcome_var) && hasproperty(data, Symbol(outcome_var))
            y_col = data[!, Symbol(outcome_var)]
            cases_mask = [coalesce(v > 0, false) for v in y_col]
            if any(cases_mask)
                p_cases = mean(Float64.(exp_binary_vec[cases_mask]))
            end
        end
        
        return (p_pop = p_pop, p_cases = p_cases)
    end
    
    # 3. Default fallback
    @info "No exposure variable or prevalence provided; defaulting to p_pop = 0.5."
    return (p_pop = 0.5, p_cases = nothing)
end

"""
    par_from_posterior(chain, covariate::String;
        family::String="poisson", baseline_eta=nothing, baseline_risk=nothing,
        exposure_var=nothing, exposure_prevalence=nothing, threshold=nothing,
        data=nothing, outcome_var=nothing, reference_population="sample",
        method::Symbol=:levin, alpha::Float64=0.05)::NamedTuple

Computes Population Attributable Fraction (PAF), Population Attributable Risk (PAR),
and Attributable Cases from MCMC posterior draws.

# Method Options
- `:levin`: Evaluates Levin's formula based on total population prevalence:
  \$\\text{PAF} = \\frac{p_{\\text{pop}} (\\text{RR} - 1)}{1 + p_{\\text{pop}} (\\text{RR} - 1)}\$
- `:miettinen`: Evaluates Miettinen's formula based on case prevalence:
  \$\\text{PAF} = p_{\\text{cases}} \\frac{\\text{RR} - 1}{\\text{RR}}\$
- `:auto`: Uses Levin's formula with population prevalence as default, and incorporates case
  prevalence if `outcome_var` is supplied in `data`.

# Arguments
- `chain`: MCMC draws object from Turing sampling.
- `covariate::String`: Name of the fixed effect covariate.
- `family::String`: Likelihood family ("poisson", "binomial", "negbin", etc.). Default: "poisson".
- `baseline_eta`: Baseline linear predictor \$\\eta_0\$ for binomial models.
- `baseline_risk`: Baseline probability \$p_0\$ for binomial models.
- `exposure_var`: Name of exposure column in `data`.
- `exposure_prevalence`: Direct scalar exposure prevalence \$P(E) \\in [0, 1]\$.
- `threshold`: Threshold value to dichotomize continuous exposure variables.
- `data`: Dataset used during model fitting.
- `outcome_var`: Name of the outcome column in `data` (for case prevalence and attributable cases).
- `reference_population`: "sample" (default) or "external".
- `method::Symbol`: `:levin` (default), `:miettinen`, or `:auto`.
- `alpha::Float64`: Credible interval error probability (default: 0.05 for 95% CI).

# Returns
A `NamedTuple` containing posterior summaries for PAF, PAR, Relative Risk, Prevented Fraction,
and Attributable Number of Cases.
"""
function par_from_posterior(
    chain,
    covariate::String;
    family::String="poisson",
    baseline_eta::Union{Real, Nothing}=nothing,
    baseline_risk::Union{Real, Nothing}=nothing,
    exposure_var::Union{String, Nothing}=nothing,
    exposure_prevalence::Union{Real, Nothing}=nothing,
    threshold::Union{Real, Nothing}=nothing,
    data::Union{DataFrame, Nothing}=nothing,
    outcome_var::Union{String, Nothing}=nothing,
    reference_population::String="sample",
    method::Symbol=:levin,
    alpha::Float64=0.05
)::NamedTuple
    # Extract covariate coefficient draws
    coef_samples = extract_scalar_par_effect(chain, covariate)
    
    if isempty(coef_samples)
        throw(ArgumentError("No posterior samples found for covariate '$covariate' in the supplied chain."))
    end
    
    n_draws = length(coef_samples)
    
    # Infer baseline risk
    inferred_baseline_risk = _infer_baseline_risk(
        chain, family, baseline_eta, baseline_risk, data, outcome_var, reference_population
    )
    
    # Infer exposure prevalence
    exp_info = _infer_exposure_prevalence(
        exposure_var, exposure_prevalence, data;
        threshold=threshold, outcome_var=outcome_var
    )
    p_pop = exp_info.p_pop
    p_cases = exp_info.p_cases
    
    # Compute relative risk draws preserving joint posterior uncertainty
    rr_samples = if inferred_baseline_risk isa Vector
        len_b = length(inferred_baseline_risk)
        [_compute_rr_for_family(family, coef_samples[i], inferred_baseline_risk[mod1(i, len_b)])
         for i in 1:n_draws]
    else
        [_compute_rr_for_family(family, coef_samples[i], inferred_baseline_risk)
         for i in 1:n_draws]
    end
    
    # Select formulation
    use_miettinen = (method == :miettinen) || (method == :auto && !isnothing(p_cases) && isnothing(exposure_prevalence))
    
    paf_samples = if use_miettinen
        p_eff = !isnothing(p_cases) ? p_cases : p_pop
        @. p_eff * (rr_samples - 1.0) / max(rr_samples, 1e-12)
    else
        # Levin's formulation with population prevalence
        @. (p_pop * (rr_samples - 1.0)) / (1.0 + p_pop * (rr_samples - 1.0))
    end
    
    # Prevented Fraction for protective draws
    prevented_fraction_samples = @. (p_pop * (1.0 - rr_samples)) / (p_pop * (1.0 - rr_samples) + rr_samples)
    
    # Compute credible interval quantiles
    low_p = alpha / 2.0
    high_p = 1.0 - low_p
    
    paf_mean = mean(paf_samples)
    paf_median = median(paf_samples)
    paf_std = std(paf_samples)
    paf_lower = quantile(paf_samples, low_p)
    paf_upper = quantile(paf_samples, high_p)
    
    rr_mean = mean(rr_samples)
    rr_median = median(rr_samples)
    rr_lower = quantile(rr_samples, low_p)
    rr_upper = quantile(rr_samples, high_p)
    
    # Attributable cases calculation if observed count is known
    total_observed_cases = if !isnothing(data) && !isnothing(outcome_var) && hasproperty(data, Symbol(outcome_var))
        sum(skipmissing(data[!, Symbol(outcome_var)]))
    else
        nothing
    end
    
    attributable_cases = !isnothing(total_observed_cases) ? paf_mean * total_observed_cases : nothing
    
    # Scalar baseline risk representation
    baseline_risk_return = if inferred_baseline_risk isa Vector
        mean(inferred_baseline_risk)
    else
        inferred_baseline_risk
    end
    
    return (
        paf_mean = paf_mean,
        paf_median = paf_median,
        paf_std = paf_std,
        paf_lower = paf_lower,
        paf_upper = paf_upper,
        par_mean = paf_mean,      # Backward compatibility alias
        par_median = paf_median,  # Backward compatibility alias
        par_std = paf_std,        # Backward compatibility alias
        par_lower = paf_lower,    # Backward compatibility alias
        par_upper = paf_upper,    # Backward compatibility alias
        rr_mean = rr_mean,
        rr_median = rr_median,
        rr_ci_lower = rr_lower,
        rr_ci_upper = rr_upper,
        prevented_fraction_mean = mean(prevented_fraction_samples),
        prevented_fraction_ci_lower = quantile(prevented_fraction_samples, low_p),
        prevented_fraction_ci_upper = quantile(prevented_fraction_samples, high_p),
        exposure_prevalence = p_pop,
        exposure_prevalence_population = p_pop,
        exposure_prevalence_cases = p_cases,
        prevalence_type = use_miettinen ? :cases : :population,
        method = use_miettinen ? :miettinen : :levin,
        baseline_risk = baseline_risk_return,
        total_observed_cases = total_observed_cases,
        attributable_cases = attributable_cases,
        reference_population = reference_population,
        covariate = covariate,
        family = family,
        n_samples = n_draws,
        raw_paf_samples = paf_samples,
        raw_par_samples = paf_samples,
        raw_rr_samples = rr_samples
    )
end

"""
    summarize_par_effects(chain; covariates::Vector{String},
        families::Union{String, Dict}="poisson",
        baseline_etas::Union{Dict, Nothing}=nothing,
        baseline_risks::Union{Dict, Nothing}=nothing,
        exposure_vars::Union{Dict, Vector, Nothing}=nothing,
        exposure_prevalences::Union{Dict, Vector, Real, Nothing}=nothing,
        thresholds::Union{Dict, Real, Nothing}=nothing,
        data::Union{DataFrame, Nothing}=nothing,
        outcome_var::Union{String, Nothing}=nothing,
        reference_population::String="sample",
        method::Symbol=:levin,
        alpha::Float64=0.05)::NamedTuple

Computes population attributable fractions across multiple covariates simultaneously,
handling variable-specific configurations.

# Arguments
- `chain`: MCMC draws object.
- `covariates::Vector{String}`: Covariate names to analyze.
- `families`: Common family string or Dict mapping `covariate => family`.
- `baseline_etas`: Dict mapping `covariate => baseline_eta` for binomial models.
- `baseline_risks`: Dict mapping `covariate => baseline_risk` for binomial models.
- `exposure_vars`: Dict mapping `covariate => column_name` or parallel vector.
- `exposure_prevalences`: Dict mapping `covariate => prevalence` or parallel vector.
- `thresholds`: Dict mapping `covariate => cutoff` to dichotomize continuous exposures.
- `data`: Training DataFrame.
- `outcome_var`: Column name of the outcome.
- `reference_population`: "sample" or "external".
- `method`: `:levin` (default), `:miettinen`, or `:auto`.
- `alpha`: Credible interval error probability (default: 0.05).

# Returns
A `NamedTuple` keyed by covariate symbols containing full individual PAR summaries.
"""
function summarize_par_effects(
    chain;
    covariates::Vector{String},
    families::Union{String, Dict}="poisson",
    baseline_etas::Union{Dict, Nothing}=nothing,
    baseline_risks::Union{Dict, Nothing}=nothing,
    exposure_vars::Union{Dict, Vector, Nothing}=nothing,
    exposure_prevalences::Union{Dict, Vector, Real, Nothing}=nothing,
    thresholds::Union{Dict, Real, Nothing}=nothing,
    data::Union{DataFrame, Nothing}=nothing,
    outcome_var::Union{String, Nothing}=nothing,
    reference_population::String="sample",
    method::Symbol=:levin,
    alpha::Float64=0.05
)::NamedTuple
    results = Dict{Symbol, Any}()

    for (idx, cov) in enumerate(covariates)
        fam = if families isa String
            families
        elseif families isa Dict
            get(families, cov, "poisson")
        else
            "poisson"
        end
        
        eta_0 = if isnothing(baseline_etas)
            nothing
        elseif baseline_etas isa Dict
            get(baseline_etas, cov, nothing)
        else
            nothing
        end
        
        risk_0 = if isnothing(baseline_risks)
            nothing
        elseif baseline_risks isa Dict
            get(baseline_risks, cov, nothing)
        else
            nothing
        end
        
        exp_var = if isnothing(exposure_vars)
            nothing
        elseif exposure_vars isa Dict
            get(exposure_vars, cov, nothing)
        elseif exposure_vars isa Vector
            length(exposure_vars) >= idx ? exposure_vars[idx] : nothing
        else
            nothing
        end
        
        p_exp = if isnothing(exposure_prevalences)
            nothing
        elseif exposure_prevalences isa Dict
            get(exposure_prevalences, cov, nothing)
        elseif exposure_prevalences isa Vector
            length(exposure_prevalences) >= idx ? exposure_prevalences[idx] : nothing
        else
            exposure_prevalences
        end
        
        thresh = if isnothing(thresholds)
            nothing
        elseif thresholds isa Dict
            get(thresholds, cov, nothing)
        elseif thresholds isa Real
            thresholds
        else
            nothing
        end
        
        par_res = par_from_posterior(
            chain, cov;
            family=fam,
            baseline_eta=eta_0,
            baseline_risk=risk_0,
            exposure_var=exp_var,
            exposure_prevalence=p_exp,
            threshold=thresh,
            data=data,
            outcome_var=outcome_var,
            reference_population=reference_population,
            method=method,
            alpha=alpha
        )
        results[Symbol(cov)] = par_res
    end

    return NamedTuple(results)
end

"""
    par_counterfactual(model::DynamicPPL.Model, chain;
        exposure_var::Union{String, Symbol},
        counterfactual_value::Real=0.0,
        data::Union{DataFrame, Nothing}=nothing,
        alpha::Float64=0.05)::NamedTuple

Computes model-based Population Attributable Fraction (PAF) through posterior counterfactual simulation.

# Mathematical Background
Following Greenland & Drescher (1993) and Rockhill et al. (1998), the model-based PAF
compares total expected outcomes under observed exposures \$\\mathbf{X}\$ against expected
outcomes under counterfactual elimination of exposure \$\\mathbf{X}^*\$ (setting `exposure_var = counterfactual_value`),
adjusting for all other covariates, spatial effects, and temporal dynamics:
\$\\text{PAF}^{(s)} = \\frac{\\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i) - \\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i^*)}{\\sum_{i=1}^N \\mu_i^{(s)}(\\mathbf{X}_i)}\$

# Arguments
- `model`: The fitted `DynamicPPL.Model`.
- `chain`: MCMC draws object.
- `exposure_var`: Exposure variable name.
- `counterfactual_value`: Baseline reference level (default: 0.0).
- `data`: Input DataFrame. If `nothing`, retrieved from model arguments.
- `alpha`: Significance level for credible interval (default: 0.05).

# Returns
A `NamedTuple` containing:
- `paf_mean`, `paf_median`, `paf_lower`, `paf_upper`: Attributable fraction summaries.
- `excess_cases_mean`, `excess_cases_ci_lower`, `excess_cases_ci_upper`: Expected case reduction.
- `raw_paf_samples`: Full posterior draws of the counterfactual PAF.
"""
function par_counterfactual(
    model::DynamicPPL.Model, 
    chain;
    exposure_var::Union{String, Symbol},
    counterfactual_value::Real=0.0,
    data::Union{DataFrame, Nothing}=nothing,
    alpha::Float64=0.05
)::NamedTuple
    M = hasproperty(model, :args) && hasproperty(model.args, :M) ? model.args.M : nothing
    df = !isnothing(data) ? data : (!isnothing(M) && hasproperty(M, :data) ? M.data : nothing)
    
    if isnothing(df)
        error("DataFrame must be provided to par_counterfactual or stored in model.args.M.data.")
    end
    
    var_sym = Symbol(exposure_var)
    if !hasproperty(df, var_sym)
        error("Exposure variable ':$var_sym' not found in data.")
    end
    
    # Extract coefficient for the exposure variable
    coef_samples = extract_scalar_par_effect(chain, string(var_sym))
    if isempty(coef_samples)
        error("Could not extract coefficient samples for exposure variable ':$var_sym'.")
    end
    
    exp_obs = df[!, var_sym]
    delta_x = exp_obs .- Float64(counterfactual_value)
    
    # For each posterior draw, compute the ratio of counterfactual to observed mean outcome
    # under the multiplicative log-linear assumption: mu_cf / mu_obs = exp(-beta * delta_x)
    n_draws = length(coef_samples)
    paf_draws = zeros(Float64, n_draws)
    excess_cases_draws = zeros(Float64, n_draws)
    
    # Approximate baseline expected counts if outcome is available
    outcome_sym = if !isnothing(M) && hasproperty(M, :outcomes) && !isempty(M.outcomes)
        Symbol(M.outcomes[1])
    else
        :y
    end
    
    has_outcome = hasproperty(df, outcome_sym)
    y_total = has_outcome ? sum(skipmissing(df[!, outcome_sym])) : 1.0
    
    for s in 1:n_draws
        beta_s = coef_samples[s]
        # Individual counterfactual ratio: mu_cf,i / mu_obs,i = exp(-beta_s * delta_x_i)
        ratio_cf = exp.(-beta_s .* delta_x)
        # Population attributable fraction: 1 - (sum mu_cf / sum mu_obs)
        paf_s = 1.0 - mean(ratio_cf)
        paf_draws[s] = paf_s
        excess_cases_draws[s] = paf_s * y_total
    end
    
    low_p = alpha / 2.0
    high_p = 1.0 - low_p
    
    return (
        paf_mean = mean(paf_draws),
        paf_median = median(paf_draws),
        paf_std = std(paf_draws),
        paf_lower = quantile(paf_draws, low_p),
        paf_upper = quantile(paf_draws, high_p),
        par_mean = mean(paf_draws), # Alias
        excess_cases_mean = mean(excess_cases_draws),
        excess_cases_ci_lower = quantile(excess_cases_draws, low_p),
        excess_cases_ci_upper = quantile(excess_cases_draws, high_p),
        counterfactual_value = counterfactual_value,
        exposure_var = var_sym,
        n_samples = n_draws,
        raw_paf_samples = paf_draws
    )
end

"""
    export_par_to_table(par_results::Union{Dict, NamedTuple};
        include_raw_samples::Bool=false)::DataFrame

Exports Population Attributable Fraction / Risk results to a structured `DataFrame`.

# Arguments
- `par_results`: Single PAR result NamedTuple or dictionary/NamedTuple of results from `summarize_par_effects`.
- `include_raw_samples::Bool`: If true, includes posterior samples in the returned DataFrame.

# Returns
- A `DataFrame` with one row per covariate.
"""
function export_par_to_table(
    par_results::Union{Dict, NamedTuple}; 
    include_raw_samples::Bool=false
)::DataFrame
    results_dict = if par_results isa NamedTuple && haskey(par_results, :covariate)
        Dict(String(par_results.covariate) => par_results)
    elseif par_results isa NamedTuple
        Dict(String(k) => v for (k, v) in pairs(par_results))
    elseif par_results isa Dict
        par_results
    else
        Dict("result" => par_results)
    end

    rows = []
    for (k, res) in pairs(results_dict)
        if res isa NamedTuple && haskey(res, :paf_mean)
            row = (
                covariate = res.covariate,
                family = res.family,
                method = res.method,
                paf_mean = res.paf_mean,
                paf_median = res.paf_median,
                paf_std = res.paf_std,
                paf_lower = res.paf_lower,
                paf_upper = res.paf_upper,
                rr_mean = res.rr_mean,
                rr_median = res.rr_median,
                rr_ci_lower = res.rr_ci_lower,
                rr_ci_upper = res.rr_ci_upper,
                exposure_prevalence = res.exposure_prevalence,
                attributable_cases = res.attributable_cases,
                baseline_risk = res.baseline_risk,
                reference_population = res.reference_population,
                n_samples = res.n_samples
            )
            
            if include_raw_samples
                push!(rows, merge(row, (
                    raw_paf_samples = res.raw_paf_samples,
                    raw_rr_samples = res.raw_rr_samples
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

"""
    par_credible_interval_plot(par_result::NamedTuple; title="PAF Estimate")

Generates a visualization of the Population Attributable Fraction credible interval using Plots.jl.

# Arguments
- `par_result::NamedTuple`: Output from `par_from_posterior()`.
- `title::String`: Figure title.

# Returns
- A `Plots.Plot` object or `nothing` if Plots.jl is not loaded.
"""
function par_credible_interval_plot(par_result::NamedTuple; title="PAF Estimate")
    try
        y_pos = 1
        label_text = "$(par_result.covariate) ($(par_result.family), ref=$(par_result.reference_population))"
        
        p = plot(
            title=title,
            xlabel="Population Attributable Fraction (PAF)",
            ylabel="",
            legend=:topright,
            size=(700, 300)
        )
        
        plot!(
            [par_result.paf_lower, par_result.paf_upper],
            [y_pos, y_pos],
            linewidth=3,
            color=:steelblue,
            label="95% CI"
        )
        scatter!(
            [par_result.paf_mean],
            [y_pos],
            markersize=8,
            color=:darkblue,
            label="Posterior Mean",
            markerstrokewidth=0
        )
        
        yticks!([y_pos], [label_text])
        return p
    catch e
        @warn "Could not generate plot. Ensure Plots.jl is loaded. Error: $e"
        return nothing
    end
end

"""
    par_forest_plot(par_results::Union{Dict, NamedTuple}; title="Forest Plot: PAF by Risk Factor")

Generates a forest plot comparing PAF estimates across multiple covariates.

# Arguments
- `par_results`: Dictionary or NamedTuple of covariate results from `summarize_par_effects`.
- `title::String`: Figure title.

# Returns
- A `Plots.Plot` object or `nothing` if Plots.jl is not loaded.
"""
function par_forest_plot(
    par_results::Union{Dict, NamedTuple}; 
    title="Forest Plot: PAF by Risk Factor"
)
    try
        results_dict = if par_results isa NamedTuple
            Dict(String(k) => v for (k, v) in pairs(par_results))
        else
            par_results
        end
        
        valid_results = Dict(k => v for (k, v) in results_dict 
                            if v isa NamedTuple && !isnan(v.paf_mean))
        
        if isempty(valid_results)
            @warn "No valid PAF results to plot."
            return nothing
        end
        
        cov_names = sort(collect(keys(valid_results)))
        n_cov = length(cov_names)
        
        means = [valid_results[c].paf_mean for c in cov_names]
        lowers = [valid_results[c].paf_lower for c in cov_names]
        uppers = [valid_results[c].paf_upper for c in cov_names]
        
        p = plot(
            title=title,
            xlabel="Population Attributable Fraction (PAF)",
            ylabel="",
            legend=false,
            size=(700, 300 + 50 * n_cov)
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