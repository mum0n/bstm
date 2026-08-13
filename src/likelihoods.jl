
"""
    bstm_Likelihood

Defines the observation model for `bstm`, parameterized by the linear predictor `eta`.

# Rationale for Update
This version refactors the struct to be parameterized by the linear predictor `eta`
instead of the observed data `y`. The `y_obs` field is renamed to `param` to reflect
this change. This is a critical fix to align with the standard `Distributions.jl` API
and enable correct gradient calculations for AD-based samplers. The `Base.length` and
`Base.size` methods have also been updated to use `d.param`, resolving a `MethodError`
introduced by the field rename.
"""
struct bstm_Likelihood{F, Z, C, W, P, R, S, PR, TL, TU, HT, EX} <: ContinuousMultivariateDistribution
    family::F
    param::PR # Renamed from y_obs; now holds the linear predictor `eta`.
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    phi_hurdle::P
    r_nb::R
    sigma_y::S
    trial::Int
    censor_lower::TL
    censor_upper::TU
    hurdle::HT
    extra_params::EX
end

Base.length(d::bstm_Likelihood) = length(d.param)
Base.size(d::bstm_Likelihood) = (length(d.param),)

function get_model_family(model_family::String)
    # Purpose: Maps a string identifier to its corresponding concrete `AbstractBSTM_Family` type.
    # Rationale: This version is updated to be more robust to parsing artifacts. It now strips
    #            leading/trailing whitespace (including newlines) from the input string before
    #            looking it up in the registry. This resolves an error where a family name like
    #            "binomial \n" would cause a lookup failure.
    # v1.0.1 (2026-08-02)
    # Inputs:
    #   - model_family: The string name of the family.
    # Outputs: An instance of a concrete subtype of `AbstractBSTM_Family`.
    family_key = lowercase(strip(model_family))
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$(model_family)'. Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end
 

# Distribution reference generators forced to promote inputs to V
function get_dist_ref(::PoissonFamily, d, eta, sig)    
    # Clamp the rate parameter lambda to a small positive value (1e-9) to avoid
    # numerical instability (log(0)) during gradient-based sampling when eta is very small.
    return Poisson(clamp(exp(eta), 1e-9, 1e9))
end    


function get_dist_ref(::DirichletFamily, d, eta, sig); error("The Dirichlet likelihood is for compositional outcomes and is not supported in the current univariate response framework."); end
function get_dist_ref(::InverseWishartFamily, d, eta, sig); error("The Inverse-Wishart likelihood is for covariance matrix outcomes and is not supported in the current univariate response framework."); end

function get_dist_ref(::GaussianFamily, d, eta::V, sig::S) where {V<:Real, S<:Real}
    return Normal(eta, V(sig) + V(1e-9))
end


function get_dist_ref(::LogNormalFamily, d, eta::V, sig::S) where {V<:Real, S<:Real}
    mu = eta - (V(sig)^2) / V(2.0)
    return LogNormal(mu, V(sig) + V(1e-9))
end


function get_dist_ref(::NegativeBinomialFamily, d, eta::V, sig) where {V<:Real}
    r = V(d.r_nb)
    mu = clamp(exp(eta), V(1e-9), V(1e9))
    p = clamp(r / (r + mu), V(1e-12), V(1.0 - 1e-12))
    return NegativeBinomial(r, p)
end



function get_dist_ref(::BinomialFamily, d, eta::V, sig) where {V<:Real}
    n = d.trial isa AbstractVector ? d.trial[1] : d.trial
    return Binomial(Int(n), LogExpFunctions.logistic(eta))
end



function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, clamp(exp(eta), 1e-9, 1e9)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = LogExpFunctions.logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(clamp(mu*phi, 1e-9, Inf), clamp((1.0-mu)*phi, 1e-9, Inf)); end
function get_dist_ref(::InverseGaussianFamily, d, eta, sig); mu = clamp(exp(eta), 1e-9, 1e9); lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return InverseGaussian(mu, lambda); end
function get_dist_ref(::StudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return LocationScale(eta, max(sig, 1e-9), TDist(nu)); end
function get_dist_ref(::HalfNormalFamily, d, eta, sig); return truncated(Normal(0.0, max(sig, 1e-9)), 0.0, Inf); end
function get_dist_ref(::HalfStudentTFamily, d, eta, sig); nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0; return truncated(LocationScale(0.0, max(sig, 1e-9), TDist(nu)), 0.0, Inf); end
function get_dist_ref(::LaplaceFamily, d, eta, sig); return Laplace(eta, max(sig, 1e-9)); end
function get_dist_ref(::ParetoFamily, d, eta, sig)
    shape = d.extra_params isa Number && d.extra_params > 1.0 ? d.extra_params : 1.1
    mean_val = clamp(exp(eta), 1e-9, 1e9)
    scale = mean_val * (shape - 1.0) / shape
    return Pareto(shape, scale)
end



# Version 1.5.2 (2026-08-06)
# Purpose: Creates a DirichletMultinomial distribution instance.
# Rationale: This version is updated to use `d.trial` to get the total number of trials,
#            making it consistent with the Binomial family and decoupling it from the
#            observation data `y_obs`, which is no longer stored in the distribution object.
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)
    alpha_0 = max(sig, 1e-6)
    mean_probs = NNlib.softmax(eta_vec)
    alpha_params = alpha_0 .* mean_probs
    # The total number of trials is now passed via the `trial` keyword argument.
    n_total = d.trial
    return DirichletMultinomial(Int(n_total), alpha_params)
end

# # Add trait for Dirichlet family

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end
function is_discrete_family(::AbstractBSTM_Family)
    return false
end
 

function bstm_kernel(fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec)
    dist = get_dist_ref(fam, d, eta_vec, sig)
    return logpdf(dist, y_vec)
end
 

# Version 1.5.2 (2026-08-06)
# Purpose: Constructor for bstm_Likelihood.
# Rationale: This version is updated to accept the linear predictor parameter `param`
#            instead of the observation `y_obs`. This aligns the constructor with the
#            refactored struct definition and the correct `logpdf` evaluation flow.
function bstm_Likelihood(family_input::Union{String, Symbol}, param;
    zi_state=nothing, censoring_state=nothing, weight=1.0,
    phi_zi=-Inf, phi_hurdle=-Inf, r_nb=1.0, sigma_y=1.0, trial::Int=1,
    censor_lower=-Inf, censor_upper=Inf, hurdle=-Inf, extra_params=nothing
)
    # This function constructs the bstm_Likelihood object, which acts as a wrapper
    # for various distributions and handles censoring, zero-inflation, and hurdles.
    f_trait = get_model_family(string(family_input))
    
    # Promote all numeric parameters to a common type to ensure type stability.
    # This will become `ForwardDiff.Dual` if any input parameter is a Dual number.
    promoted_type = promote_type(typeof(weight), typeof(phi_zi), typeof(phi_hurdle), typeof(r_nb), typeof(sigma_y), typeof(censor_lower), typeof(censor_upper), typeof(hurdle))

    h_val = isnothing(hurdle) ? promoted_type(-Inf) : promoted_type(hurdle)
    zi_trait = promoted_type(phi_zi) > promoted_type(-Inf) ? ZeroInflated() : NonZeroInflated()
    yL_val = isnothing(censor_lower) ? promoted_type(-Inf) : promoted_type(censor_lower)
    yU_val = isnothing(censor_upper) ? promoted_type(Inf) : promoted_type(censor_upper)

    censor_trait = if !isfinite(yL_val) && !isfinite(yU_val)
        Uncensored()
    elseif isfinite(yL_val) && !isfinite(yU_val)
        RightCensored()
    elseif !isfinite(yL_val) && isfinite(yU_val)
        LeftCensored()
    else 
        IntervalCensored() 
    end

    # Ensure the parameter `param` (eta) is stored as a vector.
    param_vec = param isa AbstractVector ? param : [param]

    return bstm_Likelihood(f_trait, param_vec, zi_trait, censor_trait, promoted_type(weight), promoted_type(phi_zi), promoted_type(phi_hurdle), promoted_type(r_nb), promoted_type(sigma_y), trial, yL_val, yU_val, h_val, extra_params)
end

# Version 1.5.2 (2026-08-06)
# Purpose: Computes the log-probability for a vector of observations.
# Rationale: The signature is changed to `_logpdf(d, y)` to conform to the standard API.
#            It now extracts the linear predictor `eta` from `d.param` and evaluates the
#            log-pdf at the given observation vector `y`. This is only used for vector-
#            valued likelihoods like DirichletMultinomial.
function Distributions._logpdf(d::bstm_Likelihood, y::AbstractVector{V}) where {V<:Real}
    logp = zero(V) 
    
    if d.family isa DirichletMultinomialFamily
        eta = d.param
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y) * w
    else
        # This path is not typically used for univariate models which loop outside.
        for i in 1:length(y)
            eta_i = d.param isa AbstractVector ? d.param[i] : d.param
            sig_i = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            w_i = d.weight isa AbstractVector ? d.weight[i] : d.weight
            logp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta_i, sig_i, y[i]) * w_i
        end
        return logp
    end
end


# Version 1.5.2 (2026-08-06)
# Purpose: Public scalar overload for `logpdf`.
# Rationale: The signature is changed to `logpdf(d, y)` to conform to the standard API.
#            It now extracts the linear predictor `eta` from `d.param` and evaluates the
#            log-pdf at the given observation `y`.
function Distributions.logpdf(d::bstm_Likelihood, y::Real)
    # This method handles the logpdf evaluation for a single scalar observation.
    if d.family isa DirichletMultinomialFamily
        error("DirichletMultinomial likelihood requires a vector of observations, but received a scalar.")
    end
    
    # Extract the linear predictor `eta` from the distribution object.
    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    
    # Delegate to the appropriate kernel based on family, censoring, and zero-inflation traits.
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y) * w
end

# Version 1.5.2 (2026-08-06)
# Purpose: Public vector overload to maintain `MultivariateDistribution` compliance.
# Rationale: Delegates to the internal `_logpdf` implementation.
function Distributions.logpdf(d::bstm_Likelihood, y::AbstractVector{<:Real})
    return Distributions._logpdf(d, y)
end


# Version 1.5.3 (2026-08-06)
# Purpose: Computes the log-probability for an uncensored observation.
# Rationale: Ensures all intermediate values (log_phi, etc.) respect type V.
#            The comparison `y == 0.0` is changed to `y == zero(V)` for type stability.
function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zero_inflated::AbstractZIState, d, eta::V, sig, y) where {V<:Real}
    dist = get_dist_ref(fam, d, eta, sig)

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_zi)) # Cast to V

        if y == zero(V) # Changed from y == 0.0
            if is_discrete_family(fam)
                return LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logpdf(dist, zero(V)))
            else
                return log_phi
            end
        else
            return log_one_minus_phi + logpdf(dist, V(y))
        end
    elseif d.phi_hurdle > V(-Inf) # Cast -Inf to V
        log_phi = V(log(d.phi_hurdle)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_hurdle)) # Cast to V

        if y <= V(d.hurdle) # Cast d.hurdle to V
            return log_one_minus_phi
        else
            logp_truncated = logpdf(dist, V(y)) - logccdf(dist, V(d.hurdle))
            return log_phi + logp_truncated
        end
    else
        return logpdf(dist, V(y))
    end
end


# Version 1.5.1 (2026-08-06)
# Purpose: Computes the log-probability for a left-censored observation.
# Rationale: Correctly calculates the cumulative probability for standard, ZI, and hurdle models.
function bstm_kernel(fam::AbstractBSTM_Family, ::LeftCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y) where {V<:Real}
    dist = get_dist_ref(fam, d, eta, sig)
    upper_bound = V(d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper) # Cast to V

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_zi)) # Cast to V
        lp_base = logcdf(dist, upper_bound)
        if upper_bound >= V(0.0) # Cast 0.0 to V
            return LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            return log_one_minus_phi + lp_base
        end
    elseif d.phi_hurdle > V(-Inf) # Cast -Inf to V
        log_phi = V(log(d.phi_hurdle)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_hurdle)) # Cast to V
        if upper_bound <= V(d.hurdle) # Cast d.hurdle to V
            return log_one_minus_phi
        end
        log_prob_in_interval_given_hurdle = _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, V(d.hurdle))) - logccdf(dist, V(d.hurdle)) # Cast d.hurdle to V
        return LogExpFunctions.logsumexp(log_one_minus_phi, log_phi + log_prob_in_interval_given_hurdle)
    else
        return logcdf(dist, upper_bound)
    end
end

# Version 1.5.1 (2026-08-06)
# Purpose: Computes the log-probability for a right-censored observation.
# Rationale: Correctly calculates the complementary cumulative probability for all model types.
function bstm_kernel(fam::AbstractBSTM_Family, ::RightCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y) where {V<:Real}
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = V(d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower) # Cast to V
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound # Cast 1.0 to V

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_zi)) # Cast to V
        
        log_p_le_L = if lower_bound < V(0.0) # Cast 0.0 to V
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return LogExpFunctions.log1mexp(log_p_le_L)

    elseif d.phi_hurdle > V(-Inf) # Cast -Inf to V
        log_phi = V(log(d.phi_hurdle)) # Cast to V
        adj_hurdle = is_discrete_family(fam) ? V(d.hurdle) - V(1.0) : V(d.hurdle) # Cast d.hurdle and 1.0 to V

        if lower_bound > V(d.hurdle) # Cast d.hurdle to V
            return log_phi + logccdf(dist, adj_L) - logccdf(dist, adj_hurdle)
        else
            return log_phi
        end
    else
        return logccdf(dist, adj_L)
    end
end

# Version 1.5.1 (2026-08-06)
# Purpose: Computes the log-probability for an interval-censored observation.
# Rationale: Calculates the probability mass within the interval [censor_lower, censor_upper].
function bstm_kernel(fam::AbstractBSTM_Family, ::IntervalCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y) where {V<:Real}
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = V(d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower) # Cast to V
    upper_bound = V(d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper) # Cast to V
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound # Cast 1.0 to V

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_zi)) # Cast to V

        log_p_le_U = if upper_bound < V(0.0) # Cast 0.0 to V
            log_one_minus_phi + logcdf(dist, upper_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, upper_bound))
        end

        log_p_le_L = if lower_bound < V(0.0) # Cast 0.0 to V
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return _stable_logsubexp(log_p_le_U, log_p_le_L)

    elseif d.phi_hurdle > V(-Inf) # Cast -Inf to V
        log_phi = V(log(d.phi_hurdle)) # Cast to V
        adj_hurdle = is_discrete_family(fam) ? V(d.hurdle) - V(1.0) : V(d.hurdle) # Cast d.hurdle and 1.0 to V

        if upper_bound <= V(d.hurdle) # Cast d.hurdle to V
            return V(-Inf) # Cast -Inf to V
        end

        effective_lower = max(adj_L, adj_hurdle)
        log_prob_in_interval = _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, effective_lower))
        log_normalizer = logccdf(dist, adj_hurdle)
        return log_phi + log_prob_in_interval - log_normalizer
    else
        return _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, adj_L))
    end
end
