 
abstract type AbstractBSTM_Family end
struct PoissonFamily <: AbstractBSTM_Family end
struct GaussianFamily <: AbstractBSTM_Family end
struct LogNormalFamily <: AbstractBSTM_Family end
struct NegativeBinomialFamily <: AbstractBSTM_Family end
struct BinomialFamily <: AbstractBSTM_Family end
struct GammaFamily <: AbstractBSTM_Family end
struct ExponentialFamily <: AbstractBSTM_Family end
struct BetaFamily <: AbstractBSTM_Family end
struct InverseGaussianFamily <: AbstractBSTM_Family end
struct StudentTFamily <: AbstractBSTM_Family end
struct HalfNormalFamily <: AbstractBSTM_Family end
struct HalfStudentTFamily <: AbstractBSTM_Family end
struct LaplaceFamily <: AbstractBSTM_Family end
struct ParetoFamily <: AbstractBSTM_Family end
struct DirichletFamily <: AbstractBSTM_Family end
struct InverseWishartFamily <: AbstractBSTM_Family end
struct DirichletMultinomialFamily <: AbstractBSTM_Family end
struct OrdinalFamily <: AbstractBSTM_Family end

abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end


abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end
struct LeftCensored <: AbstractCensoringState end
struct RightCensored <: AbstractCensoringState end
struct IntervalCensored <: AbstractCensoringState end

 

# Version 1.5.1 (2026-08-06)
# Purpose: Defines the observation model for bstm.
# Rationale: This version corrects the type of the `trial` field to `Int`
#            as it represents a count for binomial models. It also ensures
#            that all numeric fields are correctly typed to prevent `Float64`
#            contamination during automatic differentiation.
struct bstm_Likelihood{F, Z, C, W, P, R, S, TR, TL, TU, HT, EX} <: ContinuousMultivariateDistribution
    family::F
    y_obs::TR
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    phi_hurdle::P
    r_nb::R
    sigma_y::S
    trial::Int # Corrected type to Int
    censor_lower::TL
    censor_upper::TU
    hurdle::HT
    extra_params::EX
end




Base.length(d::bstm_Likelihood) = length(d.y_obs)
Base.size(d::bstm_Likelihood) = (length(d.y_obs),)

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
function get_dist_ref(::PoissonFamily, d, eta::V, sig) where {V<:Real}
    return Poisson(exp(eta))
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
    mu = exp(eta)
    return NegativeBinomial(mu, V(d.r_nb))
end



function get_dist_ref(::BinomialFamily, d, eta::V, sig) where {V<:Real}
    n = d.trial isa AbstractVector ? d.trial[1] : d.trial
    return Binomial(Int(n), logistic(eta))
end



function get_dist_ref(::GammaFamily, d, eta, sig); alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0; return Gamma(alpha, clamp(exp(eta), 1e-9, 1e9)/alpha); end
function get_dist_ref(::ExponentialFamily, d, eta, sig); return Exponential(clamp(exp(eta), 1e-9, 1e9)); end
function get_dist_ref(::BetaFamily, d, eta, sig); mu = logistic(eta); phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0; return Beta(clamp(mu*phi, 1e-9, Inf), clamp((1.0-mu)*phi, 1e-9, Inf)); end
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
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)
    alpha_0 = max(sig, 1e-6)
    mean_probs = softmax(eta_vec)
    alpha_params = alpha_0 .* mean_probs
    n_total = sum(d.y_obs)
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
 

# Constructor handles the conversion of trait names to concrete trait types
# Version 1.5.1 (2026-08-06)
# Purpose: Constructor for bstm_Likelihood.
# Rationale: Ensures all numeric parameters are correctly typed to prevent
#            `Float64` contamination during automatic differentiation.
function bstm_Likelihood(family_input::Union{String, Symbol}, y_obs;
    zi_state=nothing, censoring_state=nothing, weight=1.0,
    phi_zi=-Inf, phi_hurdle=-Inf, r_nb=1.0, sigma_y=1.0, trial::Int=1, # trial is explicitly Int
    censor_lower=-Inf, censor_upper=Inf, hurdle=-Inf, extra_params=nothing
)
    f_trait = get_model_family(string(family_input))
    # Ensure all numeric values are promoted to a common type, typically Float64,
    # but will be Dual if any input is Dual.
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

    y_vec = y_obs isa AbstractVector ? y_obs : [y_obs]

    return bstm_Likelihood(f_trait, y_vec, zi_trait, censor_trait, promoted_type(weight), promoted_type(phi_zi), promoted_type(phi_hurdle), promoted_type(r_nb), promoted_type(sigma_y), trial, yL_val, yU_val, h_val, extra_params)
end

# Specialized logpdf using a where {V<:Real} clause to capture the Dual type during NUTS sampling
# Version 1.5.1 (2026-08-06)
# Purpose: Computes the log-probability for a vector of observations.
# Rationale: Initializes `logp` with `zero(V)` to ensure type stability.
function Distributions._logpdf(d::bstm_Likelihood, eta::AbstractVector{V}) where {V<:Real}
    # CRITICAL: Initialize logp with zero of type V to prevent Float64 conversion error
    logp = zero(V) 
    
    if d.family isa DirichletMultinomialFamily
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs) * w
    else
        for i in 1:length(eta)
            sig = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            w = d.weight isa AbstractVector ? d.weight[i] : d.weight
            # Kernel must also be generic over V
            logp += bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta[i], sig, d.y_obs[i]) * w
        end
        return logp
    end
end

# Version 1.5.1 (2026-08-06)
# Purpose: Public scalar overload for `logpdf`.
# Rationale: Provides a convenient interface for single-observation likelihood evaluation.
function Distributions.logpdf(d::bstm_Likelihood, eta::Real)
    # This method is for a single scalar observation. It is not used by the DirichletMultinomial path.
    # It is preserved for backward compatibility with existing univariate likelihoods.
    if d.family isa DirichletMultinomialFamily
        error("DirichletMultinomial likelihood requires a vector of linear predictors, but received a scalar.")
    end

    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, d.y_obs[1]) * w
end

# Version 1.5.1 (2026-08-06)
# Purpose: Public vector overload to maintain `MultivariateDistribution` compliance.
# Rationale: Delegates to the internal `_logpdf` implementation.
function Distributions.logpdf(d::bstm_Likelihood, y::AbstractVector{<:Real})
    return Distributions._logpdf(d, y)
end

# Version 1.5.1 (2026-08-06)
# Purpose: Computes the log-probability for an uncensored observation.
# Rationale: Ensures all intermediate values (log_phi, etc.) respect type V.
function bstm_kernel(fam::AbstractBSTM_Family, ::Uncensored, zero_inflated::AbstractZIState, d, eta::V, sig, y) where {V<:Real}
    dist = get_dist_ref(fam, d, eta, sig)

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi)) # Cast to V
        log_one_minus_phi = V(log1p(-d.phi_zi)) # Cast to V

        if y == 0.0
            if is_discrete_family(fam)
                return logsumexp(log_phi, log_one_minus_phi + logpdf(dist, zero(V)))
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
            return logsumexp(log_phi, log_one_minus_phi + lp_base)
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
        return logsumexp(log_one_minus_phi, log_phi + log_prob_in_interval_given_hurdle)
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
            logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return log1mexp(log_p_le_L)

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
            logsumexp(log_phi, log_one_minus_phi + logcdf(dist, upper_bound))
        end

        log_p_le_L = if lower_bound < V(0.0) # Cast 0.0 to V
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
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

