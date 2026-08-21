"""
    likelihoods.jl

Observation models, likelihood evaluation kernels, censoring, and hurdle transformations
for Bayesian Spatio-Temporal Models (BSTM).

Version: v1.0.0
"""

"""
    bstm_Likelihood

Observation likelihood distribution parameterized by the linear predictor `param` (`eta`),
supporting zero-inflation, hurdles, left/right/interval censoring, and observation weights.
"""
struct bstm_Likelihood{
    F, Z, C, W, P, R, S, PR, TL, TU, HT, EX
} <: ContinuousMultivariateDistribution
    family::F
    param::PR
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

"""
    get_model_family(model_family::String)::AbstractBSTM_Family

Maps a string identifier to its corresponding concrete `AbstractBSTM_Family` singleton instance.
"""
function get_model_family(model_family::String)
    family_key = lowercase(strip(model_family))
    if haskey(BSTM_FAMILY_REGISTRY, family_key)
        return BSTM_FAMILY_REGISTRY[family_key]
    else
        error("Unknown model_family: '$(model_family)'. " *
              "Supported families are: $(keys(BSTM_FAMILY_REGISTRY))")
    end
end

# --- Distribution Reference Generators ---

function get_dist_ref(::PoissonFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Poisson(1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Poisson(clamp(exp(eta_clamped), V(1e-9), V(1e9)))
end    

function get_dist_ref(::PoissonFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Poisson(1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Poisson(clamp(exp(eta_clamped), 1e-9, 1e9))
end

function get_dist_ref(::DirichletFamily, d, eta, sig)
    error("The Dirichlet likelihood is for compositional outcomes and is not supported " *
          "in the univariate framework.")
end

function get_dist_ref(::InverseWishartFamily, d, eta, sig)
    error("The Inverse-Wishart likelihood is for covariance matrix outcomes and is not " *
          "supported in the univariate framework.")
end

function get_dist_ref(::GaussianFamily, d, eta::V, sig::S) where {V<:Real, S<:Real}
    if isnan(eta) || isinf(eta)
        return Normal(0.0, 1.0)
    end
    return Normal(eta, V(sig) + V(1e-9))
end

function get_dist_ref(::LogNormalFamily, d, eta::V, sig::S) where {V<:Real, S<:Real}
    if isnan(eta) || isinf(eta)
        return LogNormal(0.0, 1.0)
    end
    mu = clamp(eta - (V(sig)^2) / V(2.0), V(-30.0), V(30.0))
    return LogNormal(mu, V(sig) + V(1e-9))
end

function get_dist_ref(::NegativeBinomialFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return NegativeBinomial(1, 0.5)
    end
    r = V(d.r_nb)
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    mu = clamp(exp(eta_clamped), V(1e-9), V(1e9))
    p = clamp(r / (r + mu), V(1e-12), V(1.0 - 1e-12))
    return NegativeBinomial(r, p)
end

function get_dist_ref(::BinomialFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Binomial(1, 0.5)
    end
    n = d.trial isa AbstractVector ? d.trial[1] : d.trial
    return Binomial(Int(n), LogExpFunctions.logistic(eta))
end

function get_dist_ref(::GammaFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Gamma(1.0, 1.0)
    end
    alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Gamma(alpha, clamp(exp(eta_clamped), V(1e-9), V(1e9)) / alpha)
end

function get_dist_ref(::GammaFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Gamma(1.0, 1.0)
    end
    alpha = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Gamma(alpha, clamp(exp(eta_clamped), 1e-9, 1e9) / alpha)
end

function get_dist_ref(::ExponentialFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Exponential(1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Exponential(clamp(exp(eta_clamped), V(1e-9), V(1e9)))
end

function get_dist_ref(::ExponentialFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Exponential(1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Exponential(clamp(exp(eta_clamped), 1e-9, 1e9))
end

function get_dist_ref(::BetaFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Beta(1.0, 1.0)
    end
    mu = LogExpFunctions.logistic(eta)
    phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0
    return Beta(clamp(mu * phi, V(1e-9), V(1e9)), clamp((V(1.0) - mu) * phi, V(1e-9), V(1e9)))
end

function get_dist_ref(::BetaFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return Beta(1.0, 1.0)
    end
    mu = LogExpFunctions.logistic(eta)
    phi = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 10.0
    return Beta(clamp(mu * phi, 1e-9, 1e9), clamp((1.0 - mu) * phi, 1e-9, 1e9))
end

function get_dist_ref(::InverseGaussianFamily, d, eta::V, sig) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return InverseGaussian(1.0, 1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    mu = clamp(exp(eta_clamped), V(1e-9), V(1e9))
    lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    return InverseGaussian(mu, lambda)
end

function get_dist_ref(::InverseGaussianFamily, d, eta, sig)
    if isnan(eta) || isinf(eta)
        return InverseGaussian(1.0, 1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    mu = clamp(exp(eta_clamped), 1e-9, 1e9)
    lambda = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 1.0
    return InverseGaussian(mu, lambda)
end

function get_dist_ref(::StudentTFamily, d, eta, sig)
    nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0
    loc = isnan(eta) || isinf(eta) ? 0.0 : eta
    return LocationScale(loc, max(sig, 1e-9), TDist(nu))
end

function get_dist_ref(::HalfNormalFamily, d, eta, sig)
    return truncated(Normal(0.0, max(sig, 1e-9)), 0.0, Inf)
end

function get_dist_ref(::HalfStudentTFamily, d, eta, sig)
    nu = d.extra_params isa Number && d.extra_params > 0 ? d.extra_params : 5.0
    return truncated(LocationScale(0.0, max(sig, 1e-9), TDist(nu)), 0.0, Inf)
end

function get_dist_ref(::LaplaceFamily, d, eta, sig)
    loc = isnan(eta) || isinf(eta) ? 0.0 : eta
    return Laplace(loc, max(sig, 1e-9))
end

function get_dist_ref(::ParetoFamily, d, eta, sig)
    shape = d.extra_params isa Number && d.extra_params > 1.0 ? d.extra_params : 1.1
    eta_clamped = clamp(eta, -30.0, 30.0)
    mean_val = clamp(exp(eta_clamped), 1e-9, 1e9)
    scale = mean_val * (shape - 1.0) / shape
    return Pareto(shape, scale)
end




"""
    get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)

Constructs a `DirichletMultinomial` distribution instance parameterized by `d.trial` and
  `softmax(eta_vec)`.
"""
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig)
    alpha_0 = max(sig, 1e-6)
    mean_probs = NNlib.softmax(eta_vec)
    alpha_params = alpha_0 .* mean_probs
    n_total = d.trial
    return DirichletMultinomial(Int(n_total), alpha_params)
end

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily})
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    return false
end

function bstm_kernel(
    fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec
)
    dist = get_dist_ref(fam, d, eta_vec, sig)
    return logpdf(dist, y_vec)
end

"""
    bstm_Likelihood(family_input, param; kwargs...)

Constructs a `bstm_Likelihood` distribution wrapper parameterized by linear predictor `param`
(`eta`), incorporating censoring, zero-inflation, hurdles, dispersion, and observation weights.
"""
function bstm_Likelihood(
    family_input::Union{String, Symbol},
    param;
    zi_state = nothing,
    censoring_state = nothing,
    weight = 1.0,
    phi_zi = -Inf,
    phi_hurdle = -Inf,
    r_nb = 1.0,
    sigma_y = 1.0,
    trial::Int = 1,
    censor_lower = -Inf,
    censor_upper = Inf,
    hurdle = -Inf,
    extra_params = nothing
)
    f_trait = get_model_family(string(family_input))
    
    extract_scalar(x) = x isa AbstractArray ? (isempty(x) ? -Inf : x[1]) : x
    w_s = extract_scalar(weight)
    pzi_s = extract_scalar(phi_zi)
    phu_s = extract_scalar(phi_hurdle)
    rnb_s = extract_scalar(r_nb)
    sig_s = extract_scalar(sigma_y)
    cl_s = extract_scalar(censor_lower)
    cu_s = extract_scalar(censor_upper)
    hu_s = extract_scalar(hurdle)

    promoted_type = promote_type(
        typeof(w_s), typeof(pzi_s), typeof(phu_s), typeof(rnb_s),
        typeof(sig_s), typeof(cl_s), typeof(cu_s), typeof(hu_s)
    )

    h_val = isnothing(hu_s) ? promoted_type(-Inf) : promoted_type(hu_s)
    zi_trait = promoted_type(pzi_s) > promoted_type(-Inf) ? ZeroInflated() : NonZeroInflated()
    yL_val = isnothing(cl_s) ? promoted_type(-Inf) : promoted_type(cl_s)
    yU_val = isnothing(cu_s) ? promoted_type(Inf) : promoted_type(cu_s)

    censor_trait = if !isfinite(yL_val) && !isfinite(yU_val)
        Uncensored()
    elseif isfinite(yL_val) && !isfinite(yU_val)
        RightCensored()
    elseif !isfinite(yL_val) && isfinite(yU_val)
        LeftCensored()
    else 
        IntervalCensored() 
    end

    param_vec = param isa AbstractVector ? param : [param]

    return bstm_Likelihood(
        f_trait, param_vec, zi_trait, censor_trait,
        promoted_type(w_s), promoted_type(pzi_s), promoted_type(phu_s),
        promoted_type(rnb_s), promoted_type(sig_s), trial,
        yL_val, yU_val, h_val, extra_params
    )
end

function Distributions._logpdf(d::bstm_Likelihood, y::AbstractVector{V}) where {V<:Real}
    logp = zero(V) 
    
    if d.family isa DirichletMultinomialFamily
        eta = d.param
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y) * w
    else
        for i in 1:length(y)
            eta_i = d.param isa AbstractVector ? d.param[i] : d.param
            sig_i = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            w_i = d.weight isa AbstractVector ? d.weight[i] : d.weight
            k_val = bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta_i, sig_i, y[i])
            logp += k_val * w_i
        end
        return logp
    end
end

function Distributions.logpdf(d::bstm_Likelihood, y::Real)
    if d.family isa DirichletMultinomialFamily
        error("DirichletMultinomial likelihood requires a vector of observations, " *
              "but received a scalar.")
    end
    
    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y) * w
end

function Distributions.logpdf(d::bstm_Likelihood, y::AbstractVector{<:Real})
    return Distributions._logpdf(d, y)
end

function Base.rand(rng::Random.AbstractRNG, d::bstm_Likelihood)
    if d.family isa DirichletMultinomialFamily
        dist = get_dist_ref(d.family, d, d.param, d.sigma_y)
        return rand(rng, dist)
    end

    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y

    # Zero-inflation draw
    if d.zi_state isa ZeroInflated
        if rand(rng) < d.phi_zi
            return zero(Float64)
        end
    end

    # Hurdle draw
    if d.phi_hurdle > -Inf
        if rand(rng) < (1.0 - d.phi_hurdle)
            return Float64(d.hurdle)
        end
    end

    dist = get_dist_ref(d.family, d, eta, sig)
    raw_draw = Float64(rand(rng, dist))

    # Censoring bounds
    if d.censoring_state isa LeftCensored
        u_b = d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper
        return min(raw_draw, Float64(u_b))
    elseif d.censoring_state isa RightCensored
        l_b = d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower
        return max(raw_draw, Float64(l_b))
    elseif d.censoring_state isa IntervalCensored
        l_b = d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower
        u_b = d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper
        return clamp(raw_draw, Float64(l_b), Float64(u_b))
    end

    return raw_draw
end

Base.rand(d::bstm_Likelihood) = rand(Random.default_rng(), d)

function Distributions._rand!(rng::Random.AbstractRNG, d::bstm_Likelihood, x::AbstractArray)
    for i in eachindex(x)
        x[i] = rand(rng, d)
    end
    return x
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::Uncensored, zero_inflated::AbstractZIState, d, eta::V, sig, y
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    dist = get_dist_ref(fam, d, eta, sig)

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi))
        log_one_minus_phi = V(log1p(-d.phi_zi))

        if y == zero(V)
            if is_discrete_family(fam)
                lp0 = log_one_minus_phi + logpdf(dist, zero(V))
                return LogExpFunctions.logsumexp(log_phi, lp0)
            else
                return log_phi
            end
        else
            return log_one_minus_phi + logpdf(dist, V(y))
        end
    elseif d.phi_hurdle > V(-Inf)
        log_phi = V(log(d.phi_hurdle))
        log_one_minus_phi = V(log1p(-d.phi_hurdle))

        if y <= V(d.hurdle)
            return log_one_minus_phi
        else
            logp_truncated = logpdf(dist, V(y)) - logccdf(dist, V(d.hurdle))
            return log_phi + logp_truncated
        end
    else
        return logpdf(dist, V(y))
    end
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::LeftCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    dist = get_dist_ref(fam, d, eta, sig)
    upper_bound = V(d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper)

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi))
        log_one_minus_phi = V(log1p(-d.phi_zi))
        lp_base = logcdf(dist, upper_bound)
        if upper_bound >= V(0.0)
            return LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            return log_one_minus_phi + lp_base
        end
    elseif d.phi_hurdle > V(-Inf)
        log_phi = V(log(d.phi_hurdle))
        log_one_minus_phi = V(log1p(-d.phi_hurdle))
        if upper_bound <= V(d.hurdle)
            return log_one_minus_phi
        end
        log_prob_interval = _stable_logsubexp(
            logcdf(dist, upper_bound), logcdf(dist, V(d.hurdle))
        ) - logccdf(dist, V(d.hurdle))
        return LogExpFunctions.logsumexp(log_one_minus_phi, log_phi + log_prob_interval)
    else
        return logcdf(dist, upper_bound)
    end
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::RightCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = V(d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower)
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi))
        log_one_minus_phi = V(log1p(-d.phi_zi))
        
        log_p_le_L = if lower_bound < V(0.0)
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return LogExpFunctions.log1mexp(log_p_le_L)

    elseif d.phi_hurdle > V(-Inf)
        log_phi = V(log(d.phi_hurdle))
        adj_hurdle = is_discrete_family(fam) ? V(d.hurdle) - V(1.0) : V(d.hurdle)

        if lower_bound > V(d.hurdle)
            return log_phi + logccdf(dist, adj_L) - logccdf(dist, adj_hurdle)
        else
            return log_phi
        end
    else
        return logccdf(dist, adj_L)
    end
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::IntervalCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    dist = get_dist_ref(fam, d, eta, sig)
    lower_bound = V(d.censor_lower isa AbstractVector ? d.censor_lower[1] : d.censor_lower)
    upper_bound = V(d.censor_upper isa AbstractVector ? d.censor_upper[1] : d.censor_upper)
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound

    if zero_inflated isa ZeroInflated
        log_phi = V(log(d.phi_zi))
        log_one_minus_phi = V(log1p(-d.phi_zi))

        log_p_le_U = if upper_bound < V(0.0)
            log_one_minus_phi + logcdf(dist, upper_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, upper_bound))
        end

        log_p_le_L = if lower_bound < V(0.0)
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return _stable_logsubexp(log_p_le_U, log_p_le_L)

    elseif d.phi_hurdle > V(-Inf)
        log_phi = V(log(d.phi_hurdle))
        adj_hurdle = is_discrete_family(fam) ? V(d.hurdle) - V(1.0) : V(d.hurdle)

        if upper_bound <= V(d.hurdle)
            return V(-Inf)
        end

        effective_lower = max(adj_L, adj_hurdle)
        log_prob_in_interval = _stable_logsubexp(
            logcdf(dist, upper_bound), logcdf(dist, effective_lower)
        )
        log_normalizer = logccdf(dist, adj_hurdle)
        return log_phi + log_prob_in_interval - log_normalizer
    else
        return _stable_logsubexp(logcdf(dist, upper_bound), logcdf(dist, adj_L))
    end
end

"""
    _stable_logsubexp(a::Real, b::Real)

Computes `log(exp(a) - exp(b))` in a numerically stable manner via `a + log1mexp(b - a)`.
Returns `-Inf` if `a <= b`.
"""
function _stable_logsubexp(a::Real, b::Real)
    if a <= b
        return -Inf
    end
    return a + LogExpFunctions.log1mexp(b - a)
end
