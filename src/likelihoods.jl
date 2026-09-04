"""
    likelihoods.jl

Observation models, likelihood evaluation kernels, censoring, and hurdle transformations
for Bayesian Spatio-Temporal Models (BSTM).

Version: v1.0.0
"""


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
struct MultinomialFamily <: AbstractBSTM_Family end
struct CategoricalFamily <: AbstractBSTM_Family end
struct DirichletMultinomialFamily <: AbstractBSTM_Family end
struct OrdinalFamily <: AbstractBSTM_Family end
struct CategoricalMovementFamily <: AbstractBSTM_Family end
struct MvNormalFamily <: AbstractBSTM_Family end

abstract type AbstractZIState end
struct NonZeroInflated <: AbstractZIState end
struct ZeroInflated <: AbstractZIState end


abstract type AbstractCensoringState end
struct Uncensored <: AbstractCensoringState end
struct LeftCensored <: AbstractCensoringState end
struct RightCensored <: AbstractCensoringState end
struct IntervalCensored <: AbstractCensoringState end


const BSTM_FAMILY_REGISTRY = Dict{String, AbstractBSTM_Family}(
    "poisson"               => PoissonFamily(),
    "gaussian"              => GaussianFamily(),
    "lognormal"             => LogNormalFamily(),
    "bernoulli"             => BinomialFamily(),
    "binomial"              => BinomialFamily(),
    "negbin"                => NegativeBinomialFamily(),
    "gamma"                 => GammaFamily(),
    "exponential"           => ExponentialFamily(),
    "beta"                  => BetaFamily(),
    "inverse_gaussian"      => InverseGaussianFamily(),
    "student_t"             => StudentTFamily(),
    "half_normal"           => HalfNormalFamily(),
    "half_student_t"        => HalfStudentTFamily(),
    "laplace"               => LaplaceFamily(),
    "pareto"                => ParetoFamily(),
    "dirichlet"             => DirichletFamily(),
    "inverse_wishart"       => InverseWishartFamily(),
    "multinomial"           => MultinomialFamily(),
    "categorical"           => CategoricalFamily(),
    "dirichlet_multinomial" => DirichletMultinomialFamily(),
    "ordinal"               => OrdinalFamily(),
    "categorical_movement"  => CategoricalMovementFamily(),
    "mvnormal"              => MvNormalFamily(),
    "multivariate_normal"   => MvNormalFamily()
)

const STATSMODELS_CONTRASTS = Dict(
    :dummy => StatsModels.DummyCoding(),
    :effects => StatsModels.EffectsCoding(),
    :helmert => StatsModels.HelmertCoding(),
    :treatment => StatsModels.DummyCoding()
)




"""
    bstm_Likelihood

Observation likelihood distribution parameterized by the linear predictor `param` (`eta`),
supporting zero-inflation, hurdles, left/right/interval censoring, and observation weights.
"""
struct bstm_Likelihood{
    F, Z, C, W, P, PH, R, S, PR, TR, TL, TU, HT, EX
} <: ContinuousMultivariateDistribution
    family::F
    param::PR
    zi_state::Z
    censoring_state::C
    weight::W
    phi_zi::P
    phi_hurdle::PH
    r_nb::R
    sigma_y::S
    trial::TR
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

@inline _get_obs(v, i::Int) = v isa AbstractVector ? v[i] : v

# Generic fallback: distribution reference ignores obs_idx unless specialized
get_dist_ref(fam::AbstractBSTM_Family, d, eta, sig, obs_idx::Int) =
    get_dist_ref(fam, d, eta, sig)

function get_dist_ref(::PoissonFamily, d, eta::V, sig, obs_idx::Int=1) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Poisson(1.0)
    end
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    return Poisson(clamp(exp(eta_clamped), V(1e-9), V(1e9)))
end    

function get_dist_ref(::PoissonFamily, d, eta, sig, obs_idx::Int=1)
    if isnan(eta) || isinf(eta)
        return Poisson(1.0)
    end
    eta_clamped = clamp(eta, -30.0, 30.0)
    return Poisson(clamp(exp(eta_clamped), 1e-9, 1e9))
end

function get_dist_ref(::InverseWishartFamily, d, eta, sig, obs_idx::Int=1)
    error("The Inverse-Wishart likelihood is for covariance matrix outcomes and is not " *
          "supported in the univariate framework.")
end

function get_dist_ref(::GaussianFamily, d, eta::V, sig::S, obs_idx::Int=1) where {V<:Real, S<:Real}
    if isnan(eta) || isinf(eta)
        return Normal(0.0, 1.0)
    end
    sig_val = _get_obs(sig, obs_idx)
    return Normal(eta, V(sig_val) + V(1e-9))
end

function get_dist_ref(::LogNormalFamily, d, eta::V, sig::S, obs_idx::Int=1) where {V<:Real, S<:Real}
    if isnan(eta) || isinf(eta)
        return LogNormal(0.0, 1.0)
    end
    sig_val = _get_obs(sig, obs_idx)
    mu = clamp(eta - (V(sig_val)^2) / V(2.0), V(-30.0), V(30.0))
    return LogNormal(mu, V(sig_val) + V(1e-9))
end

function get_dist_ref(::NegativeBinomialFamily, d, eta::V, sig, obs_idx::Int=1) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return NegativeBinomial(1, 0.5)
    end
    r = V(_get_obs(d.r_nb, obs_idx))
    eta_clamped = clamp(eta, V(-30.0), V(30.0))
    mu = clamp(exp(eta_clamped), V(1e-9), V(1e9))
    p = clamp(r / (r + mu), V(1e-12), V(1.0 - 1e-12))
    return NegativeBinomial(r, p)
end

function get_dist_ref(::BinomialFamily, d, eta::V, sig, obs_idx::Int=1) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return Binomial(1, 0.5)
    end
    n = _get_obs(d.trial, obs_idx)
    return Binomial(Int(n), LogExpFunctions.logistic(eta))
end

function get_dist_ref(::BinomialFamily, d, eta, sig, obs_idx::Int=1)
    if isnan(eta) || isinf(eta)
        return Binomial(1, 0.5)
    end
    n = _get_obs(d.trial, obs_idx)
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
    get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `DirichletMultinomial` distribution instance parameterized by `d.trial` and
  `softmax(eta_vec)`.
"""
function get_dist_ref(::DirichletMultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)
    sig_val = _get_obs(sig, obs_idx)
    alpha_0 = max(sig_val, 1e-4)
    mean_probs = NNlib.softmax(eta_vec)
    alpha_params = max.(alpha_0 .* mean_probs, 1e-6)
    n_total = _get_obs(d.trial, obs_idx)
    return DirichletMultinomial(Int(n_total), alpha_params)
end

function bstm_kernel(
    fam::DirichletMultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

"""
    get_dist_ref(::MultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `Multinomial` distribution instance parameterized by `d.trial` and `softmax(eta_vec)`.
"""
function get_dist_ref(::MultinomialFamily, d, eta_vec, sig, obs_idx::Int=1)
    probs = NNlib.softmax(eta_vec)
    n_total = _get_obs(d.trial, obs_idx)
    return Multinomial(Int(n_total), probs)
end

function bstm_kernel(
    fam::MultinomialFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

"""
    get_dist_ref(::CategoricalFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `Categorical` distribution instance parameterized by `softmax(eta_vec)`.
"""
function get_dist_ref(::CategoricalFamily, d, eta_vec, sig, obs_idx::Int=1)
    probs = NNlib.softmax(eta_vec)
    return Categorical(probs)
end

function bstm_kernel(
    fam::CategoricalFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_val,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, Int(y_val))
end

"""
    get_dist_ref(::DirichletFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a `Dirichlet` distribution instance parameterized by `alpha = phi .* softmax(eta_vec)`.
"""
function get_dist_ref(::DirichletFamily, d, eta_vec, sig, obs_idx::Int=1)
    sig_val = _get_obs(sig, obs_idx)
    phi = max(sig_val, 1e-4)
    probs = NNlib.softmax(eta_vec)
    alpha_params = max.(phi .* probs, 1e-6)
    return Dirichlet(alpha_params)
end

function bstm_kernel(
    fam::DirichletFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

"""
    get_dist_ref(::MvNormalFamily, d, eta_vec, sig, obs_idx::Int=1)

Constructs a multivariate normal (`MvNormal`) distribution instance.
Mean vector is given by `eta_vec`. The covariance structure is resolved from
`d.extra_params[:cov]`, correlation Cholesky factor `d.extra_params[:L_corr]`,
matrix `sig`, or diagonal elements from vector/scalar `sig`.

# Mathematical Formulation
\$ Y \\sim \\text{MvNormal}(\\mu, \\Sigma) \$
where \$\\mu = \\eta\$ and \$\\Sigma = D(s) L L^T D(s)\$ or \$\\text{diag}(s^2)\$.
"""
function get_dist_ref(::MvNormalFamily, d, eta_vec, sig, obs_idx::Int=1)
    K = length(eta_vec)
    cov_mat = if !isnothing(d.extra_params) && haskey(d.extra_params, :cov)
        d.extra_params[:cov]
    elseif !isnothing(d.extra_params) && haskey(d.extra_params, :L_corr)
        L = d.extra_params[:L_corr]
        s = sig isa AbstractVector ? sig : fill(sig, K)
        Diagonal(s) * (L * L') * Diagonal(s)
    elseif sig isa AbstractMatrix
        sig
    elseif sig isa AbstractVector
        Diagonal(sig .^ 2)
    else
        Diagonal(fill(Float64(sig)^2, K))
    end
    return MvNormal(collect(eta_vec), Symmetric(cov_mat))
end

function bstm_kernel(
    fam::MvNormalFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_vec,
    obs_idx::Int=1
)
    dist = get_dist_ref(fam, d, eta_vec, sig, obs_idx)
    return logpdf(dist, y_vec)
end

function is_discrete_family(::Union{PoissonFamily, NegativeBinomialFamily, BinomialFamily,
                                   MultinomialFamily, CategoricalFamily,
                                   DirichletMultinomialFamily, CategoricalMovementFamily})
    return true
end

function is_discrete_family(::AbstractBSTM_Family)
    return false
end

"""
    bstm_Likelihood(family_input, param; kwargs...)

Constructs a `bstm_Likelihood` distribution wrapper parameterized by linear predictor `param`
(`eta`), incorporating censoring, zero-inflation, hurdles, dispersion, and observation weights.
Supports both scalar and vectorized per-observation modifiers with rigorous validation.
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
    trial = 1,
    censor_lower = -Inf,
    censor_upper = Inf,
    hurdle = -Inf,
    extra_params = nothing
)
    f_trait = get_model_family(string(family_input))
    param_vec = param isa AbstractVector ? param : [param]
    n_obs = length(param_vec)
    
    function _validate_obs_param(x, name::Symbol)
        if x isa AbstractArray
            if isempty(x)
                throw(ArgumentError("Parameter ':$name' in bstm_Likelihood cannot be an empty array."))
            elseif length(x) == 1
                return first(x)
            elseif length(x) != n_obs
                throw(DimensionMismatch(
                    "Parameter ':$name' has length $(length(x)), but expected scalar or " *
                    "vector matching observation count ($n_obs)."
                ))
            else
                return x
            end
        end
        return x
    end

    function _validate_int_param(x, name::Symbol)
        val = _validate_obs_param(x, name)
        return val isa AbstractArray ? Int.(val) : Int(val)
    end

    tr_s = _validate_int_param(trial, :trial)
    w_s = _validate_obs_param(weight, :weight)
    pzi_s = _validate_obs_param(phi_zi, :phi_zi)
    phu_s = _validate_obs_param(phi_hurdle, :phi_hurdle)
    rnb_s = _validate_obs_param(r_nb, :r_nb)
    sig_s = _validate_obs_param(sigma_y, :sigma_y)
    cl_s = _validate_obs_param(censor_lower, :censor_lower)
    cu_s = _validate_obs_param(censor_upper, :censor_upper)
    hu_s = _validate_obs_param(hurdle, :hurdle)

    # Validate mutual exclusivity of zero-inflation and hurdle formulations
    has_zi = (zi_state isa ZeroInflated) ||
             (pzi_s isa AbstractVector ? any(v -> v > -Inf, pzi_s) : pzi_s > -Inf)
    has_hurdle = (phu_s isa AbstractVector ? any(v -> v > -Inf, phu_s) : phu_s > -Inf) ||
                 (hu_s isa AbstractVector ? any(v -> isfinite(v) && v > -Inf, hu_s) :
                  (isfinite(hu_s) && hu_s > -Inf))
    if has_zi && has_hurdle
        throw(ArgumentError(
            "Likelihood specification error: Zero-inflation and hurdle models are mutually exclusive. " *
            "Received zero-inflation (phi_zi = $(pzi_s)) with hurdle (phi_hurdle = $(phu_s), hurdle = $(hu_s)). " *
            "Specify either zero-inflation or a hurdle threshold, not both."
        ))
    end

    zi_trait = if !isnothing(zi_state)
        zi_state
    else
        has_zi ? ZeroInflated() : NonZeroInflated()
    end

    has_lower_censor = cl_s isa AbstractVector ? any(isfinite, cl_s) : isfinite(cl_s)
    has_upper_censor = cu_s isa AbstractVector ? any(isfinite, cu_s) : isfinite(cu_s)

    # Validate censoring bounds: lower must be strictly less than upper (Item 15)
    if has_lower_censor && has_upper_censor
        if cl_s isa AbstractVector && cu_s isa AbstractVector
            if length(cl_s) != length(cu_s)
                throw(DimensionMismatch(
                    "Censoring bounds vector length mismatch: censor_lower has $(length(cl_s)) " *
                    "elements but censor_upper has $(length(cu_s)) elements."
                ))
            end
            for idx in eachindex(cl_s, cu_s)
                if isfinite(cl_s[idx]) && isfinite(cu_s[idx]) && cl_s[idx] >= cu_s[idx]
                    throw(ArgumentError(
                        "Censoring bounds invalid at index $(idx): lower bound ($(cl_s[idx])) " *
                        "must be strictly less than upper bound ($(cu_s[idx]))."
                    ))
                end
            end
        elseif cl_s isa AbstractVector
            for idx in eachindex(cl_s)
                if isfinite(cl_s[idx]) && isfinite(cu_s) && cl_s[idx] >= cu_s
                    throw(ArgumentError(
                        "Censoring bounds invalid at index $(idx): lower bound ($(cl_s[idx])) " *
                        "must be strictly less than upper bound ($(cu_s))."
                    ))
                end
            end
        elseif cu_s isa AbstractVector
            for idx in eachindex(cu_s)
                if isfinite(cl_s) && isfinite(cu_s[idx]) && cl_s >= cu_s[idx]
                    throw(ArgumentError(
                        "Censoring bounds invalid at index $(idx): lower bound ($(cl_s)) " *
                        "must be strictly less than upper bound ($(cu_s[idx]))."
                    ))
                end
            end
        else
            if isfinite(cl_s) && isfinite(cu_s) && cl_s >= cu_s
                throw(ArgumentError(
                    "Censoring bounds invalid: lower bound ($(cl_s)) must be strictly " *
                    "less than upper bound ($(cu_s))."
                ))
            end
        end
    end

    censor_trait = if !isnothing(censoring_state)
        censoring_state
    elseif !has_lower_censor && !has_upper_censor
        Uncensored()
    elseif has_lower_censor && !has_upper_censor
        RightCensored()
    elseif !has_lower_censor && has_upper_censor
        LeftCensored()
    else 
        IntervalCensored() 
    end

    return bstm_Likelihood(
        f_trait, param_vec, zi_trait, censor_trait,
        w_s, pzi_s, phu_s, rnb_s, sig_s, tr_s,
        cl_s, cu_s, hu_s, extra_params
    )
end

function Distributions._logpdf(d::bstm_Likelihood, y::AbstractVector{V}) where {V<:Real}
    logp = zero(V) 
    
    if d.family isa MvNormalFamily
        eta = d.param
        sig = d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y, 1) * w
    elseif d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily}
        eta = d.param
        sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
        w = d.weight isa AbstractVector ? d.weight[1] : d.weight
        return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y, 1) * w
    else
        for i in 1:length(y)
            eta_i = d.param isa AbstractVector ? d.param[i] : d.param
            sig_i = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            w_i = d.weight isa AbstractVector ? d.weight[i] : d.weight
            k_val = bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta_i, sig_i, y[i], i)
            logp += k_val * w_i
        end
        return logp
    end
end

"""
    Distributions.logpdf(d::bstm_Likelihood, Y::AbstractMatrix{<:Real})

Evaluates joint log-likelihood for a batch of \$N\$ multivariate observations
represented as rows of matrix \$Y \\in \\mathbb{R}^{N \\times K}\$.
"""
function Distributions.logpdf(d::bstm_Likelihood, Y::AbstractMatrix{<:Real})
    if d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily, MvNormalFamily}
        logp = 0.0
        N = size(Y, 1)
        for i in 1:N
            eta_i = d.param isa AbstractMatrix ? d.param[i, :] : d.param
            sig_i = if d.family isa MvNormalFamily
                (d.sigma_y isa AbstractMatrix && size(d.sigma_y, 1) == N) ? d.sigma_y[i, :] : d.sigma_y
            else
                d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
            end
            w_i = d.weight isa AbstractVector ? d.weight[i] : d.weight
            k_val = bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta_i, sig_i, Y[i, :], i)
            logp += k_val * w_i
        end
        return logp
    else
        throw(ArgumentError(
            "Matrix observation evaluation is only supported for multivariate likelihood families, " *
            "got $(typeof(d.family))."
        ))
    end
end

function Distributions.logpdf(d::bstm_Likelihood, y::Real)
    if d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily, MvNormalFamily}
        error("$(typeof(d.family)) likelihood requires a vector of observations, " *
              "but received a scalar.")
    end
    
    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    w = d.weight isa AbstractVector ? d.weight[1] : d.weight
    
    return bstm_kernel(d.family, d.censoring_state, d.zi_state, d, eta, sig, y, 1) * w
end

function Distributions.logpdf(d::bstm_Likelihood, y::AbstractVector{<:Real})
    return Distributions._logpdf(d, y)
end

function Base.rand(rng::Random.AbstractRNG, d::bstm_Likelihood)
    if d.family isa Union{DirichletMultinomialFamily, MultinomialFamily, DirichletFamily,
                          CategoricalFamily, MvNormalFamily}
        dist = get_dist_ref(d.family, d, d.param, d.sigma_y, 1)
        return rand(rng, dist)
    end

    eta = d.param isa AbstractVector ? d.param[1] : d.param
    sig = d.sigma_y isa AbstractVector ? d.sigma_y[1] : d.sigma_y
    phi_zi = _get_obs(d.phi_zi, 1)
    phi_hu = _get_obs(d.phi_hurdle, 1)
    hu_val = _get_obs(d.hurdle, 1)

    # Zero-inflation draw
    if d.zi_state isa ZeroInflated
        if rand(rng) < phi_zi
            return zero(Float64)
        end
    elseif phi_hu > -Inf
        if rand(rng) < (1.0 - phi_hu)
            return Float64(hu_val)
        end
    end

    dist = get_dist_ref(d.family, d, eta, sig, 1)
    raw_draw = Float64(rand(rng, dist))

    # Censoring bounds
    if d.censoring_state isa LeftCensored
        l_b = _get_obs(d.censor_lower, 1)
        return max(raw_draw, Float64(l_b))
    elseif d.censoring_state isa RightCensored
        u_b = _get_obs(d.censor_upper, 1)
        return min(raw_draw, Float64(u_b))
    elseif d.censoring_state isa IntervalCensored
        l_b = _get_obs(d.censor_lower, 1)
        u_b = _get_obs(d.censor_upper, 1)
        return clamp(raw_draw, Float64(l_b), Float64(u_b))
    end

    return raw_draw
end

Base.rand(d::bstm_Likelihood) = rand(Random.default_rng(), d)

function Distributions._rand!(rng::Random.AbstractRNG, d::bstm_Likelihood, x::AbstractArray)
    for i in eachindex(x)
        eta_i = d.param isa AbstractVector ? d.param[i] : d.param
        sig_i = d.sigma_y isa AbstractVector ? d.sigma_y[i] : d.sigma_y
        phi_zi_i = _get_obs(d.phi_zi, i)
        phi_hu_i = _get_obs(d.phi_hurdle, i)
        hu_i = _get_obs(d.hurdle, i)
        
        draw = if d.zi_state isa ZeroInflated && rand(rng) < phi_zi_i
            zero(Float64)
        elseif phi_hu_i > -Inf && rand(rng) < (1.0 - phi_hu_i)
            Float64(hu_i)
        else
            dist_i = get_dist_ref(d.family, d, eta_i, sig_i, i)
            Float64(rand(rng, dist_i))
        end
        
        cl_i = _get_obs(d.censor_lower, i)
        cu_i = _get_obs(d.censor_upper, i)
        if d.censoring_state isa LeftCensored
            x[i] = max(draw, Float64(cl_i))
        elseif d.censoring_state isa RightCensored
            x[i] = min(draw, Float64(cu_i))
        elseif d.censoring_state isa IntervalCensored
            x[i] = clamp(draw, Float64(cl_i), Float64(cu_i))
        else
            x[i] = draw
        end
    end
    return x
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::Uncensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))

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
    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        log_one_minus_phi = V(log1p(-phi_hu_val))

        if y <= hu_val
            return log_one_minus_phi
        else
            logp_truncated = logpdf(dist, V(y)) - logccdf(dist, hu_val)
            return log_phi + logp_truncated
        end
    else
        return logpdf(dist, V(y))
    end
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::LeftCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    upper_bound = V(_get_obs(d.censor_upper, obs_idx))
    if !isfinite(upper_bound)
        return bstm_kernel(fam, Uncensored(), zero_inflated, d, eta, sig, y, obs_idx)
    end

    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))
        lp_base = logcdf(dist, upper_bound)
        if upper_bound >= V(0.0)
            return LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + lp_base)
        else
            return log_one_minus_phi + lp_base
        end
    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        log_one_minus_phi = V(log1p(-phi_hu_val))
        if upper_bound <= hu_val
            return log_one_minus_phi
        end
        log_prob_interval = _stable_logsubexp(
            logcdf(dist, upper_bound), logcdf(dist, hu_val)
        ) - logccdf(dist, hu_val)
        return LogExpFunctions.logsumexp(log_one_minus_phi, log_phi + log_prob_interval)
    else
        return logcdf(dist, upper_bound)
    end
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::RightCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    lower_bound = V(_get_obs(d.censor_lower, obs_idx))
    if !isfinite(lower_bound)
        return bstm_kernel(fam, Uncensored(), zero_inflated, d, eta, sig, y, obs_idx)
    end

    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))
        
        log_p_le_L = if lower_bound < V(0.0)
            log_one_minus_phi + logcdf(dist, lower_bound)
        else
            LogExpFunctions.logsumexp(log_phi, log_one_minus_phi + logcdf(dist, lower_bound))
        end
        return LogExpFunctions.log1mexp(log_p_le_L)

    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        adj_hurdle = is_discrete_family(fam) ? hu_val - V(1.0) : hu_val

        if lower_bound > hu_val
            return log_phi + logccdf(dist, adj_L) - logccdf(dist, adj_hurdle)
        else
            return log_phi
        end
    else
        return logccdf(dist, adj_L)
    end
end

function bstm_kernel(
    fam::AbstractBSTM_Family, ::IntervalCensored, zero_inflated::AbstractZIState, d, eta::V, sig, y,
    obs_idx::Int = 1
) where {V<:Real}
    if isnan(eta) || isinf(eta)
        return V(-Inf)
    end
    lower_bound = V(_get_obs(d.censor_lower, obs_idx))
    upper_bound = V(_get_obs(d.censor_upper, obs_idx))

    if !isfinite(lower_bound) && !isfinite(upper_bound)
        return bstm_kernel(fam, Uncensored(), zero_inflated, d, eta, sig, y, obs_idx)
    elseif isfinite(lower_bound) && !isfinite(upper_bound)
        return bstm_kernel(fam, RightCensored(), zero_inflated, d, eta, sig, y, obs_idx)
    elseif !isfinite(lower_bound) && isfinite(upper_bound)
        return bstm_kernel(fam, LeftCensored(), zero_inflated, d, eta, sig, y, obs_idx)
    end

    dist = get_dist_ref(fam, d, eta, sig, obs_idx)
    adj_L = is_discrete_family(fam) ? lower_bound - V(1.0) : lower_bound
    phi_zi_val = V(_get_obs(d.phi_zi, obs_idx))
    phi_hu_val = V(_get_obs(d.phi_hurdle, obs_idx))
    hu_val = V(_get_obs(d.hurdle, obs_idx))

    if zero_inflated isa ZeroInflated
        log_phi = V(log(phi_zi_val))
        log_one_minus_phi = V(log1p(-phi_zi_val))

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

    elseif phi_hu_val > V(-Inf)
        log_phi = V(log(phi_hu_val))
        adj_hurdle = is_discrete_family(fam) ? hu_val - V(1.0) : hu_val

        if upper_bound <= hu_val
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


function get_dist_ref(::CategoricalMovementFamily, d, eta_vec, sig)
    # eta_vec acts as the pre-normalized probability vector p over S spatial units
    p_safe = max.(eta_vec, 1e-12)
    p_norm = p_safe ./ sum(p_safe)
    return Categorical(p_norm)
end

function bstm_kernel(fam::CategoricalMovementFamily, ::Uncensored, ::NonZeroInflated, d, eta_vec, sig, y_scalar)
    dist = get_dist_ref(fam, d, eta_vec, sig)
    # y_scalar represents the integer index of the recapture location (1 to S)
    return logpdf(dist, Int(y_scalar))
end

