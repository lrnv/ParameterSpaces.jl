module ParameterSpacesDistributionsExt

using Distributions
using ParameterSpaces: AbstractParameterSpace,
                       Between,
                       Id,
                       Lower,
                       NIG,
                       Neg,
                       NonNeg,
                       Ordered,
                       Pos,
                       PosOrdered,
                       PosVec,
                       Prefixed,
                       Prob,
                       ProbOpen,
                       ProbOpenLeft,
                       ProbOpenRight,
                       ProbVec,
                       RealMat,
                       RealVec,
                       SPD,
                       Simplex
import ParameterSpaces: param_space

# ---------------------------------------------------------------------------
# Distribution integration helpers
# ---------------------------------------------------------------------------

function _pd_space(symbol::Symbol, A)
    n = size(A, 1)
    name = nameof(typeof(A))

    if name === :ScalMat
        return Pos(symbol)
    elseif name === :PDiagMat
        return PosVec(symbol, n)
    else
        return SPD(symbol, n)
    end
end


function _first_distribution_field(d)
    for name in propertynames(d)
        value = getproperty(d, name)
        value isa Distribution && return value
    end
    throw(ArgumentError("could not find an underlying distribution in $(typeof(d))"))
end


function _distribution_container(d)
    for name in propertynames(d)
        value = getproperty(d, name)

        if value isa NamedTuple
            vals = values(value)
            !isempty(vals) && all(x -> x isa Distribution, vals) && return value
        elseif value isa AbstractArray
            !isempty(value) && all(x -> x isa Distribution, value) && return value
        elseif value isa Tuple
            !isempty(value) && all(x -> x isa Distribution, value) && return value
        end
    end

    throw(ArgumentError("could not find component distributions in $(typeof(d))"))
end


function _product_param_space(d)
    container = _distribution_container(d)

    if container isa NamedTuple
        names = keys(container)
        vals = values(container)
        spaces = map(names, vals) do name, dist
            Prefixed(name, param_space(dist))
        end
        return Tuple(spaces)
    end

    vals = collect(container)
    spaces = Vector{AbstractParameterSpace}(undef, length(vals))
    for i in eachindex(vals)
        spaces[i] = Prefixed(Symbol("component", i), param_space(vals[i]))
    end
    return Tuple(spaces)
end

# ---------------------------------------------------------------------------
# Distributions.jl integration
# ---------------------------------------------------------------------------

# These mappings depend only on the distribution type. Instance-level calls are
# delegated to them by the generic Distribution fallback near the end of the
# extension. Mappings whose space depends on dimensions, values, or nested
# distributions remain defined on instances below.

# Ordered endpoints
param_space(::Type{<:Uniform})                  = Ordered(:a, :b)
param_space(::Type{<:Arcsine})                  = Ordered(:a, :b)
param_space(::Type{<:LogUniform})               = PosOrdered(:a, :b)
param_space(::Type{<:TriangularDist})           = Between(:a, :b, :c)

# Location / scale families
param_space(::Type{<:Normal})                   = (Id(:μ), NonNeg(:σ))
param_space(::Type{<:LogNormal})                = (Id(:μ), NonNeg(:σ))
param_space(::Type{<:LogitNormal})              = (Id(:μ), NonNeg(:σ))
param_space(::Type{<:Cauchy})                   = (Id(:μ), Pos(:σ))
param_space(::Type{<:Laplace})                  = (Id(:μ), Pos(:θ))
param_space(::Type{<:Logistic})                 = (Id(:μ), Pos(:θ))
param_space(::Type{<:Gumbel})                   = (Id(:μ), Pos(:θ))
param_space(::Type{<:Levy})                     = (Id(:μ), Pos(:σ))
param_space(::Type{<:Biweight})                 = (Id(:μ), Pos(:σ))
param_space(::Type{<:Cosine})                   = (Id(:μ), Pos(:σ))
param_space(::Type{<:Epanechnikov})             = (Id(:μ), Pos(:σ))
param_space(::Type{<:SymTriangularDist})        = (Id(:μ), Pos(:σ))
param_space(::Type{<:Triweight})                = (Id(:μ), Pos(:σ))

# Positive scalar parameters
param_space(::Type{<:Exponential})              = Pos(:θ)
param_space(::Type{<:Rayleigh})                 = Pos(:σ)
param_space(::Type{<:Chi})                      = Pos(:ν)
param_space(::Type{<:Chisq})                    = Pos(:ν)
param_space(::Type{<:TDist})                    = Pos(:ν)
param_space(::Type{<:Lindley})                  = Pos(:θ)
param_space(::Type{<:Semicircle})               = Pos(:r)

# Positive pairs / triples
param_space(::Type{<:Gamma})                    = (Pos(:α), Pos(:θ))
param_space(::Type{<:Beta})                     = (Pos(:α), Pos(:β))
param_space(::Type{<:BetaPrime})                = (Pos(:α), Pos(:β))
param_space(::Type{<:Frechet})                  = (Pos(:α), Pos(:θ))
param_space(::Type{<:InverseGamma})             = (Pos(:α), Pos(:θ))
param_space(::Type{<:InverseGaussian})          = (Pos(:μ), Pos(:λ))
param_space(::Type{<:Kumaraswamy})              = (Pos(:a), Pos(:b))
param_space(::Type{<:LogLogistic})              = (Pos(:α), Pos(:β))
param_space(::Type{<:Pareto})                   = (Pos(:α), Pos(:θ))
param_space(::Type{<:Weibull})                  = (Pos(:α), Pos(:θ))
param_space(::Type{<:FDist})                    = (Pos(:ν1), Pos(:ν2))
param_space(::Type{<:PGeneralizedGaussian})     = (Id(:μ), Pos(:α), Pos(:p))

# Unconstrained shape / location parameters combined with positive scales
param_space(::Type{<:GeneralizedExtremeValue})  = (Id(:μ), Pos(:σ), Id(:ξ))
param_space(::Type{<:GeneralizedPareto})        = (Id(:μ), Pos(:σ), Id(:ξ))
param_space(::Type{<:SkewNormal})               = (Id(:ξ), Pos(:ω), Id(:α))
param_space(::Type{<:SkewedExponentialPower})   = (Id(:μ), Pos(:σ), Pos(:p), ProbOpen(:α))
param_space(::Type{<:JohnsonSU})                = (Id(:ξ), Pos(:λ), Id(:γ), Pos(:δ))
param_space(::Type{<:NormalInverseGaussian})    = NIG(:μ, :α, :β, :δ)
param_space(::Type{<:StudentizedRange})         = (Pos(:ν), Lower(:k, 1.0))

# Noncentral families
param_space(::Type{<:NoncentralBeta})           = (Pos(:α), Pos(:β), NonNeg(:λ))
param_space(::Type{<:NoncentralChisq})          = (Pos(:ν), NonNeg(:λ))
param_space(::Type{<:NoncentralF})              = (Pos(:ν1), Pos(:ν2), NonNeg(:λ))
param_space(::Type{<:NoncentralT})              = (Pos(:ν), Id(:λ))

# Alternative normal parameterization
param_space(::Type{<:NormalCanon})              = (Id(:η), Pos(:λ))

# Circular / radial families
param_space(::Type{<:Rician})                   = (NonNeg(:ν), Pos(:σ))
param_space(::Type{<:VonMises})                 = (Id(:μ), NonNeg(:κ))

# Discrete distributions with continuous parameters
param_space(::Type{<:Bernoulli})                = Prob(:p)
param_space(::Type{<:BernoulliLogit})           = Id(:logitp)
param_space(::Type{<:Binomial})                 = Prob(:p)
param_space(::Type{<:Geometric})                = ProbOpenLeft(:p)
param_space(::Type{<:NegativeBinomial})         = (Pos(:r), ProbOpenLeft(:p))
param_space(::Type{<:Poisson})                  = NonNeg(:λ)
param_space(::Type{<:Skellam})                  = (NonNeg(:μ1), NonNeg(:μ2))
param_space(d::PoissonBinomial)                 = ProbVec(:p, length(params(d)[1]))
param_space(::Type{<:Soliton})                  = (ProbOpen(:δ), ProbOpenRight(:atol))

# Distributions with structural discrete parameters
param_space(::Type{<:BetaBinomial})             = (Pos(:α), Pos(:β))
param_space(::Type{<:Erlang})                   = Pos(:θ)
param_space(::Type{<:Chernoff})                 = ()
param_space(::Type{<:DiscreteUniform})          = ()
param_space(::Type{<:Hypergeometric})           = ()
param_space(::Type{<:Kolmogorov})               = ()
param_space(::Type{<:KSDist})                   = ()
param_space(::Type{<:KSOneSided})               = ()

# Degenerate / vector-parameter distributions
param_space(::Type{<:Dirac})                    = Id(:x)
param_space(d::DiscreteNonParametric)           = Simplex(:p, probs(d))
param_space(d::Dirichlet)                       = PosVec(:α, length(d))

# Simplex-valued probability parameters. The trial count of Multinomial
# remains structural and is therefore not part of the optimization space.
param_space(d::Categorical)                     = Simplex(:p, probs(d))
param_space(d::Multinomial)                     = Simplex(:p, probs(d))

# Noncentral hypergeometric families: population/sample sizes are structural.
if isdefined(Distributions, :FisherNoncentralHypergeometric)
    @eval param_space(::Type{<:Distributions.FisherNoncentralHypergeometric}) = Pos(:ω)
end
if isdefined(Distributions, :WalleniusNoncentralHypergeometric)
    @eval param_space(::Type{<:Distributions.WalleniusNoncentralHypergeometric}) = Pos(:ω)
end

# Affine wrapper. The sign of the nonzero scale determines the connected chart.
function param_space(d::Distributions.AffineDistribution)
    scale_space = d.σ > zero(d.σ) ? Pos(:σ) : Neg(:σ)
    return (Id(:μ), scale_space, Prefixed(:base, param_space(d.ρ)))
end

# Multivariate normal families.
function param_space(d::MvNormal)
    μ, Σ = params(d)
    return (RealVec(:μ, length(μ)), _pd_space(:Σ, Σ))
end

function param_space(d::MvNormalCanon)
    h, J = params(d)
    return (RealVec(:h, length(h)), _pd_space(:J, J))
end

function param_space(d::MvLogNormal)
    μ, Σ = params(d)
    return (RealVec(:μ, length(μ)), _pd_space(:Σ, Σ))
end

param_space(d::MvLogitNormal) =
    Prefixed(:normal, param_space(d.normal))

# Matrix-variate distributions.
function param_space(d::MatrixNormal)
    M, U, V = params(d)
    m, n = size(M)
    return (
        RealMat(:M, m, n),
        _pd_space(:U, U),
        _pd_space(:V, V),
    )
end

function param_space(d::Wishart)
    ν, S = params(d)
    p = size(d, 1)

    if ν > p - 1
        return (Lower(:ν, p - 1), _pd_space(:S, S))
    else
        # Singular Wishart requires integer degrees of freedom; keep ν structural.
        return _pd_space(:S, S)
    end
end

function param_space(d::InverseWishart)
    ν, Ψ = params(d)
    p = size(d, 1)
    return (Lower(:ν, p - 1), _pd_space(:Ψ, Ψ))
end

function param_space(d::MatrixTDist)
    ν, M, Σ, Ω = params(d)
    m, n = size(M)
    return (
        Pos(:ν),
        RealMat(:M, m, n),
        _pd_space(:Σ, Σ),
        _pd_space(:Ω, Ω),
    )
end

function param_space(d::MatrixBeta)
    p = size(d, 1)
    return (Lower(:n1, p - 1), Lower(:n2, p - 1))
end

function param_space(d::MatrixFDist)
    ps = params(d)
    p = size(d, 1)
    B = ps[end]
    return (
        Lower(:n1, p - 1),
        Lower(:n2, p - 1),
        _pd_space(:B, B),
    )
end

param_space(::Type{<:LKJ}) = Pos(:η)
param_space(::Type{<:LKJCholesky}) = Pos(:η)

# Generic wrappers. Truncation/censoring bounds, reshape dimensions, and
# order-statistic ranks/sample sizes are treated as structural.
param_space(d::Distributions.Truncated) =
    Prefixed(:base, param_space(d.untruncated))

param_space(d::Distributions.Censored) =
    Prefixed(:base, param_space(d.uncensored))

param_space(d::Distributions.OrderStatistic) =
    Prefixed(:base, param_space(_first_distribution_field(d)))

param_space(d::Distributions.JointOrderStatistics) =
    Prefixed(:base, param_space(_first_distribution_field(d)))

param_space(d::Distributions.ReshapedDistribution) =
    Prefixed(:base, param_space(_first_distribution_field(d)))

# Mixture models: all component parameters plus a simplex of mixing weights.
function param_space(d::Distributions.AbstractMixtureModel)
    cs = components(d)
    component_spaces = ntuple(length(cs)) do i
        Prefixed(Symbol("component", i), param_space(cs[i]))
    end
    return (component_spaces..., Simplex(:π, probs(d)))
end

# Product distributions.
param_space(d::Distributions.ProductDistribution) = _product_param_space(d)
param_space(d::Distributions.ProductNamedTupleDistribution) = _product_param_space(d)

# Product is deprecated but still present in Distributions 0.25.
if isdefined(Distributions, :Product)
    @eval param_space(d::Distributions.Product) = _product_param_space(d)
end

function param_space(d::Distribution)
    T = typeof(d)
    applicable(param_space, T) && return param_space(T)
    throw(ArgumentError("parameter space not implemented for $T"))
end


end
