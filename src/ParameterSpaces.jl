module ParameterSpaces

export AbstractParameterSpace,
       param_space,
       Id,
       Pos,
       NonNeg,
       Neg,
       Prob,
       ProbOpen,
       ProbOpenLeft,
       ProbOpenRight,
       Lower,
       LowerClosed,
       Bounded,
       BoundedOpen,
       BoundedOpenLeft,
       BoundedOpenRight,
       Ordered,
       PosOrdered,
       Between,
       BilinearQuad,
       Simplex,
       SPD,
       Prefixed,
       PosVec,
       ProbVec,
       RealVec,
       RealMat,
       dimension,
       constrained_dimension,
       unconstrained_example,
       constrained_example,
       constrain,
       unconstrain,
       parameter_symbols,
       constrained_namedtuple

"""
    param_space(object)

Return the parameter-space description associated with `object`.

`param_space` is an open generic function. ParameterSpaces.jl provides the
space constructors and transformations, while downstream packages define
methods for their own objects.
"""
function param_space end

abstract type AbstractParameterSpace end
abstract type AbstractScalarDomain end

struct IdentityDomain <: AbstractScalarDomain end
struct ExpDomain{AllowZero} <: AbstractScalarDomain end
struct NegativeExpDomain <: AbstractScalarDomain end
struct ProbabilityDomain{LeftClosed,RightClosed} <: AbstractScalarDomain end
struct LowerDomain{AllowEqual,T} <: AbstractScalarDomain
    lower::T
end
struct BoundedDomain{LeftClosed,RightClosed,L,U} <: AbstractScalarDomain
    lower::L
    upper::U
end

struct ScalarSpace{S,D<:AbstractScalarDomain} <: AbstractParameterSpace
    domain::D
end

struct ElementwiseSpace{S,D<:AbstractScalarDomain,N} <: AbstractParameterSpace
    domain::D
    dims::NTuple{N,Int}
end

struct OrderedSpace{A,B,D<:AbstractScalarDomain} <: AbstractParameterSpace
    first_domain::D
end

struct Between{A,B,C} <: AbstractParameterSpace end

struct BilinearQuad{A,B,T} <: AbstractParameterSpace
    p00::NTuple{2,T}
    p10::NTuple{2,T}
    p01::NTuple{2,T}
    p11::NTuple{2,T}
end

# Internal specialized space used by the Distributions.jl extension.
struct NIG{M,A,B,D} <: AbstractParameterSpace end

struct Simplex{S} <: AbstractParameterSpace
    n::Int
    anchor::Int

    function Simplex{S}(n::Integer, anchor::Integer) where {S}
        n >= 1 || throw(ArgumentError("simplex must contain at least one coordinate"))
        1 <= anchor <= n || throw(ArgumentError("anchor must belong to 1:n"))
        new{S}(Int(n), Int(anchor))
    end
end

struct SPD{S} <: AbstractParameterSpace
    n::Int

    function SPD{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("matrix dimension must be positive"))
        new{S}(Int(n))
    end
end

struct Prefixed{P,S} <: AbstractParameterSpace
    space::S
end

const ProductParameterSpace = Tuple{Vararg{AbstractParameterSpace}}

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

_scalar(s::Symbol, d::D) where {D<:AbstractScalarDomain} = ScalarSpace{s,D}(d)

_elementwise(s::Symbol, d::D, dims::NTuple{N,Int}) where {D<:AbstractScalarDomain,N} =
    ElementwiseSpace{s,D,N}(d, dims)

_ordered(a::Symbol, b::Symbol, d::D) where {D<:AbstractScalarDomain} =
    OrderedSpace{a,b,D}(d)

function _bounded(s::Symbol, lower, upper, ::Val{LC}, ::Val{RC}) where {LC,RC}
    lower < upper || throw(ArgumentError("lower bound must be smaller than upper bound"))
    d = BoundedDomain{LC,RC,typeof(lower),typeof(upper)}(lower, upper)
    return _scalar(s, d)
end

"""A scalar parameter named `name` with values in `ℝ`."""
Id(name::Symbol) = _scalar(name, IdentityDomain())

"""A strictly positive scalar parameter."""
Pos(name::Symbol) = _scalar(name, ExpDomain{false}())

"""A nonnegative scalar parameter; zero maps back to `-Inf`."""
NonNeg(name::Symbol) = _scalar(name, ExpDomain{true}())

"""A strictly negative scalar parameter."""
Neg(name::Symbol) = _scalar(name, NegativeExpDomain())

"""A probability parameter in `[0, 1]`."""
Prob(name::Symbol) = _scalar(name, ProbabilityDomain{true,true}())

"""A probability parameter in `(0, 1)`."""
ProbOpen(name::Symbol) = _scalar(name, ProbabilityDomain{false,false}())

"""A probability parameter in `(0, 1]`."""
ProbOpenLeft(name::Symbol) = _scalar(name, ProbabilityDomain{false,true}())

"""A probability parameter in `[0, 1)`."""
ProbOpenRight(name::Symbol) = _scalar(name, ProbabilityDomain{true,false}())

"""A scalar parameter strictly greater than `lower`."""
Lower(name::Symbol, lower) = _scalar(name, LowerDomain{false,typeof(lower)}(lower))

"""A scalar parameter greater than or equal to `lower`."""
LowerClosed(name::Symbol, lower) = _scalar(name, LowerDomain{true,typeof(lower)}(lower))

"""A scalar parameter in `[lower, upper]`."""
Bounded(name::Symbol, lower, upper) = _bounded(name, lower, upper, Val(true), Val(true))

"""A scalar parameter in `(lower, upper)`."""
BoundedOpen(name::Symbol, lower, upper) = _bounded(name, lower, upper, Val(false), Val(false))

"""A scalar parameter in `(lower, upper]`."""
BoundedOpenLeft(name::Symbol, lower, upper) = _bounded(name, lower, upper, Val(false), Val(true))

"""A scalar parameter in `[lower, upper)`."""
BoundedOpenRight(name::Symbol, lower, upper) = _bounded(name, lower, upper, Val(true), Val(false))

"""Two real scalar parameters satisfying `first < second`."""
Ordered(first::Symbol, second::Symbol) = _ordered(first, second, IdentityDomain())

"""Two positive scalar parameters satisfying `0 < first < second`."""
PosOrdered(first::Symbol, second::Symbol) = _ordered(first, second, ExpDomain{false}())

"""Three scalar parameters satisfying `lower ≤ value ≤ upper`."""
Between(lower::Symbol, upper::Symbol, value::Symbol) = Between{lower,upper,value}()

@inline _cross2(a, b) = a[1] * b[2] - a[2] * b[1]

function _point2(p)
    length(p) == 2 || throw(DimensionMismatch("quadrilateral corners must have two coordinates"))
    return (p[1], p[2])
end

function _bilinear_basis(p::BilinearQuad)
    e = (p.p10[1] - p.p00[1], p.p10[2] - p.p00[2])
    f = (p.p01[1] - p.p00[1], p.p01[2] - p.p00[2])
    g = (
        p.p11[1] - p.p10[1] - p.p01[1] + p.p00[1],
        p.p11[2] - p.p10[2] - p.p01[2] + p.p00[2],
    )
    return e, f, g
end

function _validate_bilinear_quad(p::BilinearQuad)
    e, f, g = _bilinear_basis(p)
    eg = (e[1] + g[1], e[2] + g[2])
    fg = (f[1] + g[1], f[2] + g[2])
    dets = (
        _cross2(e, f),
        _cross2(e, fg),
        _cross2(eg, f),
        _cross2(eg, fg),
    )
    positive = all(d -> d > zero(d), dets)
    negative = all(d -> d < zero(d), dets)
    positive || negative || throw(ArgumentError(
        "BilinearQuad corners must define a nondegenerate, non-folded quadrilateral",
    ))
    return p
end

"""
    BilinearQuad(first, second, p00, p10, p01, p11)

Two coupled scalar parameters obtained by mapping the open unit square
bilinearly onto a non-folded quadrilateral.
"""
function BilinearQuad(a::Symbol, b::Symbol, p00, p10, p01, p11)
    raw = (_point2(p00), _point2(p10), _point2(p01), _point2(p11))
    T = promote_type((typeof(x) for point in raw for x in point)...)
    corners = ntuple(4) do i
        (convert(T, raw[i][1]), convert(T, raw[i][2]))
    end
    return _validate_bilinear_quad(BilinearQuad{a,b,T}(corners...))
end

NIG(μ::Symbol, α::Symbol, β::Symbol, δ::Symbol) = NIG{μ,α,β,δ}()

"""An `n`-component vector parameter with strictly positive entries."""
function PosVec(name::Symbol, n::Integer)
    n > 0 || throw(ArgumentError("dimension must be positive"))
    return _elementwise(name, ExpDomain{false}(), (Int(n),))
end

"""An `n`-component vector parameter with entries in `[0, 1]`."""
function ProbVec(name::Symbol, n::Integer)
    n > 0 || throw(ArgumentError("dimension must be positive"))
    return _elementwise(name, ProbabilityDomain{true,true}(), (Int(n),))
end

"""An unconstrained real vector parameter with `n` entries."""
function RealVec(name::Symbol, n::Integer)
    n >= 0 || throw(ArgumentError("dimension must be nonnegative"))
    return _elementwise(name, IdentityDomain(), (Int(n),))
end

"""An unconstrained real `m × n` matrix parameter."""
function RealMat(name::Symbol, m::Integer, n::Integer)
    m >= 0 || throw(ArgumentError("number of rows must be nonnegative"))
    n >= 0 || throw(ArgumentError("number of columns must be nonnegative"))
    return _elementwise(name, IdentityDomain(), (Int(m), Int(n)))
end

"""A probability vector of length `n` constrained to the simplex."""
Simplex(name::Symbol, n::Integer; anchor::Integer=n) = Simplex{name}(n, anchor)

function Simplex(name::Symbol, p::AbstractVector)
    isempty(p) && throw(ArgumentError("probability vector must be nonempty"))
    all(x -> x >= zero(x), p) || throw(DomainError(p, "probabilities must be nonnegative"))
    total = sum(p)
    isapprox(total, one(total)) || throw(DomainError(p, "probabilities must sum to one"))
    anchor = argmax(p)
    p[anchor] > zero(p[anchor]) || throw(DomainError(p, "at least one probability must be positive"))
    return Simplex(name, length(p); anchor)
end

"""An `n × n` symmetric positive-definite matrix parameter."""
SPD(name::Symbol, n::Integer) = SPD{name}(n)

"""Prefix the logical parameter names of `space` without changing its values."""
Prefixed(prefix::Symbol, space) = Prefixed{prefix,typeof(space)}(space)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

parameter_symbols(::ScalarSpace{S}) where {S} = (S,)
parameter_symbols(::ElementwiseSpace{S}) where {S} = (S,)
parameter_symbols(::OrderedSpace{A,B}) where {A,B} = (A, B)
parameter_symbols(::Between{A,B,C}) where {A,B,C} = (A, B, C)
parameter_symbols(::BilinearQuad{A,B}) where {A,B} = (A, B)
parameter_symbols(::NIG{M,A,B,D}) where {M,A,B,D} = (M, A, B, D)
parameter_symbols(::Simplex{S}) where {S} = (S,)
parameter_symbols(::SPD{S}) where {S} = (S,)

function parameter_symbols(p::ProductParameterSpace)
    return Tuple(s for q in p for s in parameter_symbols(q))
end

parameter_symbols(p::Prefixed{P}) where {P} =
    Tuple(Symbol(P, "_", s) for s in parameter_symbols(p.space))

dimension(::ScalarSpace) = 1
dimension(::OrderedSpace) = 2
dimension(::Between) = 3
dimension(::BilinearQuad) = 2
dimension(::NIG) = 4
dimension(p::Simplex) = p.n - 1
dimension(p::ElementwiseSpace) = prod(p.dims)
dimension(p::SPD) = p.n * (p.n + 1) ÷ 2
dimension(p::Prefixed) = dimension(p.space)
dimension(p::ProductParameterSpace) = sum(dimension, p; init=0)

"""
    constrained_dimension(space)

Number of independent scalar coordinates in the constrained value. This is
metadata only: `constrain` returns the natural scalar/vector/matrix/tuple value,
not a flattened vector.
"""
constrained_dimension(::ScalarSpace) = 1
constrained_dimension(::OrderedSpace) = 2
constrained_dimension(::Between) = 3
constrained_dimension(::BilinearQuad) = 2
constrained_dimension(::NIG) = 4
constrained_dimension(p::Simplex) = p.n
constrained_dimension(p::ElementwiseSpace) = prod(p.dims)
constrained_dimension(p::SPD) = p.n * (p.n + 1) ÷ 2
constrained_dimension(p::Prefixed) = constrained_dimension(p.space)
constrained_dimension(p::ProductParameterSpace) = sum(constrained_dimension, p; init=0)

_parameter_count(p) = length(parameter_symbols(p))

# A single parameter may itself be a vector or matrix. Coupled spaces with
# several logical parameters use a tuple, while product spaces flatten only the
# logical parameter blocks, never their contents.
function _parameter_values(p::AbstractParameterSpace, η)
    n = _parameter_count(p)
    n == 0 && return ()
    n == 1 && return (η,)
    η isa Tuple || throw(ArgumentError("expected $n constrained parameter values"))
    length(η) == n || throw(DimensionMismatch("expected $n constrained parameter values"))
    return η
end

_parameter_values(p::ProductParameterSpace, η::Tuple) = η
_parameter_values(p::Prefixed, η) = _parameter_values(p.space, η)

function _from_parameter_values(p::AbstractParameterSpace, values::Tuple)
    n = _parameter_count(p)
    length(values) == n || throw(DimensionMismatch("expected $n constrained parameter values"))
    n == 0 && return ()
    n == 1 && return values[1]
    return values
end

_from_parameter_values(p::ProductParameterSpace, values::Tuple) = values
_from_parameter_values(p::Prefixed, values::Tuple) = _from_parameter_values(p.space, values)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _check_dimension(p, x)
    length(x) == dimension(p) || throw(DimensionMismatch(
        "expected $(dimension(p)) unconstrained parameters, got $(length(x))",
    ))
    return nothing
end

function _promoted_vector(x)
    isempty(x) && return Float64[]
    T = promote_type(map(typeof, x)...)
    return T[xi for xi in x]
end

function _concatenate_vectors(vs)
    total = sum(length, vs; init=0)
    total == 0 && return Float64[]
    Ts = [eltype(v) for v in vs if !isempty(v)]
    T = isempty(Ts) ? Float64 : promote_type(Ts...)
    result = Vector{T}(undef, total)
    offset = 0
    for v in vs, x in v
        offset += 1
        result[offset] = x
    end
    return result
end

function _check_tuple(η, n, message="constrained parameter")
    η isa Tuple || throw(ArgumentError("$message must be a tuple of length $n"))
    length(η) == n || throw(DimensionMismatch("expected $n constrained values"))
    return nothing
end

# ---------------------------------------------------------------------------
# Scalar transforms
# ---------------------------------------------------------------------------

_constrain_scalar(::IdentityDomain, θ) = θ
_unconstrain_scalar(::IdentityDomain, η) = η

_constrain_scalar(::ExpDomain, θ) = exp(θ)

function _unconstrain_scalar(::ExpDomain{AllowZero}, η) where {AllowZero}
    valid = AllowZero ? η >= zero(η) : η > zero(η)
    valid || throw(DomainError(
        η,
        AllowZero ? "parameter must be nonnegative" : "parameter must be strictly positive",
    ))
    return log(η)
end

_constrain_scalar(::NegativeExpDomain, θ) = -exp(θ)

function _unconstrain_scalar(::NegativeExpDomain, η)
    η < zero(η) || throw(DomainError(η, "parameter must be strictly negative"))
    return log(-η)
end

function _constrain_scalar(::ProbabilityDomain, θ)
    if θ >= zero(θ)
        z = exp(-θ)
        return inv(one(θ) + z)
    else
        z = exp(θ)
        return z / (one(θ) + z)
    end
end

function _unconstrain_scalar(::ProbabilityDomain{LC,RC}, η) where {LC,RC}
    left_ok = LC ? η >= zero(η) : η > zero(η)
    right_ok = RC ? η <= one(η) : η < one(η)
    left_ok && right_ok || throw(DomainError(
        η,
        "parameter must belong to " *
        (LC ? "[0, 1" : "(0, 1") *
        (RC ? "]" : ")"),
    ))
    return log(η) - log1p(-η)
end

_constrain_scalar(d::LowerDomain, θ) = d.lower + exp(θ)

function _unconstrain_scalar(d::LowerDomain{AllowEqual}, η) where {AllowEqual}
    valid = AllowEqual ? η >= d.lower : η > d.lower
    valid || throw(DomainError(
        η,
        AllowEqual ?
            "parameter must be greater than or equal to $(d.lower)" :
            "parameter must be strictly greater than $(d.lower)",
    ))
    return log(η - d.lower)
end

function _constrain_scalar(d::BoundedDomain, θ)
    q = _constrain_scalar(ProbabilityDomain{true,true}(), θ)
    return d.lower + (d.upper - d.lower) * q
end

function _unconstrain_scalar(d::BoundedDomain{LC,RC}, η) where {LC,RC}
    left_ok = LC ? η >= d.lower : η > d.lower
    right_ok = RC ? η <= d.upper : η < d.upper
    left_ok && right_ok || throw(DomainError(
        η,
        "parameter must belong to " *
        (LC ? "[$(d.lower), $(d.upper)" : "($(d.lower), $(d.upper)") *
        (RC ? "]" : ")"),
    ))
    q = (η - d.lower) / (d.upper - d.lower)
    return _unconstrain_scalar(ProbabilityDomain{LC,RC}(), q)
end

# ---------------------------------------------------------------------------
# Scalar and elementwise spaces
# ---------------------------------------------------------------------------

function constrain(p::ScalarSpace, θ)
    _check_dimension(p, θ)
    return _constrain_scalar(p.domain, θ[1])
end

unconstrain(p::ScalarSpace, η::Number) = [_unconstrain_scalar(p.domain, η)]

function constrain(p::ElementwiseSpace, θ)
    _check_dimension(p, θ)
    values = [_constrain_scalar(p.domain, x) for x in θ]
    length(p.dims) == 1 && return values
    return copy(reshape(values, p.dims))
end

function unconstrain(p::ElementwiseSpace, η::AbstractArray)
    size(η) == p.dims || throw(DimensionMismatch(
        "expected constrained parameter with size $(p.dims), got $(size(η))",
    ))
    return _promoted_vector([_unconstrain_scalar(p.domain, x) for x in vec(η)])
end

# ---------------------------------------------------------------------------
# Product spaces
# ---------------------------------------------------------------------------

function constrain(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)
    return _constrain_product(p, θ, 0)
end

_constrain_product(::Tuple{}, θ, offset) = ()

function _constrain_product(p::Tuple{Q,Vararg{AbstractParameterSpace}}, θ, offset) where {Q<:AbstractParameterSpace}
    q = first(p)
    n = dimension(q)
    ηq = constrain(q, view(θ, (offset + 1):(offset + n)))
    return (_parameter_values(q, ηq)..., _constrain_product(Base.tail(p), θ, offset + n)...)
end

function unconstrain(p::ProductParameterSpace, η::Tuple)
    length(η) == _parameter_count(p) || throw(DimensionMismatch(
        "expected $(_parameter_count(p)) constrained parameter values, got $(length(η))",
    ))
    values = Any[]
    offset = 0
    for q in p
        n = _parameter_count(q)
        qvalues = ntuple(i -> η[offset + i], n)
        push!(values, unconstrain(q, _from_parameter_values(q, qvalues)))
        offset += n
    end
    return _concatenate_vectors(values)
end

function unconstrain(p, η::NamedTuple)
    names = parameter_symbols(p)
    keys(η) == names || throw(ArgumentError(
        "expected constrained parameter names $names, got $(keys(η))",
    ))
    vals = Tuple(Base.values(η))
    return unconstrain(p, _from_parameter_values(p, vals))
end

# ---------------------------------------------------------------------------
# Coupled scalar spaces
# ---------------------------------------------------------------------------

function constrain(p::OrderedSpace, θ)
    _check_dimension(p, θ)
    a = _constrain_scalar(p.first_domain, θ[1])
    return (a, a + exp(θ[2]))
end

function unconstrain(p::OrderedSpace, η)
    _check_tuple(η, 2)
    a, b = η
    b > a || throw(DomainError(η, "parameters must satisfy first < second"))
    return _promoted_vector((_unconstrain_scalar(p.first_domain, a), log(b - a)))
end

function constrain(p::Between, θ)
    _check_dimension(p, θ)
    a = θ[1]
    width = exp(θ[2])
    q = _constrain_scalar(ProbabilityDomain{true,true}(), θ[3])
    return (a, a + width, a + width * q)
end

function unconstrain(p::Between, η)
    _check_tuple(η, 3)
    a, b, c = η
    a <= c <= b || throw(DomainError(η, "parameters must satisfy lower <= value <= upper"))
    if b == a
        c == a || throw(DomainError(η, "degenerate bounds require lower == upper == value"))
        return _promoted_vector((a, -Inf, 0.0))
    end
    width = b - a
    q = (c - a) / width
    z = _unconstrain_scalar(ProbabilityDomain{true,true}(), q)
    return _promoted_vector((a, log(width), z))
end

@inline function _bilinear_eval(p::BilinearQuad, u, v)
    e, f, g = _bilinear_basis(p)
    return (
        p.p00[1] + u * e[1] + v * f[1] + u * v * g[1],
        p.p00[2] + u * e[2] + v * f[2] + u * v * g[2],
    )
end

function constrain(p::BilinearQuad, θ)
    _check_dimension(p, θ)
    u = _constrain_scalar(ProbabilityDomain{true,true}(), θ[1])
    v = _constrain_scalar(ProbabilityDomain{true,true}(), θ[2])
    return _bilinear_eval(p, u, v)
end

function _bilinear_inverse_candidate(p::BilinearQuad, y, u)
    e, f, g = _bilinear_basis(p)
    h = (f[1] + u * g[1], f[2] + u * g[2])
    rhs = (
        y[1] - p.p00[1] - u * e[1],
        y[2] - p.p00[2] - u * e[2],
    )
    if abs(h[1]) >= abs(h[2])
        iszero(h[1]) && return nothing
        v = rhs[1] / h[1]
    else
        iszero(h[2]) && return nothing
        v = rhs[2] / h[2]
    end
    point = _bilinear_eval(p, u, v)
    residual = abs(point[1] - y[1]) + abs(point[2] - y[2])
    return (u=u, v=v, residual=residual)
end

function _bilinear_inverse(p::BilinearQuad, y)
    e, f, g = _bilinear_basis(p)
    r = (y[1] - p.p00[1], y[2] - p.p00[2])
    A = -_cross2(e, g)
    B = _cross2(r, g) - _cross2(e, f)
    C = _cross2(r, f)

    roots = if iszero(A)
        iszero(B) && throw(DomainError(y, "point is not uniquely invertible in this quadrilateral"))
        (-C / B,)
    else
        disc = B * B - 4 * A * C
        scale = max(abs(B * B), abs(4 * A * C), one(abs(disc)))
        tol = 64 * eps(float(one(disc))) * scale
        disc < -tol && throw(DomainError(y, "point lies outside the quadrilateral"))
        disc = max(disc, zero(disc))
        root = sqrt(disc)
        ((-B - root) / (2 * A), (-B + root) / (2 * A))
    end

    candidates = Any[]
    for u in roots
        candidate = _bilinear_inverse_candidate(p, y, u)
        isnothing(candidate) && continue
        tol = 64 * sqrt(eps(float(one(candidate.u))))
        if -tol <= candidate.u <= one(candidate.u) + tol &&
                -tol <= candidate.v <= one(candidate.v) + tol
            push!(candidates, candidate)
        end
    end
    isempty(candidates) && throw(DomainError(y, "point lies outside the quadrilateral"))
    candidate = argmin(c -> c.residual, candidates)
    u = clamp(candidate.u, zero(candidate.u), one(candidate.u))
    v = clamp(candidate.v, zero(candidate.v), one(candidate.v))
    return u, v
end

function unconstrain(p::BilinearQuad, η)
    _check_tuple(η, 2)
    u, v = _bilinear_inverse(p, η)
    q = ProbabilityDomain{true,true}()
    return _promoted_vector((_unconstrain_scalar(q, u), _unconstrain_scalar(q, v)))
end

function constrain(p::NIG, θ)
    _check_dimension(p, θ)
    μ = θ[1]
    γ = exp(θ[2])
    β = θ[3]
    δ = exp(θ[4])
    α = hypot(β, γ)
    return (μ, α, β, δ)
end

function unconstrain(p::NIG, η)
    _check_tuple(η, 4)
    μ, α, β, δ = η
    α > abs(β) || throw(DomainError(η, "parameters must satisfy α > |β|"))
    δ > zero(δ) || throw(DomainError(η, "δ must be strictly positive"))
    γ2 = (α - abs(β)) * (α + abs(β))
    return _promoted_vector((μ, log(γ2) / 2, β, log(δ)))
end

# ---------------------------------------------------------------------------
# Simplex
# ---------------------------------------------------------------------------

function constrain(p::Simplex, θ)
    _check_dimension(p, θ)
    p.n == 1 && return [1.0]

    m = max(zero(θ[1]), maximum(θ))
    anchor_weight = exp(-m)
    free_weights = exp.(θ .- m)
    denom = anchor_weight + sum(free_weights)

    T = promote_type(typeof(anchor_weight), eltype(free_weights))
    η = Vector{T}(undef, p.n)
    j = 1
    for i in 1:p.n
        if i == p.anchor
            η[i] = anchor_weight / denom
        else
            η[i] = free_weights[j] / denom
            j += 1
        end
    end
    return η
end

function unconstrain(p::Simplex, η::AbstractVector)
    length(η) == p.n || throw(DimensionMismatch("expected simplex of length $(p.n)"))
    all(x -> x >= zero(x), η) || throw(DomainError(η, "probabilities must be nonnegative"))
    total = sum(η)
    isapprox(total, one(total)) || throw(DomainError(η, "probabilities must sum to one"))
    anchor = η[p.anchor]
    anchor > zero(anchor) || throw(DomainError(
        η,
        "the anchor probability must be strictly positive for this simplex chart",
    ))
    log_anchor = log(anchor)
    return _promoted_vector([
        log(η[i]) - log_anchor for i in 1:p.n if i != p.anchor
    ])
end

# ---------------------------------------------------------------------------
# Symmetric positive-definite matrices
# ---------------------------------------------------------------------------

function _spd_pairs(n::Integer)
    pairs = Tuple{Int,Int}[]
    for i in 1:n, j in 1:i
        push!(pairs, (i, j))
    end
    return pairs
end

function _spd_cholesky_from_theta(p::SPD, θ)
    _check_dimension(p, θ)
    T = isempty(θ) ? Float64 : promote_type(map(typeof, θ)...)
    L = zeros(T, p.n, p.n)
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        L[i, j] = i == j ? exp(θ[q]) : θ[q]
    end
    return L
end

function constrain(p::SPD, θ)
    L = _spd_cholesky_from_theta(p, θ)
    return L * transpose(L)
end

function _spd_cholesky_from_matrix(p::SPD, S)
    size(S) == (p.n, p.n) || throw(DimensionMismatch("expected a $(p.n)×$(p.n) matrix"))
    for i in 1:p.n, j in 1:(i - 1)
        isapprox(S[i, j], S[j, i]) || throw(DomainError(S, "matrix must be symmetric"))
    end

    T = eltype(S)
    L = zeros(T, p.n, p.n)
    for i in 1:p.n, j in 1:i
        s = S[i, j]
        for k in 1:(j - 1)
            s -= L[i, k] * L[j, k]
        end
        if i == j
            s > zero(s) || throw(DomainError(S, "matrix must be positive definite"))
            L[i, j] = sqrt(s)
        else
            L[i, j] = s / L[j, j]
        end
    end
    return L
end

function unconstrain(p::SPD, S)
    L = _spd_cholesky_from_matrix(p, S)
    θ = Vector{eltype(L)}(undef, dimension(p))
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        θ[q] = i == j ? log(L[i, j]) : L[i, j]
    end
    return θ
end

# Prefixes alter names only, never values or transformations.
constrain(p::Prefixed, θ) = constrain(p.space, θ)
unconstrain(p::Prefixed, η) = unconstrain(p.space, η)

# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

unconstrained_example(p) = zeros(dimension(p))
constrained_example(p) = constrain(p, unconstrained_example(p))

function constrained_namedtuple(p, θ)
    η = constrain(p, θ)
    return NamedTuple{parameter_symbols(p)}(_parameter_values(p, η))
end

end # module
