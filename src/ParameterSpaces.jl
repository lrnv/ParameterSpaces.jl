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
       constrain_jac,
       unconstrain_jac,
       constrain_with_jac,
       unconstrain_with_jac,
       logabsdet_constrain_jac,
       logabsdet_unconstrain_jac,
       parameter_symbols,
       constrained_namedtuple


# ---------------------------------------------------------------------------
# Core parameter-space types
# ---------------------------------------------------------------------------

"""
    param_space(object)

Return the parameter-space description associated with `object`.

`param_space` is an open generic function: `ParameterSpaces.jl` provides the
space constructors and transformation machinery, while users and package
extensions define methods for their own object types.
"""
function param_space end

abstract type AbstractParameterSpace end
abstract type AbstractSeparableParameterSpace <: AbstractParameterSpace end
abstract type AbstractScalarDomain end

# Scalar domains. Boundary flags affect the inverse-domain check only; finite
# unconstrained coordinates still map to the interior through exp/logistic.
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

struct ScalarSpace{S,D<:AbstractScalarDomain} <: AbstractSeparableParameterSpace
    domain::D
end

struct ElementwiseSpace{S,D<:AbstractScalarDomain,N} <: AbstractSeparableParameterSpace
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


# ---------------------------------------------------------------------------
# Public parameter-space constructors
# ---------------------------------------------------------------------------

_scalar(s::Symbol, d::D) where {D<:AbstractScalarDomain} =
    ScalarSpace{s,D}(d)

_elementwise(s::Symbol, d::D, dims::NTuple{N,Int}) where {D<:AbstractScalarDomain,N} =
    ElementwiseSpace{s,D,N}(d, dims)

_ordered(a::Symbol, b::Symbol, d::D) where {D<:AbstractScalarDomain} =
    OrderedSpace{a,b,D}(d)

function _bounded(s::Symbol, lower, upper, ::Val{LC}, ::Val{RC}) where {LC,RC}
    lower < upper || throw(ArgumentError("lower bound must be smaller than upper bound"))
    d = BoundedDomain{LC,RC,typeof(lower),typeof(upper)}(lower, upper)
    return _scalar(s, d)
end

"""
    Id(name::Symbol)

A scalar parameter named `name` with values in `ℝ`.
"""
Id(s::Symbol) = _scalar(s, IdentityDomain())

"""
    Pos(name::Symbol)

A strictly positive scalar parameter named `name`.
"""
Pos(s::Symbol) = _scalar(s, ExpDomain{false}())

"""
    NonNeg(name::Symbol)

A nonnegative scalar parameter named `name`. Finite unconstrained coordinates
map to positive values; the boundary value zero maps back to `-Inf`.
"""
NonNeg(s::Symbol) = _scalar(s, ExpDomain{true}())

"""
    Neg(name::Symbol)

A strictly negative scalar parameter named `name`.
"""
Neg(s::Symbol) = _scalar(s, NegativeExpDomain())

"""
    Prob(name::Symbol)

A scalar probability parameter named `name` in `[0, 1]`. Finite unconstrained
coordinates map to `(0, 1)`; the endpoints map back to infinite coordinates.
"""
Prob(s::Symbol) = _scalar(s, ProbabilityDomain{true,true}())

"""
    ProbOpen(name::Symbol)

A scalar probability parameter named `name` in `(0, 1)`.
"""
ProbOpen(s::Symbol) = _scalar(s, ProbabilityDomain{false,false}())

"""
    ProbOpenLeft(name::Symbol)

A scalar probability parameter named `name` in `(0, 1]`.
"""
ProbOpenLeft(s::Symbol) = _scalar(s, ProbabilityDomain{false,true}())

"""
    ProbOpenRight(name::Symbol)

A scalar probability parameter named `name` in `[0, 1)`.
"""
ProbOpenRight(s::Symbol) = _scalar(s, ProbabilityDomain{true,false}())

"""
    Lower(name::Symbol, lower)

A scalar parameter named `name` constrained to be strictly greater than
`lower`.
"""
Lower(s::Symbol, lower) = _scalar(s, LowerDomain{false,typeof(lower)}(lower))

"""
    LowerClosed(name::Symbol, lower)

A scalar parameter named `name` constrained to be greater than or equal to
`lower`. Finite unconstrained coordinates map strictly above the bound; the
boundary value maps back to `-Inf`.
"""
LowerClosed(s::Symbol, lower) = _scalar(s, LowerDomain{true,typeof(lower)}(lower))

"""
    Bounded(name::Symbol, lower, upper)

A scalar parameter in the closed interval `[lower, upper]`. Finite
unconstrained coordinates map to the open interval; endpoints map back to
infinite coordinates.
"""
Bounded(s::Symbol, lower, upper) = _bounded(s, lower, upper, Val(true), Val(true))

"""
    BoundedOpen(name::Symbol, lower, upper)

A scalar parameter in the open interval `(lower, upper)`.
"""
BoundedOpen(s::Symbol, lower, upper) = _bounded(s, lower, upper, Val(false), Val(false))

"""
    BoundedOpenLeft(name::Symbol, lower, upper)

A scalar parameter in `(lower, upper]`.
"""
BoundedOpenLeft(s::Symbol, lower, upper) = _bounded(s, lower, upper, Val(false), Val(true))

"""
    BoundedOpenRight(name::Symbol, lower, upper)

A scalar parameter in `[lower, upper)`.
"""
BoundedOpenRight(s::Symbol, lower, upper) = _bounded(s, lower, upper, Val(true), Val(false))

"""
    Ordered(first::Symbol, second::Symbol)

Two real scalar parameters satisfying `first < second`.
"""
Ordered(a::Symbol, b::Symbol) = _ordered(a, b, IdentityDomain())

"""
    PosOrdered(first::Symbol, second::Symbol)

Two scalar parameters satisfying `0 < first < second`.
"""
PosOrdered(a::Symbol, b::Symbol) = _ordered(a, b, ExpDomain{false}())

"""
    Between(lower::Symbol, upper::Symbol, value::Symbol)

Three scalar parameters satisfying `lower ≤ value ≤ upper`. Finite
unconstrained coordinates map to the interior `lower < value < upper`.
"""
Between(a::Symbol, b::Symbol, c::Symbol) = Between{a,b,c}()

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
    BilinearQuad(first::Symbol, second::Symbol, p00, p10, p01, p11)

A two-parameter space obtained by mapping the open unit square bilinearly onto
a nondegenerate quadrilateral. `p00`, `p10`, `p01`, and `p11` are its two-
coordinate corners corresponding to latent square coordinates `(0,0)`,
`(1,0)`, `(0,1)`, and `(1,1)`.

Finite unconstrained coordinates first pass through logistic maps, so they land
in the quadrilateral interior. Feasible boundary points are accepted by
`unconstrain` and map to infinite coordinates. The constructor rejects corner
orders for which the bilinear map folds or becomes singular inside the square.
"""
function BilinearQuad(a::Symbol, b::Symbol, p00, p10, p01, p11)
    raw = (_point2(p00), _point2(p10), _point2(p01), _point2(p11))
    T = promote_type((typeof(x) for p in raw for x in p)...)
    corners = ntuple(4) do i
        (convert(T, raw[i][1]), convert(T, raw[i][2]))
    end
    p = BilinearQuad{a,b,T}(corners...)
    return _validate_bilinear_quad(p)
end

# Internal specialized space used by the Distributions.jl extension.
NIG(μ::Symbol, α::Symbol, β::Symbol, δ::Symbol) = NIG{μ,α,β,δ}()

"""
    PosVec(name::Symbol, n::Integer)

An `n`-component vector parameter with strictly positive entries.
"""
function PosVec(s::Symbol, n::Integer)
    n > 0 || throw(ArgumentError("dimension must be positive"))
    return _elementwise(s, ExpDomain{false}(), (Int(n),))
end

"""
    ProbVec(name::Symbol, n::Integer)

An `n`-component vector whose entries are independently constrained to
`[0, 1]`. Use [`Simplex`](@ref) when the entries must also sum to one.
"""
function ProbVec(s::Symbol, n::Integer)
    n > 0 || throw(ArgumentError("dimension must be positive"))
    return _elementwise(s, ProbabilityDomain{true,true}(), (Int(n),))
end

"""
    RealVec(name::Symbol, n::Integer)

An unconstrained real vector parameter with `n` entries.
"""
function RealVec(s::Symbol, n::Integer)
    n >= 0 || throw(ArgumentError("dimension must be nonnegative"))
    return _elementwise(s, IdentityDomain(), (Int(n),))
end

"""
    RealMat(name::Symbol, m::Integer, n::Integer)

An unconstrained real `m × n` matrix parameter, represented as a flattened
vector in column-major order.
"""
function RealMat(s::Symbol, m::Integer, n::Integer)
    m >= 0 || throw(ArgumentError("number of rows must be nonnegative"))
    n >= 0 || throw(ArgumentError("number of columns must be nonnegative"))
    return _elementwise(s, IdentityDomain(), (Int(m), Int(n)))
end

"""
    Simplex(name::Symbol, n::Integer; anchor=n)
    Simplex(name::Symbol, probabilities::AbstractVector)

An `n`-component probability vector constrained to the simplex. The space has
`n - 1` unconstrained coordinates. `anchor` selects the component used as the
reference coordinate for the chart.

When a probability vector is supplied, its largest component is selected as
the anchor.
"""
Simplex(s::Symbol, n::Integer; anchor::Integer=n) = Simplex{s}(n, anchor)

function Simplex(s::Symbol, p::AbstractVector)
    isempty(p) && throw(ArgumentError("probability vector must be nonempty"))
    all(x -> x >= zero(x), p) ||
        throw(DomainError(p, "probabilities must be nonnegative"))

    total = sum(p)
    isapprox(total, one(total)) ||
        throw(DomainError(p, "probabilities must sum to one"))

    anchor = argmax(p)
    p[anchor] > zero(p[anchor]) ||
        throw(DomainError(p, "at least one probability must be strictly positive"))

    return Simplex(s, length(p); anchor=anchor)
end

"""
    SPD(name::Symbol, n::Integer)

An `n × n` symmetric positive-definite matrix parameter. The constrained
representation contains the flattened lower triangle; the unconstrained chart
uses a Cholesky factor with log-transformed diagonal entries.
"""
SPD(s::Symbol, n::Integer) = SPD{s}(n)

"""
    Prefixed(prefix::Symbol, space)

Wrap a parameter space and prefix all names returned by `parameter_symbols`
with `prefix`. The transformation itself is unchanged.
"""
Prefixed(prefix::Symbol, space) = Prefixed{prefix,typeof(space)}(space)

const ProductParameterSpace = Tuple{Vararg{AbstractParameterSpace}}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

parameter_symbol(::ScalarSpace{S}) where {S} = S
parameter_symbols(p::ScalarSpace) = (parameter_symbol(p),)

parameter_symbols(::OrderedSpace{A,B}) where {A,B} = (A, B)
parameter_symbols(::Between{A,B,C}) where {A,B,C} = (A, B, C)
parameter_symbols(::BilinearQuad{A,B}) where {A,B} = (A, B)
parameter_symbols(::NIG{M,A,B,D}) where {M,A,B,D} = (M, A, B, D)

parameter_symbols(p::Simplex{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

parameter_symbols(p::ElementwiseSpace{S,D,1}) where {S,D} =
    ntuple(i -> Symbol(S, "_", i), p.dims[1])

function parameter_symbols(p::ElementwiseSpace{S,D,2}) where {S,D}
    m, n = p.dims
    return ntuple(m * n) do k
        i = mod1(k, m)
        j = (k - 1) ÷ m + 1
        Symbol(S, "_", i, "_", j)
    end
end

function parameter_symbols(p::SPD{S}) where {S}
    result = Symbol[]
    for i in 1:p.n, j in 1:i
        push!(result, Symbol(S, "_", i, "_", j))
    end
    return Tuple(result)
end

function parameter_symbols(p::ProductParameterSpace)
    result = Symbol[]
    for q in p
        append!(result, parameter_symbols(q))
    end
    return Tuple(result)
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

constrained_dimension(p) = length(parameter_symbols(p))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _check_dimension(p, x)
    length(x) == dimension(p) ||
        throw(DimensionMismatch(
            "expected $(dimension(p)) unconstrained parameters, got $(length(x))"
        ))
    return nothing
end

function _check_constrained_dimension(p, x)
    length(x) == constrained_dimension(p) ||
        throw(DimensionMismatch(
            "expected $(constrained_dimension(p)) constrained parameters, got $(length(x))"
        ))
    return nothing
end

function _promoted_vector(x)
    isempty(x) && return Float64[]
    T = promote_type(map(typeof, x)...)
    return T[xi for xi in x]
end

function _diagonal_matrix(d)
    n = length(d)
    n == 0 && return zeros(Float64, 0, 0)
    T = promote_type(map(typeof, d)...)
    J = zeros(T, n, n)
    for i in 1:n
        J[i, i] = d[i]
    end
    return J
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

function _blockdiag(blocks)
    isempty(blocks) && return zeros(Float64, 0, 0)

    nr = sum(B -> size(B, 1), blocks; init=0)
    nc = sum(B -> size(B, 2), blocks; init=0)
    T = promote_type((eltype(B) for B in blocks)...)
    result = zeros(T, nr, nc)

    r0 = c0 = 0
    for B in blocks
        m, n = size(B)
        result[(r0 + 1):(r0 + m), (c0 + 1):(c0 + n)] .= B
        r0 += m
        c0 += n
    end
    return result
end

function _invert_lower_triangular(A)
    n, m = size(A)
    n == m || throw(DimensionMismatch("matrix must be square"))
    n == 0 && return zeros(eltype(A), 0, 0)

    T = eltype(A)
    B = zeros(T, n, n)
    for j in 1:n, i in j:n
        if i == j
            B[i, j] = inv(A[i, i])
        else
            s = zero(T)
            for k in j:(i - 1)
                s += A[i, k] * B[k, j]
            end
            B[i, j] = -s / A[i, i]
        end
    end
    return B
end


# ---------------------------------------------------------------------------
# Scalar-domain transforms
# ---------------------------------------------------------------------------

_constrain_scalar(::IdentityDomain, θ) = (θ, one(θ))
_unconstrain_scalar(::IdentityDomain, η) = η

function _constrain_scalar(::ExpDomain, θ)
    η = exp(θ)
    return η, η
end

function _unconstrain_scalar(::ExpDomain{AllowZero}, η) where {AllowZero}
    valid = AllowZero ? η >= zero(η) : η > zero(η)
    valid || throw(DomainError(
        η,
        AllowZero ? "parameter must be nonnegative" : "parameter must be strictly positive",
    ))
    return log(η)
end

function _constrain_scalar(::NegativeExpDomain, θ)
    η = -exp(θ)
    return η, η
end

function _unconstrain_scalar(::NegativeExpDomain, η)
    η < zero(η) || throw(DomainError(η, "parameter must be strictly negative"))
    return log(-η)
end

function _constrain_scalar(::ProbabilityDomain, θ)
    if θ >= zero(θ)
        z = exp(-θ)
        η = inv(one(θ) + z)
    else
        z = exp(θ)
        η = z / (one(θ) + z)
    end
    return η, η * (one(η) - η)
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

function _constrain_scalar(d::LowerDomain, θ)
    w = exp(θ)
    return d.lower + w, w
end

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
    q, dq = _constrain_scalar(ProbabilityDomain{true,true}(), θ)
    span = d.upper - d.lower
    return d.lower + span * q, span * dq
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
# Separable scalar / elementwise spaces
# ---------------------------------------------------------------------------

function constrain_with_jac(p::ScalarSpace, θ)
    _check_dimension(p, θ)
    η, dηdθ = _constrain_scalar(p.domain, θ[1])
    return [η], reshape([dηdθ], 1, 1)
end

function unconstrain(p::ScalarSpace, η)
    _check_constrained_dimension(p, η)
    return [_unconstrain_scalar(p.domain, η[1])]
end

function constrain_with_jac(p::ElementwiseSpace, θ)
    _check_dimension(p, θ)
    result = [_constrain_scalar(p.domain, x) for x in θ]
    η = _promoted_vector(first.(result))
    dηdθ = _promoted_vector(last.(result))
    return η, _diagonal_matrix(dηdθ)
end

function unconstrain(p::ElementwiseSpace, η)
    _check_constrained_dimension(p, η)
    return _promoted_vector([_unconstrain_scalar(p.domain, x) for x in η])
end


# ---------------------------------------------------------------------------
# Product spaces
# ---------------------------------------------------------------------------

function constrain_with_jac(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)
    values = Any[]
    blocks = Any[]
    offset = 0

    for q in p
        n = dimension(q)
        ηq, Jq = constrain_with_jac(q, view(θ, (offset + 1):(offset + n)))
        push!(values, ηq)
        push!(blocks, Jq)
        offset += n
    end
    return _concatenate_vectors(values), _blockdiag(blocks)
end

function unconstrain(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)
    values = Any[]
    offset = 0

    for q in p
        n = constrained_dimension(q)
        push!(values, unconstrain(q, view(η, (offset + 1):(offset + n))))
        offset += n
    end
    return _concatenate_vectors(values)
end


# ---------------------------------------------------------------------------
# Ordered pairs: a < b, optionally with a > 0
# ---------------------------------------------------------------------------

function constrain_with_jac(p::OrderedSpace, θ)
    _check_dimension(p, θ)
    a, da = _constrain_scalar(p.first_domain, θ[1])
    w = exp(θ[2])
    b = a + w
    return _promoted_vector((a, b)), [da zero(w); da w]
end

function unconstrain(p::OrderedSpace, η)
    _check_constrained_dimension(p, η)
    a, b = η
    θa = _unconstrain_scalar(p.first_domain, a)
    b > a || throw(DomainError(η, "parameters must satisfy a < b"))
    return _promoted_vector((θa, log(b - a)))
end

function unconstrain_with_jac(p::OrderedSpace, η)
    θ = unconstrain(p, η)
    _, da = _constrain_scalar(p.first_domain, θ[1])
    iw = inv(η[2] - η[1])
    return θ, [inv(da) zero(da); -iw iw]
end


# ---------------------------------------------------------------------------
# Bounded interior point: a <= c <= b
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Between, θ)
    _check_dimension(p, θ)
    a = θ[1]
    w = exp(θ[2])
    q, dq = _constrain_scalar(ProbabilityDomain{true,true}(), θ[3])
    b = a + w
    c = a + w * q
    return _promoted_vector((a, b, c)), [
        one(w)  zero(w)  zero(w)
        one(w)  w        zero(w)
        one(w)  w*q      w*dq
    ]
end

function unconstrain(p::Between, η)
    _check_constrained_dimension(p, η)
    a, b, c = η
    a <= c <= b || throw(DomainError(η, "parameters must satisfy a <= c <= b"))

    if b == a
        c == a || throw(DomainError(η, "degenerate bounds require a == b == c"))
        return _promoted_vector((a, -Inf, 0.0))
    end

    w = b - a
    q = (c - a) / w
    z = _unconstrain_scalar(ProbabilityDomain{true,true}(), q)
    return _promoted_vector((a, log(w), z))
end

function unconstrain_with_jac(p::Between, η)
    θ = unconstrain(p, η)
    a, b, c = η
    w = b - a
    w > zero(w) || throw(DomainError(η, "inverse Jacobian is undefined for a == b"))
    ca, bc = c - a, b - c
    return θ, [
        one(w)    zero(w)           zero(w)
        -inv(w)   inv(w)            zero(w)
        -inv(ca)  -inv(bc)          inv(ca)+inv(bc)
    ]
end


# ---------------------------------------------------------------------------
# Bilinear maps from the unit square to a quadrilateral
# ---------------------------------------------------------------------------

@inline function _bilinear_eval(p::BilinearQuad, u, v)
    e, f, g = _bilinear_basis(p)
    point = (
        p.p00[1] + u * e[1] + v * f[1] + u * v * g[1],
        p.p00[2] + u * e[2] + v * f[2] + u * v * g[2],
    )
    du = (e[1] + v * g[1], e[2] + v * g[2])
    dv = (f[1] + u * g[1], f[2] + u * g[2])
    return point, du, dv
end

function constrain_with_jac(p::BilinearQuad, θ)
    _check_dimension(p, θ)
    u, du = _constrain_scalar(ProbabilityDomain{true,true}(), θ[1])
    v, dv = _constrain_scalar(ProbabilityDomain{true,true}(), θ[2])
    point, dpoint_du, dpoint_dv = _bilinear_eval(p, u, v)
    J = [
        dpoint_du[1] * du   dpoint_dv[1] * dv
        dpoint_du[2] * du   dpoint_dv[2] * dv
    ]
    return _promoted_vector(point), J
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
    point, _, _ = _bilinear_eval(p, u, v)
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
    _check_constrained_dimension(p, η)
    u, v = _bilinear_inverse(p, η)
    q = ProbabilityDomain{true,true}()
    return _promoted_vector((_unconstrain_scalar(q, u), _unconstrain_scalar(q, v)))
end

function unconstrain_with_jac(p::BilinearQuad, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)
    a, b, c, d = J[1, 1], J[1, 2], J[2, 1], J[2, 2]
    det = a * d - b * c
    iszero(det) && throw(DomainError(η, "inverse Jacobian is undefined on the quadrilateral boundary"))
    return θ, [d -b; -c a] / det
end


# ---------------------------------------------------------------------------
# Normal-inverse Gaussian parameters: α > |β|, δ > 0
# ---------------------------------------------------------------------------

function constrain_with_jac(p::NIG, θ)
    _check_dimension(p, θ)
    μ = θ[1]
    γ = exp(θ[2])
    β = θ[3]
    δ = exp(θ[4])
    α = hypot(β, γ)
    return _promoted_vector((μ, α, β, δ)), [
        one(α)   zero(α)     zero(α)  zero(α)
        zero(α)  γ*γ/α       β/α      zero(α)
        zero(α)  zero(α)     one(α)   zero(α)
        zero(α)  zero(α)     zero(α)  δ
    ]
end

function unconstrain(p::NIG, η)
    _check_constrained_dimension(p, η)
    μ, α, β, δ = η
    α > abs(β) || throw(DomainError(η, "parameters must satisfy α > |β|"))
    δ > zero(δ) || throw(DomainError(η, "δ must be strictly positive"))
    γ2 = (α - abs(β)) * (α + abs(β))
    return _promoted_vector((μ, log(γ2)/2, β, log(δ)))
end

function unconstrain_with_jac(p::NIG, η)
    θ = unconstrain(p, η)
    _, α, β, δ = η
    γ2 = (α - abs(β)) * (α + abs(β))
    return θ, [
        one(α)   zero(α)   zero(α)    zero(α)
        zero(α)  α/γ2      -β/γ2      zero(α)
        zero(α)  zero(α)   one(α)     zero(α)
        zero(α)  zero(α)   zero(α)    inv(δ)
    ]
end


# ---------------------------------------------------------------------------
# Simplex chart (K constrained coordinates, K-1 free coordinates)
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Simplex, θ)
    _check_dimension(p, θ)
    k = p.n
    k == 1 && return [1.0], zeros(Float64, 1, 0)

    m = max(zero(θ[1]), maximum(θ))
    anchor_weight = exp(-m)
    free_weights = exp.(θ .- m)
    denom = anchor_weight + sum(free_weights)

    T = promote_type(typeof(anchor_weight), eltype(free_weights))
    η = Vector{T}(undef, k)
    j = 1
    for i in 1:k
        if i == p.anchor
            η[i] = anchor_weight / denom
        else
            η[i] = free_weights[j] / denom
            j += 1
        end
    end

    J = zeros(T, k, k - 1)
    j = 1
    for col in 1:k
        col == p.anchor && continue
        for i in 1:k
            J[i, j] = η[i] * ((i == col ? one(T) : zero(T)) - η[col])
        end
        j += 1
    end
    return η, J
end

function unconstrain(p::Simplex, η)
    _check_constrained_dimension(p, η)
    all(x -> x >= zero(x), η) ||
        throw(DomainError(η, "probabilities must be nonnegative"))

    total = sum(η)
    isapprox(total, one(total)) ||
        throw(DomainError(η, "probabilities must sum to one"))

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

function unconstrain_with_jac(p::Simplex, η)
    θ = unconstrain(p, η)
    k = p.n
    k == 1 && return θ, zeros(Float64, 0, 1)

    T = promote_type(map(typeof, η)...)
    J = zeros(T, k - 1, k)
    j = 1
    for i in 1:k
        i == p.anchor && continue
        J[j, i] = inv(η[i])
        J[j, p.anchor] = -inv(η[p.anchor])
        j += 1
    end
    return θ, J
end

# ---------------------------------------------------------------------------
# Symmetric positive-definite matrices, represented by lower triangles
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

function _spd_eta_from_cholesky(L)
    n = size(L, 1)
    pairs = _spd_pairs(n)
    η = Vector{eltype(L)}(undef, length(pairs))
    for (q, (i, j)) in enumerate(pairs)
        s = zero(eltype(L))
        for k in 1:j
            s += L[i, k] * L[j, k]
        end
        η[q] = s
    end
    return η
end

function _spd_matrix_from_eta(p::SPD, η)
    _check_constrained_dimension(p, η)
    T = isempty(η) ? Float64 : promote_type(map(typeof, η)...)
    S = zeros(T, p.n, p.n)
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        S[i, j] = S[j, i] = η[q]
    end
    return S
end

function _spd_cholesky_from_eta(p::SPD, η)
    S = _spd_matrix_from_eta(p, η)
    T = typeof(sqrt(one(eltype(S))))
    L = zeros(T, p.n, p.n)

    for i in 1:p.n, j in 1:i
        s = convert(T, S[i, j])
        for k in 1:(j - 1)
            s -= L[i, k] * L[j, k]
        end
        if i == j
            s > zero(s) || throw(DomainError(η, "matrix must be positive definite"))
            L[i, j] = sqrt(s)
        else
            L[i, j] = s / L[j, j]
        end
    end
    return L
end

function constrain_with_jac(p::SPD, θ)
    L = _spd_cholesky_from_theta(p, θ)
    η = _spd_eta_from_cholesky(L)
    pairs = _spd_pairs(p.n)
    T = eltype(L)
    J = zeros(T, length(pairs), length(pairs))

    for (q, (i, j)) in enumerate(pairs), (r, (a, b)) in enumerate(pairs)
        dL = a == b ? L[a, b] : one(T)
        v = zero(T)
        a == i && b <= j && (v += dL * L[j, b])
        a == j && b <= j && (v += L[i, b] * dL)
        J[q, r] = v
    end
    return η, J
end

function unconstrain(p::SPD, η)
    L = _spd_cholesky_from_eta(p, η)
    pairs = _spd_pairs(p.n)
    θ = Vector{eltype(L)}(undef, length(pairs))
    for (q, (i, j)) in enumerate(pairs)
        θ[q] = i == j ? log(L[i, j]) : L[i, j]
    end
    return θ
end

function unconstrain_with_jac(p::SPD, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)
    return θ, _invert_lower_triangular(J)
end


# ---------------------------------------------------------------------------
# Prefix wrapper and derived operations
# ---------------------------------------------------------------------------

constrain_with_jac(p::Prefixed, θ) = constrain_with_jac(p.space, θ)
unconstrain(p::Prefixed, η) = unconstrain(p.space, η)
unconstrain_with_jac(p::Prefixed, η) = unconstrain_with_jac(p.space, η)

constrain(p, θ) = first(constrain_with_jac(p, θ))
constrain_jac(p, θ) = last(constrain_with_jac(p, θ))
unconstrain_jac(p, η) = last(unconstrain_with_jac(p, η))

function unconstrain_with_jac(p::AbstractSeparableParameterSpace, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)
    return θ, _diagonal_matrix([inv(J[i, i]) for i in 1:dimension(p)])
end

function unconstrain_with_jac(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)
    values = Any[]
    blocks = Any[]
    offset = 0

    for q in p
        n = constrained_dimension(q)
        θq, Jq = unconstrain_with_jac(q, view(η, (offset + 1):(offset + n)))
        push!(values, θq)
        push!(blocks, Jq)
        offset += n
    end
    return _concatenate_vectors(values), _blockdiag(blocks)
end

unconstrained_example(p) = zeros(dimension(p))
constrained_example(p) = constrain(p, unconstrained_example(p))


# ---------------------------------------------------------------------------
# Jacobian determinants
# ---------------------------------------------------------------------------

function logabsdet_constrain_jac(p::AbstractSeparableParameterSpace, θ)
    _check_dimension(p, θ)
    isempty(θ) && return 0.0
    _, J = constrain_with_jac(p, θ)
    s = zero(J[1, 1])
    for i in 1:dimension(p)
        s += log(abs(J[i, i]))
    end
    return s
end

logabsdet_unconstrain_jac(p::AbstractSeparableParameterSpace, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)
    s = 0.0
    offset = 0
    for q in p
        n = dimension(q)
        s += logabsdet_constrain_jac(q, view(θ, (offset + 1):(offset + n)))
        offset += n
    end
    return s
end

function logabsdet_unconstrain_jac(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)
    s = 0.0
    offset = 0
    for q in p
        n = constrained_dimension(q)
        s += logabsdet_unconstrain_jac(q, view(η, (offset + 1):(offset + n)))
        offset += n
    end
    return s
end

function logabsdet_constrain_jac(p::OrderedSpace, θ)
    _check_dimension(p, θ)
    _, da = _constrain_scalar(p.first_domain, θ[1])
    return log(abs(da)) + θ[2]
end

logabsdet_unconstrain_jac(p::OrderedSpace, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::Between, θ)
    _check_dimension(p, θ)
    _, dq = _constrain_scalar(ProbabilityDomain{true,true}(), θ[3])
    return 2*θ[2] + log(dq)
end

logabsdet_unconstrain_jac(p::Between, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::BilinearQuad, θ)
    _check_dimension(p, θ)
    _, J = constrain_with_jac(p, θ)
    return log(abs(J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]))
end

logabsdet_unconstrain_jac(p::BilinearQuad, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::NIG, θ)
    _check_dimension(p, θ)
    γ = exp(θ[2])
    α = hypot(θ[3], γ)
    return 2*θ[2] - log(α) + θ[4]
end

logabsdet_unconstrain_jac(p::NIG, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::SPD, θ)
    _check_dimension(p, θ)
    s = p.n * log(2.0)
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        i == j && (s += (p.n - i + 2) * θ[q])
    end
    return s
end

logabsdet_unconstrain_jac(p::SPD, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

logabsdet_constrain_jac(p::Prefixed, θ) =
    logabsdet_constrain_jac(p.space, θ)
logabsdet_unconstrain_jac(p::Prefixed, η) =
    logabsdet_unconstrain_jac(p.space, η)

function logabsdet_constrain_jac(p::Simplex, θ)
    _check_dimension(p, θ)
    throw(ArgumentError(
        "logabsdet is not defined for the rectangular simplex Jacobian; " *
        "use constrain_with_jac to access the K×(K-1) Jacobian"
    ))
end

function logabsdet_unconstrain_jac(p::Simplex, η)
    _check_constrained_dimension(p, η)
    throw(ArgumentError(
        "logabsdet is not defined for the rectangular simplex Jacobian; " *
        "use unconstrain_with_jac to access the (K-1)×K Jacobian"
    ))
end


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

function constrained_namedtuple(p, θ)
    η = constrain(p, θ)
    return NamedTuple{parameter_symbols(p)}(Tuple(η))
end


end # module