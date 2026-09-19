```@meta
CurrentModule = Paramorph
```

# Paramorph

`Paramorph.jl` describes parameter geometry with a deliberately tiny
interface. A parameter space connects a flat unconstrained vector, suitable for
optimization, to the natural constrained Julia values used by a model.

There is **one open hook and five operations**:

```julia
p = param_space(x)

dimension(p)
names(p)
example(p)
η = constrain(p, θ)
θ = unconstrain(p, η)
```

That is the complete conceptual API. Constructors such as `Pos`, `Simplex`,
`SPD`, and `Correlation` are only vocabulary for describing different parameter
geometries. Specialized spaces may additionally expose operations tied to their
geometry; for example, `correlation_factor` returns the triangular factor used
by a `Correlation` chart.

## The complete interface

```@docs
param_space
dimension
names
example
constrain
unconstrain
```

## Representation model

The package deliberately keeps only two mathematical representations:

```text
flat unconstrained vector  <---->  natural constrained value
                          constrain
                        unconstrain
```

The unconstrained side is always a vector. The constrained side can be a
scalar, vector, matrix, or tuple depending on the parameter space. `names(p)` is
metadata attached to the logical constrained parameters; it does not force the
constrained value into a `NamedTuple`.

For example:

```julia
p = (
    RealVec(:μ, 3),
    Pos(:σ),
    Correlation(:R, 3),
)

θ = zeros(dimension(p))
η = constrain(p, θ)

names(p)
# (:μ, :σ, :R)

dimension(p)
# 7

unconstrain(p, η) ≈ θ
# true
```

## Space constructors

### Scalar spaces

```@docs
Id
Pos
NonNeg
Neg
Prob
ProbOpen
ProbOpenLeft
ProbOpenRight
Lower
LowerClosed
Bounded
BoundedOpen
BoundedOpenLeft
BoundedOpenRight
```

### Related scalar parameters

```@docs
Ordered
PosOrdered
Between
```

### Dependent scalar parameters

`DependentProduct` is a small sequential product for cases where a later scalar
parameter has a bound determined by an earlier constrained scalar. `GreaterThan`
and `LowerThan` can also intersect that dynamic bound with fixed intrinsic bounds.
This keeps simple conditional geometries explicit without introducing a general
constraint solver.

```@docs
DependentProduct
GreaterThan
LowerThan
```

For example, the second parameter below is constrained to remain at least as
large as the first while also staying in `[0, 1]`:

```julia
p = DependentProduct(
    Bounded(:parent, 0.0, 1.0),
    GreaterThan(:child, :parent; lower=0.0, upper=1.0),
)

parent, child = constrain(p, zeros(2))
parent <= child
# true
```

### Vector, matrix, and structured spaces

```@docs
RealVec
PosVec
ProbVec
NonNegVec
LowerClosedVec
RealMat
Simplex
SPD
Correlation
correlation_factor
```

`SPD(:Σ, n)` describes an arbitrary symmetric positive-definite matrix.
`Correlation(:R, n)` describes a positive-definite correlation matrix with unit
diagonal. Both return full matrices on the constrained side.

### Naming wrappers

```@docs
Prefixed
```

Tuples of spaces form Cartesian products. `Prefixed` changes logical names but
not the underlying transformation.

## Distributions.jl extension

The bundled extension associates common `Distributions.jl` models with parameter
spaces through the same `param_space` hook used by downstream packages. The core
package itself has no dependency on `Distributions.jl`.

```@example distribution-types
using Paramorph
using Distributions

X = MvNormal(zeros(3), [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0])
p = param_space(X)

(names(p), dimension(p))
```

The constrained values retain their constructor-level shapes:

```@example distribution-types
μ, Σ = example(p)
(size(μ), size(Σ))
```

For `MvNormalCanon`, the matrix `J` is a precision matrix. Geometrically it is
an arbitrary symmetric positive-definite matrix, so its parameter space is
`SPD(:J, n)`.

## Automatic differentiation

There is no differentiation-specific extension and no Jacobian API. `constrain`
is implemented with ordinary generic Julia operations, so AD packages can pass
through it directly.

```julia
using ForwardDiff, Paramorph

p = (Id(:μ), Pos(:σ))

ForwardDiff.gradient([0.3, -0.2]) do θ
    μ, σ = constrain(p, θ)
    μ^2 + σ
end
```

```@index
```