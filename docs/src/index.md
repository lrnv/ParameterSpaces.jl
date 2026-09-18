```@meta
CurrentModule = ParameterSpaces
```

# ParameterSpaces

`ParameterSpaces.jl` describes parameter geometry with a deliberately tiny
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
geometries.

## The complete interface

### `param_space`

```@docs
param_space
```

`param_space(x)` associates an object or type with its parameter-space
description. It is an open generic function and is normally the only function a
downstream package extends.

```julia
import ParameterSpaces: param_space

struct MyModel
    n::Int
end

param_space(m::MyModel) = (
    Id(:μ),
    Pos(:σ),
    Simplex(:weights, m.n),
)
```

### `dimension`

`dimension(p)` is the number of scalar coordinates in the **unconstrained**
representation. Equivalently, `constrain(p, θ)` expects
`length(θ) == dimension(p)`.

A structured constrained parameter does not change this rule. For example, a
`3 × 3` correlation matrix has three free coordinates:

```julia
p = Correlation(:R, 3)
dimension(p) == 3
```

### `names`

`names(p)` returns the names of the **logical constrained parameters**. A vector
or matrix parameter has one name, not one name per scalar entry.

```julia
p = (RealVec(:μ, 3), SPD(:Σ, 3))
names(p)
# (:μ, :Σ)
```

Thus `dimension(p)` counts optimizer coordinates, while `names(p)` describes the
model-level parameter blocks. They answer different questions and need not have
the same length.

### `example`

`example(p)` returns one canonical value in the constrained representation. It
is defined from the origin of the unconstrained chart:

```julia
example(p) == constrain(p, zeros(dimension(p)))
```

It is intended as a convenient valid representative of the space, not as a
statistical default or fitted value.

### `constrain`

`constrain(p, θ)` maps a flat unconstrained vector to its natural constrained
representation.

The output keeps the model-level Julia structure:

```julia
constrain(Pos(:σ), [0.0])
# 1.0

constrain(RealVec(:μ, 3), zeros(3))
# [0.0, 0.0, 0.0]

constrain(SPD(:Σ, 2), zeros(3))
# [1.0 0.0; 0.0 1.0]
```

For a Cartesian product of spaces, the constrained result is a tuple of the
logical parameter values:

```julia
p = (RealVec(:μ, 3), Pos(:σ), Correlation(:R, 3))
η = constrain(p, zeros(dimension(p)))
# (μ_vector, σ_scalar, R_matrix)
```

The constrained side is therefore not flattened merely for the convenience of
an optimizer.

### `unconstrain`

`unconstrain(p, η)` performs the inverse transformation: it accepts the natural
constrained value and returns the flat unconstrained vector.

For interior points of a chart,

```julia
θ = randn(dimension(p))
unconstrain(p, constrain(p, θ)) ≈ θ
```

Closed boundaries may naturally correspond to infinite unconstrained
coordinates; this is part of the geometry of the chosen space.

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

### Vector, matrix, and structured spaces

```@docs
RealVec
PosVec
ProbVec
RealMat
Simplex
SPD
Correlation
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
using ParameterSpaces
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
using ForwardDiff, ParameterSpaces

p = (Id(:μ), Pos(:σ))

ForwardDiff.gradient([0.3, -0.2]) do θ
    μ, σ = constrain(p, θ)
    μ^2 + σ
end
```

```@index
```
