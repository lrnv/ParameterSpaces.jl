```@meta
CurrentModule = ParameterSpaces
```

# ParameterSpaces

`ParameterSpaces.jl` describes constrained model parameters and maps a flat
unconstrained vector to the parameters in their natural Julia representation.
The unconstrained side is deliberately simple for optimizers; the constrained
side preserves scalars, vectors, matrices, and coupled tuples.

```julia
using ParameterSpaces

p = (Id(:μ), Pos(:σ), Simplex(:weights, 3))
θ = zeros(dimension(p))
η = constrain(p, θ)
# (0.0, 1.0, [1/3, 1/3, 1/3])
```

Associate a space with an arbitrary object by extending [`param_space`](@ref):

```julia
import ParameterSpaces: param_space

struct MyModel
    n::Int
end

param_space(m::MyModel) = (Id(:μ), Pos(:σ), Simplex(:weights, m.n))
```

The bundled `Distributions.jl` extension uses exactly this mechanism. The core
package itself has no dependency on `Distributions.jl`.

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
BilinearQuad
```

### Vector, matrix, and structured spaces

```@docs
RealVec
PosVec
ProbVec
RealMat
Simplex
SPD
```

### Naming wrappers

```@docs
Prefixed
```

Tuples of spaces form Cartesian products. Vector and matrix spaces remain one
logical parameter each: `parameter_symbols(RealVec(:μ, 3)) == (:μ,)`, and
`parameter_symbols(SPD(:Σ, 3)) == (:Σ,)`.

## Object mapping

```@docs
param_space
```

### Distributions.jl

When a space is determined by a `Distributions.jl` type, both type-level and
instance-level mappings are available. Runtime shape information still requires
an instance for multivariate, matrix-variate, mixture, and similar models.

```@example distribution-types
using ParameterSpaces
using Distributions

parameter_symbols(param_space(Normal))
```

For structured parameters the constrained result has the constructor-level
shape:

```@example distribution-types
X = MvNormal(zeros(3), [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0])
p = param_space(X)
θ = unconstrained_example(p)
η = constrained_example(p)
(size(η[1]), size(η[2]), keys(constrained_namedtuple(p, θ)))
```

## Transformation interface

The public transformation interface is intentionally first-order and small:

```julia
η = constrain(p, θ)
θ = unconstrain(p, η)
```

`θ` is always a flat vector. `η` has the natural constrained representation:
a scalar parameter is a scalar, a vector parameter is a vector, an SPD
parameter is a full symmetric matrix, and a Cartesian product is a tuple of
logical parameter values.

Use `dimension` for the optimizer dimension, `constrained_dimension` for the
number of independent scalar constrained coordinates, and `parameter_symbols`
for logical parameter names. `constrained_namedtuple(p, θ)` combines those
logical names with the natural constrained values.

### Automatic differentiation

There is no differentiation-specific extension. `constrain` is implemented
with ordinary generic Julia operations, so AD packages can differentiate
through it directly.

```julia
using ForwardDiff, ParameterSpaces

p = (Id(:μ), Pos(:σ))
θ = [0.3, -0.2]

ForwardDiff.gradient(θ) do x
    μ, σ = constrain(p, x)
    μ^2 + σ
end
```

```@index
```
