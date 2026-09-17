```@meta
CurrentModule = ParameterSpaces
```

# ParameterSpaces

`ParameterSpaces.jl` provides a small public vocabulary for describing
constrained parameter spaces and mapping them to unconstrained Euclidean
coordinates for optimization.

A space can be constructed directly:

```julia
using ParameterSpaces

p = (Id(:μ), Pos(:σ), Simplex(:weights, 3))
η = constrain(p, [0.0, 0.0, 0.0, 0.0])
```

or associated with an arbitrary object by extending [`param_space`](@ref):

```julia
import ParameterSpaces: param_space

struct MyModel
    n::Int
end

param_space(m::MyModel) = (Id(:μ), Pos(:σ), Simplex(:weights, m.n))
```

The bundled `Distributions.jl` package extension uses exactly this mechanism.
The core package itself has no dependency on `Distributions.jl`.

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
Prefixed
```

Tuples of parameter spaces form Cartesian product spaces. For example,
`(Id(:μ), Pos(:σ))` describes an unconstrained location and a positive scale.

## Object mapping

```@docs
param_space
```

## Transformation API

```@docs
AbstractParameterSpace
dimension
constrained_dimension
parameter_symbols
constrain
unconstrain
constrain_jac
unconstrain_jac
constrain_with_jac
unconstrain_with_jac
logabsdet_constrain_jac
logabsdet_unconstrain_jac
unconstrained_example
constrained_example
constrained_namedtuple
```

```@index
```
