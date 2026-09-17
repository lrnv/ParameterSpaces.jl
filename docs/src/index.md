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
```

### Naming wrappers

```@docs
Prefixed
```

Tuples of parameter spaces form Cartesian product spaces. For example,
`(Id(:μ), Pos(:σ))` describes an unconstrained location and a positive scale.

## Object mapping

```@docs
param_space
```

### Distributions.jl types

When the parameter space is completely determined by a `Distributions.jl`
distribution type, the extension supports both the type and an instance:

```@example distribution-types
using ParameterSpaces
using Distributions

parameter_symbols(param_space(Normal))
```

```@example distribution-types
parameter_symbols(param_space(typeof(Gamma(2.0, 3.0))))
```

The same type-level API is available for distributions whose omitted constructor
arguments are purely structural, for example `Binomial` and `Erlang`.

An instance is still required when the space depends on stored values or runtime
shape information. Examples include `Dirichlet`, `Categorical`, `Multinomial`,
`PoissonBinomial`, multivariate and matrix-variate distributions, mixtures, and
wrappers whose space depends on their contained distribution.

## Transformation interface

Once a space `p` is available, the main public operations are:

```julia
η = constrain(p, θ)
θ = unconstrain(p, η)

J = constrain_jac(p, θ)
Jinv = unconstrain_jac(p, η)

η, J = constrain_with_jac(p, θ)
θ, Jinv = unconstrain_with_jac(p, η)
```

Use `dimension`, `constrained_dimension`, and `parameter_symbols` to inspect the
space, and `logabsdet_constrain_jac` / `logabsdet_unconstrain_jac` for
change-of-variables calculations when the Jacobian is square.

### ForwardDiff integration

When `ForwardDiff.jl` is loaded together with `ParameterSpaces.jl`, the optional
ForwardDiff extension propagates dual-number partials through `constrain` using
the analytical Jacobian already provided by the parameter space. This is the
hot direction for unconstrained optimization: optimizer coordinates are mapped
to valid constrained parameters before evaluating the objective. No additional
user API is required:

```julia
using ForwardDiff, ParameterSpaces

p = (Id(:μ), Pos(:σ))
θ = [0.3, -0.2]

ForwardDiff.jacobian(x -> constrain(p, x), θ)
# equivalent to constrain_jac(p, θ)
```

The rule also composes with nested ForwardDiff differentiation, so higher-order
derivatives of objectives that call `constrain` remain available. `unconstrain`
keeps its ordinary implementation; it is primarily used to initialize optimizer
coordinates from constrained parameters rather than inside the optimization
hot path.

```@index
```
