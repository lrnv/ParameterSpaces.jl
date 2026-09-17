# ParameterSpaces.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/ParameterSpaces.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/ParameterSpaces.jl/dev/)
[![Build Status](https://github.com/lrnv/ParameterSpaces.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lrnv/ParameterSpaces.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lrnv/ParameterSpaces.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lrnv/ParameterSpaces.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/D/ParameterSpaces.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/D/ParameterSpaces.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

---

`ParameterSpaces.jl` describes constrained parameter spaces and provides
transformations between unconstrained optimization coordinates `θ ∈ ℝⁿ` and
valid constrained parameters `η`, together with analytic Jacobians.

The package has two layers:

- the **core package** provides public constructors for parameter spaces and the
  transformation/Jacobian machinery, with no dependencies;
- optional **package extensions** can define `param_space` mappings for external
  object types. The bundled `Distributions.jl` extension is one such mapping.

A parameter space can therefore be constructed directly:

```julia
using ParameterSpaces

p = (Id(:μ), Pos(:σ))

θ = [1.5, log(2.0)]
η = constrain(p, θ)
# [1.5, 2.0]

unconstrain(p, η)
# ≈ θ
```

or associated with an object through the open `param_space` interface:

```julia
p = param_space(object)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/lrnv/ParameterSpaces.jl")
using ParameterSpaces
```

`Distributions.jl` is optional. Install and load it only when you want the
bundled distribution mappings:

```julia
Pkg.add("Distributions")
using Distributions
```

Loading both packages automatically activates the package extension.

## Defining a parameter space for your own type

`param_space` is deliberately an open generic function. To integrate your own
object, define a method returning one parameter space or a tuple of parameter
spaces.

```julia
using ParameterSpaces
import ParameterSpaces: param_space

struct MyModel
    μ::Float64
    σ::Float64
    weights::Vector{Float64}
end

param_space(m::MyModel) = (
    Id(:μ),
    Pos(:σ),
    Simplex(:weights, m.weights),
)

model = MyModel(0.0, 1.0, [0.2, 0.3, 0.5])
p = param_space(model)

dimension(p)
# 4

constrained_dimension(p)
# 5

parameter_symbols(p)
# (:μ, :σ, :weights_1, :weights_2, :weights_3)
```

The object-to-space mapping is separate from the transformation engine. The
core package does not need to know anything about `MyModel` itself.

## Public space constructors

The constructors below are the public vocabulary for describing parameter
constraints. Their concrete implementation types are intentionally kept
internal so downstream code can depend on the semantic constructors rather
than on representation details.

| Constructor | Constrained parameter space |
| --- | --- |
| `Id(:x)` | `x ∈ ℝ` |
| `Pos(:x)` | `x > 0` |
| `NonNeg(:x)` | `x ≥ 0` |
| `Neg(:x)` | `x < 0` |
| `Prob(:p)` | `p ∈ [0, 1]` |
| `ProbOpen(:p)` | `p ∈ (0, 1)` |
| `ProbOpenLeft(:p)` | `p ∈ (0, 1]` |
| `ProbOpenRight(:p)` | `p ∈ [0, 1)` |
| `Lower(:x, a)` | `x > a` |
| `Ordered(:a, :b)` | `a < b` |
| `PosOrdered(:a, :b)` | `0 < a < b` |
| `Between(:a, :b, :c)` | `a ≤ c ≤ b` |
| `RealVec(:x, n)` | real vector of length `n` |
| `PosVec(:x, n)` | positive vector of length `n` |
| `ProbVec(:p, n)` | vector with entries in `[0, 1]` |
| `RealMat(:A, m, n)` | real `m × n` matrix |
| `Simplex(:p, n)` | `n` probabilities summing to one |
| `SPD(:Σ, n)` | symmetric positive-definite `n × n` matrix |
| `Prefixed(:base, p)` | space `p` with prefixed parameter names |

Finite unconstrained coordinates always map to the interior of the constrained
domain. Some inverse transforms accept boundary values and represent them with
infinite unconstrained coordinates. For example:

```julia
p = Prob(:p)

unconstrain(p, [0.0])
# [-Inf]

unconstrain(p, [1.0])
# [Inf]
```

### Product spaces

A tuple of spaces represents their Cartesian product, so heterogeneous models
can be built compositionally:

```julia
p = (
    Id(:location),
    Pos(:scale),
    Prob(:mixing_probability),
)

parameter_symbols(p)
# (:location, :scale, :mixing_probability)

dimension(p)
# 3
```

### Vector and matrix spaces

Vector and matrix parameters are represented by flat constrained vectors.
`parameter_symbols` records the ordering:

```julia
parameter_symbols(RealVec(:x, 3))
# (:x_1, :x_2, :x_3)

parameter_symbols(RealMat(:A, 2, 2))
# (:A_1_1, :A_2_1, :A_1_2, :A_2_2)
```

`ProbVec` constrains each component independently. If the components must sum to
one, use `Simplex` instead.

### Simplex spaces

A simplex with `K` constrained probabilities has only `K - 1` unconstrained
degrees of freedom:

```julia
p = Simplex(:p, 3)

dimension(p)
# 2

constrained_dimension(p)
# 3

θ = unconstrained_example(p)
η, J = constrain_with_jac(p, θ)

sum(η)
# ≈ 1

size(J)
# (3, 2)
```

The optional `anchor` keyword chooses the reference component for the chart:

```julia
p = Simplex(:p, 4; anchor=2)
```

Passing an existing probability vector chooses its largest component as the
anchor, which is useful when the chart is selected from an existing object:

```julia
p = Simplex(:p, [0.1, 0.6, 0.3])
```

### Positive-definite matrices

`SPD(:Σ, n)` represents a symmetric positive-definite matrix through
unconstrained Cholesky coordinates. The constrained representation is the
flattened lower triangle:

```julia
p = SPD(:Σ, 3)

dimension(p)
# 6

parameter_symbols(p)
# (:Σ_1_1, :Σ_2_1, :Σ_2_2, :Σ_3_1, :Σ_3_2, :Σ_3_3)
```

## Transformation interface

For any parameter space `p`:

```julia
η = constrain(p, θ)
θ = unconstrain(p, η)
```

`dimension(p)` is the number of unconstrained optimization coordinates, while
`constrained_dimension(p)` is the length of the constrained representation.
They are equal for ordinary square transformations and differ for spaces such
as a simplex.

`parameter_symbols(p)` returns names for the constrained representation.

Round trips hold whenever the point belongs to the relevant chart:

```julia
θ ≈ unconstrain(p, constrain(p, θ))
η ≈ constrain(p, unconstrain(p, η))
```

### Jacobians

```julia
J = constrain_jac(p, θ)
Jinv = unconstrain_jac(p, η)
```

with

```math
J = \frac{\partial \eta}{\partial \theta}.
```

To avoid evaluating a transformation twice, use:

```julia
η, J = constrain_with_jac(p, θ)
θ, Jinv = unconstrain_with_jac(p, η)
```

For example, if `gη` is a gradient with respect to the constrained parameters,

```julia
η, J = constrain_with_jac(p, θ)
gθ = J' * gη
```

gives the gradient with respect to the unconstrained coordinates.

### Log absolute Jacobian determinant

```julia
logabsdet_constrain_jac(p, θ)
logabsdet_unconstrain_jac(p, η)
```

These are useful for change-of-variables calculations. For spaces with
rectangular Jacobians, such as `Simplex`, `logabsdet_*` is intentionally
undefined and throws an `ArgumentError`.

### Example points and named output

```julia
θ = unconstrained_example(p)
η = constrained_example(p)
```

return simple valid example points.

`constrained_namedtuple` combines a transformation with the constrained
parameter names:

```julia
p = (Id(:μ), Pos(:σ))

constrained_namedtuple(p, [1.0, log(2.0)])
# (μ = 1.0, σ = 2.0)
```

## Distributions.jl extension

The optional integration lives in `ext/ParameterSpacesDistributionsExt.jl`.
`Distributions` is a weak dependency, so the core transformation engine remains
independent of it.

```julia
using ParameterSpaces, Distributions

d = Gamma(2.0, 3.0)
p = param_space(d)

parameter_symbols(p)
# (:α, :θ)

θ = [log(2.0), log(3.0)]
constrain(p, θ)
# ≈ [2.0, 3.0]
```

The extension covers a broad range of `Distributions.jl` families, including
common univariate distributions, multivariate and matrix-variate families,
mixtures, product distributions, simplex-valued parameters, positive-definite
matrix parameters, and common wrappers.

For this extension, `param_space(d)` describes the **continuous parameters to
optimize**, not necessarily every constructor argument. Discrete or structural
parameters such as trial counts, reshape dimensions, truncation bounds, and
order-statistic indices can therefore be omitted from the optimization space.

Unsupported distribution types fail explicitly instead of silently guessing a
parameterization.

## Design principle

The intended extension point is the semantic space-construction API:

```julia
param_space(object) = (
    Id(:location),
    Pos(:scale),
    Simplex(:weights, 3),
)
```

Downstream code should construct spaces with the public constructors above and
use the generic transformation interface. Internal concrete types and scalar
domain implementations are not part of the public API.
