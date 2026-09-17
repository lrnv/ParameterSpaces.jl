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

`ParameterSpaces.jl` provides transformations between unconstrained optimization
coordinates `θ ∈ ℝⁿ` and constrained continuous parameters `η`, together with
analytic Jacobians.

The package is split into two layers:

- the **core package**, which contains parameter-space representations, associated
  transformation and Jacobians, with no dependencies.
- a **`Distributions.jl` package extension** as a proof of concept, which adds `param_space` methods
  for supported distribution types when `Distributions.jl` is loaded.

The main convention is

```julia
p = param_space(object)
η = constrain(p, θ)
θ = unconstrain(p, η)
```

`param_space` is an open generic function. The core package provides the
parameter-space machinery, while concrete object-to-space mappings can be added
by package extensions or by the user. The extension shipped here covers almmost all `Distributions.jl` objects.

## Installation

Install and the core package with

```julia
using Pkg
Pkg.add(url="https://github.com/lrnv/ParameterSpaces.jl")
using ParameterSpaces
```

## Quick start with Distributions.jl

```julia
using ParameterSpaces, Distributions

d = Gamma(2.0, 3.0)
p = param_space(d)

dimension(p)
# 2

parameter_symbols(p)
# (:α, :θ)

θ = [log(2.0), log(3.0)]

η = constrain(p, θ)
# ≈ [2.0, 3.0]

unconstrain(p, η)
# ≈ θ
```

The parameter-space object returned by `param_space` can generally be treated
as opaque. The exported interface is intended to be sufficient for downstream
optimization code.

## Public interface

### `param_space`

```julia
p = param_space(object)
```

Return the parameter space associated with an object when a mapping is
available.

`ParameterSpaces.jl` itself defines the generic function. Concrete mappings are
provided by extensions. In particular, loading `Distributions.jl` activates the
bundled distribution extension.

For distribution instances, the concrete instance can matter because the
parameter-space dimension or structure may depend on values stored in the
distribution, for example for `Dirichlet`, `Categorical`, `Multinomial`,
multivariate distributions, and mixture models.

The bundled extension deliberately does not guess a mapping for unsupported
distribution types: `param_space(d)` throws an `ArgumentError` when no specific
mapping is implemented.

### Dimensions

```julia
dimension(p)
constrained_dimension(p)
```

`dimension(p)` is the number of unconstrained optimization coordinates.

`constrained_dimension(p)` is the length of the constrained continuous
parameter vector returned by `constrain`.

For most parameter spaces the two are equal. They differ for constrained
manifolds such as a simplex.

For example:

```julia
p = param_space(Categorical([0.2, 0.3, 0.5]))

dimension(p)
# 2

constrained_dimension(p)
# 3
```

A categorical distribution with `K` probabilities has only `K - 1`
independent degrees of freedom.

### Parameter names

```julia
parameter_symbols(p)
```

Return names for the constrained continuous parameters.

```julia
p = param_space(Normal(0.0, 1.0))

parameter_symbols(p)
# (:μ, :σ)
```

Vector and matrix parameters are flattened and receive indexed names.

### Transformations

```julia
η = constrain(p, θ)
θ = unconstrain(p, η)
```

`constrain` maps unconstrained optimization coordinates to valid constrained
parameters.

`unconstrain` applies the inverse chart.

Round trips hold whenever the point is in the interior of the relevant chart:

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

To avoid evaluating a transformation twice, use

```julia
η, J = constrain_with_jac(p, θ)
θ, Jinv = unconstrain_with_jac(p, η)
```

For maximum-likelihood optimization, if `gη` is the gradient with respect to
the constrained parameters,

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

These are useful for change-of-variables calculations.

They are not required merely to optimize a likelihood after reparameterizing
its parameters.

For parameter spaces with rectangular Jacobians, such as a simplex,
`logabsdet_*` is intentionally undefined and throws an `ArgumentError`.

### Examples

```julia
θ = unconstrained_example(p)
η = constrained_example(p)
```

Return simple valid example points.

### Named constrained parameters

```julia
constrained_namedtuple(p, θ)
```

Example:

```julia
p = param_space(Gamma(2.0, 3.0))

constrained_namedtuple(p, [log(2.0), log(3.0)])
# (α = 2.0, θ = 3.0)
```

## Distributions.jl extension

The `Distributions.jl` integration lives in
`ext/ParameterSpacesDistributionsExt.jl` and is declared through Julia's package
extension mechanism. `Distributions` is therefore a weak dependency rather than
a core dependency.

This keeps the transformation engine usable on its own while retaining the
convenient distribution mappings when `Distributions.jl` is present.

The extension currently covers a broad range of distribution families,
including:

- common continuous and discrete univariate distributions,
- ordered and bounded parameterizations,
- simplex-valued probability parameters,
- multivariate normal families,
- positive-definite matrix parameters,
- matrix-variate distributions,
- mixture models,
- product distributions,
- affine distributions,
- truncated and censored distributions,
- order-statistic and reshaped wrappers.

Coverage can be queried directly by calling `param_space(d)`. Unsupported types
fail explicitly instead of silently guessing a parameterization.

## Simplex-valued parameters

Probability vectors are represented using `K - 1` unconstrained coordinates
for `K` constrained probabilities.

```julia
d = Categorical([0.2, 0.3, 0.5])
p = param_space(d)

θ = unconstrained_example(p)
η, J = constrain_with_jac(p, θ)

length(θ)
# 2

length(η)
# 3

size(J)
# (3, 2)

sum(η)
# ≈ 1
```

The same principle is used for distributions such as `Multinomial` and for
mixture weights.

## Positive-definite matrix parameters

Positive-definite matrix parameters are represented through unconstrained
Cholesky coordinates.

For example, multivariate normal covariance or precision matrices are mapped
to a Euclidean vector while preserving positive definiteness after
`constrain`.

The constrained representation exposed by this package is flattened; use
`parameter_symbols` to inspect its ordering.

## Structural parameters

For `Distributions.jl`, `param_space` describes the **continuous parameters to
optimize**, not necessarily every constructor argument of a distribution.

Discrete or structural parameters are intentionally omitted when appropriate.

Examples include:

```julia
Binomial(n, p)       # n is structural, p is optimized
Multinomial(n, p)    # n is structural, p is optimized
Erlang(k, θ)         # k is structural, θ is optimized
```

Likewise, truncation/censoring bounds, reshape dimensions, order-statistic
indices, and similar metadata are treated as structural by the corresponding
wrapper mappings.

A distribution may therefore legitimately have

```julia
dimension(param_space(d)) == 0
```

when it contains no continuous parameter to optimize.

## Boundary behavior

Finite unconstrained coordinates map to the interior of constrained domains.

Some inverse transformations also accept valid boundary values and map them to
infinite unconstrained coordinates. For example:

```julia
p = param_space(Bernoulli(0.5))

unconstrain(p, [0.0])
# [-Inf]

unconstrain(p, [1.0])
# [Inf]
```

This makes the constrained domain faithfully represent distributions that allow
boundary values while retaining an unconstrained Euclidean chart for finite
optimization coordinates.

## Design goals

The package aims to keep the optimization-facing interface small:

```julia
p = param_space(object)

η, J = constrain_with_jac(p, θ)
θ = unconstrain(p, η)
```

The transformations and Jacobians are implemented analytically. The core
package does not require `Distributions.jl`; distribution-specific mappings are
loaded only through the optional package extension.
