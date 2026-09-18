# Paramorph.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/Paramorph.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/Paramorph.jl/dev/)
[![Build Status](https://github.com/lrnv/Paramorph.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lrnv/Paramorph.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lrnv/Paramorph.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lrnv/Paramorph.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/D/Paramorph.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/D/Paramorph.html)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

---

`Paramorph.jl` does one thing: it describes how model parameters are
represented in unconstrained Euclidean coordinates and how to map them back to
their natural constrained Julia values.

The public interface is intentionally tiny: **one open hook and five operations**.

```julia
p = param_space(x)      # associate an object with a parameter space

dimension(p)            # number of unconstrained scalar coordinates
names(p)                # logical names of the constrained parameters
example(p)              # one canonical constrained value
η = constrain(p, θ)     # flat unconstrained vector -> natural constrained value
θ = unconstrain(p, η)   # natural constrained value -> flat unconstrained vector
```

That is the whole interface. Everything else exported by the package (`Pos`,
`Simplex`, `SPD`, `Correlation`, ...) is vocabulary for describing a parameter
space.

## Other approaches

Several Julia packages address related parts of the parameter-representation problem, but with different abstractions and goals:

- [`TransformVariables.jl`](https://github.com/tpapp/TransformVariables.jl) provides composable transformations from unconstrained Euclidean vectors to constrained, structured values, with first-class support for inverses and log-Jacobian determinants. It is particularly well suited to change-of-variables problems in Bayesian inference and MCMC.
- [`ParameterHandling.jl`](https://github.com/JuliaGaussianProcesses/ParameterHandling.jl) starts from structured parameter values and provides recursive `flatten`/`unflatten` machinery, together with parameter wrappers such as positive or bounded parameters. Its main goal is to bridge convenient Julia data structures and flat representations used by optimizers and inference algorithms.
- [`DifferentiableFlatten.jl`](https://github.com/JuliaNonconvex/DifferentiableFlatten.jl) focuses on differentiable flattening and reconstruction of arbitrary Julia data structures. It handles the structural problem of mapping nested objects to vectors, rather than describing the geometry or constraints of a model's parameter space.
- [`ModelWrappers.jl`](https://github.com/paschermayr/ModelWrappers.jl) provides a higher-level framework around structured model parameters, combining flattening, constraints, bijectors, Jacobian corrections, mutable model state, and automatic differentiation utilities.

Paramorph takes a different starting point. Rather than asking *how to flatten this value* or *which transformation should be applied to this vector*, it asks:

> **What is the parameter space of this object?**

`param_space` is an open interface associating a Julia object or type with a declarative description of its logical parameters and their geometry. From that description, Paramorph derives the unconstrained dimension, parameter names, a canonical example, and the maps between flat Euclidean coordinates and natural constrained Julia values.

This makes the parameter space independent of any particular current parameter value, optimizer, inference algorithm, prior distribution, AD backend, or Jacobian convention. Paramorph is therefore intended primarily as a small interoperability layer: model packages describe their parameter spaces once, and generic numerical tools can consume that description without knowing how the model itself is implemented.

One last difference is that `Paramorph` has no dependencies.


## The two representations

A parameter space connects two deliberately different representations:

- the **unconstrained** representation is always a flat vector, convenient for
  optimization and automatic differentiation;
- the **constrained** representation keeps the natural Julia shape of each
  parameter: a scalar stays a scalar, a vector stays a vector, a matrix stays a
  matrix, and a product of logical parameters is a tuple.

```julia
using Paramorph

p = (
    RealVec(:μ, 3),
    Pos(:σ),
    Correlation(:R, 3),
)

names(p)
# (:μ, :σ, :R)

dimension(p)
# 7

θ = zeros(dimension(p))
η = constrain(p, θ)
# ([0.0, 0.0, 0.0], 1.0, [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])

unconstrain(p, η) ≈ θ
# true
```

`names` describes the **logical parameters**, not the scalar coordinates used by
an optimizer. Thus a vector or matrix parameter has one name:

```julia
names((RealVec(:μ, 3), SPD(:Σ, 3)))
# (:μ, :Σ)
```

`example(p)` is simply a convenient canonical constrained value. It is obtained
from the origin of the unconstrained chart:

```julia
example(p) == constrain(p, zeros(dimension(p)))
```

## Open object interface

`param_space` is the only hook downstream packages normally extend:

```julia
import Paramorph: param_space

struct MyModel end

param_space(::Type{MyModel}) = (
    Id(:location),
    Pos(:scale),
)
```

The bundled `Distributions.jl` extension uses exactly this mechanism while the
core package remains independent of `Distributions.jl`.

```julia
using Distributions, Paramorph

X = MvNormal(3, 1 / 2)
p = param_space(X)

names(p)
# (:μ, :Σ)

μ, Σ = example(p)
size(μ)  # (3,)
size(Σ)  # (3, 3)
```

## Space vocabulary

The package provides reusable spaces for common geometries: unconstrained,
positive and negative scalars; probabilities and bounded intervals; ordered or
coupled scalars; vectors and matrices; simplexes; symmetric positive-definite
matrices; and correlation matrices. Tuples of spaces form Cartesian products.

These types only describe geometry. The transformation interface remains the
same five functions regardless of which spaces are composed.

## Automatic differentiation

`Paramorph.jl` does not implement differentiation rules or maintain an AD
extension. `constrain` is ordinary generic Julia code, so differentiation can
pass through naturally:

```julia
using ForwardDiff

p = (Id(:μ), Pos(:σ))
f(θ) = begin
    μ, σ = constrain(p, θ)
    μ^2 + σ
end

ForwardDiff.gradient(f, [0.3, -0.2])
```

There is no Jacobian API to learn or maintain.
