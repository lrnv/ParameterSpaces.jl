# ParameterSpaces.jl

`ParameterSpaces.jl` provides a small vocabulary for describing constrained
model parameters and mapping them to unconstrained Euclidean coordinates for
optimization.

The design is intentionally asymmetric:

- the **unconstrained** representation is always a flat vector;
- the **constrained** representation keeps the natural Julia shape of each
  parameter: scalar, vector, matrix, or coupled tuple.

```julia
using ParameterSpaces

p = (Id(:μ), Pos(:σ), Simplex(:weights, 3))
θ = zeros(dimension(p))

constrain(p, θ)
# (0.0, 1.0, [1/3, 1/3, 1/3])

NamedTuple{names(p)}(constrain(p, θ))
# (μ = 0.0, σ = 1.0, weights = [1/3, 1/3, 1/3])
```

The inverse transformation consumes the same natural representation:

```julia
η = constrain(p, θ)
unconstrain(p, η) ≈ θ
```

## Structured parameters

Vector and matrix spaces are logical parameters, not collections of scalar
parameter names.

```julia
p = (RealVec(:μ, 3), SPD(:Σ, 3))
names(p)
# (:μ, :Σ)

μ, Σ = example(p)
size(μ) # (3,)
size(Σ) # (3, 3)
```

`SPD` returns the full symmetric positive-definite matrix. `Simplex` returns a
probability vector. `RealMat` returns a matrix. The flat coordinates needed by
an optimizer remain entirely on the unconstrained side.

## Open object interface

`param_space` is an open generic function:

```julia
import ParameterSpaces: param_space

struct MyModel end
param_space(::Type{MyModel}) = (Id(:location), Pos(:scale))
```

A `Distributions.jl` extension supplies mappings for common distributions while
keeping the core package independent of `Distributions.jl`.

```julia
using Distributions, ParameterSpaces

X = MvNormal(zeros(3), [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0])
p = param_space(X)
θ = unconstrain(p,example(p))

NamedTuple{names(p)}(constrain(p, θ))
# (μ = [...], Σ = [...])
```

## Automatic differentiation

ParameterSpaces does not implement or override differentiation rules.
`constrain` uses ordinary generic Julia operations, so automatic differentiation
can pass through naturally:

```julia
using ForwardDiff

p = (Id(:μ), Pos(:σ))
f(θ) = begin
    μ, σ = constrain(p, θ)
    μ^2 + σ
end

ForwardDiff.gradient(f, [0.3, -0.2])
```

There is no Jacobian API or ForwardDiff extension to maintain.
