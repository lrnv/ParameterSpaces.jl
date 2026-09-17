module ParameterSpacesForwardDiffExt

using ForwardDiff
import ParameterSpaces
import ParameterSpaces: constrain

const _DualVector{T,V,N} = AbstractVector{<:ForwardDiff.Dual{T,V,N}}

@inline function _pushforward_partials(J, x, i, k)
    return mapreduce(
        j -> J[i, j] * ForwardDiff.partials(x[j])[k],
        +,
        axes(J, 2);
        init=zero(eltype(J)),
    )
end

function _lift_duals(::Type{T}, ::Val{N}, x, y, J) where {T,N}
    return [
        ForwardDiff.Dual{T}(
            y[i],
            ntuple(k -> _pushforward_partials(J, x, i, k), N),
        )
        for i in eachindex(y)
    ]
end

# Separable scalar maps are cheaper to trace directly than to materialize a
# Jacobian, multiply seed partials through it, and reconstruct Duals.  This is
# especially important for small product spaces used in optimizer hot loops.
function constrain(
    p::ParameterSpaces.ScalarSpace,
    θ::_DualVector{T,V,N},
) where {T,V,N}
    ParameterSpaces._check_dimension(p, θ)
    η, _ = ParameterSpaces._constrain_scalar(p.domain, θ[1])
    return [η]
end

function constrain(
    p::ParameterSpaces.ElementwiseSpace,
    θ::_DualVector{T,V,N},
) where {T,V,N}
    ParameterSpaces._check_dimension(p, θ)
    return [first(ParameterSpaces._constrain_scalar(p.domain, x)) for x in θ]
end

# A product is differentiated block-by-block.  Separable children take the
# direct path above while genuinely coupled children still use their analytic
# Jacobian through the generic fallback below.  Avoiding the global block
# diagonal Jacobian removes a large allocation penalty for tiny products.
function constrain(
    p::ParameterSpaces.ProductParameterSpace,
    θ::_DualVector{T,V,N},
) where {T,V,N}
    ParameterSpaces._check_dimension(p, θ)
    η = similar(θ, ParameterSpaces.constrained_dimension(p))
    θoffset = 0
    ηoffset = 0

    for q in p
        nθ = ParameterSpaces.dimension(q)
        nη = ParameterSpaces.constrained_dimension(q)
        ηq = constrain(q, view(θ, (θoffset + 1):(θoffset + nθ)))
        copyto!(η, ηoffset + 1, ηq, 1, nη)
        θoffset += nθ
        ηoffset += nη
    end
    return η
end

constrain(
    p::ParameterSpaces.Prefixed,
    θ::_DualVector{T,V,N},
) where {T,V,N} = constrain(p.space, θ)

# Coupled spaces use the authoritative analytic Jacobian supplied by
# ParameterSpaces.  Only one ForwardDiff layer is stripped; nested Dual values
# remain in the primal calculation, so higher-order differentiation composes.
function constrain(
    p,
    θ::_DualVector{T,V,N},
) where {T,V,N}
    primal = ForwardDiff.value.(θ)
    η, J = ParameterSpaces.constrain_with_jac(p, primal)
    return _lift_duals(T, Val(N), θ, η, J)
end

end
