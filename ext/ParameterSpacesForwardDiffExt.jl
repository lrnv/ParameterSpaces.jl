module ParameterSpacesForwardDiffExt

using ForwardDiff
import ParameterSpaces
import ParameterSpaces: constrain, unconstrain

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

function constrain(
    p,
    θ::AbstractVector{<:ForwardDiff.Dual{T,V,N}},
) where {T,V,N}
    primal = ForwardDiff.value.(θ)
    η, J = ParameterSpaces.constrain_with_jac(p, primal)
    return _lift_duals(T, Val(N), θ, η, J)
end

function unconstrain(
    p,
    η::AbstractVector{<:ForwardDiff.Dual{T,V,N}},
) where {T,V,N}
    primal = ForwardDiff.value.(η)
    θ, Jinv = ParameterSpaces.unconstrain_with_jac(p, primal)
    return _lift_duals(T, Val(N), η, θ, Jinv)
end

end
