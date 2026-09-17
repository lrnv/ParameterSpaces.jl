export Correlation

"""An `n × n` positive-definite correlation-matrix parameter."""
struct Correlation{S} <: AbstractParameterSpace
    n::Int

    function Correlation{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("matrix dimension must be positive"))
        new{S}(Int(n))
    end
end

Correlation(name::Symbol, n::Integer) = Correlation{name}(n)

parameter_symbols(::Correlation{S}) where {S} = (S,)
dimension(p::Correlation) = p.n * (p.n - 1) ÷ 2
constrained_dimension(p::Correlation) = dimension(p)

function constrain(p::Correlation, θ)
    _check_dimension(p, θ)
    T = isempty(θ) ? Float64 : promote_type(map(typeof, θ)...)
    L = zeros(T, p.n, p.n)
    q = 0

    for i in 1:p.n
        scale = i == 1 ? one(T) : one(L[i - 1, i - 1])
        for j in 1:(i - 1)
            q += 1
            z = tanh(θ[q])
            L[i, j] = scale * z
            scale *= sqrt(one(z) - z * z)
        end
        L[i, i] = scale
    end

    return L * transpose(L)
end

function _check_correlation_matrix(p::Correlation, R)
    size(R) == (p.n, p.n) || throw(DimensionMismatch("expected a $(p.n)×$(p.n) matrix"))
    for i in 1:p.n
        isapprox(R[i, i], one(R[i, i])) || throw(DomainError(R, "correlation matrix must have unit diagonal"))
        for j in 1:(i - 1)
            isapprox(R[i, j], R[j, i]) || throw(DomainError(R, "correlation matrix must be symmetric"))
        end
    end
    return nothing
end

function unconstrain(p::Correlation, R)
    _check_correlation_matrix(p, R)
    L = _spd_cholesky_from_matrix(SPD{:correlation}(p.n), R)
    θ = Vector{eltype(L)}(undef, dimension(p))
    q = 0

    for i in 2:p.n
        scale = one(L[i, i])
        for j in 1:(i - 1)
            z = L[i, j] / scale
            abs(z) < one(z) || throw(DomainError(R, "correlation matrix lies on the boundary of the chart"))
            q += 1
            θ[q] = atanh(z)
            scale *= sqrt(one(z) - z * z)
        end
    end

    return θ
end
