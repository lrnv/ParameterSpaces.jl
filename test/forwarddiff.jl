struct AnalyticOnlySpace <: AbstractParameterSpace end

ParameterSpaces.dimension(::AnalyticOnlySpace) = 1
ParameterSpaces.constrained_dimension(::AnalyticOnlySpace) = 1
ParameterSpaces.parameter_symbols(::AnalyticOnlySpace) = (:x,)

function ParameterSpaces.constrain_with_jac(
    ::AnalyticOnlySpace,
    x::AbstractVector{<:AbstractFloat},
)
    y = x[1]^3
    return [y], reshape([3x[1]^2], 1, 1)
end

@testset "ForwardDiff extension" begin
    @test Base.get_extension(ParameterSpaces, :ParameterSpacesForwardDiffExt) !== nothing

    @testset "analytic Jacobians drive Dual propagation" begin
        p = AnalyticOnlySpace()
        x = [1.7]

        @test ForwardDiff.jacobian(z -> constrain(p, z), x) ≈
              constrain_jac(p, x)
    end

    @testset "built-in spaces" begin
        cases = (
            (Pos(:x), [0.3]),
            ((Id(:μ), Pos(:σ)), [0.2, -0.4]),
            (Simplex(:p, 3), [0.2, -0.4]),
            (SPD(:Σ, 2), [log(1.2), 0.3, log(0.8)]),
        )

        for (p, x) in cases
            _, J = constrain_with_jac(p, x)
            @test ForwardDiff.jacobian(z -> constrain(p, z), x) ≈ J
        end
    end

    @testset "nested ForwardDiff" begin
        p = (Id(:μ), Pos(:σ))
        x = [0.3, -0.2]
        f(z) = sum(abs2, constrain(p, z))

        @test ForwardDiff.hessian(f, x) ≈ [
            2.0  0.0
            0.0  4exp(2x[2])
        ]
    end
end
