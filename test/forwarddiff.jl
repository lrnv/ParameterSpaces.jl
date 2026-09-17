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

@testset "Additional core spaces" begin
    @testset "closed lower bound" begin
        p = LowerClosed(:x, 1.0)
        @test constrain(p, [log(2.0)]) ≈ [3.0]
        @test unconstrain(p, [1.0]) == [-Inf]
        @test_throws DomainError unconstrain(p, [0.9])
    end

    @testset "bounded scalar intervals" begin
        p = Bounded(:x, -2.0, 4.0)
        @test constrain(p, [0.0]) ≈ [1.0]
        @test unconstrain(p, [-2.0]) == [-Inf]
        @test unconstrain(p, [4.0]) == [Inf]
        @test unconstrain(p, [1.0]) ≈ [0.0]

        @test_throws DomainError unconstrain(BoundedOpen(:x, -2.0, 4.0), [-2.0])
        @test_throws DomainError unconstrain(BoundedOpen(:x, -2.0, 4.0), [4.0])
        @test_throws DomainError unconstrain(BoundedOpenLeft(:x, -2.0, 4.0), [-2.0])
        @test unconstrain(BoundedOpenLeft(:x, -2.0, 4.0), [4.0]) == [Inf]
        @test unconstrain(BoundedOpenRight(:x, -2.0, 4.0), [-2.0]) == [-Inf]
        @test_throws DomainError unconstrain(BoundedOpenRight(:x, -2.0, 4.0), [4.0])
    end

    @testset "bilinear quadrilateral" begin
        p = BilinearQuad(
            :θ₁,
            :θ₂,
            (0.0, 0.0),
            (1.5, -0.5),
            (0.0, 0.5),
            (1.0, 0.0),
        )
        x = [-0.7, 0.9]
        y, J = constrain_with_jac(p, x)
        x₂, Jinv = unconstrain_with_jac(p, y)

        @test parameter_symbols(p) == (:θ₁, :θ₂)
        @test x₂ ≈ x
        @test Jinv * J ≈ [1.0 0.0; 0.0 1.0]
        @test isapprox(
            logabsdet_constrain_jac(p, x) + logabsdet_unconstrain_jac(p, y),
            0.0;
            atol=1e-12,
            rtol=0.0,
        )

        @test unconstrain(p, [0.0, 0.0]) == [-Inf, -Inf]
        @test_throws DomainError unconstrain(p, [2.0, 2.0])
        @test_throws ArgumentError BilinearQuad(
            :x,
            :y,
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        )
    end
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
            ((Pos(:θ), LowerClosed(:δ, 1.0)), [0.2, -0.4]),
            (Bounded(:ρ, -0.5, 1.0), [0.3]),
            (BilinearQuad(
                :θ₁,
                :θ₂,
                (0.0, 0.0),
                (1.5, -0.5),
                (0.0, 0.5),
                (1.0, 0.0),
            ), [-0.4, 0.6]),
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
