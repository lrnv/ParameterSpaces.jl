using ForwardDiff

@testset "Automatic differentiation passes through constrain" begin
    @testset "scalar product" begin
        p = (Id(:μ), Pos(:σ))
        θ = [0.3, -0.2]
        f(x) = begin
            μ, σ = constrain(p, x)
            μ^2 + σ
        end
        g = ForwardDiff.gradient(f, θ)
        @test g ≈ [0.6, exp(-0.2)]
    end

    @testset "elementwise vector" begin
        p = PosVec(:x, 3)
        θ = [0.1, -0.2, 0.3]
        J = ForwardDiff.jacobian(x -> constrain(p, x), θ)
        expected = [
            exp(θ[1])  0.0        0.0
            0.0        exp(θ[2])  0.0
            0.0        0.0        exp(θ[3])
        ]
        @test J ≈ expected
    end

    @testset "simplex" begin
        p = Simplex(:p, 3)
        θ = [0.2, -0.4]
        J = ForwardDiff.jacobian(x -> constrain(p, x), θ)
        @test size(J) == (3, 2)
        @test vec(sum(J; dims=1)) ≈ zeros(2) atol=1e-12
        @test all(isfinite, J)
    end

    @testset "SPD matrix" begin
        p = SPD(:Σ, 3)
        θ = [0.1, 0.2, -0.1, 0.3, -0.2, 0.4]
        f(x) = sum(abs2, constrain(p, x))
        g = ForwardDiff.gradient(f, θ)
        @test length(g) == dimension(p)
        @test all(isfinite, g)
    end

    @testset "correlation matrix" begin
        p = Correlation(:R, 4)
        θ = [0.2, -0.3, 0.4, 0.1, -0.2, 0.35]
        R = constrain(p, θ)

        @test parameter_symbols(p) == (:R,)
        @test dimension(p) == 6
        @test size(R) == (4, 4)
        @test R ≈ R'
        @test all(isapprox(R[i, i], 1.0) for i in axes(R, 1))
        @test unconstrain(p, R) ≈ θ
        @test constrained_namedtuple(p, θ).R ≈ R

        g = ForwardDiff.gradient(x -> sum(abs2, constrain(p, x)), θ)
        @test length(g) == dimension(p)
        @test all(isfinite, g)

        @test_throws DomainError unconstrain(Correlation(:R, 2), [2.0 0.0; 0.0 1.0])
        @test_throws DomainError unconstrain(Correlation(:R, 2), [1.0 0.2; 0.1 1.0])
        @test_throws DomainError unconstrain(Correlation(:R, 2), [1.0 1.2; 1.2 1.0])
    end

    @testset "coupled bilinear space" begin
        p = BilinearQuad(:x, :y, (0.0, 0.0), (1.5, -0.5), (0.0, 0.5), (1.0, 0.0))
        f(x) = sum(constrain(p, x))
        g = ForwardDiff.gradient(f, [0.2, -0.3])
        @test length(g) == 2
        @test all(isfinite, g)
    end
end
