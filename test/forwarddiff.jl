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

    @testset "coupled bilinear space" begin
        p = BilinearQuad(:x, :y, (0.0, 0.0), (1.5, -0.5), (0.0, 0.5), (1.0, 0.0))
        f(x) = sum(constrain(p, x))
        g = ForwardDiff.gradient(f, [0.2, -0.3])
        @test length(g) == 2
        @test all(isfinite, g)
    end
end
