using Aqua
using Distributions
using ParameterSpaces
using Test

@testset "Aqua" begin
    Aqua.test_all(ParameterSpaces)
end

@testset "Core parameter spaces" begin
    @testset "scalar spaces" begin
        p = Pos(:σ)
        @test dimension(p) == 1
        @test parameter_symbols(p) == (:σ,)
        @test constrain(p, [0.0]) == 1.0
        @test unconstrain(p, 1.0) == [0.0]
        @test constrained_example(p) == 1.0
        @test constrained_namedtuple(p, [0.0]) == (; σ=1.0)

        @test unconstrain(NonNeg(:x), 0.0) == [-Inf]
        @test unconstrain(Prob(:p), 0.0) == [-Inf]
        @test unconstrain(Prob(:p), 1.0) == [Inf]
        @test_throws DomainError unconstrain(Pos(:x), 0.0)
        @test_throws DomainError unconstrain(ProbOpen(:p), 0.0)
    end

    @testset "natural vector and matrix values" begin
        pv = RealVec(:μ, 3)
        @test dimension(pv) == 3
        @test parameter_symbols(pv) == (:μ,)
        @test constrain(pv, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
        @test constrained_namedtuple(pv, [1.0, 2.0, 3.0]) == (; μ=[1.0, 2.0, 3.0])
        @test unconstrain(pv, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

        pm = RealMat(:A, 2, 3)
        A = constrain(pm, collect(1.0:6.0))
        @test A isa AbstractMatrix
        @test size(A) == (2, 3)
        @test A == [1.0 3.0 5.0; 2.0 4.0 6.0]
        @test parameter_symbols(pm) == (:A,)
        @test unconstrain(pm, A) == collect(1.0:6.0)
    end

    @testset "product spaces preserve parameter blocks" begin
        p = (Id(:μ), Pos(:σ), RealVec(:β, 2))
        θ = [0.5, log(2.0), 3.0, 4.0]
        η = constrain(p, θ)

        @test η == (0.5, 2.0, [3.0, 4.0])
        @test parameter_symbols(p) == (:μ, :σ, :β)
        @test dimension(p) == 4
        @test unconstrain(p, η) ≈ θ

        nt = constrained_namedtuple(p, θ)
        @test nt == (; μ=0.5, σ=2.0, β=[3.0, 4.0])
    end

    @testset "ordered, between, and bilinear spaces" begin
        po = Ordered(:a, :b)
        ηo = constrain(po, [1.0, log(2.0)])
        @test ηo == (1.0, 3.0)
        @test unconstrain(po, ηo) ≈ [1.0, log(2.0)]

        pb = Between(:a, :b, :c)
        ηb = constrain(pb, [1.0, log(4.0), 0.0])
        @test collect(ηb) ≈ [1.0, 5.0, 3.0]
        @test unconstrain(pb, ηb) ≈ [1.0, log(4.0), 0.0]

        pq = BilinearQuad(:x, :y, (0.0, 0.0), (1.5, -0.5), (0.0, 0.5), (1.0, 0.0))
        θq = [0.3, -0.4]
        ηq = constrain(pq, θq)
        @test ηq isa Tuple
        @test length(ηq) == 2
        @test unconstrain(pq, ηq) ≈ θq
    end

    @testset "simplex values are vectors" begin
        p = Simplex(:weights, 3)
        θ = [0.2, -0.4]
        η = constrain(p, θ)
        @test η isa AbstractVector
        @test length(η) == 3
        @test sum(η) ≈ 1.0
        @test all(>(0), η)
        @test parameter_symbols(p) == (:weights,)
        @test unconstrain(p, η) ≈ θ
        @test constrained_namedtuple(p, θ).weights ≈ η
    end

    @testset "SPD values are full matrices" begin
        p = SPD(:Σ, 3)
        θ = [log(1.2), 0.2, log(0.8), -0.1, 0.3, log(1.5)]
        Σ = constrain(p, θ)

        @test Σ isa AbstractMatrix
        @test size(Σ) == (3, 3)
        @test Σ ≈ Σ'
        @test all(Σ[i, i] > 0 for i in axes(Σ, 1))
        @test parameter_symbols(p) == (:Σ,)
        @test dimension(p) == 6
        @test unconstrain(p, Σ) ≈ θ
        @test constrained_namedtuple(p, θ).Σ ≈ Σ

        @test_throws DomainError unconstrain(SPD(:Σ, 2), [1.0 2.0; 2.0 1.0])
        @test_throws DomainError unconstrain(SPD(:Σ, 2), [1.0 0.1; 0.2 1.0])
    end

    @testset "prefixes only change names" begin
        p = Prefixed(:base, (Id(:μ), Pos(:σ)))
        θ = [1.0, 0.0]
        @test parameter_symbols(p) == (:base_μ, :base_σ)
        @test constrain(p, θ) == (1.0, 1.0)
        @test constrained_namedtuple(p, θ) == (; base_μ=1.0, base_σ=1.0)
    end
end

@testset "Distributions.jl extension" begin
    @testset "simple distributions" begin
        cases = (
            (Normal(0.0, 1.0), (:μ, :σ), 2),
            (Exponential(1.0), (:θ,), 1),
            (Gamma(2.0, 3.0), (:α, :θ), 2),
            (Beta(2.0, 3.0), (:α, :β), 2),
            (Bernoulli(0.5), (:p,), 1),
            (Uniform(-1.0, 2.0), (:a, :b), 2),
        )

        for (d, names, n) in cases
            p = param_space(d)
            @test parameter_symbols(p) == names
            @test dimension(p) == n
            θ = unconstrained_example(p)
            η = constrained_example(p)
            @test unconstrain(p, η) ≈ θ
            @test keys(constrained_namedtuple(p, θ)) == names
        end
    end

    @testset "Dirichlet keeps one vector parameter" begin
        p = param_space(Dirichlet([1.0, 2.0, 3.0]))
        @test parameter_symbols(p) == (:α,)
        @test dimension(p) == 3
        α = constrain(p, log.([1.0, 2.0, 3.0]))
        @test α ≈ [1.0, 2.0, 3.0]
        @test constrained_namedtuple(p, log.([1.0, 2.0, 3.0])) == (; α=α)
    end

    @testset "multivariate normal families keep natural parameter shapes" begin
        Σ0 = [1.0 0.2 0.1; 0.2 1.5 0.3; 0.1 0.3 2.0]
        X = MvNormal(zeros(3), Σ0)
        p = param_space(X)

        @test parameter_symbols(p) == (:μ, :Σ)
        @test dimension(p) == 9

        θ = unconstrained_example(p)
        η = constrained_example(p)
        @test η isa Tuple
        @test length(η) == 2
        @test η[1] isa AbstractVector
        @test size(η[1]) == (3,)
        @test η[2] isa AbstractMatrix
        @test size(η[2]) == (3, 3)
        @test η[2] ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

        nt = constrained_namedtuple(p, θ)
        @test keys(nt) == (:μ, :Σ)
        @test nt.μ == zeros(3)
        @test nt.Σ ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

        Xcanon = MvNormalCanon([0.1, -0.2], [2.0 0.2; 0.2 1.2])
        pcanon = param_space(Xcanon)
        @test parameter_symbols(pcanon) == (:h, :J)
        @test dimension(pcanon) == 5
        ηcanon = constrained_example(pcanon)
        @test size(ηcanon[1]) == (2,)
        @test size(ηcanon[2]) == (2, 2)
        @test unconstrain(pcanon, ηcanon) ≈ unconstrained_example(pcanon)
    end

    @testset "matrix-variate shapes" begin
        M = zeros(2, 3)
        U = [1.0 0.2; 0.2 1.0]
        V = [1.0 0.1 0.0; 0.1 1.0 0.2; 0.0 0.2 1.0]
        d = MatrixNormal(M, U, V)
        p = param_space(d)
        @test parameter_symbols(p) == (:M, :U, :V)
        η = constrained_example(p)
        @test size(η[1]) == (2, 3)
        @test size(η[2]) == (2, 2)
        @test size(η[3]) == (3, 3)
    end
end

include("forwarddiff.jl")
