using Aqua
using Distributions
using Paramorph
using ForwardDiff
using Test

@testset "Aqua" begin
    Aqua.test_all(Paramorph)
end

@testset "Core parameter spaces" begin
    @testset "scalar spaces" begin
        p = Pos(:σ)
        @test dimension(p) == 1
        @test names(p) == (:σ,)
        @test constrain(p, [0.0]) == 1.0
        @test unconstrain(p, 1.0) == [0.0]
        @test example(p) == 1.0

        @test unconstrain(NonNeg(:x), 0.0) == [-Inf]
        @test unconstrain(Prob(:p), 0.0) == [-Inf]
        @test unconstrain(Prob(:p), 1.0) == [Inf]
        @test_throws DomainError unconstrain(Pos(:x), 0.0)
        @test_throws DomainError unconstrain(ProbOpen(:p), 0.0)

        for θ in (-1000.0, 1000.0)
            q = constrain(ProbOpen(:p), [θ])
            @test 0.0 < q < 1.0
            x = constrain(BoundedOpen(:x, -1.0, 1.0), [θ])
            @test -1.0 < x < 1.0
        end
    end

    @testset "scalar spaces tuple entries" begin
        p = Pos(:σ)
        @test unconstrain(p, (1.0,)) == [0.0]
    end

    @testset "natural vector and matrix values" begin
        pv = RealVec(:μ, 3)
        @test dimension(pv) == 3
        @test names(pv) == (:μ,)
        @test constrain(pv, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
        @test unconstrain(pv, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

        pm = RealMat(:A, 2, 3)
        A = constrain(pm, collect(1.0:6.0))
        @test A isa AbstractMatrix
        @test size(A) == (2, 3)
        @test A == [1.0 3.0 5.0; 2.0 4.0 6.0]
        @test names(pm) == (:A,)
        @test unconstrain(pm, A) == collect(1.0:6.0)
    end

    @testset "product spaces preserve parameter blocks" begin
        p = (Id(:μ), Pos(:σ), RealVec(:β, 2))
        θ = [0.5, log(2.0), 3.0, 4.0]
        η = constrain(p, θ)

        @test η == (0.5, 2.0, [3.0, 4.0])
        @test names(p) == (:μ, :σ, :β)
        @test dimension(p) == 4
        @test unconstrain(p, η) ≈ θ
        nt = NamedTuple{names(p)}(η)
        @test nt == (; μ=0.5, σ=2.0, β=[3.0, 4.0])
    end

    @testset "ordered and between" begin
        po = Ordered(:a, :b)
        ηo = constrain(po, [1.0, log(2.0)])
        @test ηo == (1.0, 3.0)
        @test unconstrain(po, ηo) ≈ [1.0, log(2.0)]

        pb = Between(:a, :b, :c)
        ηb = constrain(pb, [1.0, log(4.0), 0.0])
        @test collect(ηb) ≈ [1.0, 5.0, 3.0]
        @test unconstrain(pb, ηb) ≈ [1.0, log(4.0), 0.0]
    end

    @testset "simplex values are vectors" begin
        p = Simplex(:weights, 3)
        θ = [0.2, -0.4]
        η = constrain(p, θ)
        @test η isa AbstractVector
        @test length(η) == 3
        @test sum(η) ≈ 1.0
        @test all(>(0), η)
        @test names(p) == (:weights,)
        @test unconstrain(p, η) ≈ θ
    end

    @testset "SPD values are full matrices" begin
        p = SPD(:Σ, 3)
        θ = [log(1.2), 0.2, log(0.8), -0.1, 0.3, log(1.5)]
        Σ = constrain(p, θ)

        @test Σ isa AbstractMatrix
        @test size(Σ) == (3, 3)
        @test Σ ≈ Σ'
        @test all(Σ[i, i] > 0 for i in axes(Σ, 1))
        @test names(p) == (:Σ,)
        @test dimension(p) == 6
        @test unconstrain(p, Σ) ≈ θ

        @test_throws DomainError unconstrain(SPD(:Σ, 2), [1.0 2.0; 2.0 1.0])
        @test_throws DomainError unconstrain(SPD(:Σ, 2), [1.0 0.1; 0.2 1.0])
    end

    @testset "prefixes only change names" begin
        p = Prefixed(:base, (Id(:μ), Pos(:σ)))
        θ = [1.0, 0.0]
        @test names(p) == (:base_μ, :base_σ)
        @test constrain(p, θ) == (1.0, 1.0)
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

        for (d, nms, n) in cases
            p = param_space(d)
            @test names(p) == nms
            @test dimension(p) == n
            θ = unconstrain(p,example(p))
            η = example(p)
            @test unconstrain(p, η) ≈ θ
        end
    end

    @testset "mapping smoke tests" begin
        distributions = (
            Cauchy(0.0, 1.0),
            LogNormal(0.0, 1.0),
            LogitNormal(0.0, 1.0),
            Rayleigh(1.0),
            Chi(2.0),
            Chisq(2.0),
            TDist(3.0),
            FDist(2.0, 3.0),
            BernoulliLogit(0.3),
            BetaBinomial(10, 2.0, 3.0),
            BetaPrime(2.0, 3.0),
            Laplace(0.0, 1.0),
            Logistic(0.0, 1.0),
            Gumbel(0.0, 1.0),
            Levy(0.0, 1.0),
            Frechet(2.0, 3.0),
            InverseGamma(2.0, 3.0),
            InverseGaussian(2.0, 3.0),
            Kumaraswamy(2.0, 3.0),
            LogLogistic(2.0, 3.0),
            Pareto(2.0, 3.0),
            Weibull(2.0, 3.0),
            GeneralizedExtremeValue(0.0, 1.0, 0.2),
            GeneralizedPareto(0.0, 1.0, 0.2),
            JohnsonSU(0.0, 1.0, 0.0, 1.0),
            Lindley(1.0),
            Semicircle(1.0),
            NormalCanon(0.0, 1.0),
            NoncentralBeta(2.0, 3.0, 0.5),
            NoncentralChisq(2.0, 0.5),
            NoncentralF(2.0, 3.0, 0.5),
            NoncentralT(3.0, 0.5),
            PGeneralizedGaussian(0.0, 1.0, 2.0),
            Rician(0.5, 1.0),
            SkewNormal(0.0, 1.0, 0.5),
            VonMises(0.0, 1.0),
            Geometric(0.4),
            NegativeBinomial(2.0, 0.4),
            Poisson(1.5),
            Skellam(1.0, 2.0),
            Dirac(0.0),
            Erlang(2, 1.0),
            PoissonBinomial([0.2, 0.5, 0.8]),
            TriangularDist(0.0, 2.0, 1.0),
            NormalInverseGaussian(0.5, 2.0, 0.5, 1.5),
        )

        for d in distributions
            p = param_space(d)
            θ = unconstrain(p,example(p))
            η = constrain(p, θ)
            @test unconstrain(p, η) ≈ θ
        end
    end

    @testset "structural-only parameter spaces" begin
        for d in (
            Chernoff(),
            DiscreteUniform(1, 4),
            Hypergeometric(5, 6, 3),
            Kolmogorov(),
            KSDist(10),
            KSOneSided(10),
        )
            p = param_space(d)
            @test dimension(p) == 0
            @test names(p) == ()
            @test example(p) == ()
            @test unconstrain(p, ()) == Float64[]
        end
    end

    @testset "Dirichlet keeps one vector parameter" begin
        p = param_space(Dirichlet([1.0, 2.0, 3.0]))
        @test names(p) == (:α,)
        @test dimension(p) == 3
        α = constrain(p, log.([1.0, 2.0, 3.0]))
        @test α ≈ [1.0, 2.0, 3.0]
    end

    @testset "multivariate normal families keep natural parameter shapes" begin
        Σ0 = [1.0 0.2 0.1; 0.2 1.5 0.3; 0.1 0.3 2.0]
        X = MvNormal(zeros(3), Σ0)
        p = param_space(X)

        @test names(p) == (:μ, :Σ)
        @test dimension(p) == 9

        θ = unconstrain(p,example(p))
        η = example(p)
        @test η isa Tuple
        @test length(η) == 2
        @test η[1] isa AbstractVector
        @test size(η[1]) == (3,)
        @test η[2] isa AbstractMatrix
        @test size(η[2]) == (3, 3)
        @test η[2] ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

        Xcanon = MvNormalCanon([0.1, -0.2], [2.0 0.2; 0.2 1.2])
        pcanon = param_space(Xcanon)
        @test names(pcanon) == (:h, :J)
        @test dimension(pcanon) == 5
        ηcanon = example(pcanon)
        @test size(ηcanon[1]) == (2,)
        @test size(ηcanon[2]) == (2, 2)
        @test unconstrain(pcanon, ηcanon) ≈ unconstrain(pcanon,example(pcanon))

        Xlog = MvLogNormal(MvNormal([0.0, 0.3], [1.0 0.2; 0.2 1.5]))
        plog = param_space(Xlog)
        @test names(plog) == (:μ, :Σ)
        @test size(example(plog)[2]) == (2, 2)
    end

    @testset "matrix-variate shapes" begin
        M = zeros(2, 3)
        U = [1.0 0.2; 0.2 1.0]
        V = [1.0 0.1 0.0; 0.1 1.0 0.2; 0.0 0.2 1.0]
        d = MatrixNormal(M, U, V)
        p = param_space(d)
        @test names(p) == (:M, :U, :V)
        η = example(p)
        @test size(η[1]) == (2, 3)
        @test size(η[2]) == (2, 2)
        @test size(η[3]) == (3, 3)

        S = [2.0 0.2; 0.2 1.5]
        for d in (
            Wishart(4.0, S),
            InverseWishart(4.0, S),
            MatrixTDist(5.0, zeros(2, 2), S, S),
            MatrixFDist(3.0, 4.0, S),
        )
            p = param_space(d)
            θ = unconstrain(p,example(p))
            @test unconstrain(p, constrain(p, θ)) ≈ θ
        end
    end

    @testset "wrappers, mixtures, and products" begin
        wrapped = (
            Distributions.AffineDistribution(2.0, 3.0, Gamma(2.0, 1.0)),
            truncated(Normal(0.0, 1.0), -1.0, 2.0),
            censored(Normal(0.0, 1.0), -1.0, 2.0),
            OrderStatistic(Gamma(2.0, 1.0), 10, 3),
            JointOrderStatistics(Normal(), 10, (1, 5, 10)),
        )
        for d in wrapped
            p = param_space(d)
            θ = unconstrain(p,example(p))
            @test unconstrain(p, constrain(p, θ)) ≈ θ
        end

        dmix = MixtureModel(
            Normal[Normal(-1.0, 1.0), Normal(2.0, 0.5)],
            [0.4, 0.6],
        )
        pmix = param_space(dmix)
        @test names(pmix) == (:component1_μ, :component1_σ, :component2_μ, :component2_σ, :π)
        θmix = unconstrain(pmix, example(pmix))
        @test unconstrain(pmix, constrain(pmix, θmix)) ≈ θmix

        dprod = product_distribution(UnivariateDistribution[Normal(), Exponential()])
        pprod = param_space(dprod)
        @test names(pprod) == (:component1_μ, :component1_σ, :component2_θ)
        θprod = unconstrain(pprod, example(pprod))
        @test unconstrain(pprod, constrain(pprod, θprod)) ≈ θprod

        dnamed = product_distribution((x=Normal(), y=Gamma(2.0, 1.0)))
        pnamed = param_space(dnamed)
        @test names(pnamed) == (:x_μ, :x_σ, :y_α, :y_θ)
        θnamed = unconstrain(pnamed, example(pnamed))
        @test unconstrain(pnamed, constrain(pnamed, θnamed)) ≈ θnamed
    end
end

@testset "ForwardDiff passthrough" begin
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

        @test names(p) == (:R,)
        @test dimension(p) == 6
        @test size(R) == (4, 4)
        @test R ≈ R'
        @test all(isapprox(R[i, i], 1.0) for i in axes(R, 1))
        @test unconstrain(p, R) ≈ θ

        g = ForwardDiff.gradient(x -> sum(abs2, constrain(p, x)), θ)
        @test length(g) == dimension(p)
        @test all(isfinite, g)

        @test_throws DomainError unconstrain(Correlation(:R, 2), [2.0 0.0; 0.0 1.0])
        @test_throws DomainError unconstrain(Correlation(:R, 2), [1.0 0.2; 0.1 1.0])
        @test_throws DomainError unconstrain(Correlation(:R, 2), [1.0 1.2; 1.2 1.0])
    end

end

