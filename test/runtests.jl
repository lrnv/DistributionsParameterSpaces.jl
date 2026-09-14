using Aqua
using Distributions
using DistributionsParameterSpaces
using Test

@testset "Aqua" begin
    Aqua.test_all(DistributionsParameterSpaces)
end

@testset "DistributionsParameterSpaces.jl" begin

    @testset "Normal" begin
        d = Normal(0.0, 1.0)
        p = param_space(d)

        @test dimension(p) == 2
        @test parameter_symbols(p) == (:μ, :σ)

        θ = [1.5, log(2.0)]

        η, J = constrain_with_jac(p, θ)

        @test η ≈ [1.5, 2.0]
        @test J ≈ [
            1.0  0.0
            0.0  2.0
        ]

        @test constrain(p, θ) ≈ η
        @test constrain_jac(p, θ) ≈ J

        @test unconstrain(p, η) ≈ θ

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ [
            1.0  0.0
            0.0  0.5
        ]

        @test unconstrain_jac(p, η) ≈ Jinv

        @test constrain(p, unconstrain(p, η)) ≈ η
        @test unconstrain(p, constrain(p, θ)) ≈ θ

        @test logabsdet_constrain_jac(p, θ) ≈ log(2.0)
        @test logabsdet_unconstrain_jac(p, η) ≈ -log(2.0)
    end


    @testset "Exponential" begin
        d = Exponential(1.0)
        p = param_space(d)

        @test dimension(p) == 1
        @test parameter_symbols(p) == (:θ,)

        θ = [log(3.0)]

        η, J = constrain_with_jac(p, θ)

        @test η ≈ [3.0]
        @test J ≈ reshape([3.0], 1, 1)

        @test unconstrain(p, η) ≈ θ
        @test constrain(p, unconstrain(p, η)) ≈ η

        @test logabsdet_constrain_jac(p, θ) ≈ log(3.0)
        @test logabsdet_unconstrain_jac(p, η) ≈ -log(3.0)
    end


    @testset "Gamma" begin
        d = Gamma(2.0, 3.0)
        p = param_space(d)

        @test dimension(p) == 2
        @test parameter_symbols(p) == (:α, :θ)

        θ = [log(2.0), log(3.0)]

        η, J = constrain_with_jac(p, θ)

        @test η ≈ [2.0, 3.0]
        @test J ≈ [
            2.0  0.0
            0.0  3.0
        ]

        @test constrain(p, θ) ≈ η
        @test constrain_jac(p, θ) ≈ J
        @test unconstrain(p, η) ≈ θ

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ [
            0.5      0.0
            0.0  1 / 3
        ]

        @test logabsdet_constrain_jac(p, θ) ≈ log(6.0)
        @test logabsdet_unconstrain_jac(p, η) ≈ -log(6.0)
    end


    @testset "Beta" begin
        d = Beta(2.0, 3.0)
        p = param_space(d)

        @test dimension(p) == 2
        @test parameter_symbols(p) == (:α, :β)

        θ = [log(2.0), log(4.0)]

        η, J = constrain_with_jac(p, θ)

        @test η ≈ [2.0, 4.0]
        @test J ≈ [
            2.0  0.0
            0.0  4.0
        ]

        @test constrain(p, θ) ≈ η
        @test unconstrain(p, η) ≈ θ

        @test logabsdet_constrain_jac(p, θ) ≈ log(8.0)
        @test logabsdet_unconstrain_jac(p, η) ≈ -log(8.0)
    end


    @testset "Bernoulli" begin
        d = Bernoulli(0.5)
        p = param_space(d)

        @test dimension(p) == 1
        @test parameter_symbols(p) == (:p,)

        θ = [0.0]

        η, J = constrain_with_jac(p, θ)

        @test η ≈ [0.5]
        @test J ≈ reshape([0.25], 1, 1)

        @test constrain(p, θ) ≈ η
        @test constrain_jac(p, θ) ≈ J
        @test unconstrain(p, η) ≈ θ

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ reshape([4.0], 1, 1)

        @test unconstrain_jac(p, η) ≈ Jinv

        @test unconstrain(p, [0.0]) == [-Inf]
        @test unconstrain(p, [1.0]) == [Inf]

        @test_throws DomainError unconstrain(p, [-0.1])
        @test_throws DomainError unconstrain(p, [1.1])
    end


    @testset "Dirichlet" begin
        d = Dirichlet([1.0, 2.0, 3.0])
        p = param_space(d)

        @test dimension(p) == 3
        @test parameter_symbols(p) == (:α_1, :α_2, :α_3)

        θ = log.([1.0, 2.0, 3.0])

        η, J = constrain_with_jac(p, θ)

        @test η ≈ [1.0, 2.0, 3.0]
        @test J ≈ [
            1.0  0.0  0.0
            0.0  2.0  0.0
            0.0  0.0  3.0
        ]

        @test constrain(p, θ) ≈ η
        @test constrain_jac(p, θ) ≈ J

        @test unconstrain(p, η) ≈ θ

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ [
            1.0  0.0      0.0
            0.0  0.5      0.0
            0.0  0.0  1 / 3
        ]

        @test unconstrain_jac(p, η) ≈ Jinv

        @test logabsdet_constrain_jac(p, θ) ≈
              sum(log, η)

        @test logabsdet_unconstrain_jac(p, η) ≈
              -sum(log, η)

        @test_throws DomainError unconstrain(
            p,
            [1.0, 0.0, 2.0],
        )

        @test_throws DimensionMismatch constrain(
            p,
            [1.0, 2.0],
        )

        @test_throws DimensionMismatch unconstrain(
            p,
            [1.0, 2.0],
        )
    end


    @testset "Examples" begin
        distributions = (
            Normal(0.0, 1.0),
            Exponential(1.0),
            Gamma(2.0, 1.0),
            Beta(2.0, 2.0),
            Bernoulli(0.5),
            Dirichlet([1.0, 1.0, 1.0]),
        )

        for d in distributions
            p = param_space(d)

            θ = unconstrained_example(p)
            η = constrained_example(p)

            @test length(θ) == dimension(p)
            @test length(η) == dimension(p)
            @test length(parameter_symbols(p)) == dimension(p)

            @test constrain(p, θ) ≈ η
            @test unconstrain(p, η) ≈ θ
        end
    end


    @testset "Round-trip" begin
        cases = (
            (
                Normal(0.0, 1.0),
                [1.2, -0.7],
            ),
            (
                Exponential(1.0),
                [0.4],
            ),
            (
                Gamma(2.0, 3.0),
                [-0.5, 1.2],
            ),
            (
                Beta(2.0, 3.0),
                [0.2, -1.0],
            ),
            (
                Bernoulli(0.5),
                [1.4],
            ),
            (
                Dirichlet([1.0, 1.0, 1.0]),
                [-1.0, 0.2, 2.0],
            ),
        )

        for (d, θ) in cases
            p = param_space(d)

            η = constrain(p, θ)
            θ₂ = unconstrain(p, η)

            @test θ₂ ≈ θ
        end
    end


    @testset "Jacobian consistency" begin
        cases = (
            (
                Normal(0.0, 1.0),
                [0.4, -0.2],
            ),
            (
                Gamma(2.0, 3.0),
                [0.3, 0.8],
            ),
            (
                Beta(2.0, 3.0),
                [-0.4, 0.7],
            ),
            (
                Bernoulli(0.5),
                [0.3],
            ),
            (
                Dirichlet([1.0, 1.0, 1.0]),
                [0.2, -0.3, 0.7],
            ),
        )

        for (d, θ) in cases
            p = param_space(d)

            η, J = constrain_with_jac(p, θ)
            θ₂, Jinv = unconstrain_with_jac(p, η)

            @test θ₂ ≈ θ

            n = dimension(p)

            # Since the currently implemented spaces are separable,
            # J and Jinv are diagonal.
            for i in 1:n
                @test J[i, i] * Jinv[i, i] ≈ 1.0

                for j in 1:n
                    if i != j
                        @test J[i, j] == 0
                        @test Jinv[i, j] == 0
                    end
                end
            end
        end
    end


    @testset "Dimension mismatch" begin
        p = param_space(Normal(0.0, 1.0))

        @test_throws DimensionMismatch constrain(
            p,
            [1.0],
        )

        @test_throws DimensionMismatch constrain(
            p,
            [1.0, 2.0, 3.0],
        )

        @test_throws DimensionMismatch unconstrain(
            p,
            [1.0],
        )
    end

    @testset "Additional separable distributions" begin
        cases = (
            (
                Cauchy(0.0, 1.0),
                (:μ, :σ),
                [0.3, log(2.0)],
            ),
            (
                LogNormal(0.0, 1.0),
                (:μ, :σ),
                [-0.4, log(1.5)],
            ),
            (
                Rayleigh(1.0),
                (:σ,),
                [log(2.0)],
            ),
            (
                Chi(2.0),
                (:ν,),
                [log(3.0)],
            ),
            (
                Chisq(2.0),
                (:ν,),
                [log(4.0)],
            ),
            (
                TDist(3.0),
                (:ν,),
                [log(5.0)],
            ),
            (
                FDist(2.0, 3.0),
                (:ν1, :ν2),
                [log(2.5), log(4.0)],
            ),
            (
                Binomial(10, 0.5),
                (:p,),
                [0.7],
            ),
        )

        for (d, symbols, θ) in cases
            p = param_space(d)

            @test dimension(p) == length(θ)
            @test parameter_symbols(p) == symbols

            η, J = constrain_with_jac(p, θ)

            @test constrain(p, θ) ≈ η
            @test constrain_jac(p, θ) ≈ J
            @test unconstrain(p, η) ≈ θ

            θ₂, Jinv = unconstrain_with_jac(p, η)

            @test θ₂ ≈ θ
            @test unconstrain_jac(p, η) ≈ Jinv

            # These parameter spaces are currently separable, so the
            # Jacobians are diagonal inverses of each other.
            for i in 1:dimension(p)
                @test J[i, i] * Jinv[i, i] ≈ 1.0

                for j in 1:dimension(p)
                    if i != j
                        @test J[i, j] == 0
                        @test Jinv[i, j] == 0
                    end
                end
            end
        end
    end


    @testset "Extended distribution mappings" begin
        cases = (
            (BernoulliLogit(0.3), (:logitp,), 1),
            (BetaBinomial(10, 2.0, 3.0), (:α, :β), 2),
            (BetaPrime(2.0, 3.0), (:α, :β), 2),
            (LogitNormal(0.0, 1.0), (:μ, :σ), 2),
            (Laplace(0.0, 1.0), (:μ, :θ), 2),
            (Logistic(0.0, 1.0), (:μ, :θ), 2),
            (Gumbel(0.0, 1.0), (:μ, :θ), 2),
            (Levy(0.0, 1.0), (:μ, :σ), 2),
            (Biweight(0.0, 1.0), (:μ, :σ), 2),
            (Cosine(0.0, 1.0), (:μ, :σ), 2),
            (Epanechnikov(0.0, 1.0), (:μ, :σ), 2),
            (SymTriangularDist(0.0, 1.0), (:μ, :σ), 2),
            (Triweight(0.0, 1.0), (:μ, :σ), 2),
            (Frechet(2.0, 3.0), (:α, :θ), 2),
            (InverseGamma(2.0, 3.0), (:α, :θ), 2),
            (InverseGaussian(2.0, 3.0), (:μ, :λ), 2),
            (Kumaraswamy(2.0, 3.0), (:a, :b), 2),
            (LogLogistic(2.0, 3.0), (:α, :β), 2),
            (Pareto(2.0, 3.0), (:α, :θ), 2),
            (Weibull(2.0, 3.0), (:α, :θ), 2),
            (GeneralizedExtremeValue(0.0, 1.0, 0.2), (:μ, :σ, :ξ), 3),
            (GeneralizedPareto(0.0, 1.0, 0.2), (:μ, :σ, :ξ), 3),
            (JohnsonSU(0.0, 1.0, 0.0, 1.0), (:ξ, :λ, :γ, :δ), 4),
            (Lindley(1.0), (:θ,), 1),
            (Semicircle(1.0), (:r,), 1),
            (NormalCanon(0.0, 1.0), (:η, :λ), 2),
            (NoncentralBeta(2.0, 3.0, 0.5), (:α, :β, :λ), 3),
            (NoncentralChisq(2.0, 0.5), (:ν, :λ), 2),
            (NoncentralF(2.0, 3.0, 0.5), (:ν1, :ν2, :λ), 3),
            (NoncentralT(3.0, 0.5), (:ν, :λ), 2),
            (PGeneralizedGaussian(0.0, 1.0, 2.0), (:μ, :α, :p), 3),
            (Rician(0.5, 1.0), (:ν, :σ), 2),
            (SkewNormal(0.0, 1.0, 0.5), (:ξ, :ω, :α), 3),
            (VonMises(0.0, 1.0), (:μ, :κ), 2),
            (Geometric(0.4), (:p,), 1),
            (NegativeBinomial(2.0, 0.4), (:r, :p), 2),
            (Poisson(1.5), (:λ,), 1),
            (Skellam(1.0, 2.0), (:μ1, :μ2), 2),
            (Dirac(0.0), (:x,), 1),
            (Erlang(2, 1.0), (:θ,), 1),
            (PoissonBinomial([0.2, 0.5, 0.8]), (:p_1, :p_2, :p_3), 3),
        )

        for (d, symbols, n) in cases
            p = param_space(d)
            @test dimension(p) == n
            @test parameter_symbols(p) == symbols

            θ = unconstrained_example(p)
            η, J = constrain_with_jac(p, θ)

            @test length(η) == n
            @test size(J) == (n, n)
            @test unconstrain(p, η) ≈ θ
        end
    end

    @testset "Structural-only parameter spaces" begin
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
            @test parameter_symbols(p) == ()
            @test constrain(p, Float64[]) == Float64[]
            @test unconstrain(p, Float64[]) == Float64[]
            @test logabsdet_constrain_jac(p, Float64[]) == 0.0
        end
    end


    @testset "Half-open probability spaces" begin
        p = param_space(Geometric(0.5))
        @test_throws DomainError unconstrain(p, [0.0])
        @test unconstrain(p, [1.0]) == [Inf]

        p = param_space(NegativeBinomial(2.0, 0.5))
        @test_throws DomainError unconstrain(p, [2.0, 0.0])
        @test unconstrain(p, [2.0, 1.0])[2] == Inf
    end

    @testset "Ordered parameter spaces" begin
        p = param_space(Uniform(0.0, 1.0))

        @test dimension(p) == 2
        @test parameter_symbols(p) == (:a, :b)

        θ = [1.0, log(2.0)]
        η, J = constrain_with_jac(p, θ)

        @test η ≈ [1.0, 3.0]
        @test J ≈ [
            1.0  0.0
            1.0  2.0
        ]

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ [
             1.0  0.0
            -0.5  0.5
        ]

        @test logabsdet_constrain_jac(p, θ) ≈ log(2.0)
        @test logabsdet_unconstrain_jac(p, η) ≈ -log(2.0)

        @test constrain(p, unconstrain(p, η)) ≈ η
        @test_throws DomainError unconstrain(p, [1.0, 1.0])
        @test_throws DomainError unconstrain(p, [2.0, 1.0])

        # Arcsine has the same a < b geometry.
        p = param_space(Arcsine(-1.0, 2.0))
        @test parameter_symbols(p) == (:a, :b)
        @test unconstrain(p, constrain(p, [-0.3, 0.8])) ≈ [-0.3, 0.8]
    end


    @testset "Positive ordered parameter spaces" begin
        p = param_space(LogUniform(1.0, 10.0))

        @test dimension(p) == 2
        @test parameter_symbols(p) == (:a, :b)

        θ = [log(2.0), log(3.0)]
        η, J = constrain_with_jac(p, θ)

        @test η ≈ [2.0, 5.0]
        @test J ≈ [
            2.0  0.0
            2.0  3.0
        ]

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ [
             0.5      0.0
            -1 / 3  1 / 3
        ]

        @test logabsdet_constrain_jac(p, θ) ≈ log(6.0)
        @test logabsdet_unconstrain_jac(p, η) ≈ -log(6.0)

        @test_throws DomainError unconstrain(p, [0.0, 1.0])
        @test_throws DomainError unconstrain(p, [2.0, 1.0])
    end


    @testset "Triangular parameter space" begin
        d = TriangularDist(0.0, 2.0, 1.0)
        p = param_space(d)

        @test dimension(p) == 3
        @test constrained_dimension(p) == 3
        @test parameter_symbols(p) == (:a, :b, :c)

        θ = [1.0, log(2.0), 0.0]
        η, J = constrain_with_jac(p, θ)

        @test η ≈ [1.0, 3.0, 2.0]
        @test J ≈ [
            1.0  0.0  0.0
            1.0  2.0  0.0
            1.0  1.0  0.5
        ]

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv ≈ [
             1.0   0.0   0.0
            -0.5   0.5   0.0
            -1.0  -1.0   2.0
        ]

        @test Jinv * J ≈ [
            1.0  0.0  0.0
            0.0  1.0  0.0
            0.0  0.0  1.0
        ]

        @test logabsdet_constrain_jac(p, θ) ≈ 0.0
        @test logabsdet_unconstrain_jac(p, η) ≈ 0.0

        @test unconstrain(p, [0.0, 2.0, 0.0])[3] == -Inf
        @test unconstrain(p, [0.0, 2.0, 2.0])[3] == Inf

        # Distributions.jl also allows the degenerate a == b == c case.
        @test unconstrain(p, [1.0, 1.0, 1.0]) == [1.0, -Inf, 0.0]

        @test_throws DomainError unconstrain(p, [0.0, 2.0, 3.0])
        @test_throws DomainError unconstrain(p, [2.0, 0.0, 1.0])
    end


    @testset "Simplex parameter space" begin
        d = Categorical([0.2, 0.3, 0.5])
        p = param_space(d)

        @test dimension(p) == 2
        @test constrained_dimension(p) == 3
        @test parameter_symbols(p) == (:p_1, :p_2, :p_3)

        θ = log.([0.2 / 0.5, 0.3 / 0.5])
        η, J = constrain_with_jac(p, θ)

        @test η ≈ [0.2, 0.3, 0.5]
        @test size(J) == (3, 2)
        @test J ≈ [
             0.16  -0.06
            -0.06   0.21
            -0.10  -0.15
        ]

        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test size(Jinv) == (2, 3)
        @test Jinv ≈ [
            5.0       0.0      -2.0
            0.0   10 / 3       -2.0
        ]

        @test Jinv * J ≈ [
            1.0  0.0
            0.0  1.0
        ]

        @test constrain(p, θ₂) ≈ η

        nt = constrained_namedtuple(p, θ)
        @test nt.p_1 ≈ 0.2
        @test nt.p_2 ≈ 0.3
        @test nt.p_3 ≈ 0.5

        # Boundary coordinates are representable when the chosen anchor
        # remains strictly positive.
        d_boundary = Categorical([0.0, 0.4, 0.6])
        p_boundary = param_space(d_boundary)
        θ_boundary = unconstrain(p_boundary, probs(d_boundary))
        @test θ_boundary[1] == -Inf
        @test constrain(p_boundary, θ_boundary) ≈ probs(d_boundary)

        # The anchor is chosen from the instance, so a zero final
        # probability is not a problem either.
        d_other_boundary = Categorical([0.7, 0.3, 0.0])
        p_other_boundary = param_space(d_other_boundary)
        θ_other_boundary = unconstrain(p_other_boundary, probs(d_other_boundary))
        @test constrain(p_other_boundary, θ_other_boundary) ≈ probs(d_other_boundary)

        # A one-category categorical has no free coordinate.
        p1 = param_space(Categorical([1.0]))
        @test dimension(p1) == 0
        @test constrained_dimension(p1) == 1
        @test constrain(p1, Float64[]) == [1.0]
        @test unconstrain(p1, [1.0]) == Float64[]
        @test size(constrain_jac(p1, Float64[])) == (1, 0)
        @test size(unconstrain_jac(p1, [1.0])) == (0, 1)

        @test_throws DomainError unconstrain(p, [0.2, 0.3, 0.4])
        @test_throws ArgumentError logabsdet_constrain_jac(p, θ)
        @test_throws ArgumentError logabsdet_unconstrain_jac(p, η)
    end


    @testset "Multinomial simplex mapping" begin
        d = Multinomial(10, [0.2, 0.3, 0.5])
        p = param_space(d)

        @test dimension(p) == 2
        @test constrained_dimension(p) == 3
        @test parameter_symbols(p) == (:p_1, :p_2, :p_3)

        θ = unconstrain(p, probs(d))
        @test constrain(p, θ) ≈ probs(d)
    end


    @testset "Remaining documented univariate distributions" begin
        # DiscreteNonParametric: support points are structural; only probabilities vary.
        d = DiscreteNonParametric([-1.0, 2.0, 4.0], [0.2, 0.3, 0.5])
        p = param_space(d)
        @test dimension(p) == 2
        @test constrained_dimension(p) == 3
        @test parameter_symbols(p) == (:p_1, :p_2, :p_3)
        θ = unconstrain(p, probs(d))
        @test constrain(p, θ) ≈ probs(d)

        # Skewed exponential power: α is strictly inside (0, 1).
        d = SkewedExponentialPower(0.3, 1.2, 2.5, 0.4)
        p = param_space(d)
        @test parameter_symbols(p) == (:μ, :σ, :p, :α)
        η = [0.3, 1.2, 2.5, 0.4]
        θ = unconstrain(p, η)
        @test constrain(p, θ) ≈ η
        @test_throws DomainError unconstrain(p, [0.0, 1.0, 2.0, 0.0])
        @test_throws DomainError unconstrain(p, [0.0, 1.0, 2.0, 1.0])

        # StudentizedRange: ν > 0 and k > 1.
        d = StudentizedRange(10.0, 3.0)
        p = param_space(d)
        @test parameter_symbols(p) == (:ν, :k)
        θ = [log(10.0), log(2.0)]
        @test constrain(p, θ) ≈ [10.0, 3.0]
        @test unconstrain(p, [10.0, 3.0]) ≈ θ
        @test_throws DomainError unconstrain(p, [10.0, 1.0])

        # Soliton: K and M are structural; δ ∈ (0,1), atol ∈ [0,1).
        d = Soliton(100, 10, 0.1, 0.0)
        p = param_space(d)
        @test dimension(p) == 2
        @test parameter_symbols(p) == (:δ, :atol)
        θ = unconstrain(p, [0.1, 0.0])
        @test θ[2] == -Inf
        @test constrain(p, θ) ≈ [0.1, 0.0]
        @test_throws DomainError unconstrain(p, [0.0, 0.0])
        @test_throws DomainError unconstrain(p, [1.0, 0.0])
        @test_throws DomainError unconstrain(p, [0.1, 1.0])

        # NormalInverseGaussian: α > |β| and δ > 0.
        d = NormalInverseGaussian(0.5, 2.0, 0.5, 1.5)
        p = param_space(d)
        @test dimension(p) == 4
        @test parameter_symbols(p) == (:μ, :α, :β, :δ)

        η = [0.5, 2.0, 0.5, 1.5]
        θ = unconstrain(p, η)
        η₂, J = constrain_with_jac(p, θ)
        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test η₂ ≈ η
        @test θ₂ ≈ θ
        @test Jinv * J ≈ [
            1.0  0.0  0.0  0.0
            0.0  1.0  0.0  0.0
            0.0  0.0  1.0  0.0
            0.0  0.0  0.0  1.0
        ]
        @test isapprox(
            logabsdet_constrain_jac(p, θ) +
            logabsdet_unconstrain_jac(p, η),
            0.0; atol=1e-12, rtol=0.0,
        )

        @test_throws DomainError unconstrain(p, [0.0, 0.5, 0.5, 1.0])
        @test_throws DomainError unconstrain(p, [0.0, 1.0, -1.0, 1.0])
        @test_throws DomainError unconstrain(p, [0.0, 2.0, 0.5, 0.0])
    end

    @testset "SPD parameter space" begin
        p = DistributionsParameterSpaces.SPD(:Σ, 3)

        @test dimension(p) == 6
        @test constrained_dimension(p) == 6
        @test parameter_symbols(p) == (
            :Σ_1_1,
            :Σ_2_1, :Σ_2_2,
            :Σ_3_1, :Σ_3_2, :Σ_3_3,
        )

        θ = [log(1.2), 0.2, log(0.8), -0.1, 0.3, log(1.5)]
        η, J = constrain_with_jac(p, θ)
        θ₂, Jinv = unconstrain_with_jac(p, η)

        @test θ₂ ≈ θ
        @test Jinv * J ≈ [
            1.0  0.0  0.0  0.0  0.0  0.0
            0.0  1.0  0.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  0.0  0.0
            0.0  0.0  0.0  1.0  0.0  0.0
            0.0  0.0  0.0  0.0  1.0  0.0
            0.0  0.0  0.0  0.0  0.0  1.0
        ]

        @test isapprox(
            logabsdet_constrain_jac(p, θ) +
            logabsdet_unconstrain_jac(p, η),
            0.0; atol=1e-12, rtol=0.0,
        )

        @test_throws DomainError unconstrain(
            DistributionsParameterSpaces.SPD(:Σ, 2),
            [1.0, 2.0, 1.0],
        )
    end


    @testset "Multivariate normal families" begin
        Σ = [2.0 0.3; 0.3 1.5]

        d = MvNormal([0.2, -0.4], Σ)
        p = param_space(d)
        @test dimension(p) == 5
        @test constrained_dimension(p) == 5
        @test parameter_symbols(p) ==
              (:μ_1, :μ_2, :Σ_1_1, :Σ_2_1, :Σ_2_2)

        θ = unconstrained_example(p)
        η, J = constrain_with_jac(p, θ)
        θ₂, Jinv = unconstrain_with_jac(p, η)
        @test θ₂ ≈ θ
        @test Jinv * J ≈ [
            1.0  0.0  0.0  0.0  0.0
            0.0  1.0  0.0  0.0  0.0
            0.0  0.0  1.0  0.0  0.0
            0.0  0.0  0.0  1.0  0.0
            0.0  0.0  0.0  0.0  1.0
        ]

        dc = MvNormalCanon([0.1, -0.2], [2.0 0.2; 0.2 1.2])
        pc = param_space(dc)
        @test dimension(pc) == 5
        @test parameter_symbols(pc) ==
              (:h_1, :h_2, :J_1_1, :J_2_1, :J_2_2)

        dl = MvLogNormal(MvNormal([0.0, 0.3], Σ))
        pl = param_space(dl)
        @test dimension(pl) == 5
        @test parameter_symbols(pl) ==
              (:μ_1, :μ_2, :Σ_1_1, :Σ_2_1, :Σ_2_2)

        dlogit = MvLogitNormal(MvNormal([0.0, 0.3], Σ))
        plogit = param_space(dlogit)
        @test dimension(plogit) == 5
        @test first(parameter_symbols(plogit)) == :normal_μ_1
    end


    @testset "Matrix-variate distributions" begin
        A = [2.0 0.2; 0.2 1.5]
        B = [1.3 0.1; 0.1 1.1]
        M = [0.0 1.0; -1.0 0.5]

        d = MatrixNormal(M, A, B)
        p = param_space(d)
        @test dimension(p) == 10
        @test constrained_dimension(p) == 10
        @test parameter_symbols(p)[1:4] ==
              (:M_1_1, :M_2_1, :M_1_2, :M_2_2)

        θ = unconstrained_example(p)
        η = constrain(p, θ)
        @test unconstrain(p, η) ≈ θ

        dw = Wishart(4.0, A)
        pw = param_space(dw)
        @test dimension(pw) == 4
        @test parameter_symbols(pw)[1] == :ν

        dws = Wishart(1, A)
        pws = param_space(dws)
        @test dimension(pws) == 3
        @test parameter_symbols(pws) ==
              (:S_1_1, :S_2_1, :S_2_2)

        diw = InverseWishart(4.0, A)
        piw = param_space(diw)
        @test dimension(piw) == 4

        dmt = MatrixTDist(5.0, M, A, B)
        pmt = param_space(dmt)
        @test dimension(pmt) == 11
        @test first(parameter_symbols(pmt)) == :ν

        dmb = MatrixBeta(2, 3.0, 4.0)
        pmb = param_space(dmb)
        @test dimension(pmb) == 2
        @test parameter_symbols(pmb) == (:n1, :n2)

        dmf = MatrixFDist(3.0, 4.0, A)
        pmf = param_space(dmf)
        @test dimension(pmf) == 5
        @test parameter_symbols(pmf)[1:2] == (:n1, :n2)

        @test dimension(param_space(LKJ(3, 2.0))) == 1
        @test parameter_symbols(param_space(LKJ(3, 2.0))) == (:η,)

        @test dimension(param_space(LKJCholesky(3, 2.0))) == 1
        @test parameter_symbols(param_space(LKJCholesky(3, 2.0))) == (:η,)
    end


    @testset "Affine and derived distributions" begin
        da = Distributions.AffineDistribution(2.0, 3.0, Gamma(2.0, 1.0))
        pa = param_space(da)
        @test dimension(pa) == 4
        @test parameter_symbols(pa) == (:μ, :σ, :base_α, :base_θ)

        da_neg = Distributions.AffineDistribution(2.0, -3.0, Normal())
        pa_neg = param_space(da_neg)
        @test dimension(pa_neg) == 4
        η = [2.0, -3.0, 0.0, 1.0]
        θ = unconstrain(pa_neg, η)
        @test constrain(pa_neg, θ) ≈ η

        dt = truncated(Normal(0.0, 1.0), -1.0, 2.0)
        pt = param_space(dt)
        @test dimension(pt) == 2
        @test parameter_symbols(pt) == (:base_μ, :base_σ)

        dc = censored(Normal(0.0, 1.0), -1.0, 2.0)
        pc = param_space(dc)
        @test dimension(pc) == 2
        @test parameter_symbols(pc) == (:base_μ, :base_σ)

        dos = OrderStatistic(Gamma(2.0, 1.0), 10, 3)
        pos = param_space(dos)
        @test dimension(pos) == 2
        @test parameter_symbols(pos) == (:base_α, :base_θ)

        djos = JointOrderStatistics(Normal(), 10, (1, 5, 10))
        pjos = param_space(djos)
        @test dimension(pjos) == 2
        @test parameter_symbols(pjos) == (:base_μ, :base_σ)
    end


    @testset "Mixtures and products" begin
        dmix = MixtureModel(
            Normal[
                Normal(-1.0, 1.0),
                Normal(2.0, 0.5),
            ],
            [0.4, 0.6],
        )

        pmix = param_space(dmix)
        @test dimension(pmix) == 5
        @test constrained_dimension(pmix) == 6
        @test parameter_symbols(pmix) == (
            :component1_μ,
            :component1_σ,
            :component2_μ,
            :component2_σ,
            :π_1,
            :π_2,
        )

        θmix = unconstrained_example(pmix)
        ηmix = constrain(pmix, θmix)
        @test unconstrain(pmix, ηmix) ≈ θmix

        dprod = product_distribution(UnivariateDistribution[Normal(), Exponential()])
        pprod = param_space(dprod)
        @test dimension(pprod) == 3
        @test parameter_symbols(pprod) == (
            :component1_μ,
            :component1_σ,
            :component2_θ,
        )

        dnamed = product_distribution((
            x = Normal(),
            y = Gamma(2.0, 1.0),
        ))
        pnamed = param_space(dnamed)
        @test dimension(pnamed) == 4
        @test parameter_symbols(pnamed) ==
              (:x_μ, :x_σ, :y_α, :y_θ)

        dr = reshape(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), 1, 2)
        pr = param_space(dr)
        @test dimension(pr) == 5
    end


    @testset "Noncentral hypergeometric distributions" begin
        if isdefined(Distributions, :FisherNoncentralHypergeometric)
            d = Distributions.FisherNoncentralHypergeometric(5, 7, 4, 2.0)
            @test dimension(param_space(d)) == 1
            @test parameter_symbols(param_space(d)) == (:ω,)
        end

        if isdefined(Distributions, :WalleniusNoncentralHypergeometric)
            d = Distributions.WalleniusNoncentralHypergeometric(5, 7, 4, 2.0)
            @test dimension(param_space(d)) == 1
            @test parameter_symbols(param_space(d)) == (:ω,)
        end
    end

end
