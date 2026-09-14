# using Aqua
using Distributions
using DistributionsParameterSpaces
using Test


# @testset "Aqua" begin
#     Aqua.test_all(DistributionsParameterSpaces)
# end


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


    @testset "Unsupported distribution" begin
        @test_throws ArgumentError param_space(Poisson(1.0))
    end

end