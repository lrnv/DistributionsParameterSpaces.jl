using Test
using Distributions
using DistributionsParameterSpaces


# using Aqua
# @testset "Code quality (Aqua.jl)" begin
#     Aqua.test_all(DistributionsParameterSpaces)
# end

@testset "Normal" begin
    p = param_space(Normal)

    @test dimension(p) == 2

    θ = [1.5, log(2.0)]

    η, J = constrain_with_jac(p, θ)

    @test η ≈ [1.5, 2.0]
    @test J ≈ [
        1.0  0.0
        0.0  2.0
    ]

    @test unconstrain(p, η) ≈ θ

    θ₂, Jinv = unconstrain_with_jac(p, η)

    @test θ₂ ≈ θ
    @test Jinv ≈ [
        1.0  0.0
        0.0  0.5
    ]

    @test constrain(p, unconstrain(p, η)) ≈ η
end


@testset "Gamma" begin
    p = param_space(Gamma)

    θ = [log(2.0), log(3.0)]

    η, J = constrain_with_jac(p, θ)

    @test η ≈ [2.0, 3.0]
    @test J ≈ [
        2.0  0.0
        0.0  3.0
    ]

    @test unconstrain(p, η) ≈ θ
end


@testset "Bernoulli" begin
    p = param_space(Bernoulli)

    η, J = constrain_with_jac(p, [0.0])

    @test η ≈ [0.5]
    @test J ≈ reshape([0.25], 1, 1)

    @test unconstrain(p, η) ≈ [0.0]

    # Boundary values correspond to infinite unconstrained coordinates.
    @test unconstrain(p, [0.0]) == [-Inf]
    @test unconstrain(p, [1.0]) == [Inf]
end


@testset "Dirichlet" begin
    d = Dirichlet([1.0, 2.0, 3.0])

    p = param_space(d)

    @test dimension(p) == 3

    θ = log.([1.0, 2.0, 3.0])

    η, J = constrain_with_jac(p, θ)

    @test η ≈ [1.0, 2.0, 3.0]
    @test J ≈ [
        1.0  0.0  0.0
        0.0  2.0  0.0
        0.0  0.0  3.0
    ]

    @test unconstrain(p, η) ≈ θ

    @test logabsdet_constrain_jac(p, θ) ≈
          sum(log, η)
end


@testset "Examples" begin
    for D in (Normal, Exponential, Gamma, Beta, Bernoulli)
        p = param_space(D)

        θ = unconstrained_example(p)
        η = constrained_example(p)

        @test length(θ) == dimension(p)
        @test length(η) == dimension(p)
        @test unconstrain(p, η) ≈ θ
    end
end