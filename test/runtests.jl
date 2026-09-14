using DistributionsParameterSpaces
using Test
using Aqua

@testset "DistributionsParameterSpaces.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DistributionsParameterSpaces)
    end
    # Write your tests here.
end
