module DistributionsParameterSpaces

using Distributions

export AbstractParameterSpace,
       param_space,
       dimension,
       unconstrained_example,
       constrained_example,
       constrain,
       unconstrain,
       constrain_jac,
       unconstrain_jac,
       constrain_with_jac,
       unconstrain_with_jac,
       logabsdet_constrain_jac,
       logabsdet_unconstrain_jac,
       parameter_symbols,
       constrained_namedtuple


# ---------------------------------------------------------------------------
# Parameter spaces
# ---------------------------------------------------------------------------

abstract type AbstractParameterSpace end
abstract type AbstractScalarParameterSpace <: AbstractParameterSpace end


struct Id{S} <: AbstractScalarParameterSpace end
struct Pos{S} <: AbstractScalarParameterSpace end
struct NonNeg{S} <: AbstractScalarParameterSpace end
struct Prob{S} <: AbstractScalarParameterSpace end
struct ProbOpenLeft{S} <: AbstractScalarParameterSpace end

struct Ordered{A,B} <: AbstractParameterSpace end
struct PosOrdered{A,B} <: AbstractParameterSpace end

struct PosVec{S} <: AbstractParameterSpace
    n::Int

    function PosVec{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("dimension must be positive"))
        new{S}(Int(n))
    end
end


struct ProbVec{S} <: AbstractParameterSpace
    n::Int

    function ProbVec{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("dimension must be positive"))
        new{S}(Int(n))
    end
end


Id(s::Symbol)     = Id{s}()
Pos(s::Symbol)    = Pos{s}()
NonNeg(s::Symbol) = NonNeg{s}()
Prob(s::Symbol)   = Prob{s}()
ProbOpenLeft(s::Symbol) = ProbOpenLeft{s}()
Ordered(a::Symbol, b::Symbol) = Ordered{a,b}()
PosOrdered(a::Symbol, b::Symbol) = PosOrdered{a,b}()
PosVec(s::Symbol, n::Integer) = PosVec{s}(n)
ProbVec(s::Symbol, n::Integer) = ProbVec{s}(n)


const ProductParameterSpace = Tuple{Vararg{AbstractScalarParameterSpace}}
const SeparableParameterSpace =
    Union{AbstractScalarParameterSpace, ProductParameterSpace, PosVec, ProbVec}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

parameter_symbol(::Id{S}) where {S} = S
parameter_symbol(::Pos{S}) where {S} = S
parameter_symbol(::NonNeg{S}) where {S} = S
parameter_symbol(::Prob{S}) where {S} = S
parameter_symbol(::ProbOpenLeft{S}) where {S} = S

parameter_symbols(::Ordered{A,B}) where {A,B} = (A, B)
parameter_symbols(::PosOrdered{A,B}) where {A,B} = (A, B)

parameter_symbols(p::AbstractScalarParameterSpace) =
    (parameter_symbol(p),)

parameter_symbols(p::ProductParameterSpace) =
    map(parameter_symbol, p)

parameter_symbols(p::PosVec{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

parameter_symbols(p::ProbVec{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)


dimension(::AbstractScalarParameterSpace) = 1
dimension(::Union{Ordered,PosOrdered}) = 2
dimension(p::ProductParameterSpace) = length(p)
dimension(p::PosVec) = p.n
dimension(p::ProbVec) = p.n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _check_dimension(p, x)
    length(x) == dimension(p) ||
        throw(DimensionMismatch(
            "expected $(dimension(p)) parameters, got $(length(x))"
        ))
    return nothing
end


function _promoted_vector(x)
    isempty(x) && return Float64[]
    T = promote_type(map(typeof, x)...)
    return T[xi for xi in x]
end


function _diagonal_matrix(d)
    n = length(d)
    n == 0 && return Matrix{Float64}(undef, 0, 0)

    T = promote_type(map(typeof, d)...)
    J = zeros(T, n, n)

    for i in 1:n
        J[i, i] = d[i]
    end

    return J
end


# ---------------------------------------------------------------------------
# Scalar transforms
# ---------------------------------------------------------------------------

_constrain_with_jac(::Id, θ) = (θ, one(θ))
_unconstrain(::Id, η) = η


function _constrain_with_jac(::Union{Pos,NonNeg}, θ)
    η = exp(θ)
    return η, η
end

function _unconstrain(::Pos, η)
    η > zero(η) ||
        throw(DomainError(η, "parameter must be strictly positive"))
    return log(η)
end

function _unconstrain(::NonNeg, η)
    η >= zero(η) ||
        throw(DomainError(η, "parameter must be nonnegative"))
    return log(η)
end


function _constrain_with_jac(::Union{Prob,ProbOpenLeft}, θ)
    # Numerically stable logistic transform
    if θ >= zero(θ)
        z = exp(-θ)
        η = inv(one(θ) + z)
    else
        z = exp(θ)
        η = z / (one(θ) + z)
    end

    return η, η * (one(η) - η)
end

function _unconstrain(::Prob, η)
    zero(η) <= η <= one(η) ||
        throw(DomainError(η, "parameter must belong to [0, 1]"))

    return log(η) - log1p(-η)
end


function _unconstrain(::ProbOpenLeft, η)
    zero(η) < η <= one(η) ||
        throw(DomainError(η, "parameter must belong to (0, 1]"))

    return log(η) - log1p(-η)
end


# ---------------------------------------------------------------------------
# Scalar parameter spaces
# ---------------------------------------------------------------------------

function constrain_with_jac(p::AbstractScalarParameterSpace, θ)
    _check_dimension(p, θ)

    η, dηdθ = _constrain_with_jac(p, θ[1])

    return [η], reshape([dηdθ], 1, 1)
end


function unconstrain(p::AbstractScalarParameterSpace, η)
    _check_dimension(p, η)

    return [_unconstrain(p, η[1])]
end


# ---------------------------------------------------------------------------
# Product parameter spaces
# ---------------------------------------------------------------------------

function constrain_with_jac(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)

    result = ntuple(length(p)) do i
        _constrain_with_jac(p[i], θ[i])
    end

    η = _promoted_vector(first.(result))
    dηdθ = _promoted_vector(last.(result))

    return η, _diagonal_matrix(dηdθ)
end


function unconstrain(p::ProductParameterSpace, η)
    _check_dimension(p, η)

    θ = ntuple(length(p)) do i
        _unconstrain(p[i], η[i])
    end

    return _promoted_vector(θ)
end


# ---------------------------------------------------------------------------
# Ordered pairs
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Ordered, θ)
    _check_dimension(p, θ)

    a = θ[1]
    w = exp(θ[2])
    b = a + w

    η = _promoted_vector((a, b))
    J = [
        one(w)  zero(w)
        one(w)  w
    ]

    return η, J
end


function unconstrain(p::Ordered, η)
    _check_dimension(p, η)

    a, b = η
    b > a ||
        throw(DomainError(η, "parameters must satisfy a < b"))

    return _promoted_vector((a, log(b - a)))
end


function constrain_with_jac(p::PosOrdered, θ)
    _check_dimension(p, θ)

    a = exp(θ[1])
    w = exp(θ[2])
    b = a + w

    η = _promoted_vector((a, b))
    J = [
        a  zero(a)
        a  w
    ]

    return η, J
end


function unconstrain(p::PosOrdered, η)
    _check_dimension(p, η)

    a, b = η
    a > zero(a) ||
        throw(DomainError(η, "first parameter must be strictly positive"))
    b > a ||
        throw(DomainError(η, "parameters must satisfy 0 < a < b"))

    return _promoted_vector((log(a), log(b - a)))
end


# ---------------------------------------------------------------------------
# Positive vectors
# ---------------------------------------------------------------------------

function constrain_with_jac(p::PosVec, θ)
    _check_dimension(p, θ)

    η = exp.(θ)

    return η, _diagonal_matrix(η)
end


function unconstrain(p::PosVec, η)
    _check_dimension(p, η)

    all(x -> x > zero(x), η) ||
        throw(DomainError(
            η,
            "all parameters must be strictly positive",
        ))

    return log.(η)
end


# ---------------------------------------------------------------------------
# Probability vectors
# ---------------------------------------------------------------------------

function constrain_with_jac(p::ProbVec{S}, θ) where {S}
    _check_dimension(p, θ)

    result = [_constrain_with_jac(Prob{S}(), x) for x in θ]
    η = _promoted_vector(first.(result))
    dηdθ = _promoted_vector(last.(result))

    return η, _diagonal_matrix(dηdθ)
end

function unconstrain(p::ProbVec{S}, η) where {S}
    _check_dimension(p, η)

    θ = [_unconstrain(Prob{S}(), x) for x in η]
    return _promoted_vector(θ)
end


# ---------------------------------------------------------------------------
# Derived operations
# ---------------------------------------------------------------------------

constrain(p, θ) =
    first(constrain_with_jac(p, θ))

constrain_jac(p, θ) =
    last(constrain_with_jac(p, θ))


function unconstrain_with_jac(p::SeparableParameterSpace, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)

    n = dimension(p)
    d = [inv(J[i, i]) for i in 1:n]

    return θ, _diagonal_matrix(d)
end


function unconstrain_with_jac(p::Ordered, η)
    θ = unconstrain(p, η)

    w = η[2] - η[1]
    iw = inv(w)

    J = [
        one(w)  zero(w)
        -iw     iw
    ]

    return θ, J
end


function unconstrain_with_jac(p::PosOrdered, η)
    θ = unconstrain(p, η)

    a = η[1]
    w = η[2] - η[1]
    iw = inv(w)

    J = [
        inv(a)  zero(a)
        -iw     iw
    ]

    return θ, J
end


unconstrain_jac(p, η) =
    last(unconstrain_with_jac(p, η))


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

unconstrained_example(p) =
    zeros(dimension(p))

constrained_example(p) =
    constrain(p, unconstrained_example(p))


# ---------------------------------------------------------------------------
# Jacobian determinants
# ---------------------------------------------------------------------------

function logabsdet_constrain_jac(p::SeparableParameterSpace, θ)
    _check_dimension(p, θ)

    n = dimension(p)
    n == 0 && return 0.0

    _, J = constrain_with_jac(p, θ)
    s = zero(J[1, 1])

    for i in 1:n
        s += log(abs(J[i, i]))
    end

    return s
end


function logabsdet_unconstrain_jac(p::SeparableParameterSpace, η)
    θ = unconstrain(p, η)
    return -logabsdet_constrain_jac(p, θ)
end


function logabsdet_constrain_jac(p::Ordered, θ)
    _check_dimension(p, θ)
    return θ[2]
end

function logabsdet_unconstrain_jac(p::Ordered, η)
    _check_dimension(p, η)

    a, b = η
    b > a ||
        throw(DomainError(η, "parameters must satisfy a < b"))

    return -log(b - a)
end


function logabsdet_constrain_jac(p::PosOrdered, θ)
    _check_dimension(p, θ)
    return θ[1] + θ[2]
end

function logabsdet_unconstrain_jac(p::PosOrdered, η)
    _check_dimension(p, η)

    a, b = η
    a > zero(a) ||
        throw(DomainError(η, "first parameter must be strictly positive"))
    b > a ||
        throw(DomainError(η, "parameters must satisfy 0 < a < b"))

    return -log(a) - log(b - a)
end


# Faster specialization for positive vectors
function logabsdet_constrain_jac(p::PosVec, θ)
    _check_dimension(p, θ)
    return sum(θ)
end

function logabsdet_unconstrain_jac(p::PosVec, η)
    _check_dimension(p, η)

    all(x -> x > zero(x), η) ||
        throw(DomainError(
            η,
            "all parameters must be strictly positive",
        ))

    return -sum(log, η)
end


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

function constrained_namedtuple(p, θ)
    η = constrain(p, θ)
    return NamedTuple{parameter_symbols(p)}(Tuple(η))
end

# ---------------------------------------------------------------------------
# Distributions.jl integration
# ---------------------------------------------------------------------------

# Ordered endpoints
param_space(::Uniform)                  = Ordered(:a, :b)
param_space(::Arcsine)                  = Ordered(:a, :b)
param_space(::LogUniform)               = PosOrdered(:a, :b)

# Location / scale families
param_space(::Normal)                   = (Id(:μ), NonNeg(:σ))
param_space(::LogNormal)                = (Id(:μ), NonNeg(:σ))
param_space(::LogitNormal)              = (Id(:μ), NonNeg(:σ))
param_space(::Cauchy)                   = (Id(:μ), Pos(:σ))
param_space(::Laplace)                  = (Id(:μ), Pos(:θ))
param_space(::Logistic)                 = (Id(:μ), Pos(:θ))
param_space(::Gumbel)                   = (Id(:μ), Pos(:θ))
param_space(::Levy)                     = (Id(:μ), Pos(:σ))
param_space(::Biweight)                 = (Id(:μ), Pos(:σ))
param_space(::Cosine)                   = (Id(:μ), Pos(:σ))
param_space(::Epanechnikov)             = (Id(:μ), Pos(:σ))
param_space(::SymTriangularDist)        = (Id(:μ), Pos(:σ))
param_space(::Triweight)                = (Id(:μ), Pos(:σ))

# Positive scalar parameters
param_space(::Exponential)              = Pos(:θ)
param_space(::Rayleigh)                 = Pos(:σ)
param_space(::Chi)                      = Pos(:ν)
param_space(::Chisq)                    = Pos(:ν)
param_space(::TDist)                    = Pos(:ν)
param_space(::Lindley)                  = Pos(:θ)
param_space(::Semicircle)               = Pos(:r)

# Positive pairs / triples
param_space(::Gamma)                    = (Pos(:α), Pos(:θ))
param_space(::Beta)                     = (Pos(:α), Pos(:β))
param_space(::BetaPrime)                = (Pos(:α), Pos(:β))
param_space(::Frechet)                  = (Pos(:α), Pos(:θ))
param_space(::InverseGamma)             = (Pos(:α), Pos(:θ))
param_space(::InverseGaussian)          = (Pos(:μ), Pos(:λ))
param_space(::Kumaraswamy)              = (Pos(:a), Pos(:b))
param_space(::LogLogistic)              = (Pos(:α), Pos(:β))
param_space(::Pareto)                   = (Pos(:α), Pos(:θ))
param_space(::Weibull)                  = (Pos(:α), Pos(:θ))
param_space(::FDist)                    = (Pos(:ν1), Pos(:ν2))
param_space(::PGeneralizedGaussian)     = (Id(:μ), Pos(:α), Pos(:p))

# Unconstrained shape / location parameters combined with positive scales
param_space(::GeneralizedExtremeValue)  = (Id(:μ), Pos(:σ), Id(:ξ))
param_space(::GeneralizedPareto)        = (Id(:μ), Pos(:σ), Id(:ξ))
param_space(::SkewNormal)               = (Id(:ξ), Pos(:ω), Id(:α))
param_space(::JohnsonSU)                = (Id(:ξ), Pos(:λ), Id(:γ), Pos(:δ))

# Noncentral families
param_space(::NoncentralBeta)           = (Pos(:α), Pos(:β), NonNeg(:λ))
param_space(::NoncentralChisq)          = (Pos(:ν), NonNeg(:λ))
param_space(::NoncentralF)              = (Pos(:ν1), Pos(:ν2), NonNeg(:λ))
param_space(::NoncentralT)              = (Pos(:ν), Id(:λ))

# Alternative normal parameterization
param_space(::NormalCanon)              = (Id(:η), Pos(:λ))

# Circular / radial families
param_space(::Rician)                   = (NonNeg(:ν), Pos(:σ))
param_space(::VonMises)                 = (Id(:μ), NonNeg(:κ))

# Discrete distributions with continuous parameters
param_space(::Bernoulli)                = Prob(:p)
param_space(::BernoulliLogit)           = Id(:logitp)
param_space(::Binomial)                 = Prob(:p)
param_space(::Geometric)                = ProbOpenLeft(:p)
param_space(::NegativeBinomial)         = (Pos(:r), ProbOpenLeft(:p))
param_space(::Poisson)                  = NonNeg(:λ)
param_space(::Skellam)                  = (NonNeg(:μ1), NonNeg(:μ2))
param_space(d::PoissonBinomial)         = ProbVec(:p, length(params(d)[1]))

# Distributions with structural discrete parameters
param_space(::BetaBinomial)             = (Pos(:α), Pos(:β))
param_space(::Erlang)                   = Pos(:θ)
param_space(::Chernoff)                 = ()
param_space(::DiscreteUniform)          = ()
param_space(::Hypergeometric)           = ()
param_space(::Kolmogorov)               = ()
param_space(::KSDist)                   = ()
param_space(::KSOneSided)               = ()

# Degenerate / vector-parameter distributions
param_space(::Dirac)                    = Id(:x)
param_space(d::Dirichlet)               = PosVec(:α, length(d))

param_space(d::Distribution) =
    throw(ArgumentError("parameter space not implemented for $(typeof(d))"))


end # module