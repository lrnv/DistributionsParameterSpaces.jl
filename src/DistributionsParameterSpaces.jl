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

struct PosVec{S} <: AbstractParameterSpace
    n::Int

    function PosVec{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("dimension must be positive"))
        new{S}(Int(n))
    end
end


Id(s::Symbol)     = Id{s}()
Pos(s::Symbol)    = Pos{s}()
NonNeg(s::Symbol) = NonNeg{s}()
Prob(s::Symbol)   = Prob{s}()
PosVec(s::Symbol, n::Integer) = PosVec{s}(n)


const ProductParameterSpace = Tuple{Vararg{AbstractScalarParameterSpace}}
const SeparableParameterSpace =
    Union{AbstractScalarParameterSpace, ProductParameterSpace, PosVec}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

parameter_symbol(::Id{S}) where {S} = S
parameter_symbol(::Pos{S}) where {S} = S
parameter_symbol(::NonNeg{S}) where {S} = S
parameter_symbol(::Prob{S}) where {S} = S

parameter_symbols(p::AbstractScalarParameterSpace) =
    (parameter_symbol(p),)

parameter_symbols(p::ProductParameterSpace) =
    map(parameter_symbol, p)

parameter_symbols(p::PosVec{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)


dimension(::AbstractScalarParameterSpace) = 1
dimension(p::ProductParameterSpace) = length(p)
dimension(p::PosVec) = p.n


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


function _constrain_with_jac(::Prob, θ)
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
    _, J = constrain_with_jac(p, θ)

    s = zero(J[1, 1])

    for i in 1:dimension(p)
        s += log(abs(J[i, i]))
    end

    return s
end


function logabsdet_unconstrain_jac(p::SeparableParameterSpace, η)
    θ = unconstrain(p, η)
    return -logabsdet_constrain_jac(p, θ)
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

# ---------------------------------------------------------------------------
# Distributions.jl integration
# ---------------------------------------------------------------------------

param_space(::Normal)      = (Id(:μ), NonNeg(:σ))
param_space(::LogNormal)   = (Id(:μ), NonNeg(:σ))
param_space(::Cauchy)      = (Id(:μ), Pos(:σ))

param_space(::Exponential) = Pos(:θ)
param_space(::Rayleigh)    = Pos(:σ)

param_space(::Gamma)       = (Pos(:α), Pos(:θ))
param_space(::Beta)        = (Pos(:α), Pos(:β))

param_space(::Chi)         = Pos(:ν)
param_space(::Chisq)       = Pos(:ν)
param_space(::TDist)       = Pos(:ν)
param_space(::FDist)       = (Pos(:ν1), Pos(:ν2))

param_space(::Bernoulli)   = Prob(:p)
param_space(::Binomial)    = Prob(:p)

param_space(d::Dirichlet)  = PosVec(:α, length(d))

param_space(d::Distribution) =
    throw(ArgumentError("parameter space not implemented for $(typeof(d))"))


end # module