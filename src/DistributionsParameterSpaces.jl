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
       logabsdet_unconstrain_jac
       parameter_symbols,
       constrained_namedtuple


# ---------------------------------------------------------------------------
# Public abstraction
# ---------------------------------------------------------------------------

abstract type AbstractParameterSpace end


"""
    dimension(p)

Dimension of the unconstrained parameter space.
"""
function dimension end


"""
    param_space(D)
    param_space(d)

Return the parameter space associated with a distribution type or instance.
"""
function param_space end


"""
    unconstrained_example(p)

Return a simple point in the unconstrained parameter space.
"""
unconstrained_example(p::AbstractParameterSpace) =
    zeros(dimension(p))


"""
    constrained_example(p)

Return the constrained representation corresponding to
`unconstrained_example(p)`.
"""
constrained_example(p::AbstractParameterSpace) =
    constrain(p, unconstrained_example(p))


# ---------------------------------------------------------------------------
# Public transformations
# ---------------------------------------------------------------------------

"""
    constrain_with_jac(p, θ) -> η, J

Map unconstrained coordinates `θ` to constrained parameters `η`, and return

    J = ∂η / ∂θ
"""
function constrain_with_jac end


"""
    unconstrain(p, η) -> θ

Map constrained parameters `η` to unconstrained coordinates `θ`.
"""
function unconstrain end


# These two are derived from the two fundamental operations above.

constrain(p::AbstractParameterSpace, θ) =
    first(constrain_with_jac(p, θ))

constrain_jac(p::AbstractParameterSpace, θ) =
    last(constrain_with_jac(p, θ))


# ---------------------------------------------------------------------------
# Elementary scalar transforms
# ---------------------------------------------------------------------------

abstract type AbstractCoordinateTransform end


"""
Unconstrained real parameter.
"""
struct IdentityTransform <: AbstractCoordinateTransform end


"""
Exponential transform.

`Closed = false`:
    constrained domain is (0, ∞)

`Closed = true`:
    the closure [0, ∞) is accepted by `unconstrain`,
    with 0 mapping to -Inf.
"""
struct ExpTransform{Closed} <: AbstractCoordinateTransform end


"""
Logistic transform.

The finite unconstrained space maps to (0, 1), while `unconstrain`
accepts the boundary values 0 and 1 and maps them to ±Inf.
"""
struct LogisticTransform <: AbstractCoordinateTransform end


# Identity

_constrain_with_jac(::IdentityTransform, θ) = (θ, one(θ))

_unconstrain(::IdentityTransform, η) = η


# Positive / nonnegative

_constrain_with_jac(::ExpTransform, θ) = begin
    η = exp(θ)
    η, η
end

function _unconstrain(::ExpTransform{false}, η)
    η > zero(η) ||
        throw(DomainError(η, "parameter must be strictly positive"))
    return log(η)
end

function _unconstrain(::ExpTransform{true}, η)
    η >= zero(η) ||
        throw(DomainError(η, "parameter must be nonnegative"))
    return log(η)
end


# Unit interval

function _constrain_with_jac(::LogisticTransform, θ)
    # Numerically stable logistic function.
    if θ >= zero(θ)
        z = exp(-θ)
        η = inv(one(θ) + z)
    else
        z = exp(θ)
        η = z / (one(θ) + z)
    end

    return η, η * (one(η) - η)
end

function _unconstrain(::LogisticTransform, η)
    zero(η) <= η <= one(η) ||
        throw(DomainError(η, "parameter must belong to [0, 1]"))

    return log(η) - log1p(-η)
end


# ---------------------------------------------------------------------------
# Separable parameter spaces
# ---------------------------------------------------------------------------

"""
    SeparableParameterSpace(transforms)

A parameter space for which every constrained coordinate depends only on
the corresponding unconstrained coordinate.
"""
struct SeparableParameterSpace{T<:Tuple,N<:Tuple} <: AbstractParameterSpace
    transforms::T
    symbols::N
end

function SeparableParameterSpace(transforms::T, symbols::N) where {T<:Tuple,N<:Tuple}
    length(transforms) == length(symbols) ||
        throw(DimensionMismatch(
            "number of transforms and symbols must match"
        ))

    all(x -> x isa Symbol, symbols) ||
        throw(ArgumentError("parameter names must be Symbols"))

    return SeparableParameterSpace{T,N}(transforms, symbols)
end

dimension(p::SeparableParameterSpace) = length(p.transforms)

parameter_symbols(p::SeparableParameterSpace) = p.symbols

function _check_dimension(p::AbstractParameterSpace, x)
    length(x) == dimension(p) ||
        throw(DimensionMismatch(
            "expected $(dimension(p)) parameters, got $(length(x))"
        ))
    return nothing
end


# Convert a collection of possibly different scalar types to their
# common promoted vector type.
function _promoted_vector(x)
    isempty(x) && return Float64[]

    T = promote_type(map(typeof, x)...)
    return T[xi for xi in x]
end


function _diagonal_matrix(d)
    n = length(d)
    n > 0 || return Matrix{Float64}(undef, 0, 0)

    T = promote_type(map(typeof, d)...)
    J = zeros(T, n, n)

    for i in 1:n
        J[i, i] = d[i]
    end

    return J
end


function constrain_with_jac(p::SeparableParameterSpace, θ)
    _check_dimension(p, θ)

    n = dimension(p)

    result = [
        _constrain_with_jac(p.transforms[i], θ[i])
        for i in 1:n
    ]

    η = _promoted_vector(first.(result))
    dηdθ = _promoted_vector(last.(result))

    return η, _diagonal_matrix(dηdθ)
end


function unconstrain(p::SeparableParameterSpace, η)
    _check_dimension(p, η)

    θ = [
        _unconstrain(p.transforms[i], η[i])
        for i in 1:dimension(p)
    ]

    return _promoted_vector(θ)
end


# For a separable transform, the inverse Jacobian is simply the
# reciprocal of the diagonal of the forward Jacobian.

function unconstrain_with_jac(p::SeparableParameterSpace, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)

    n = dimension(p)
    d = [inv(J[i, i]) for i in 1:n]

    return θ, _diagonal_matrix(d)
end

unconstrain_jac(p::SeparableParameterSpace, η) =
    last(unconstrain_with_jac(p, η))


# ---------------------------------------------------------------------------
# Log absolute determinants
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


# ---------------------------------------------------------------------------
# Dynamic positive vector space
#
# Used notably for Dirichlet.
# Avoids constructing an NTuple of N transforms, which is undesirable
# for potentially large N.
# ---------------------------------------------------------------------------

struct PositiveVectorParameterSpace <: AbstractParameterSpace
    n::Int
    symbol::Symbol

    function PositiveVectorParameterSpace(
        n::Integer,
        symbol::Symbol = :x,
    )
        n > 0 || throw(ArgumentError("dimension must be positive"))
        new(Int(n), symbol)
    end
end

parameter_symbols(p::PositiveVectorParameterSpace) =
    ntuple(i -> Symbol(p.symbol, "_", i), p.n)

dimension(p::PositiveVectorParameterSpace) = p.n


function constrain_with_jac(p::PositiveVectorParameterSpace, θ)
    _check_dimension(p, θ)

    η = exp.(θ)
    return η, _diagonal_matrix(η)
end


function unconstrain(p::PositiveVectorParameterSpace, η)
    _check_dimension(p, η)

    all(x -> x > zero(x), η) ||
        throw(DomainError(η, "all parameters must be strictly positive"))

    return log.(η)
end


function unconstrain_with_jac(p::PositiveVectorParameterSpace, η)
    θ = unconstrain(p, η)
    d = inv.(η)

    return θ, _diagonal_matrix(d)
end

unconstrain_jac(p::PositiveVectorParameterSpace, η) =
    last(unconstrain_with_jac(p, η))


logabsdet_constrain_jac(
    p::PositiveVectorParameterSpace,
    θ,
) = begin
    _check_dimension(p, θ)
    sum(θ)
end


logabsdet_unconstrain_jac(
    p::PositiveVectorParameterSpace,
    η,
) = begin
    _check_dimension(p, η)
    -sum(log, η)
end

function constrained_namedtuple(p, θ)
    η = constrain(p, θ)
    return NamedTuple{parameter_symbols(p)}(Tuple(η))
end

# ---------------------------------------------------------------------------
# Distributions.jl integration
# ---------------------------------------------------------------------------

# Exactly the convention we discussed:
param_space(d::UnivariateDistribution) = param_space(typeof(d))


param_space(::Type{<:Normal}) =
    SeparableParameterSpace(
        (
            IdentityTransform(),
            ExpTransform{true}(),
        ),
        (:μ, :σ),
    )

param_space(::Type{<:Exponential}) =
    SeparableParameterSpace(
        (ExpTransform{false}(),),
        (:θ,),
    )

param_space(::Type{<:Gamma}) =
    SeparableParameterSpace(
        (
            ExpTransform{false}(),
            ExpTransform{false}(),
        ),
        (:α, :θ),
    )

param_space(::Type{<:Beta}) =
    SeparableParameterSpace(
        (
            ExpTransform{false}(),
            ExpTransform{false}(),
        ),
        (:α, :β),
    )

param_space(::Type{<:Bernoulli}) =
    SeparableParameterSpace(
        (LogisticTransform(),),
        (:p,),
    )


# Dirichlet is dimension-dependent, and the dimension is not encoded
# in its type.
param_space(d::Dirichlet) =
    PositiveVectorParameterSpace(length(d), :α)

param_space(::Type{<:Dirichlet}, n::Integer) =
    PositiveVectorParameterSpace(n, :α)

function param_space(::Type{<:Dirichlet})
    throw(ArgumentError(
        "Dirichlet dimension is not encoded in its type; " *
        "use param_space(d::Dirichlet) or param_space(Dirichlet, n)"
    ))
end


# Useful fallback with a better error than an obscure MethodError.
function param_space(::Type{D}) where {D<:UnivariateDistribution}
    throw(ArgumentError(
        "parameter space not implemented for $D"
    ))
end


end # module