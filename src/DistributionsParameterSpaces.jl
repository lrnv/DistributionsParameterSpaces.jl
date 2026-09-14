module DistributionsParameterSpaces

using Distributions

export AbstractParameterSpace,
       param_space,
       dimension,
       constrained_dimension,
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
# Core parameter-space types
# ---------------------------------------------------------------------------

abstract type AbstractParameterSpace end
abstract type AbstractSeparableParameterSpace <: AbstractParameterSpace end
abstract type AbstractScalarDomain end

# Scalar domains. Boundary flags affect the inverse-domain check only; finite
# unconstrained coordinates still map to the interior through exp/logistic.
struct IdentityDomain <: AbstractScalarDomain end
struct ExpDomain{AllowZero} <: AbstractScalarDomain end
struct NegativeExpDomain <: AbstractScalarDomain end
struct ProbabilityDomain{LeftClosed,RightClosed} <: AbstractScalarDomain end
struct LowerDomain{T} <: AbstractScalarDomain
    lower::T
end

struct ScalarSpace{S,D<:AbstractScalarDomain} <: AbstractSeparableParameterSpace
    domain::D
end

struct ElementwiseSpace{S,D<:AbstractScalarDomain,N} <: AbstractSeparableParameterSpace
    domain::D
    dims::NTuple{N,Int}
end

struct OrderedSpace{A,B,D<:AbstractScalarDomain} <: AbstractParameterSpace
    first_domain::D
end

struct Between{A,B,C} <: AbstractParameterSpace end
struct NIG{M,A,B,D} <: AbstractParameterSpace end

struct Simplex{S} <: AbstractParameterSpace
    n::Int
    anchor::Int

    function Simplex{S}(n::Integer, anchor::Integer) where {S}
        n >= 1 || throw(ArgumentError("simplex must contain at least one coordinate"))
        1 <= anchor <= n || throw(ArgumentError("anchor must belong to 1:n"))
        new{S}(Int(n), Int(anchor))
    end
end

struct SPD{S} <: AbstractParameterSpace
    n::Int

    function SPD{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("matrix dimension must be positive"))
        new{S}(Int(n))
    end
end

struct Prefixed{P,S} <: AbstractParameterSpace
    space::S
end


# ---------------------------------------------------------------------------
# Compact constructors used by the distribution mappings
# ---------------------------------------------------------------------------

_scalar(s::Symbol, d::D) where {D<:AbstractScalarDomain} =
    ScalarSpace{s,D}(d)

_elementwise(s::Symbol, d::D, dims::NTuple{N,Int}) where {D<:AbstractScalarDomain,N} =
    ElementwiseSpace{s,D,N}(d, dims)

_ordered(a::Symbol, b::Symbol, d::D) where {D<:AbstractScalarDomain} =
    OrderedSpace{a,b,D}(d)

Id(s::Symbol)            = _scalar(s, IdentityDomain())
Pos(s::Symbol)           = _scalar(s, ExpDomain{false}())
NonNeg(s::Symbol)        = _scalar(s, ExpDomain{true}())
Neg(s::Symbol)           = _scalar(s, NegativeExpDomain())
Prob(s::Symbol)          = _scalar(s, ProbabilityDomain{true,true}())
ProbOpen(s::Symbol)      = _scalar(s, ProbabilityDomain{false,false}())
ProbOpenLeft(s::Symbol)  = _scalar(s, ProbabilityDomain{false,true}())
ProbOpenRight(s::Symbol) = _scalar(s, ProbabilityDomain{true,false}())
Lower(s::Symbol, lower)  = _scalar(s, LowerDomain(lower))

Ordered(a::Symbol, b::Symbol)    = _ordered(a, b, IdentityDomain())
PosOrdered(a::Symbol, b::Symbol) = _ordered(a, b, ExpDomain{false}())
Between(a::Symbol, b::Symbol, c::Symbol) = Between{a,b,c}()
NIG(μ::Symbol, α::Symbol, β::Symbol, δ::Symbol) = NIG{μ,α,β,δ}()

function PosVec(s::Symbol, n::Integer)
    n > 0 || throw(ArgumentError("dimension must be positive"))
    return _elementwise(s, ExpDomain{false}(), (Int(n),))
end

function ProbVec(s::Symbol, n::Integer)
    n > 0 || throw(ArgumentError("dimension must be positive"))
    return _elementwise(s, ProbabilityDomain{true,true}(), (Int(n),))
end

function RealVec(s::Symbol, n::Integer)
    n >= 0 || throw(ArgumentError("dimension must be nonnegative"))
    return _elementwise(s, IdentityDomain(), (Int(n),))
end

function RealMat(s::Symbol, m::Integer, n::Integer)
    m >= 0 || throw(ArgumentError("number of rows must be nonnegative"))
    n >= 0 || throw(ArgumentError("number of columns must be nonnegative"))
    return _elementwise(s, IdentityDomain(), (Int(m), Int(n)))
end

Simplex(s::Symbol, n::Integer; anchor::Integer=n) = Simplex{s}(n, anchor)

function Simplex(s::Symbol, p::AbstractVector)
    isempty(p) && throw(ArgumentError("probability vector must be nonempty"))
    all(x -> x >= zero(x), p) ||
        throw(DomainError(p, "probabilities must be nonnegative"))

    total = sum(p)
    isapprox(total, one(total)) ||
        throw(DomainError(p, "probabilities must sum to one"))

    anchor = argmax(p)
    p[anchor] > zero(p[anchor]) ||
        throw(DomainError(p, "at least one probability must be strictly positive"))

    return Simplex(s, length(p); anchor=anchor)
end

SPD(s::Symbol, n::Integer) = SPD{s}(n)
Prefixed(prefix::Symbol, space) = Prefixed{prefix,typeof(space)}(space)

const ProductParameterSpace = Tuple{Vararg{AbstractParameterSpace}}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

parameter_symbol(::ScalarSpace{S}) where {S} = S
parameter_symbols(p::ScalarSpace) = (parameter_symbol(p),)

parameter_symbols(::OrderedSpace{A,B}) where {A,B} = (A, B)
parameter_symbols(::Between{A,B,C}) where {A,B,C} = (A, B, C)
parameter_symbols(::NIG{M,A,B,D}) where {M,A,B,D} = (M, A, B, D)

parameter_symbols(p::Simplex{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

parameter_symbols(p::ElementwiseSpace{S,D,1}) where {S,D} =
    ntuple(i -> Symbol(S, "_", i), p.dims[1])

function parameter_symbols(p::ElementwiseSpace{S,D,2}) where {S,D}
    m, n = p.dims
    return ntuple(m * n) do k
        i = mod1(k, m)
        j = (k - 1) ÷ m + 1
        Symbol(S, "_", i, "_", j)
    end
end

function parameter_symbols(p::SPD{S}) where {S}
    result = Symbol[]
    for i in 1:p.n, j in 1:i
        push!(result, Symbol(S, "_", i, "_", j))
    end
    return Tuple(result)
end

function parameter_symbols(p::ProductParameterSpace)
    result = Symbol[]
    for q in p
        append!(result, parameter_symbols(q))
    end
    return Tuple(result)
end

parameter_symbols(p::Prefixed{P}) where {P} =
    Tuple(Symbol(P, "_", s) for s in parameter_symbols(p.space))


dimension(::ScalarSpace) = 1
dimension(::OrderedSpace) = 2
dimension(::Between) = 3
dimension(::NIG) = 4
dimension(p::Simplex) = p.n - 1
dimension(p::ElementwiseSpace) = prod(p.dims)
dimension(p::SPD) = p.n * (p.n + 1) ÷ 2
dimension(p::Prefixed) = dimension(p.space)
dimension(p::ProductParameterSpace) = sum(dimension, p; init=0)

constrained_dimension(p) = length(parameter_symbols(p))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _check_dimension(p, x)
    length(x) == dimension(p) ||
        throw(DimensionMismatch(
            "expected $(dimension(p)) unconstrained parameters, got $(length(x))"
        ))
    return nothing
end

function _check_constrained_dimension(p, x)
    length(x) == constrained_dimension(p) ||
        throw(DimensionMismatch(
            "expected $(constrained_dimension(p)) constrained parameters, got $(length(x))"
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
    n == 0 && return zeros(Float64, 0, 0)
    T = promote_type(map(typeof, d)...)
    J = zeros(T, n, n)
    for i in 1:n
        J[i, i] = d[i]
    end
    return J
end

function _concatenate_vectors(vs)
    total = sum(length, vs; init=0)
    total == 0 && return Float64[]

    Ts = [eltype(v) for v in vs if !isempty(v)]
    T = isempty(Ts) ? Float64 : promote_type(Ts...)
    result = Vector{T}(undef, total)

    offset = 0
    for v in vs, x in v
        offset += 1
        result[offset] = x
    end
    return result
end

function _blockdiag(blocks)
    isempty(blocks) && return zeros(Float64, 0, 0)

    nr = sum(B -> size(B, 1), blocks; init=0)
    nc = sum(B -> size(B, 2), blocks; init=0)
    T = promote_type((eltype(B) for B in blocks)...)
    result = zeros(T, nr, nc)

    r0 = c0 = 0
    for B in blocks
        m, n = size(B)
        result[(r0 + 1):(r0 + m), (c0 + 1):(c0 + n)] .= B
        r0 += m
        c0 += n
    end
    return result
end

function _invert_lower_triangular(A)
    n, m = size(A)
    n == m || throw(DimensionMismatch("matrix must be square"))
    n == 0 && return zeros(eltype(A), 0, 0)

    T = eltype(A)
    B = zeros(T, n, n)
    for j in 1:n, i in j:n
        if i == j
            B[i, j] = inv(A[i, i])
        else
            s = zero(T)
            for k in j:(i - 1)
                s += A[i, k] * B[k, j]
            end
            B[i, j] = -s / A[i, i]
        end
    end
    return B
end


# ---------------------------------------------------------------------------
# Scalar-domain transforms
# ---------------------------------------------------------------------------

_constrain_scalar(::IdentityDomain, θ) = (θ, one(θ))
_unconstrain_scalar(::IdentityDomain, η) = η

function _constrain_scalar(::ExpDomain, θ)
    η = exp(θ)
    return η, η
end

function _unconstrain_scalar(::ExpDomain{AllowZero}, η) where {AllowZero}
    valid = AllowZero ? η >= zero(η) : η > zero(η)
    valid || throw(DomainError(
        η,
        AllowZero ? "parameter must be nonnegative" : "parameter must be strictly positive",
    ))
    return log(η)
end

function _constrain_scalar(::NegativeExpDomain, θ)
    η = -exp(θ)
    return η, η
end

function _unconstrain_scalar(::NegativeExpDomain, η)
    η < zero(η) || throw(DomainError(η, "parameter must be strictly negative"))
    return log(-η)
end

function _constrain_scalar(::ProbabilityDomain, θ)
    if θ >= zero(θ)
        z = exp(-θ)
        η = inv(one(θ) + z)
    else
        z = exp(θ)
        η = z / (one(θ) + z)
    end
    return η, η * (one(η) - η)
end

function _unconstrain_scalar(::ProbabilityDomain{LC,RC}, η) where {LC,RC}
    left_ok = LC ? η >= zero(η) : η > zero(η)
    right_ok = RC ? η <= one(η) : η < one(η)
    left_ok && right_ok || throw(DomainError(
        η,
        "parameter must belong to " *
        (LC ? "[0, 1" : "(0, 1") *
        (RC ? "]" : ")"),
    ))
    return log(η) - log1p(-η)
end

function _constrain_scalar(d::LowerDomain, θ)
    w = exp(θ)
    return d.lower + w, w
end

function _unconstrain_scalar(d::LowerDomain, η)
    η > d.lower ||
        throw(DomainError(η, "parameter must be strictly greater than $(d.lower)"))
    return log(η - d.lower)
end


# ---------------------------------------------------------------------------
# Separable scalar / elementwise spaces
# ---------------------------------------------------------------------------

function constrain_with_jac(p::ScalarSpace, θ)
    _check_dimension(p, θ)
    η, dηdθ = _constrain_scalar(p.domain, θ[1])
    return [η], reshape([dηdθ], 1, 1)
end

function unconstrain(p::ScalarSpace, η)
    _check_constrained_dimension(p, η)
    return [_unconstrain_scalar(p.domain, η[1])]
end

function constrain_with_jac(p::ElementwiseSpace, θ)
    _check_dimension(p, θ)
    result = [_constrain_scalar(p.domain, x) for x in θ]
    η = _promoted_vector(first.(result))
    dηdθ = _promoted_vector(last.(result))
    return η, _diagonal_matrix(dηdθ)
end

function unconstrain(p::ElementwiseSpace, η)
    _check_constrained_dimension(p, η)
    return _promoted_vector([_unconstrain_scalar(p.domain, x) for x in η])
end


# ---------------------------------------------------------------------------
# Product spaces
# ---------------------------------------------------------------------------

function constrain_with_jac(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)
    values = Any[]
    blocks = Any[]
    offset = 0

    for q in p
        n = dimension(q)
        ηq, Jq = constrain_with_jac(q, view(θ, (offset + 1):(offset + n)))
        push!(values, ηq)
        push!(blocks, Jq)
        offset += n
    end
    return _concatenate_vectors(values), _blockdiag(blocks)
end

function unconstrain(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)
    values = Any[]
    offset = 0

    for q in p
        n = constrained_dimension(q)
        push!(values, unconstrain(q, view(η, (offset + 1):(offset + n))))
        offset += n
    end
    return _concatenate_vectors(values)
end


# ---------------------------------------------------------------------------
# Ordered pairs: a < b, optionally with a > 0
# ---------------------------------------------------------------------------

function constrain_with_jac(p::OrderedSpace, θ)
    _check_dimension(p, θ)
    a, da = _constrain_scalar(p.first_domain, θ[1])
    w = exp(θ[2])
    b = a + w
    return _promoted_vector((a, b)), [da zero(w); da w]
end

function unconstrain(p::OrderedSpace, η)
    _check_constrained_dimension(p, η)
    a, b = η
    θa = _unconstrain_scalar(p.first_domain, a)
    b > a || throw(DomainError(η, "parameters must satisfy a < b"))
    return _promoted_vector((θa, log(b - a)))
end

function unconstrain_with_jac(p::OrderedSpace, η)
    θ = unconstrain(p, η)
    _, da = _constrain_scalar(p.first_domain, θ[1])
    iw = inv(η[2] - η[1])
    return θ, [inv(da) zero(da); -iw iw]
end


# ---------------------------------------------------------------------------
# Bounded interior point: a <= c <= b
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Between, θ)
    _check_dimension(p, θ)
    a = θ[1]
    w = exp(θ[2])
    q, dq = _constrain_scalar(ProbabilityDomain{true,true}(), θ[3])
    b = a + w
    c = a + w * q
    return _promoted_vector((a, b, c)), [
        one(w)  zero(w)  zero(w)
        one(w)  w        zero(w)
        one(w)  w*q      w*dq
    ]
end

function unconstrain(p::Between, η)
    _check_constrained_dimension(p, η)
    a, b, c = η
    a <= c <= b || throw(DomainError(η, "parameters must satisfy a <= c <= b"))

    if b == a
        c == a || throw(DomainError(η, "degenerate bounds require a == b == c"))
        return _promoted_vector((a, -Inf, 0.0))
    end

    w = b - a
    q = (c - a) / w
    z = _unconstrain_scalar(ProbabilityDomain{true,true}(), q)
    return _promoted_vector((a, log(w), z))
end

function unconstrain_with_jac(p::Between, η)
    θ = unconstrain(p, η)
    a, b, c = η
    w = b - a
    w > zero(w) || throw(DomainError(η, "inverse Jacobian is undefined for a == b"))
    ca, bc = c - a, b - c
    return θ, [
        one(w)    zero(w)           zero(w)
        -inv(w)   inv(w)            zero(w)
        -inv(ca)  -inv(bc)          inv(ca)+inv(bc)
    ]
end


# ---------------------------------------------------------------------------
# Normal-inverse Gaussian parameters: α > |β|, δ > 0
# ---------------------------------------------------------------------------

function constrain_with_jac(p::NIG, θ)
    _check_dimension(p, θ)
    μ = θ[1]
    γ = exp(θ[2])
    β = θ[3]
    δ = exp(θ[4])
    α = hypot(β, γ)
    return _promoted_vector((μ, α, β, δ)), [
        one(α)   zero(α)     zero(α)  zero(α)
        zero(α)  γ*γ/α       β/α      zero(α)
        zero(α)  zero(α)     one(α)   zero(α)
        zero(α)  zero(α)     zero(α)  δ
    ]
end

function unconstrain(p::NIG, η)
    _check_constrained_dimension(p, η)
    μ, α, β, δ = η
    α > abs(β) || throw(DomainError(η, "parameters must satisfy α > |β|"))
    δ > zero(δ) || throw(DomainError(η, "δ must be strictly positive"))
    γ2 = (α - abs(β)) * (α + abs(β))
    return _promoted_vector((μ, log(γ2)/2, β, log(δ)))
end

function unconstrain_with_jac(p::NIG, η)
    θ = unconstrain(p, η)
    _, α, β, δ = η
    γ2 = (α - abs(β)) * (α + abs(β))
    return θ, [
        one(α)   zero(α)   zero(α)    zero(α)
        zero(α)  α/γ2      -β/γ2      zero(α)
        zero(α)  zero(α)   one(α)     zero(α)
        zero(α)  zero(α)   zero(α)    inv(δ)
    ]
end


# ---------------------------------------------------------------------------
# Simplex chart (K constrained coordinates, K-1 free coordinates)
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Simplex, θ)
    _check_dimension(p, θ)
    k = p.n
    k == 1 && return [1.0], zeros(Float64, 1, 0)

    m = max(zero(θ[1]), maximum(θ))
    anchor_weight = exp(-m)
    free_weights = exp.(θ .- m)
    denom = anchor_weight + sum(free_weights)

    T = promote_type(typeof(anchor_weight), eltype(free_weights))
    η = Vector{T}(undef, k)
    j = 1
    for i in 1:k
        if i == p.anchor
            η[i] = anchor_weight / denom
        else
            η[i] = free_weights[j] / denom
            j += 1
        end
    end

    J = zeros(T, k, k - 1)
    j = 1
    for col in 1:k
        col == p.anchor && continue
        for i in 1:k
            J[i, j] = η[i] * ((i == col ? one(T) : zero(T)) - η[col])
        end
        j += 1
    end
    return η, J
end

function unconstrain(p::Simplex, η)
    _check_constrained_dimension(p, η)
    all(x -> x >= zero(x), η) ||
        throw(DomainError(η, "probabilities must be nonnegative"))

    total = sum(η)
    isapprox(total, one(total)) ||
        throw(DomainError(η, "probabilities must sum to one"))

    anchor = η[p.anchor]
    anchor > zero(anchor) || throw(DomainError(
        η,
        "the anchor probability must be strictly positive for this simplex chart",
    ))

    log_anchor = log(anchor)
    return _promoted_vector([
        log(η[i]) - log_anchor for i in 1:p.n if i != p.anchor
    ])
end

function unconstrain_with_jac(p::Simplex, η)
    θ = unconstrain(p, η)
    k = p.n
    k == 1 && return θ, zeros(Float64, 0, 1)

    T = promote_type(map(typeof, η)...)
    J = zeros(T, k - 1, k)
    j = 1
    for i in 1:k
        i == p.anchor && continue
        J[j, i] = inv(η[i])
        J[j, p.anchor] = -inv(η[p.anchor])
        j += 1
    end
    return θ, J
end


# ---------------------------------------------------------------------------
# Symmetric positive-definite matrices, represented by lower triangles
# ---------------------------------------------------------------------------

function _spd_pairs(n::Integer)
    pairs = Tuple{Int,Int}[]
    for i in 1:n, j in 1:i
        push!(pairs, (i, j))
    end
    return pairs
end

function _spd_cholesky_from_theta(p::SPD, θ)
    _check_dimension(p, θ)
    T = isempty(θ) ? Float64 : promote_type(map(typeof, θ)...)
    L = zeros(T, p.n, p.n)
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        L[i, j] = i == j ? exp(θ[q]) : θ[q]
    end
    return L
end

function _spd_eta_from_cholesky(L)
    n = size(L, 1)
    pairs = _spd_pairs(n)
    η = Vector{eltype(L)}(undef, length(pairs))
    for (q, (i, j)) in enumerate(pairs)
        s = zero(eltype(L))
        for k in 1:j
            s += L[i, k] * L[j, k]
        end
        η[q] = s
    end
    return η
end

function _spd_matrix_from_eta(p::SPD, η)
    _check_constrained_dimension(p, η)
    T = isempty(η) ? Float64 : promote_type(map(typeof, η)...)
    S = zeros(T, p.n, p.n)
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        S[i, j] = S[j, i] = η[q]
    end
    return S
end

function _spd_cholesky_from_eta(p::SPD, η)
    S = _spd_matrix_from_eta(p, η)
    T = typeof(sqrt(one(eltype(S))))
    L = zeros(T, p.n, p.n)

    for i in 1:p.n, j in 1:i
        s = convert(T, S[i, j])
        for k in 1:(j - 1)
            s -= L[i, k] * L[j, k]
        end
        if i == j
            s > zero(s) || throw(DomainError(η, "matrix must be positive definite"))
            L[i, j] = sqrt(s)
        else
            L[i, j] = s / L[j, j]
        end
    end
    return L
end

function constrain_with_jac(p::SPD, θ)
    L = _spd_cholesky_from_theta(p, θ)
    η = _spd_eta_from_cholesky(L)
    pairs = _spd_pairs(p.n)
    T = eltype(L)
    J = zeros(T, length(pairs), length(pairs))

    for (q, (i, j)) in enumerate(pairs), (r, (a, b)) in enumerate(pairs)
        dL = a == b ? L[a, b] : one(T)
        v = zero(T)
        a == i && b <= j && (v += dL * L[j, b])
        a == j && b <= j && (v += L[i, b] * dL)
        J[q, r] = v
    end
    return η, J
end

function unconstrain(p::SPD, η)
    L = _spd_cholesky_from_eta(p, η)
    pairs = _spd_pairs(p.n)
    θ = Vector{eltype(L)}(undef, length(pairs))
    for (q, (i, j)) in enumerate(pairs)
        θ[q] = i == j ? log(L[i, j]) : L[i, j]
    end
    return θ
end

function unconstrain_with_jac(p::SPD, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)
    return θ, _invert_lower_triangular(J)
end


# ---------------------------------------------------------------------------
# Prefix wrapper and derived operations
# ---------------------------------------------------------------------------

constrain_with_jac(p::Prefixed, θ) = constrain_with_jac(p.space, θ)
unconstrain(p::Prefixed, η) = unconstrain(p.space, η)
unconstrain_with_jac(p::Prefixed, η) = unconstrain_with_jac(p.space, η)

constrain(p, θ) = first(constrain_with_jac(p, θ))
constrain_jac(p, θ) = last(constrain_with_jac(p, θ))
unconstrain_jac(p, η) = last(unconstrain_with_jac(p, η))

function unconstrain_with_jac(p::AbstractSeparableParameterSpace, η)
    θ = unconstrain(p, η)
    _, J = constrain_with_jac(p, θ)
    return θ, _diagonal_matrix([inv(J[i, i]) for i in 1:dimension(p)])
end

function unconstrain_with_jac(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)
    values = Any[]
    blocks = Any[]
    offset = 0

    for q in p
        n = constrained_dimension(q)
        θq, Jq = unconstrain_with_jac(q, view(η, (offset + 1):(offset + n)))
        push!(values, θq)
        push!(blocks, Jq)
        offset += n
    end
    return _concatenate_vectors(values), _blockdiag(blocks)
end

unconstrained_example(p) = zeros(dimension(p))
constrained_example(p) = constrain(p, unconstrained_example(p))


# ---------------------------------------------------------------------------
# Jacobian determinants
# ---------------------------------------------------------------------------

function logabsdet_constrain_jac(p::AbstractSeparableParameterSpace, θ)
    _check_dimension(p, θ)
    isempty(θ) && return 0.0
    _, J = constrain_with_jac(p, θ)
    s = zero(J[1, 1])
    for i in 1:dimension(p)
        s += log(abs(J[i, i]))
    end
    return s
end

logabsdet_unconstrain_jac(p::AbstractSeparableParameterSpace, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)
    s = 0.0
    offset = 0
    for q in p
        n = dimension(q)
        s += logabsdet_constrain_jac(q, view(θ, (offset + 1):(offset + n)))
        offset += n
    end
    return s
end

function logabsdet_unconstrain_jac(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)
    s = 0.0
    offset = 0
    for q in p
        n = constrained_dimension(q)
        s += logabsdet_unconstrain_jac(q, view(η, (offset + 1):(offset + n)))
        offset += n
    end
    return s
end

function logabsdet_constrain_jac(p::OrderedSpace, θ)
    _check_dimension(p, θ)
    _, da = _constrain_scalar(p.first_domain, θ[1])
    return log(abs(da)) + θ[2]
end

logabsdet_unconstrain_jac(p::OrderedSpace, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::Between, θ)
    _check_dimension(p, θ)
    _, dq = _constrain_scalar(ProbabilityDomain{true,true}(), θ[3])
    return 2*θ[2] + log(dq)
end

logabsdet_unconstrain_jac(p::Between, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::NIG, θ)
    _check_dimension(p, θ)
    γ = exp(θ[2])
    α = hypot(θ[3], γ)
    return 2*θ[2] - log(α) + θ[4]
end

logabsdet_unconstrain_jac(p::NIG, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

function logabsdet_constrain_jac(p::SPD, θ)
    _check_dimension(p, θ)
    s = p.n * log(2.0)
    for (q, (i, j)) in enumerate(_spd_pairs(p.n))
        i == j && (s += (p.n - i + 2) * θ[q])
    end
    return s
end

logabsdet_unconstrain_jac(p::SPD, η) =
    -logabsdet_constrain_jac(p, unconstrain(p, η))

logabsdet_constrain_jac(p::Prefixed, θ) =
    logabsdet_constrain_jac(p.space, θ)
logabsdet_unconstrain_jac(p::Prefixed, η) =
    logabsdet_unconstrain_jac(p.space, η)

function logabsdet_constrain_jac(p::Simplex, θ)
    _check_dimension(p, θ)
    throw(ArgumentError(
        "logabsdet is not defined for the rectangular simplex Jacobian; " *
        "use constrain_with_jac to access the K×(K-1) Jacobian"
    ))
end

function logabsdet_unconstrain_jac(p::Simplex, η)
    _check_constrained_dimension(p, η)
    throw(ArgumentError(
        "logabsdet is not defined for the rectangular simplex Jacobian; " *
        "use unconstrain_with_jac to access the (K-1)×K Jacobian"
    ))
end


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

function constrained_namedtuple(p, θ)
    η = constrain(p, θ)
    return NamedTuple{parameter_symbols(p)}(Tuple(η))
end

# ---------------------------------------------------------------------------
# Distribution integration helpers
# ---------------------------------------------------------------------------

function _pd_space(symbol::Symbol, A)
    n = size(A, 1)
    name = nameof(typeof(A))

    if name === :ScalMat
        return Pos(symbol)
    elseif name === :PDiagMat
        return PosVec(symbol, n)
    else
        return SPD(symbol, n)
    end
end


function _first_distribution_field(d)
    for name in propertynames(d)
        value = getproperty(d, name)
        value isa Distribution && return value
    end
    throw(ArgumentError("could not find an underlying distribution in $(typeof(d))"))
end


function _distribution_container(d)
    for name in propertynames(d)
        value = getproperty(d, name)

        if value isa NamedTuple
            vals = values(value)
            !isempty(vals) && all(x -> x isa Distribution, vals) && return value
        elseif value isa AbstractArray
            !isempty(value) && all(x -> x isa Distribution, value) && return value
        elseif value isa Tuple
            !isempty(value) && all(x -> x isa Distribution, value) && return value
        end
    end

    throw(ArgumentError("could not find component distributions in $(typeof(d))"))
end


function _product_param_space(d)
    container = _distribution_container(d)

    if container isa NamedTuple
        names = keys(container)
        vals = values(container)
        spaces = map(names, vals) do name, dist
            Prefixed(name, param_space(dist))
        end
        return Tuple(spaces)
    end

    vals = collect(container)
    spaces = Vector{AbstractParameterSpace}(undef, length(vals))
    for i in eachindex(vals)
        spaces[i] = Prefixed(Symbol("component", i), param_space(vals[i]))
    end
    return Tuple(spaces)
end


# ---------------------------------------------------------------------------
# Distributions.jl integration
# ---------------------------------------------------------------------------

# Ordered endpoints
param_space(::Uniform)                  = Ordered(:a, :b)
param_space(::Arcsine)                  = Ordered(:a, :b)
param_space(::LogUniform)               = PosOrdered(:a, :b)
param_space(::TriangularDist)           = Between(:a, :b, :c)

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
param_space(::SkewedExponentialPower)   = (Id(:μ), Pos(:σ), Pos(:p), ProbOpen(:α))
param_space(::JohnsonSU)                = (Id(:ξ), Pos(:λ), Id(:γ), Pos(:δ))
param_space(::NormalInverseGaussian)    = NIG(:μ, :α, :β, :δ)
param_space(::StudentizedRange)         = (Pos(:ν), Lower(:k, 1.0))

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
param_space(::Soliton)                  = (ProbOpen(:δ), ProbOpenRight(:atol))

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
param_space(d::DiscreteNonParametric)   = Simplex(:p, probs(d))
param_space(d::Dirichlet)               = PosVec(:α, length(d))

# Simplex-valued probability parameters. The trial count of Multinomial
# remains structural and is therefore not part of the optimization space.
param_space(d::Categorical)             = Simplex(:p, probs(d))
param_space(d::Multinomial)             = Simplex(:p, probs(d))

# Noncentral hypergeometric families: population/sample sizes are structural.
if isdefined(Distributions, :FisherNoncentralHypergeometric)
    @eval param_space(::Distributions.FisherNoncentralHypergeometric) = Pos(:ω)
end
if isdefined(Distributions, :WalleniusNoncentralHypergeometric)
    @eval param_space(::Distributions.WalleniusNoncentralHypergeometric) = Pos(:ω)
end

# Affine wrapper. The sign of the nonzero scale determines the connected chart.
function param_space(d::Distributions.AffineDistribution)
    scale_space = d.σ > zero(d.σ) ? Pos(:σ) : Neg(:σ)
    return (Id(:μ), scale_space, Prefixed(:base, param_space(d.ρ)))
end

# Multivariate normal families.
function param_space(d::MvNormal)
    μ, Σ = params(d)
    return (RealVec(:μ, length(μ)), _pd_space(:Σ, Σ))
end

function param_space(d::MvNormalCanon)
    h, J = params(d)
    return (RealVec(:h, length(h)), _pd_space(:J, J))
end

function param_space(d::MvLogNormal)
    μ, Σ = params(d)
    return (RealVec(:μ, length(μ)), _pd_space(:Σ, Σ))
end

param_space(d::MvLogitNormal) =
    Prefixed(:normal, param_space(d.normal))

# Matrix-variate distributions.
function param_space(d::MatrixNormal)
    M, U, V = params(d)
    m, n = size(M)
    return (
        RealMat(:M, m, n),
        _pd_space(:U, U),
        _pd_space(:V, V),
    )
end

function param_space(d::Wishart)
    ν, S = params(d)
    p = size(d, 1)

    if ν > p - 1
        return (Lower(:ν, p - 1), _pd_space(:S, S))
    else
        # Singular Wishart requires integer degrees of freedom; keep ν structural.
        return _pd_space(:S, S)
    end
end

function param_space(d::InverseWishart)
    ν, Ψ = params(d)
    p = size(d, 1)
    return (Lower(:ν, p - 1), _pd_space(:Ψ, Ψ))
end

function param_space(d::MatrixTDist)
    ν, M, Σ, Ω = params(d)
    m, n = size(M)
    return (
        Pos(:ν),
        RealMat(:M, m, n),
        _pd_space(:Σ, Σ),
        _pd_space(:Ω, Ω),
    )
end

function param_space(d::MatrixBeta)
    p = size(d, 1)
    return (Lower(:n1, p - 1), Lower(:n2, p - 1))
end

function param_space(d::MatrixFDist)
    ps = params(d)
    p = size(d, 1)
    B = ps[end]
    return (
        Lower(:n1, p - 1),
        Lower(:n2, p - 1),
        _pd_space(:B, B),
    )
end

param_space(::LKJ) = Pos(:η)
param_space(::LKJCholesky) = Pos(:η)

# Generic wrappers. Truncation/censoring bounds, reshape dimensions, and
# order-statistic ranks/sample sizes are treated as structural.
param_space(d::Distributions.Truncated) =
    Prefixed(:base, param_space(d.untruncated))

param_space(d::Distributions.Censored) =
    Prefixed(:base, param_space(d.uncensored))

param_space(d::Distributions.OrderStatistic) =
    Prefixed(:base, param_space(_first_distribution_field(d)))

param_space(d::Distributions.JointOrderStatistics) =
    Prefixed(:base, param_space(_first_distribution_field(d)))

param_space(d::Distributions.ReshapedDistribution) =
    Prefixed(:base, param_space(_first_distribution_field(d)))

# Mixture models: all component parameters plus a simplex of mixing weights.
function param_space(d::Distributions.AbstractMixtureModel)
    cs = components(d)
    component_spaces = ntuple(length(cs)) do i
        Prefixed(Symbol("component", i), param_space(cs[i]))
    end
    return (component_spaces..., Simplex(:π, probs(d)))
end

# Product distributions.
param_space(d::Distributions.ProductDistribution) = _product_param_space(d)
param_space(d::Distributions.ProductNamedTupleDistribution) = _product_param_space(d)

# Product is deprecated but still present in Distributions 0.25.
if isdefined(Distributions, :Product)
    @eval param_space(d::Distributions.Product) = _product_param_space(d)
end

param_space(d::Distribution) =
    throw(ArgumentError("parameter space not implemented for $(typeof(d))"))


end # module