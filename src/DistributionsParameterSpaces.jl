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
# Parameter spaces
# ---------------------------------------------------------------------------

abstract type AbstractParameterSpace end
abstract type AbstractScalarParameterSpace <: AbstractParameterSpace end


struct Id{S} <: AbstractScalarParameterSpace end
struct Pos{S} <: AbstractScalarParameterSpace end
struct NonNeg{S} <: AbstractScalarParameterSpace end
struct Prob{S} <: AbstractScalarParameterSpace end
struct ProbOpen{S} <: AbstractScalarParameterSpace end
struct ProbOpenLeft{S} <: AbstractScalarParameterSpace end
struct ProbOpenRight{S} <: AbstractScalarParameterSpace end

struct Lower{S,T} <: AbstractScalarParameterSpace
    lower::T
end

struct Ordered{A,B} <: AbstractParameterSpace end
struct PosOrdered{A,B} <: AbstractParameterSpace end
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


struct RealVec{S} <: AbstractParameterSpace
    n::Int

    function RealVec{S}(n::Integer) where {S}
        n >= 0 || throw(ArgumentError("dimension must be nonnegative"))
        new{S}(Int(n))
    end
end


struct RealMat{S} <: AbstractParameterSpace
    m::Int
    n::Int

    function RealMat{S}(m::Integer, n::Integer) where {S}
        m >= 0 || throw(ArgumentError("number of rows must be nonnegative"))
        n >= 0 || throw(ArgumentError("number of columns must be nonnegative"))
        new{S}(Int(m), Int(n))
    end
end


struct SPD{S} <: AbstractParameterSpace
    n::Int

    function SPD{S}(n::Integer) where {S}
        n > 0 || throw(ArgumentError("matrix dimension must be positive"))
        new{S}(Int(n))
    end
end


struct Neg{S} <: AbstractScalarParameterSpace end


struct Prefixed{P,S} <: AbstractParameterSpace
    space::S
end


Id(s::Symbol)     = Id{s}()
Pos(s::Symbol)    = Pos{s}()
NonNeg(s::Symbol) = NonNeg{s}()
Prob(s::Symbol)   = Prob{s}()
ProbOpen(s::Symbol) = ProbOpen{s}()
ProbOpenLeft(s::Symbol) = ProbOpenLeft{s}()
ProbOpenRight(s::Symbol) = ProbOpenRight{s}()
Lower(s::Symbol, lower) = Lower{s,typeof(lower)}(lower)
Ordered(a::Symbol, b::Symbol) = Ordered{a,b}()
PosOrdered(a::Symbol, b::Symbol) = PosOrdered{a,b}()
Between(a::Symbol, b::Symbol, c::Symbol) = Between{a,b,c}()
NIG(μ::Symbol, α::Symbol, β::Symbol, δ::Symbol) = NIG{μ,α,β,δ}()

Simplex(s::Symbol, n::Integer; anchor::Integer=n) =
    Simplex{s}(n, anchor)

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

PosVec(s::Symbol, n::Integer) = PosVec{s}(n)
ProbVec(s::Symbol, n::Integer) = ProbVec{s}(n)
RealVec(s::Symbol, n::Integer) = RealVec{s}(n)
RealMat(s::Symbol, m::Integer, n::Integer) = RealMat{s}(m, n)
SPD(s::Symbol, n::Integer) = SPD{s}(n)
Neg(s::Symbol) = Neg{s}()
Prefixed(prefix::Symbol, space) = Prefixed{prefix,typeof(space)}(space)


const ProductParameterSpace = Tuple{Vararg{AbstractParameterSpace}}
const SeparableParameterSpace =
    Union{AbstractScalarParameterSpace, PosVec, ProbVec, RealVec, RealMat}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

parameter_symbol(::Id{S}) where {S} = S
parameter_symbol(::Pos{S}) where {S} = S
parameter_symbol(::NonNeg{S}) where {S} = S
parameter_symbol(::Prob{S}) where {S} = S
parameter_symbol(::ProbOpen{S}) where {S} = S
parameter_symbol(::ProbOpenLeft{S}) where {S} = S
parameter_symbol(::ProbOpenRight{S}) where {S} = S
parameter_symbol(::Lower{S}) where {S} = S
parameter_symbol(::Neg{S}) where {S} = S

parameter_symbols(::Ordered{A,B}) where {A,B} = (A, B)
parameter_symbols(::PosOrdered{A,B}) where {A,B} = (A, B)
parameter_symbols(::Between{A,B,C}) where {A,B,C} = (A, B, C)
parameter_symbols(::NIG{M,A,B,D}) where {M,A,B,D} = (M, A, B, D)

parameter_symbols(p::Simplex{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

parameter_symbols(p::AbstractScalarParameterSpace) =
    (parameter_symbol(p),)

function parameter_symbols(p::ProductParameterSpace)
    result = Symbol[]
    for q in p
        append!(result, parameter_symbols(q))
    end
    return Tuple(result)
end

parameter_symbols(p::PosVec{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

parameter_symbols(p::ProbVec{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

parameter_symbols(p::RealVec{S}) where {S} =
    ntuple(i -> Symbol(S, "_", i), p.n)

function parameter_symbols(p::RealMat{S}) where {S}
    return ntuple(p.m * p.n) do k
        i = mod1(k, p.m)
        j = (k - 1) ÷ p.m + 1
        Symbol(S, "_", i, "_", j)
    end
end

function parameter_symbols(p::SPD{S}) where {S}
    result = Symbol[]
    for i in 1:p.n
        for j in 1:i
            push!(result, Symbol(S, "_", i, "_", j))
        end
    end
    return Tuple(result)
end

parameter_symbols(p::Prefixed{P}) where {P} =
    Tuple(Symbol(P, "_", s) for s in parameter_symbols(p.space))


dimension(::AbstractScalarParameterSpace) = 1
dimension(::Union{Ordered,PosOrdered}) = 2
dimension(::Between) = 3
dimension(::NIG) = 4
dimension(p::Simplex) = p.n - 1
function dimension(p::ProductParameterSpace)
    n = 0
    for q in p
        n += dimension(q)
    end
    return n
end
dimension(p::PosVec) = p.n
dimension(p::ProbVec) = p.n
dimension(p::RealVec) = p.n
dimension(p::RealMat) = p.m * p.n
dimension(p::SPD) = p.n * (p.n + 1) ÷ 2
dimension(p::Prefixed) = dimension(p.space)

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
    n == 0 && return Matrix{Float64}(undef, 0, 0)

    T = promote_type(map(typeof, d)...)
    J = zeros(T, n, n)

    for i in 1:n
        J[i, i] = d[i]
    end

    return J
end


function _identity_matrix(T, n::Integer)
    J = zeros(T, n, n)
    for i in 1:n
        J[i, i] = one(T)
    end
    return J
end


function _concatenate_vectors(vs)
    isempty(vs) && return Float64[]

    total = sum(length, vs)
    total == 0 && return Float64[]

    Ts = [eltype(v) for v in vs if !isempty(v)]
    T = isempty(Ts) ? Float64 : promote_type(Ts...)
    result = Vector{T}(undef, total)

    offset = 0
    for v in vs
        for x in v
            offset += 1
            result[offset] = x
        end
    end

    return result
end


function _blockdiag(blocks)
    isempty(blocks) && return zeros(Float64, 0, 0)

    nr = sum(B -> size(B, 1), blocks)
    nc = sum(B -> size(B, 2), blocks)

    Ts = [eltype(B) for B in blocks]
    T = isempty(Ts) ? Float64 : promote_type(Ts...)
    result = zeros(T, nr, nc)

    r0 = 0
    c0 = 0
    for B in blocks
        m, n = size(B)
        for j in 1:n
            for i in 1:m
                result[r0 + i, c0 + j] = B[i, j]
            end
        end
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

    for j in 1:n
        for i in j:n
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
    end

    return B
end


# ---------------------------------------------------------------------------
# Scalar transforms
# ---------------------------------------------------------------------------

_constrain_with_jac(::Id, θ) = (θ, one(θ))
_unconstrain(::Id, η) = η


function _constrain_with_jac(::Neg, θ)
    η = -exp(θ)
    return η, η
end

function _unconstrain(::Neg, η)
    η < zero(η) ||
        throw(DomainError(η, "parameter must be strictly negative"))
    return log(-η)
end


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


function _constrain_with_jac(::Union{Prob,ProbOpen,ProbOpenLeft,ProbOpenRight}, θ)
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


function _unconstrain(::ProbOpen, η)
    zero(η) < η < one(η) ||
        throw(DomainError(η, "parameter must belong to (0, 1)"))

    return log(η) - log1p(-η)
end


function _unconstrain(::ProbOpenRight, η)
    zero(η) <= η < one(η) ||
        throw(DomainError(η, "parameter must belong to [0, 1)"))

    return log(η) - log1p(-η)
end


function _constrain_with_jac(p::Lower, θ)
    w = exp(θ)
    return p.lower + w, w
end

function _unconstrain(p::Lower, η)
    η > p.lower ||
        throw(DomainError(η, "parameter must be strictly greater than $(p.lower)"))
    return log(η - p.lower)
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

    values = Vector{Any}()
    blocks = Vector{Any}()
    offset = 0

    for q in p
        n = dimension(q)
        θq = view(θ, (offset + 1):(offset + n))
        ηq, Jq = constrain_with_jac(q, θq)
        push!(values, ηq)
        push!(blocks, Jq)
        offset += n
    end

    return _concatenate_vectors(values), _blockdiag(blocks)
end


function unconstrain(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)

    values = Vector{Any}()
    offset = 0

    for q in p
        n = constrained_dimension(q)
        ηq = view(η, (offset + 1):(offset + n))
        push!(values, unconstrain(q, ηq))
        offset += n
    end

    return _concatenate_vectors(values)
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
# Bounded interior point: a <= c <= b
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Between, θ)
    _check_dimension(p, θ)

    a = θ[1]
    w = exp(θ[2])
    q, dq = _constrain_with_jac(Prob{:q}(), θ[3])

    b = a + w
    c = a + w * q

    η = _promoted_vector((a, b, c))
    J = [
        one(w)  zero(w)  zero(w)
        one(w)  w        zero(w)
        one(w)  w * q    w * dq
    ]

    return η, J
end


function unconstrain(p::Between, η)
    _check_constrained_dimension(p, η)

    a, b, c = η

    a <= c <= b ||
        throw(DomainError(η, "parameters must satisfy a <= c <= b"))

    if b == a
        c == a ||
            throw(DomainError(η, "degenerate bounds require a == b == c"))
        return _promoted_vector((a, -Inf, 0.0))
    end

    w = b - a
    q = (c - a) / w
    z = _unconstrain(Prob{:q}(), q)

    return _promoted_vector((a, log(w), z))
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

    η = _promoted_vector((μ, α, β, δ))
    J = [
        one(α)      zero(α)       zero(α)  zero(α)
        zero(α)     γ * γ / α     β / α    zero(α)
        zero(α)     zero(α)       one(α)   zero(α)
        zero(α)     zero(α)       zero(α)  δ
    ]

    return η, J
end


function unconstrain(p::NIG, η)
    _check_constrained_dimension(p, η)

    μ, α, β, δ = η
    α > abs(β) ||
        throw(DomainError(η, "parameters must satisfy α > |β|"))
    δ > zero(δ) ||
        throw(DomainError(η, "δ must be strictly positive"))

    γ2 = (α - abs(β)) * (α + abs(β))
    return _promoted_vector((μ, log(γ2) / 2, β, log(δ)))
end


# ---------------------------------------------------------------------------
# Simplex
#
# The unconstrained representation has K - 1 coordinates. One probability
# coordinate is used as an anchor and its logit is fixed to zero.
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Simplex, θ)
    _check_dimension(p, θ)

    k = p.n
    if k == 1
        return [1.0], zeros(Float64, 1, 0)
    end

    # Softmax with the anchor logit fixed to zero.
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
    for col_index in 1:k
        col_index == p.anchor && continue

        for i in 1:k
            J[i, j] = η[i] * ((i == col_index ? one(T) : zero(T)) - η[col_index])
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

    anchor_probability = η[p.anchor]
    anchor_probability > zero(anchor_probability) ||
        throw(DomainError(
            η,
            "the anchor probability must be strictly positive for this simplex chart",
        ))

    log_anchor = log(anchor_probability)
    θ = [
        log(η[i]) - log_anchor
        for i in 1:p.n
        if i != p.anchor
    ]

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
# Real vectors and matrices
# ---------------------------------------------------------------------------

function constrain_with_jac(p::Union{RealVec,RealMat}, θ)
    _check_dimension(p, θ)

    η = collect(θ)
    T = isempty(η) ? Float64 : eltype(η)
    return η, _identity_matrix(T, length(η))
end


function unconstrain(p::Union{RealVec,RealMat}, η)
    _check_constrained_dimension(p, η)
    return collect(η)
end


# ---------------------------------------------------------------------------
# Symmetric positive-definite matrices
#
# Both θ and η use the lower-triangular ordering
#
#   (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), ...
#
# θ parameterizes a lower Cholesky factor L. Its diagonal is exponentiated
# and η contains the independent lower-triangular entries of L*L'.
# ---------------------------------------------------------------------------

function _spd_pairs(n::Integer)
    pairs = Tuple{Int,Int}[]
    for i in 1:n
        for j in 1:i
            push!(pairs, (i, j))
        end
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

    T = eltype(L)
    η = Vector{T}(undef, length(pairs))

    for (q, (i, j)) in enumerate(pairs)
        s = zero(T)
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
        S[i, j] = η[q]
        S[j, i] = η[q]
    end

    return S
end


function _spd_cholesky_from_eta(p::SPD, η)
    S = _spd_matrix_from_eta(p, η)

    # sqrt(one(Int)) is Float64, while this preserves floating/AD-like types.
    T = typeof(sqrt(one(eltype(S))))
    L = zeros(T, p.n, p.n)

    for i in 1:p.n
        for j in 1:i
            s = convert(T, S[i, j])
            for k in 1:(j - 1)
                s -= L[i, k] * L[j, k]
            end

            if i == j
                s > zero(s) ||
                    throw(DomainError(η, "matrix must be positive definite"))
                L[i, j] = sqrt(s)
            else
                L[i, j] = s / L[j, j]
            end
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

    for (q, (i, j)) in enumerate(pairs)
        for (r, (a, b)) in enumerate(pairs)
            dL = a == b ? L[a, b] : one(T)
            v = zero(T)

            if a == i && b <= j
                v += dL * L[j, b]
            end
            if a == j && b <= j
                v += L[i, b] * dL
            end

            J[q, r] = v
        end
    end

    return η, J
end


function unconstrain(p::SPD, η)
    L = _spd_cholesky_from_eta(p, η)
    pairs = _spd_pairs(p.n)

    T = eltype(L)
    θ = Vector{T}(undef, length(pairs))

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
# Prefix wrapper
# ---------------------------------------------------------------------------

constrain_with_jac(p::Prefixed, θ) =
    constrain_with_jac(p.space, θ)

unconstrain(p::Prefixed, η) =
    unconstrain(p.space, η)

unconstrain_with_jac(p::Prefixed, η) =
    unconstrain_with_jac(p.space, η)


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


function unconstrain_with_jac(p::ProductParameterSpace, η)
    _check_constrained_dimension(p, η)

    values = Vector{Any}()
    blocks = Vector{Any}()
    offset = 0

    for q in p
        n = constrained_dimension(q)
        ηq = view(η, (offset + 1):(offset + n))
        θq, Jq = unconstrain_with_jac(q, ηq)
        push!(values, θq)
        push!(blocks, Jq)
        offset += n
    end

    return _concatenate_vectors(values), _blockdiag(blocks)
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


function unconstrain_with_jac(p::Between, η)
    θ = unconstrain(p, η)

    a, b, c = η
    w = b - a
    w > zero(w) ||
        throw(DomainError(η, "inverse Jacobian is undefined for a == b"))

    ca = c - a
    bc = b - c

    J = [
        one(w)         zero(w)         zero(w)
        -inv(w)        inv(w)          zero(w)
        -inv(ca)       -inv(bc)        inv(ca) + inv(bc)
    ]

    return θ, J
end


function unconstrain_with_jac(p::NIG, η)
    θ = unconstrain(p, η)

    _, α, β, δ = η
    γ2 = (α - abs(β)) * (α + abs(β))

    J = [
        one(α)   zero(α)       zero(α)       zero(α)
        zero(α)  α / γ2        -β / γ2       zero(α)
        zero(α)  zero(α)       one(α)        zero(α)
        zero(α)  zero(α)       zero(α)       inv(δ)
    ]

    return θ, J
end


function unconstrain_with_jac(p::Simplex, η)
    θ = unconstrain(p, η)

    k = p.n
    if k == 1
        return θ, zeros(Float64, 0, 1)
    end

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


function logabsdet_constrain_jac(p::ProductParameterSpace, θ)
    _check_dimension(p, θ)

    s = 0.0
    offset = 0
    for q in p
        n = dimension(q)
        θq = view(θ, (offset + 1):(offset + n))
        s += logabsdet_constrain_jac(q, θq)
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
        ηq = view(η, (offset + 1):(offset + n))
        s += logabsdet_unconstrain_jac(q, ηq)
        offset += n
    end
    return s
end


function logabsdet_constrain_jac(p::SPD, θ)
    _check_dimension(p, θ)

    pairs = _spd_pairs(p.n)
    isempty(pairs) && return 0.0

    s = p.n * log(2.0)
    for (q, (i, j)) in enumerate(pairs)
        if i == j
            s += (p.n - i + 2) * θ[q]
        end
    end
    return s
end


function logabsdet_unconstrain_jac(p::SPD, η)
    θ = unconstrain(p, η)
    return -logabsdet_constrain_jac(p, θ)
end


logabsdet_constrain_jac(p::Prefixed, θ) =
    logabsdet_constrain_jac(p.space, θ)

logabsdet_unconstrain_jac(p::Prefixed, η) =
    logabsdet_unconstrain_jac(p.space, η)


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


function logabsdet_constrain_jac(p::Between, θ)
    _check_dimension(p, θ)

    q, dq = _constrain_with_jac(Prob{:q}(), θ[3])
    return 2 * θ[2] + log(dq)
end

function logabsdet_unconstrain_jac(p::Between, η)
    θ = unconstrain(p, η)
    return -logabsdet_constrain_jac(p, θ)
end


function logabsdet_constrain_jac(p::NIG, θ)
    _check_dimension(p, θ)

    γ = exp(θ[2])
    β = θ[3]
    α = hypot(β, γ)

    return 2 * θ[2] - log(α) + θ[4]
end

function logabsdet_unconstrain_jac(p::NIG, η)
    θ = unconstrain(p, η)
    return -logabsdet_constrain_jac(p, θ)
end


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