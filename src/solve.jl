# IDEA: more iterative methods: GMRES, Arnoldi, Lanczos
################################################################################
# TODO: preconditioning, could be external by running cg on
# EAE'x = Eb where inv(E)inv(E)' = M where M is preconditioner
# Careful with ill-conditioned systems
struct ConjugateGradient{T, M, V<:AbstractVector, R<:CircularBuffer}
    mul_A!::M # application of (non-)linear operator
    b::V # target
    d::V # direction
    Ad::V # product of A and direction
    r::R # cirular buffer, containing last two residuals
    function ConjugateGradient(mul_A!, b::AbstractVector, x::AbstractVector)
        r₁ = zero(b)
        mul_A!(r₁, x)
        @. r₁ = b - r₁ # mul!(r₁, A, x, -1., 1.) # r = b - Ax
        d = copy(r₁)
        V = typeof(r₁)
        r = CircularBuffer{V}(2)
        push!(r, r₁); push!(r, zero(r₁));
        Ad = zero(b)
        new{eltype(x), typeof(mul_A!), V, typeof(r)}(mul_A!, b, d, Ad, r)
    end
end
const CG = ConjugateGradient
# A has to support mul!(b, A, x)
function CG(A::AbstractMatOrFac, b::AbstractVector, x::AbstractVector)
    mul_A!(Ad, d) = mul!(Ad, A, d)
    ConjugateGradient(mul_A!, b, x)
end
function CG(A::AbstractMatOrFac, b::AbstractVector)
    x = zeros(promote_type(eltype(A), eltype(b)), size(A, 2))
    CG(A, b, x)
end

# no allocations :)
function update!(C::ConjugateGradient, x::AbstractVector, t::Int)
    if t > length(x) # algorithm does not have guarantees after n iterations
        return x # restart?
    elseif t > 1
        # TODO: change for non-linear problems
        β = sum(abs2, C.r[1]) / sum(abs2, C.r[2]) # new over old residual norm
        @. C.d = β*C.d + C.r[1] # axpy!
    end
    C.mul_A!(C.Ad, C.d) # C.Ad = C.A * C.d or could be nonlinear
    α = sum(abs2, C.r[1]) / dot(C.d, C.Ad)
    @. x += α * C.d # this directly updates x -> <:Update
    circshift!(C.r)
    @. C.r[1] = C.r[2] - α * C.Ad
    return x
end

function cg(A::AbstractMatOrFac, b::AbstractVector; max_iter::Int = size(A, 2), min_res::Real = 0)
    x = zeros(promote_type(eltype(A), eltype(b)), size(A, 2))
    cg!(x, A, b, max_iter = max_iter, min_res = min_res)
end

function cg(A::AbstractMatOrFac, B::AbstractMatrix; max_iter::Int = size(A, 2), min_res::Real = 0)
    X = zeros(promote_type(eltype(A), eltype(B)), size(B))
    cg!(X, A, B, max_iter = max_iter, min_res = min_res)

end
function cg!(X::AbstractMatrix, A::AbstractMatOrFac, B::AbstractMatrix;
             max_iter::Int = size(A, 2), min_res::Real = 0)
    @sync for (i, b) in enumerate(eachcol(B))
        @spawn begin
            x = @view X[:, i]
            cg!(x, A, b, max_iter = max_iter, min_res = min_res)
        end
    end
    return X
end

function cg!(x::AbstractVector, A::AbstractMatOrFac, b::AbstractVector;
                                    max_iter::Int = size(A, 2), min_res::Real = 0)
    cg!(CG(A, b, x), x, max_iter = max_iter, min_res = min_res)
end

function cg!(U::CG, x::AbstractVector; max_iter::Int = size(A, 2), min_res::Real = 0)
    for i in 1:max_iter
        update!(U, x, i)
        norm(U.r[1]) > min_res || break
    end
    return x
end

################################################################################
# Necessary?
# TODO: could extend for general iterative solvers
# struct IterativeMatrix end
# wrapper type which converts all solves into conjugate gradient solves,
# with minimum residual tolerance tol
# add pre-conditioner?
struct ConjugateGradientMatrix{T, M<:AbstractMatOrFac{T}, TOL <:Real} <:LazyFactorization{T} # <: AbstractMatrix{T}
    parent::M
    tol::TOL # minimum residual tolerance
    function ConjugateGradientMatrix(A; tol::Real = 0, check::Bool = true)
        ishermitian(A) || throw("input matrix not hermitian")
        T = eltype(A)
        new{T, typeof(A), typeof(tol)}(A, convert(T, tol))
    end
end
const CGMatrix = ConjugateGradientMatrix
Base.getindex(A::CGMatrix, i...) = getindex(A.parent, i...)
Base.setindex!(A::CGMatrix, i...) = setindex!(A.parent, i...)
Base.size(A::CGMatrix) = size(A.parent)
Base.size(A::CGMatrix, i) = 1 ≤ i ≤ 2 ? size(A.parent)[i] : 1

Base.:*(A::CGMatrix, b::AbstractVector) = A.parent * b
# Base.:\(A::CGMatrix, b::AbstractVector) = cg(A.parent, b)
import LinearAlgebra: mul!, ldiv!
function mul!(y::AbstractVector, A::CGMatrix, x::AbstractVector, α::Real = 1, β::Real = 0)
    mul!(y, A.parent, x, α, β)
end
function ldiv!(y::AbstractVector, A::CGMatrix, x::AbstractVector)
    cg!(y, A.parent, x) # TODO: pre-allocate
end
