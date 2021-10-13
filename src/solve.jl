# IDEA: more iterative methods: GMRES, Arnoldi, Lanczos
################################################################################
struct ConjugateGradient{T, AT, V<:AbstractVecOrMat, RT<:CircularBuffer, RN<:AbstractVecOrMat, MT}
    A::AT # application of (non-)linear operator
    b::V # target
    d::V # direction
    Ad::V # product of A and direction
    r::RT # cirular buffer, containing last two residuals
    z::RT # circular buffer, containing last two "preconditioned" residuals
    M::MT # pre-conditioner (if any)
    r_norms::RN # holds norms of residuals over optimization path
    function ConjugateGradient(A::AbstractMatOrFac, b::AbstractVecOrMat, x::AbstractVecOrMat,
                               M::Union{Nothing, AbstractMatOrFac} = nothing) # M is pre-conditioner
        check_input(x)
        r₁ = zero(b)
        mul!(r₁, A, x)
        @. r₁ = b - r₁ # mul!(r₁, A, x, -1., 1.) # r = b - Ax
        V = typeof(r₁)
        r = CircularBuffer{V}(2)
        push!(r, r₁); push!(r, zero(r₁));
        z = r # without preconditioner, z is equivalent to r, so pointing to the same memory leads to correct and efficient behavior
        z₁ = r₁
        if !isnothing(M)
            z = CircularBuffer{V}(2)
            z₁ = M \ r₁
            push!(z, z₁); push!(z, zero(r₁));
        end
        d = copy(z₁)
        Ad = zero(b)
        norm_T = typeof(norm(r₁))
        r_norms = fill(norm_T(NaN), size(x))
        T, AT, RT, RN, MT = eltype(x), typeof(A), typeof(r), typeof(r_norms), typeof(M)
        new{T, AT, V, RT, RN, MT}(A, b, d, Ad, r, z, M, r_norms)
    end
end
const CG = ConjugateGradient
function CG(A::AbstractMatOrFac, b::AbstractVector)
    x = zeros(promote_type(eltype(A), eltype(b)), size(A, 2))
    CG(A, b, x)
end
function CG(A::AbstractMatOrFac, B::AbstractMatrix)
    T = promote_type(eltype(A), eltype(B))
    X = zeros(T, (size(A, 2), size(B, 2)))
    CG(A, B, X)
end

# NOTE: A has to support mul!(b, A, x)
# function CG(A::AbstractMatOrFac, b::AbstractVecOrMat, x::AbstractVecOrMat, M::Union{Nothing, AbstractMatOrFac} = nothing)
#     mul_A!(Ad, d) = mul!(Ad, A, d)
#     ConjugateGradient(mul_A!, b, x, M)
# end

# preconditioned version
# M is preconditioner
function update!(C::ConjugateGradient, x::AbstractVecOrMat, t::Int)
    if t > size(x, 1) # algorithm does not have guarantees after n iterations
        return x # restart?
    elseif t > 1 # IDEA: could save result of dot(C.r[1], C.z[1]) for α computation
        β = dot(C.r[1], C.z[1]) / dot(C.r[2], C.z[2]) # new over old residual norm
        @. C.d = β*C.d + C.z[1] # axpy!
    end
    mul!(C.Ad, C.A, C.d) # C.Ad = C.A * C.d or could be nonlinear

    # compute "stepsize" α, and return x if it is zero
    rz = dot(C.r[1], C.z[1])
    dAd = dot(C.d, C.Ad)
    if rz == 0 || dAd == 0
        return x
    end
    α = rz / dAd

    @. x += α * C.d # this directly updates x -> <:Update
    circshift!(C.r)
    @. C.r[1] = C.r[2] - α * C.Ad
    if !isnothing(C.M) # with no preconditioner, C.z = C.r, so the circshift has already happened in line 64
        circshift!(C.z)
        ldiv!(C.z[1], C.M, C.z[1]) # solve with pre-conditioner
    end
    for (i, ri) in enumerate(eachcol(C.r[1])) # IDEA: threads?
        C.r_norms[t, i] = norm(ri) # keep track of progression of residual norm
    end
    return x
end

function cg(A::AbstractMatOrFac, b::AbstractVector; max_iter::Int = size(A, 2), min_res::Real = 0)
    x = zeros(promote_type(eltype(A), eltype(b)), size(A, 2))
    cg!(x, A, b, max_iter = max_iter, min_res = min_res)
end

# TODO: enable matrix-matrix multiply in CG algorithm to take advantage of BLAS-3 level
function cg(A::AbstractMatOrFac, B::AbstractMatrix; max_iter::Int = size(A, 2), min_res::Real = 0)
    X = zeros(promote_type(eltype(A), eltype(B)), size(B))
    cg!(X, A, B, max_iter = max_iter, min_res = min_res)
end

function cg!(x::AbstractVector, A::AbstractMatOrFac, b::AbstractVector;
                                    max_iter::Int = size(A, 2), min_res::Real = 0)
    cg!(CG(A, b, x), x, max_iter = max_iter, min_res = min_res)
end

function cg!(X::AbstractMatrix, A::AbstractMatOrFac, B::AbstractMatrix;
                                    max_iter::Int = size(A, 2), min_res::Real = 0)
    cg!(CG(A, B, X), X, max_iter = max_iter, min_res = min_res)
end

function cg!(C::CG, x::AbstractVecOrMat; max_iter::Int = size(C.A, 2), min_res::Real = 0)
    for i in 1:max_iter
        update!(C, x, i)
        @views any(>(min_res), C.r_norms[i, :]) || break
    end
    return x
end

# pre-conditioned conjugate gradient method
# first converts to preconditioned system
# E⁻¹AE'⁻¹x̂ = E⁻¹b where x̂ = E⁻¹x and E*E' = M where M is the preconditioner
# then runs canonical cg on it
# afterwards, converts to solution of original system
function cg!(x::AbstractVecOrMat, A::AbstractMatOrFac, b::AbstractVecOrMat, M::AbstractMatOrFac;
                                    max_iter::Int = size(A, 2), min_res::Real = 0)
    C = CG(A, b, x, M)
    cg!(C, x, max_iter = max_iter, min_res = min_res) # run cg on pre-conditioned system
end

# E should be such that E*E' = M (e.g. the lower-triangular "L" component in a cholesky factorization)
# function preconditioned_system(x::AbstractVecOrMat, A::AbstractMatOrFac, b::AbstractVecOrMat, E::AbstractMatOrFac)
#     E⁻¹ = inverse(E)
#     EAE = LazyMatrixProduct(E⁻¹, A, E⁻¹') # depending on the approach could decide whether to be lazy here
#     E⁻¹b = E \ b
#     E⁻¹x = E \ x
#     return E⁻¹x, EAE, E⁻¹b
# end

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

Base.:*(A::CGMatrix, b::AbstractVecOrMat) = A.parent * b
# Base.:\(A::CGMatrix, b::AbstractVector) = cg(A.parent, b)
import LinearAlgebra: mul!, ldiv!
function mul!(y::AbstractVector, A::CGMatrix, x::AbstractVector, α::Real = 1, β::Real = 0)
    mul!(y, A.parent, x, α, β)
end
function ldiv!(y::AbstractVector, A::CGMatrix, x::AbstractVector)
    cg!(y, A.parent, x) # TODO: pre-allocate
end

function mul!(y::AbstractMatrix, A::CGMatrix, x::AbstractMatrix, α::Real = 1, β::Real = 0)
    mul!(y, A.parent, x, α, β)
end
function ldiv!(y::AbstractMatrix, A::CGMatrix, x::AbstractMatrix)
    cg!(y, A.parent, x) # TODO: pre-allocate
end

# OLD: parallizes over columns, new implementation takes advantage of BLAS-3
# function cg!(X::AbstractMatrix, A::AbstractMatOrFac, B::AbstractMatrix;
#              max_iter::Int = size(A, 2), min_res::Real = 0)
#     @sync for (i, b) in enumerate(eachcol(B))
#         @spawn begin
#             x = @view X[:, i]
#             cg!(x, A, b, max_iter = max_iter, min_res = min_res)
#         end
#     end
#     return X
# end
