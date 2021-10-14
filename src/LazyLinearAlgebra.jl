module LazyLinearAlgebra

using LinearAlgebra
using Base.Threads
using Base.Threads: @spawn, @sync

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractMatOrFacOrUni{T} = Union{AbstractMatrix{T}, Factorization{T}, UniformScaling{T}}
const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const AbstractVecOfVecOrMat{T} = AbstractVector{<:AbstractVecOrMat{T}}

# TODO: collect my other Lazy LA packages here (incorporate LazyInverse into this package)
# using LazyInverse
export LazyMatrixProduct, LazyMatrixSum, BlockFactorization

abstract type LazyFactorization{T} <: Factorization{T} end

const default_tol = 1e-6
const default_atol = default_tol # default absolute residual tolerance for cg
const default_rtol = default_tol # default relative residual tolerance for cg

include("util.jl")
include("algebra.jl")
include("block.jl")
include("solve.jl") # conjugate gradient

# by default, linear solves via cg to take advantage of laziness
function Base.:\(A::LazyFactorization, b::AbstractVector) # = factorize(A) \ b
    cg(A, b; atol = A.tol, rtol = A.tol)
end
function Base.:\(A::LazyFactorization, B::AbstractMatrix) # = factorize(A) \ b
    cg(A, B; atol = A.tol, rtol = A.tol)
end
# should we default to factorizing first?
function Base.:\(A::LazyFactorization, b::Vector{Complex{T}}) where T <: Union{Float32, Float64}
    cg(A, b; atol = A.tol, rtol = A.tol)
end
function Base.:\(A::LazyFactorization, B::Matrix{Complex{T}}) where T <: Union{Float32, Float64}
    cg(A, B; atol = A.tol, rtol = A.tol)
end
function LinearAlgebra.ldiv!(y::AbstractVector, A::LazyFactorization, x::AbstractVector)
    cg!(y, A, x; atol = A.tol, rtol = A.tol) # IDEA: pre-allocate CG?
end
function LinearAlgebra.ldiv!(y::AbstractMatrix, A::LazyFactorization, x::AbstractMatrix)
    cg!(y, A, x; atol = A.tol, rtol = A.tol) # IDEA: pre-allocate CG?
end

# factorize first computes a preconditioner to make inversion with CG easier and faster
# NOTE: this precludes LazyLinearAlgebra from being compatible with non-pos-def systems, change in the future!
# for now, let's do this specifically in places where it is called for
# function LinearAlgebra.factorize(A::LazyFactorization; k::Int = 16, sigma::Real = 1e-2)
#     P = cholesky_preconditioner(A, k, sigma)
#     return CGFact(A, P, tol = A.tol)
# end

end # module
