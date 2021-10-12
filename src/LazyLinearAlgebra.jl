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

const default_tol = 1e-6 # default residual tolerance for cg

include("util.jl")
include("algebra.jl")
include("block.jl")
include("solve.jl") # conjugate gradient

# by default, linear solves via cg to take advantage of laziness
Base.:\(A::LazyFactorization, b::AbstractVector) = cg(A, b; min_res = A.tol)
Base.:\(A::LazyFactorization, B::AbstractMatrix) = cg(A, B; min_res = A.tol)
function Base.:\(A::LazyFactorization, b::Vector{Complex{T}}) where T <: Union{Float32, Float64}
    cg(A, b; min_res = A.tol)
end
function Base.:\(A::LazyFactorization, B::Matrix{Complex{T}}) where T <: Union{Float32, Float64}
    cg(A, B; min_res = A.tol)
end
function LinearAlgebra.ldiv!(y::AbstractVector, A::LazyFactorization, x::AbstractVector)
    cg!(y, A, x; min_res = A.tol) # IDEA: pre-allocate CG?
end
function LinearAlgebra.ldiv!(y::AbstractMatrix, A::LazyFactorization, x::AbstractMatrix)
    cg!(y, A, x; min_res = A.tol) # IDEA: pre-allocate CG?
end



end # module
