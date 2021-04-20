module LazyLinearAlgebra

using LinearAlgebra
using Base.Threads
using Base.Threads: @spawn, @sync

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractVecOfVec{T} = AbstractVector{<:AbstractVector{T}}
const AbstractVecOfVecOrMat{T} = AbstractVector{<:AbstractVecOrMat{T}}

# TODO: collect my other Lazy LA packages here
export LazyMatrixProduct, LazyMatrixSum, BlockFactorization

abstract type LazyFactorization{T} <: Factorization{T} end

const default_tol = 1e-6 # default residual tolerance for cg

include("util.jl")
include("algebra.jl")
include("block.jl")
include("solve.jl") # conjugate gradient

# by default, linear solves via cg to take advantage of laziness
Base.:\(A::LazyFactorization, b::AbstractVecOrMat) = cg(A, b; min_res = A.tol)
function LinearAlgebra.ldiv!(y::AbstractVecOrMat, A::LazyFactorization, x::AbstractVecOrMat)
    cg!(y, A, x; min_res = A.tol) # IDEA: pre-allocate CG?
end

end # module
