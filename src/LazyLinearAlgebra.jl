module LazyLinearAlgebra

using LinearAlgebra
using Base.Threads

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
Base.:\(A::LazyFactorization, b::AbstractVector) = cg(A, b; min_res = A.tol)
function LinearAlgebra.ldiv!(y::AbstractVector, A::LazyFactorization, x::AbstractVector)
    cg!(A, x, y; min_res = A.tol) # IDEA: pre-allocate CG?
end

end # module
