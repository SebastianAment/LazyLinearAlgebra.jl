# implements lazy matrix products and sums
# in contrast to LazyArrays, it implements a different type of laziness
# LazyArrays is fully lazy and avoids computing temporaries, but assumes the
# entries of the component arrays can be computed easily and efficiently
# The structures below rely on the mul! functions of the constituent
# matrices / factorizations to define a potentially more efficient multiplication\
# IDEA: introduce temporaries and optimal matrix multiplication order to minimize complexity
# TODO: enable rdiv! for LazyMatrixSum, Product
################################################################################
# in contrast to the AppliedMatrix in LazyArrays, this is not completely lazy
# in that it calculates intermediate results
# observation: this is remarkably similar to the KroneckerProducts implementation
struct LazyMatrixProduct{T, AT<:Tuple{Vararg{AbstractMatOrFac}}, F} <: LazyFactorization{T}
    args::AT
    tol::F
    # temporaries
    function LazyMatrixProduct(args::Tuple; tol::Real = default_tol)
        for i in 1:length(args)-1
            size(args[i], 2) == size(args[i+1], 1) || throw(DimensionMismatch("$i"))
        end
        T = promote_type(eltype.(args)...)
        new{T, typeof(args), typeof(tol)}(args, tol)
    end
end
LazyMatrixProduct(A::AbstractMatOrFac...; tol::Real = default_tol) = LazyMatrixProduct(A; tol = tol)

function Base.size(L::LazyMatrixProduct, i::Int)
    if i == 1
        size(L.args[1], 1)
    elseif i == 2
        size(L.args[end], 2)
    else
        1
    end
end
Base.size(L::LazyMatrixProduct) = size(L, 1), size(L, 2)
Base.adjoint(L::LazyMatrixProduct) = LazyMatrixProduct(reverse(adjoint.(L.args)))
issquare(A::AbstractMatOrFac) = size(A, 1) == size(A, 2)
# allsquare(L::LazyMatrixProduct) = all(issquare, L.args)
Base.Matrix(L::LazyMatrixProduct) = prod(Matrix, L.args)
Base.AbstractMatrix(L::LazyMatrixProduct) = prod(AbstractMatrix, L.args)

Base.:*(x::AbstractMatrix, L::LazyMatrixProduct) = (L'*x')'
Base.:*(L::LazyMatrixProduct, x::AbstractVector) = mul!(zeros(eltype(x), size(L, 1)), L, x)
Base.:*(L::LazyMatrixProduct, x::AbstractMatrix) = mul!(zeros(eltype(x), size(L, 1), size(x, 2)), L, x)
function LinearAlgebra.mul!(y::AbstractVecOrMat, L::LazyMatrixProduct, x::AbstractVecOrMat, α::Real = 1, β::Real = 0)
    z = deepcopy(x)
    for A in reverse(L.args)
        z = A*z
    end
    @. y = α*z + β*y
    return y
end

################################################################################
# in contrast to the AppliedMatrix in LazyArrays, this is not completely lazy
# in that it calculates intermediate results
struct LazyMatrixSum{T, AT<:Tuple{Vararg{AbstractMatOrFac}}, F} <: LazyFactorization{T}
    args::AT
    tol::F
    function LazyMatrixSum(args::Tuple; tol::Real = default_tol)
        all(==(size(args[1])), size.(args)) || throw(DimensionMismatch())
        T = promote_type(eltype.(args)...)
        new{T, typeof(args), typeof(tol)}(args, tol)
    end
end
LazyMatrixSum(A::AbstractMatOrFac...; tol::Real = default_tol) = LazyMatrixSum(A, tol = tol)

Base.size(L::LazyMatrixSum, i...) = size(L.args[1], i...)
Base.adjoint(L::LazyMatrixSum) = LazyMatrixSum(adjoint.(L.args))
Base.Matrix(L::LazyMatrixSum) = sum(Matrix, L.args)
Base.AbstractMatrix(L::LazyMatrixSum) = sum(AbstractMatrix, L.args)

Base.:*(x::AbstractMatrix, L::LazyMatrixSum) = (L'*x')'
Base.:*(L::LazyMatrixSum, x::AbstractVector) = mul!(zeros(eltype(x), size(L, 1)), L, x)
Base.:*(L::LazyMatrixSum) = mul!(zeros(eltype(x), size(L, 1), size(x, 2)), L, x)
function LinearAlgebra.mul!(y::AbstractVecOrMat, L::LazyMatrixSum, x::AbstractVecOrMat, α::Real = 1, β::Real = 0)
    @. y = β*y
    for A in L.args
        mul!(y, A, x, α, 1)
    end
    return y
end

# IDEA: more efficient temporary allocation
# function LinearAlgebra.mul!(y::AbstractVector, L::LazyMatrixProduct, x::AbstractVector, α::Real = 1, β::Real = 0)
#     if all(issquare, L.args)
#         z1, z2 = similar(x), similar(x)
#
#         # if odd and greater than 1, we need to start with z
#         iseven(length(L.args)) ? copyto!(y, x) : copyto!(z, x)
#         for A in L.args
#             mul!(z, A, y)
#             y, z = z, y
#         end
#
#     else
#         z = copy(x)
#         for A in L.args
#             z = A*z
#         end
#         copyto!(y, z)
#     end
#     return y
# end
