module TestLazyMatrices
using LinearAlgebra
using LazyLinearAlgebra
using Test

@testset "lazy factorizations" begin
    n, m, k, l = 128, 256, 64, 32
    A = randn(n, k)
    B = randn(k, l)
    C = randn(l, m)

    # testing product
    L = LazyMatrixProduct(A, B, C)
    @test size(L) == (n, m)

    x = randn(m)
    ML = Matrix(L)
    @test ML ≈ *(A, B, C)
    y = ML*x
    @test y ≈ L*x
    @test ML' ≈ Matrix(L')

    # testing sum
    A, B, C = (randn(n, m) for _ in 1:3)
    L = LazyMatrixSum(A, B, C)
    @test size(L) == (n, m)

    x = randn(m)
    ML = Matrix(L)
    @test ML ≈ +(A, B, C)
    y = ML*x
    @test ML*x ≈ L*x
    @test ML' ≈ Matrix(L')

    # testing solve
    A, B = randn(n, n), randn(n, n) # invertible MatrixProduct
    A = A'A; B = B'B
    _, A = eigen(A); _, B = eigen(B) # making sure matrix product is well conditioned
    A = A'A; B = B'B
    L = LazyMatrixProduct(A, B, tol = 1e-6)
    x = randn(n)
    y = L*x
    z = L\y
    @test isapprox(z, x, atol = 1e-6)

    A, B, C = randn(n, n), I(n), randn(n, n) # invertible MatrixSum
    A = A'A; C = C'C; # make pos def
    L = LazyMatrixSum(A, B, C, tol = 1e-6)
    y = L*x
    z = L\y
    w = Matrix(L)\y
    @test isapprox(z, x, atol = 1e-6)
end

end
