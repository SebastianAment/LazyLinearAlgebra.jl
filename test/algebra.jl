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
    X = randn(m, 3)
    @test ML*X ≈ L*X
    @test ML' ≈ Matrix(L')
    for _ in 1:16
        i = rand(1:n) # test some random indices
        j = rand(1:m)
        @test L[i, j] ≈ ML[i, j]
    end

    # testing solve
    A = randn(n, n) / sqrt(n) + 10I(n) # invertible MatrixProduct
    tol = 1e-6
    L = LazyMatrixProduct(A', A, tol = tol)
    x = randn(n)
    y = L*x
    z = L\y
    @test isapprox(L*z, y, atol = tol, rtol = tol)

    A, B, C = randn(n, n) / sqrt(n), I(n), randn(n, n) # invertible MatrixSum
    A = A'A; C = C'C; # make pos def
    L = LazyMatrixSum(A, B, C, tol = tol)
    y = L*x
    z = L\y
    @test isapprox(L*z, y, atol = tol, rtol = tol)
end

end
