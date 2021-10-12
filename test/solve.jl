module TestLinearSolvers

using Test
using LinearAlgebra
using LazyLinearAlgebra: cg, CG, CGMatrix, cg! #, preconditioned_system

@testset "solve" begin
    @testset "conjugate gradient" begin
        n = 16
        m = n
        A = randn(m, n) / sqrt(n)
        A = A'A + I
        b = randn(n)

        # early stopping
        ε = 1e-3
        x = cg(A, b, min_res = ε)
        @test 1e-10 < norm(A*x-b) ≤ ε

        # high accuracy
        ε = 1e-10
        x = cg(A, b)
        @test norm(A*x-b) ≤ ε

        # matrix
        r = 3
        B = randn(n, r)
        X = cg(A, B)
        @test norm(A*X-B) ≤ r*ε

        # pre-conditioned conjugate gradient
        @. x = 0
        M = cholesky(A) # literally invert with preconditioner, should converge in one iteration
        cg!(x, A, b, M)
        @test norm(A*x-b) ≤ ε

        # E = M.L
        # E⁻¹x, EAE, E⁻¹b = preconditioned_system(x, A, b, E)
        # @test Matrix(EAE) ≈ I(n)

        @. x = 0
        C = CG(A, b, x, M)
        cg!(C, x)
        @test C.r_norms[1] < ε
        @test all(isnan, C.r_norms[2:end]) # means it returned after one iteration
    end

    @testset "CGMatrix" begin
        n = 16
        m = n
        A = randn(m, n) / sqrt(n)
        A = A'A + I
        b = randn(n)
        x = A\b

        CA = CGMatrix(A)
        @test CA isa CGMatrix
        @test CA\b ≈ x

        x = randn(n)
        @test CA*x ≈ A*x

        #################### mul!
        y = copy(b)
        @test mul!(y, CA, x, -1, 1) ≈ b - A*x

        y = copy(b)
        α, β = randn(2)
        @test mul!(y, CA, x, α, β) ≈ β*b + α*A*x

        #################### ldiv!
        b = A*x # with vector target
        @test CA \ b ≈ x
        @test ldiv!(y, CA, b) ≈ x

        # testing solve
        r = 32
        X = randn(n, r)
        B = A*X # with matrix target
        @test CA \ B ≈ X
        Y = zero(X)
        @test ldiv!(Y, CA, B) ≈ X

        # type-promotion in solve
        x = randn(ComplexF64, n)
        b = A*x # with vector target
        @test CA \ b ≈ x
        y = zero(x)
        @test ldiv!(y, CA, b) ≈ x

        X = randn(ComplexF64, n, r)
        B = A*X # with matrix target
        @test CA \ B ≈ X
        Y = zero(X)
        @test ldiv!(Y, CA, B) ≈ X

        # getindex
        @test CA[:, 1] ≈ A[:, 1]
        @test CA[3, 2] ≈ A[3, 2]

        # setindex!
        v = randn()
        CA[1] = v
        @test CA[1] ≈ v
        @test A[1] ≈ v # mutates original array
    end
end

end # TestLinearSolvers
