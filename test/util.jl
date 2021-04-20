module TestUtil
using Test
using LazyLinearAlgebra: CircularBuffer

@testset "CircularBuffer" begin
    cb = CircularBuffer{Float64}(2)
    push!(cb, 1.)
    push!(cb, 2.)
    @test circshift!(cb) ≈ [2., 1.]
end

end # TestUtil
