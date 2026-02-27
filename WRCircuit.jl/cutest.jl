using CUDA
using LinearAlgebra
using SparseArrays

begin
    N = 100000
    p = 0.1
    x = @time CuArray(sprand(Float32, N, N, p));
    s = @time svd(x)
end
