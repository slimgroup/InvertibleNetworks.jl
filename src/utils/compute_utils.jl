using CUDA

export convert_cu, cuzeros, cuones, array_of_array, chain_lr

convert_cu(in_a, X) =  X isa CuArray ? cu(in_a) : in_a
cuzeros(::Array{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = zeros(T, a...)
cuzeros(::CuArray{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = CUDA.zeros(T, a...)
cuzeros(x, a::Tuple) = cuzeros(x, a...)
cuones(::Array{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = ones(T, a...)
cuones(::CuArray{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = CUDA.ones(T, a...)
cuones(x, a::Tuple) = cuones(x, a...)

array_of_array(::Array, args...) = Array{Array}(undef, args...)
array_of_array(::CuArray, args...) = Array{CuArray}(undef, args...)

# for 1x1 Conv
gemm_outer!(out::Matrix{T}, tmp::Vector{T}, v::Vector{T}) where T = LinearAlgebra.BLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)
gemm_outer!(out::CuMatrix{T}, tmp::CuVector{T}, v::CuVector{T}) where T = CUDA.CUBLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)

function chain_lr(x::AbstractMatrix{T}, vi::Vararg{AbstractVector{T}, N}) where {T, N}
    out = T(1) .* x
    tmp = cuzeros(vi[1], size(x, 1))
    for v=vi
        n = -2/norm(v)^2
        mul!(tmp, out, v)
        rmul!(tmp, n)
        gemm_outer!(out, tmp, v)
    end
    out
end
