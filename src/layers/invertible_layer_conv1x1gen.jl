export Conv1x1gen

struct Conv1x1gen{T<:Real} <: InvertibleNetwork
    nc::Int64
    P::AbstractMatrix{T}
    l::Parameter
    inds_l::AbstractVector
    u::Parameter
    inds_u::AbstractVector
    s::Parameter
    inds_s::AbstractVector
    orthogonal::Bool
    logdet::Bool
end

@Flux.functor Conv1x1gen

function Conv1x1gen(nc::Int64; logdet::Bool=true, orthogonal::Bool=false, init_id::Bool=false, T::DataType=Float32)

    inds_l = findall((1:nc).>(1:nc)')
    inds_u = findall((1:nc).<(1:nc)')
    inds_s = findall((1:nc).==(1:nc)')
    W = Array(qr(randn(T, nc, nc)).Q)
    F = lu(W) # LU decomposition PA=LU
    P = F.P
    L = F.L; l = Matrix2Array(L, inds_l)
    U = F.U; u = Matrix2Array(U, inds_u)
    s = abs.(diag(U)) # make sure W is SO(nc)
    init_id && (l .= T(0); u.= T(0); s .= T(1))

    return Conv1x1gen{T}(nc, P, Parameter(l), inds_l, Parameter(u), inds_u, Parameter(s), inds_s, orthogonal, logdet)

end

function forward(X::AbstractArray{T,4}, C::Conv1x1gen{T}) where T

    W = convweight(C)
    nx, ny, _, nb = size(X)
    Y = conv1x1(X, W)
    if C.orthogonal
        C.logdet ? (return Y, T(0)) : (return Y)
    else
        C.logdet ? (return Y, logdet(C, nx,ny,nb)) : (return Y)
    end

end

function inverse(Y::AbstractArray{T,4}, C::Conv1x1gen{T}; Winv::Union{Nothing,AbstractMatrix{T}}=nothing) where T

    W = convweight(C)
    Winv === nothing && (Winv = W\idmat(Y;n=C.nc))
    return conv1x1(Y, Winv)

end

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, C::Conv1x1gen{T}) where T

    # Backpropagating input
    W = convweight(C)
    ΔX = conv1x1(ΔY, toConcreteArray(W'))
    X = inverse(Y, C)

    # Backpropagating weights
    cdims = DenseConvDims(X, reshape(W, (1,1,size(W)...)); stride=(1,1), padding=(0,0))
    ΔW = reshape(∇conv_filter(X, ΔY, cdims), C.nc, C.nc)

    # Parameter gradient
    PΔW = C.P*ΔW
    LTPΔW = (Array2Matrix(C.l.data, C.nc, C.inds_l)+idmat(X;n=C.nc))'*PΔW
    C.l.grad = Matrix2Array(PΔW*(Array2Matrix(C.u.data, C.nc, C.inds_u)+Array2Matrix(C.s.data, C.nc, C.inds_s))', C.inds_l)
    C.u.grad = Matrix2Array(LTPΔW, C.inds_u)
    C.s.grad = Matrix2Array(LTPΔW, C.inds_s)
    ~C.orthogonal && C.logdet && (C.s.grad .-= dlogdet(C, size(X,1),size(X,2),size(X,4)))

    return ΔX, X

end

# Log-det utils

logdet(C::Conv1x1gen, nx::Int64, ny::Int64, nb::Int64) = nx*ny*sum(log.(abs.(C.s.data)))/nb
dlogdet(C::Conv1x1gen, nx::Int64, ny::Int64, nb::Int64) = nx*ny./(C.s.data*nb)

# Convolutional weight utils

convweight(C::Conv1x1gen) = C.P'*(Array2Matrix(C.l.data, C.nc, C.inds_l)+idmat(C.P;n=C.nc))*(Array2Matrix(C.u.data, C.nc, C.inds_u)+Array2Matrix(C.s.data, C.nc, C.inds_s))

conv1x1(X::AbstractArray{T,4}, W::AbstractMatrix{T}) where T = conv(X, reshape(W, (1,1,size(W)...)); stride=(1,1), pad=(0,0))

function idmat(X::Array{T}; n::Union{Nothing,Int64}=nothing) where T
    n === nothing && (n = size(X,3))
    return Matrix{T}(I,n,n)
end
function idmat(X::CuArray{T}; n::Union{Nothing,Int64}=nothing) where T
    n === nothing && (n = size(X,3))
    return CuMatrix{T}(I,n,n)
end

toConcreteArray(X::Adjoint{T,Array{T,N}}) where {T,N} = Array(X)
toConcreteArray(X::Adjoint{T,CuArray{T,N,O}}) where {T,N,O} = CuArray(X)

# LU utils

function Array2Matrix(a::Array{T,1}, n::Int64, inds) where T
    A = zeros(T, n, n)
    A[inds] .= a
    return A
end

function Array2Matrix(a::CuArray{T,1}, n::Int64, inds) where T
    A = CUDA.zeros(T, n, n)
    A[inds] .= a
    return A
end

Matrix2Array(A::AbstractArray{T,2}, inds) where T = A[inds]

# Other utils

function clear_grad!(C::Conv1x1gen)
    C.l.grad = nothing
    C.u.grad = nothing
    ~C.orthogonal && (C.s.grad = nothing)
end

get_params(C::Conv1x1gen) = C.orthogonal ? (return cat(C.l, C.u; dims=1)) : (return cat(C.l, C.u, C.s; dims=1))

gpu(C::Conv1x1gen{T}) where T = Conv1x1gen{T}(C.nc, gpu(C.P), gpu(C.l), C.inds_l, gpu(C.u), C.inds_u, gpu(C.s), C.inds_s, C.orthogonal, C.logdet)
cpu(C::Conv1x1gen{T}) where T = Conv1x1gen{T}(C.nc, cpu(C.P), cpu(C.l), C.inds_l, cpu(C.u), C.inds_u, cpu(C.s), C.inds_s, C.orthogonal, C.logdet)