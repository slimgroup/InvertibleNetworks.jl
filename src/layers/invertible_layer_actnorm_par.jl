export ActNormPar, reset!

mutable struct ActNormPar{T<:Real} <: NeuralNetLayer
    nc::Integer
    s::Parameter
    b::Parameter
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor ActNormPar


function ActNormPar(nc, s::Parameter, b::Parameter, logdet, is_reversed)
     return ActNormPar{Float32}(nc, s, b, logdet, is_reversed)
end
function ActNormPar(nc; logdet=false, T::DataType=Float32)
    s = Parameter(nothing)
    b = Parameter(nothing)
    return ActNormPar{T}(nc, s, b, logdet, false)
end

function forward(X::AbstractArray{T,N}, AN::ActNormPar{T}; logdet=nothing) where {N,T}

    isnothing(logdet) ? logdet = (AN.logdet && ~AN.is_reversed) : logdet = logdet
    inds = [i!=(N-1) ? 1 : (:) for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isnothing(AN.s.data) && !AN.is_reversed
        μ = mean(X; dims=dims)[inds...]
        σ_sqr = var(X; dims=dims)[inds...]
        AN.s.data = T(1)./sqrt.(σ_sqr)
        AN.b.data = -μ./sqrt.(σ_sqr)
    end
    Y = X .* reshape(AN.s.data, inds...) .+ reshape(AN.b.data, inds...)

    # If logdet true, return as second ouput argument
    logdet ? (return Y, logdet_forward(size(X)[1:N-2]..., AN.s)) : (return Y)
end

# 2-3D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T,N}, AN::ActNormPar{T}; logdet=nothing) where {N,T}
    isnothing(logdet) ? logdet = (AN.logdet && AN.is_reversed) : logdet = logdet
    inds = [i!=(N-1) ? 1 : (:) for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isnothing(AN.s.data) && AN.is_reversed
        μ = mean(Y; dims=dims)[inds...]
        σ_sqr = var(Y; dims=dims)[inds...]
        AN.s.data = sqrt.(σ_sqr)
        AN.b.data = μ
    end
    X = (Y .- reshape(AN.b.data, inds...)) ./ reshape(AN.s.data, inds...)

    # If logdet true, return as second ouput argument
    logdet ? (return X, -logdet_forward(size(Y)[1:N-2]..., AN.s)) : (return X)
end

# 2-3D Backward pass: Input (ΔY, Y), Output (ΔY, Y)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, AN::ActNormPar{T}; set_grad::Bool = true) where {N,T}
    inds = [i!=(N-1) ? 1 : (:) for i=1:N]
    dims = collect(1:N-1); dims[end] +=1
    nn = size(ΔY)[1:N-2]

    X = inverse(Y, AN; logdet=false)
    ΔX = ΔY .* reshape(AN.s.data, inds...)
    Δs = sum(ΔY .* X, dims=dims)[inds...]
    if AN.logdet
        set_grad ? (Δs -= logdet_backward(nn..., AN.s)) : (Δs_ = logdet_backward(nn..., AN.s))
    end
    Δb = sum(ΔY, dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    else
        Δθ = [Parameter(Δs), Parameter(Δb)]
    end
    if set_grad
        return ΔX, X
    else
        AN.logdet ? (return ΔX, Δθ, X, [Parameter(Δs_), Parameter(0f0*Δb)]) : (return ΔX, Δθ, X)
    end
end

## Reverse-layer functions
# 2-3D Backward pass (inverse): Input (ΔX, X), Output (ΔX, X)
function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, AN::ActNormPar{T}; set_grad::Bool = true) where {N,T}
    inds = [i!=(N-1) ? 1 : (:) for i=1:N]
    dims = collect(1:N-1); dims[end] +=1
    nn = size(ΔX)[1:N-2]

    Y = forward(X, AN; logdet=false)
    ΔY = ΔX ./ reshape(AN.s.data, inds...)
    Δs = -sum(ΔX .* X ./ reshape(AN.s.data, inds...), dims=dims)[inds...]
    if AN.logdet
        set_grad ? (Δs += logdet_backward(nn..., AN.s)) : (∇logdet = -logdet_backward(nn..., AN.s))
    end
    Δb = -sum(ΔX ./ reshape(AN.s.data, inds...), dims=dims)[inds...]
    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
    else
        Δθ = [Parameter(Δs), Parameter(Δb)]
    end
    if set_grad
        return ΔY, Y
    else
        AN.logdet ? (return ΔY, Δθ, Y, ∇logdet) : (return ΔY, Δθ, Y)
    end
end

## Logdet utils
# 2D Logdet
logdet_forward(nx, ny, s) = nx*ny*sum(log.(abs.(s.data)))
logdet_backward(nx, ny, s) = nx*ny ./ s.data
logdet_hessian(nx, ny, s) = -nx*ny ./ s.data.^2
# 3D Logdet
logdet_forward(nx, ny, nz, s) = nx*ny*nz*sum(log.(abs.(s.data)))
logdet_backward(nx, ny, nz, s) = nx*ny*nz ./ s.data
logdet_hessian(nx, ny, nz, s) = -nx*ny*nz ./ s.data.^2

## Other utilities
# Clear gradients
function clear_grad!(AN::ActNormPar)
    AN.s.grad = nothing
    AN.b.grad = nothing
end

# Reset ActNormPar layers
function reset!(AN::ActNormPar)
    AN.s.data = nothing
    AN.b.data = nothing
end

function reset!(AN::AbstractArray{ActNormPar, 1})
    for j=1:length(AN)
        AN[j].s.data = nothing
        AN[j].b.data = nothing
    end
end

# Get parameters
get_params(AN::ActNormPar) = [AN.s, AN.b]

gpu(AN::ActNormPar{T}) where T = ActNormPar{T}(AN.nc, gpu(AN.s), gpu(AN.b), AN.logdet, AN.is_reversed)
cpu(AN::ActNormPar{T}) where T = ActNormPar{T}(AN.nc, cpu(AN.s), cpu(AN.b), AN.logdet, AN.is_reversed)