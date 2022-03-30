export ConvLayer

struct ConvLayer{T<:Real} <: NeuralNetLayer
    W::Parameter
    b::Union{Parameter,Nothing}
    stride
    padding
    invertible::Bool
end

@Flux.functor ConvLayer

function ConvLayer(nc_in, nc_out; k=3, p=1, s=1, bias::Bool=true, weight_std::Real=0.05, T::DataType=Float32)
    invertible = false

    W = Parameter(T(weight_std)*randn(T, k, k, nc_in, nc_out))
    bias ? (b = Parameter(zeros(T, 1, 1, nc_out, 1))) : (b = nothing)

    return ConvLayer{T}(W, b, s, p, invertible)
end

function forward(X::AbstractArray{T,4}, CL::ConvLayer{T}) where T
    Y = conv(X, CL.W.data; stride=CL.stride, pad=CL.padding)
    CL.b !== nothing && (Y .+= CL.b.data)
    return Y
end

function backward(ΔY::AbstractArray{T,4}, X::AbstractArray{T,4}, CL::ConvLayer{T}) where T
    cdims = DenseConvDims(X, CL.W.data; stride=CL.stride, padding=CL.padding)
    ΔX = ∇conv_data(ΔY, CL.W.data, cdims)
    CL.W.grad = ∇conv_filter(X, ΔY, cdims)
    CL.b !== nothing && (CL.b.grad = sum(ΔY, dims=(1,2,4)))

    return ΔX
end

function clear_grad!(CL::ConvLayer)
    CL.W.grad = nothing
    CL.b !== nothing && (CL.b.grad = nothing)
end

function get_params(CL::ConvLayer)
    CL.b !== nothing ? (return [CL.W, CL.b]) : (return [CL.W])
end