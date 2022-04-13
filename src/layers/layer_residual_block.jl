# Residual block from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ResidualBlock, ResidualBlock3D

"""
    RB = ResidualBlock(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false)
    RB = ResidualBlock3D(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false)

or

    RB = ResidualBlock(W1, W2, W3, b1, b2; p1=1, p2=1, s1=1, s2=1, fan=false)
    RB = ResidualBlock3D(W1, W2, W3, b1, b2; p1=1, p2=1, s1=1, s2=1, fan=false)

 Create a (non-invertible) residual block, consisting of three convolutional layers and activation functions.
 The first convolution is a downsampling operation with a stride equal to the kernel dimension. The last
 convolution is the corresponding transpose operation and upsamples the data to either its original dimensions
 or to twice the number of input channels (for `fan=true`). The first and second layer contain a bias term.

 *Input*:

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `fan`: bool to indicate whether the ouput has twice the number of input channels. For `fan=false`, the last
    activation function is a gated linear unit (thereby bringing the output back to the original dimensions).
    For `fan=true`, the last activation is a ReLU, in which case the output has twice the number of channels
    as the input.

or

 - `W1`, `W2`, `W3`: 4D tensors of convolutional weights

 - `b1`, `b2`: bias terms

 *Output*:

 - `RB`: residual block layer

 *Usage:*

 - Forward mode: `Y = RB.forward(X)`

 - Backward mode: `ΔX = RB.backward(ΔY, X)`

 *Trainable parameters:*

 - Convolutional kernel weights `RB.W1`, `RB.W2` and `RB.W3`

 - Bias terms `RB.b1` and `RB.b2`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ResidualBlock <: NeuralNetLayer
    W1::Parameter
    W2::Parameter
    W3::Parameter
    b1::Parameter
    b2::Parameter
    fan::Bool
    strides
    pad
end

@Flux.functor ResidualBlock

#######################################################################################################################
#  Constructors

# Constructor
function ResidualBlock(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false, ndims=2)

    k1 = Tuple(k1 for i=1:ndims)
    k2 = Tuple(k2 for i=1:ndims)
    # Initialize weights
    W1 = Parameter(glorot_uniform(k1..., n_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2..., n_hidden, n_hidden))
    W3 = Parameter(glorot_uniform(k1..., 2*n_in, n_hidden))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, (s1, s2), (p1, p2))
end

# Constructor for given weights
function ResidualBlock(W1, W2, W3, b1, b2; p1=1, p2=1, s1=1, s2=1, fan=false, ndims=2)

    # Make weights parameters
    W1 = Parameter(W1)
    W2 = Parameter(W2)
    W3 = Parameter(W3)
    b1 = Parameter(b1)
    b2 = Parameter(b2)

    return ResidualBlock(W1, W2, W3, b1, b2, fan, (s1, s2), (p1, p2))
end

ResidualBlock3D(args...; kw...) = ResidualBlock(args...; kw..., ndims=3)
#######################################################################################################################
# Functions

# Forward
function forward(X1::AbstractArray{T, N}, RB::ResidualBlock; save=false) where {T, N}
    inds =[i!=(N-1) ? 1 : Colon() for i=1:N]

    Y1 = conv(X1, RB.W1.data; stride=RB.strides[1], pad=RB.pad[1]) .+ reshape(RB.b1.data, inds...)
    X2 = ReLU(Y1)

    Y2 = X2 + conv(X2, RB.W2.data; stride=RB.strides[2], pad=RB.pad[2]) .+ reshape(RB.b2.data, inds...)
    X3 = ReLU(Y2)

    cdims3 = DCDims(X1, RB.W3.data; nc=2*size(X1, N-1), stride=RB.strides[1], padding=RB.pad[1])
    Y3 = ∇conv_data(X3, RB.W3.data, cdims3)
    RB.fan == true ? (X4 = ReLU(Y3)) : (X4 = GaLU(Y3))

    if save == false
        return X4
    else
        return Y1, Y2, Y3, X2, X3
    end
end

# Backward
function backward(ΔX4::AbstractArray{T, N}, X1::AbstractArray{T, N},
                  RB::ResidualBlock; set_grad::Bool=true) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Recompute forward states from input X
    Y1, Y2, Y3, X2, X3 = forward(X1, RB; save=true)

    # Cdims
    cdims2 = DenseConvDims(X2, RB.W2.data; stride=RB.strides[2], padding=RB.pad[2])
    cdims3 = DCDims(X1, RB.W3.data; nc=2*size(X1, N-1), stride=RB.strides[1], padding=RB.pad[1])

    # Backpropagate residual ΔX4 and compute gradients
    RB.fan == true ? (ΔY3 = ReLUgrad(ΔX4, Y3)) : (ΔY3 = GaLUgrad(ΔX4, Y3))
    ΔX3 = conv(ΔY3, RB.W3.data, cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, cdims3)

    ΔY2 = ReLUgrad(ΔX3, Y2)
    ΔX2 = ∇conv_data(ΔY2, RB.W2.data, cdims2) + ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, cdims2)
    Δb2 = sum(ΔY2, dims=dims)[inds...]

    cdims1 = DenseConvDims(X1, RB.W1.data; stride=RB.strides[1], padding=RB.pad[1])

    ΔY1 = ReLUgrad(ΔX2, Y1)
    ΔX1 = ∇conv_data(ΔY1, RB.W1.data, cdims1)
    ΔW1 = ∇conv_filter(X1, ΔY1, cdims1)
    Δb1 = sum(ΔY1, dims=dims)[inds...]

    # Set gradients
    if set_grad
        RB.W1.grad = ΔW1
        RB.W2.grad = ΔW2
        RB.W3.grad = ΔW3
        RB.b1.grad = Δb1
        RB.b2.grad = Δb2
    else
        Δθ = [Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3), Parameter(Δb1), Parameter(Δb2)]
    end

    set_grad ? (return ΔX1) : (return ΔX1, Δθ)
end

## Jacobian-related functions
function jacobian(ΔX1::AbstractArray{T, N}, Δθ::Array{Parameter, 1},
                  X1::AbstractArray{T, N}, RB::ResidualBlock) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    # Cdims
    cdims1 = DenseConvDims(X1, RB.W1.data; stride=RB.strides[1], padding=RB.pad[1])

    Y1 = conv(X1, RB.W1.data, cdims1) .+ reshape(RB.b1.data, inds...)
    ΔY1 = conv(ΔX1, RB.W1.data, cdims1) + conv(X1, Δθ[1].data, cdims1) .+ reshape(Δθ[4].data, inds...)
    X2 = ReLU(Y1)
    ΔX2 = ReLUgrad(ΔY1, Y1)

    cdims2 = DenseConvDims(X2, RB.W2.data; stride=RB.strides[2], padding=RB.pad[2])

    Y2 = X2 + conv(X2, RB.W2.data, cdims2) .+ reshape(RB.b2.data, inds...)
    ΔY2 = ΔX2 + conv(ΔX2, RB.W2.data, cdims2) + conv(X2, Δθ[2].data, cdims2) .+ reshape(Δθ[5].data, inds...)
    X3 = ReLU(Y2)
    ΔX3 = ReLUgrad(ΔY2, Y2)

    cdims3 = DCDims(X1, RB.W3.data; nc=2*size(X1, N-1), stride=RB.strides[1], padding=RB.pad[1])
    Y3 = ∇conv_data(X3, RB.W3.data, cdims3)
    ΔY3 = ∇conv_data(ΔX3, RB.W3.data, cdims3) + ∇conv_data(X3, Δθ[3].data, cdims3)
    if RB.fan == true
        X4 = ReLU(Y3)
        ΔX4 = ReLUgrad(ΔY3, Y3)
    else
        ΔX4, X4 = GaLUjacobian(ΔY3, Y3)
    end

    return ΔX4, X4

end
 
# 2D/3D
function adjointJacobian(ΔX4::AbstractArray{T, N}, X1::AbstractArray{T, N}, RB::ResidualBlock) where {T, N}
    return backward(ΔX4, X1, RB; set_grad=false)
end
