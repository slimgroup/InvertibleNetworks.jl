# Conditional residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalResidualBlock

"""
    RB = ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create a (non-invertible) conditional residual block, consisting of one dense and three convolutional layers
 with ReLU activation functions. The dense operator maps the data to the image space and both tensors are
 concatenated and fed to the subsequent convolutional layers.

 *Input*:

 - `nx1`, `nx2`, `nx_in`: spatial dimensions and no. of channels of input image

 - `ny1`, `ny2`, `ny_in`: spatial dimensions and no. of channels of input data

 - `n_hidden`: number of hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: strides for the first and third convolution (`s1`) and the second convolution (`s2`)

or

 *Output*:

 - `RB`: conditional residual block layer

 *Usage:*

 - Forward mode: `Zx, Zy = RB.forward(X, Y)`

 - Backward mode: `ΔX, ΔY = RB.backward(ΔZx, ΔZy, X, Y)`

 *Trainable parameters:*

 - Convolutional kernel weights `RB.W0`, `RB.W1`, `RB.W2` and `RB.W3`

 - Bias terms `RB.b0`, `RB.b1` and `RB.b2`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalResidualBlock <: NeuralNetLayer
    W0::Parameter
    W1::Parameter
    W2::Parameter
    W3::Parameter
    b0::Parameter
    b1::Parameter
    b2::Parameter
    cdims1::DenseConvDims
    cdims2::DenseConvDims
    cdims3::DenseConvDims
end

@Flux.functor ConditionalResidualBlock

# Constructor
function ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    # Initialize weights
    W0 = Parameter(glorot_uniform(nx1*nx2*nx_in, ny1*ny2*ny_in))  # Dense layer for data D
    W1 = Parameter(glorot_uniform(k1, k1, 2*nx_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2, k2, n_hidden, n_hidden))
    W3 = Parameter(glorot_uniform(k1, k1, nx_in, n_hidden))
    b0 = Parameter(zeros(Float32, nx1*nx2*nx_in))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    # Dimensions for convolutions
    cdims1 = DenseConvDims((nx1, nx2, 2*nx_in, batchsize), (k1, k1, 2*nx_in, n_hidden);
        stride=(s1,s1), padding=(p1,p1))
    cdims2 = DenseConvDims((Int(nx1/s1), Int(nx2/s1), n_hidden, batchsize),
        (k2, k2, n_hidden, n_hidden); stride=(s2,s2), padding=(p2,p2))
    cdims3 = DenseConvDims((nx1, nx2, nx_in, batchsize), (k1, k1, nx_in, n_hidden);
        stride=(s1,s1), padding=(p1,p1))

    return ConditionalResidualBlock(W0, W1, W2, W3, b0, b1, b2, cdims1, cdims2, cdims3)
end

function forward(X0::AbstractArray{T, N}, D::AbstractArray{T, N}, RB::ConditionalResidualBlock; save=false) where {T, N}

    # Dimensions of input image X
    nx1, nx2, nx_in, batchsize = size(X0)

    Y0 = RB.W0.data*reshape(D, :, batchsize) .+ RB.b0.data
    X0_ = ReLU(reshape(Y0, nx1, nx2, nx_in, batchsize))
    X1 = tensor_cat(X0, X0_)

    Y1 = conv(X1, RB.W1.data, RB.cdims1) .+ reshape(RB.b1.data, 1, 1, :, 1)
    X2 = ReLU(Y1)

    Y2 = X2 + conv(X2, RB.W2.data, RB.cdims2) .+ reshape(RB.b2.data, 1, 1, :, 1)
    X3 = ReLU(Y2)

    Y3 = ∇conv_data(X3, RB.W3.data, RB.cdims3)
    X4 = ReLU(Y3)

    if save == false
        return X4, D
    else
        return Y0, Y1, Y2, Y3, X1, X2, X3
    end
end


function backward(ΔX4::AbstractArray{T, N}, ΔD::AbstractArray{T, N}, X0::AbstractArray{T, N}, D::AbstractArray{T, N}, RB::ConditionalResidualBlock; set_grad::Bool=true) where {T, N}

    # Recompute forward states from input X
    Y0, Y1, Y2, Y3, X1, X2, X3 = forward(X0, D, RB; save=true)
    nx1, nx2, nx_in, batchsize = size(X0)

    # Backpropagate residual ΔX4 and compute gradients
    ΔY3 = ReLUgrad(ΔX4, Y3)
    ΔX3 = conv(ΔY3, RB.W3.data, RB.cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, RB.cdims3)

    ΔY2 = ReLUgrad(ΔX3, Y2)
    ΔX2 = ∇conv_data(ΔY2, RB.W2.data, RB.cdims2) + ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, RB.cdims2)
    Δb2 = sum(ΔY2, dims=[1,2,4])[1,1,:,1]

    ΔY1 = ReLUgrad(ΔX2, Y1)
    ΔX1 = ∇conv_data(ΔY1, RB.W1.data, RB.cdims1)
    ΔW1 = ∇conv_filter(X1, ΔY1, RB.cdims1)
    Δb1 = sum(ΔY1, dims=[1,2,4])[1,1,:,1]

    ΔX0, ΔX0_ = tensor_split(ΔX1)
    ΔY0 = ReLUgrad(ΔX0_, reshape(Y0, nx1, nx2, nx_in, batchsize))
    ΔD += reshape(transpose(RB.W0.data)*reshape(ΔY0, :, batchsize), size(D))
    ΔW0 = reshape(ΔY0, :, batchsize)*transpose(reshape(D, :, batchsize))
    Δb0 = vec(sum(ΔY0, dims=4))

    # Set gradients
    if set_grad
        RB.W0.grad = ΔW0
        RB.W1.grad = ΔW1
        RB.W2.grad = ΔW2
        RB.W3.grad = ΔW3
        RB.b0.grad = Δb0
        RB.b1.grad = Δb1
        RB.b2.grad = Δb2
    else
        Δθ = [Parameter(ΔW0), Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3), Parameter(Δb0), Parameter(Δb1), Parameter(Δb2)]
    end

    set_grad ? (return ΔX0, ΔD) : (return ΔX0, ΔD, Δθ)
end


## Jacobian-related utils

function jacobian(ΔX0, ΔD, Δθ, X0, D, RB::ConditionalResidualBlock)

    # Dimensions of input image X
    nx1, nx2, nx_in, batchsize = size(X0)

    Y0 = RB.W0.data*reshape(D, :, batchsize) .+ RB.b0.data
    ΔY0 = Δθ[1].data*reshape(D, :, batchsize) + RB.W0.data*reshape(ΔD, :, batchsize) .+ Δθ[5].data
    X0_ = ReLU(reshape(Y0, nx1, nx2, nx_in, batchsize))
    ΔX0_ = ReLUgrad(reshape(ΔY0, nx1, nx2, nx_in, batchsize), reshape(Y0, nx1, nx2, nx_in, batchsize))
    X1 = tensor_cat(X0, X0_)
    ΔX1 = tensor_cat(ΔX0, ΔX0_)

    Y1 = conv(X1, RB.W1.data, RB.cdims1) .+ reshape(RB.b1.data, 1, 1, :, 1)
    ΔY1 = conv(X1, Δθ[2].data, RB.cdims1) + conv(ΔX1, RB.W1.data, RB.cdims1) .+ reshape(Δθ[6].data, 1, 1, :, 1)
    X2 = ReLU(Y1)
    ΔX2 = ReLUgrad(ΔY1, Y1)

    Y2 = X2 + conv(X2, RB.W2.data, RB.cdims2) .+ reshape(RB.b2.data, 1, 1, :, 1)
    ΔY2 = ΔX2 + conv(ΔX2, RB.W2.data, RB.cdims2) + conv(X2, Δθ[3].data, RB.cdims2) .+ reshape(Δθ[7].data, 1, 1, :, 1)
    X3 = ReLU(Y2)
    ΔX3 = ReLUgrad(ΔY2, Y2)

    Y3 = ∇conv_data(X3, RB.W3.data, RB.cdims3)
    ΔY3 = ∇conv_data(ΔX3, RB.W3.data, RB.cdims3) + ∇conv_data(X3, Δθ[4].data, RB.cdims3)
    X4 = ReLU(Y3)
    ΔX4 = ReLUgrad(ΔY3, Y3)

    return ΔX4, ΔD, X4, D
end

adjointJacobian(ΔY, ΔD, X0, D, RB::ConditionalResidualBlock) = backward(ΔY, ΔD, X0, D, RB; set_grad=false)
