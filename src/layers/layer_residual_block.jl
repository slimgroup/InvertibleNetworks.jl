# Residual block from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ResidualBlock

"""
    RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (2D)

    RB = ResidualBlock(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (3D)

or

    RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (2D)

    RB = ResidualBlock(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (3D)

 Create a (non-invertible) residual block, consisting of three convolutional layers and activation functions.
 The first convolution is a downsampling operation with a stride equal to the kernel dimension. The last
 convolution is the corresponding transpose operation and upsamples the data to either its original dimensions
 or to twice the number of input channels (for `fan=true`). The first and second layer contain a bias term.

 *Input*: 

 - `nx`, `ny`, `nz`: spatial dimensions of input
 
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

 - `nx`, `ny`: spatial dimensions of input image

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
    cdims1::DenseConvDims
    cdims2::DenseConvDims
    cdims3::DenseConvDims
    forward::Function
    backward::Function
end

@Flux.functor ResidualBlock

#######################################################################################################################
# 2D Constructors

# Constructor 2D
function ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false)

    # Initialize weights
    W1 = Parameter(glorot_uniform(k1, k1, n_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2, k2, n_hidden, n_hidden))
    W3 = Parameter(glorot_uniform(k1, k1, 2*n_in, n_hidden))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    # Dimensions for convolutions
    cdims1 = DenseConvDims((nx, ny, n_in, batchsize), (k1, k1, n_in, n_hidden); 
        stride=(s1, s1), padding=(p1, p1))
    cdims2 = DenseConvDims((Int(nx/s1), Int(ny/s1), n_hidden, batchsize), 
        (k2, k2, n_hidden, n_hidden); stride=(s2, s2), padding=(p2, p2))
    cdims3 = DenseConvDims((nx, ny, 2*n_in, batchsize), (k1, k1, 2*n_in, n_hidden); 
        stride=(s1, s1), padding=(p1 ,p1))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3,
                         X -> residual_forward(X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3),
                         (ΔY, X) -> residual_backward(ΔY, X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3)
                         )
end

# Constructor for given weights 2D
function ResidualBlock(W1, W2, W3, b1, b2, nx, ny, batchsize; p1=1, p2=1, s1=1, s2=1, fan=false)

    # Make weights parameters
    W1 = Parameter(W1)
    W2 = Parameter(W2)
    W3 = Parameter(W3)
    b1 = Parameter(b1)
    b2 = Parameter(b2)

    # Dimensions for convolutions
    k1, n_in, n_hidden = size(W1)[2:4]
    k2 = size(W2)[1]
    cdims1 = DenseConvDims((nx, ny, n_in, batchsize), (k1, k1, n_in, n_hidden); 
        stride=(s1, s1), padding=(p1, p1))
    cdims2 = DenseConvDims((Int(nx/s1), Int(ny/s1), n_hidden, batchsize), 
        (k2, k2, n_hidden, n_hidden); stride=(s2, s2), padding=(p2, p2))
    cdims3 = DenseConvDims((nx, ny, 2*n_in, batchsize), (k1, k1, 2*n_in, n_hidden); 
        stride=(s1, s1), padding=(p1, p1))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3,
                         X -> residual_forward(X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3),
                         (ΔY, X) -> residual_backward(ΔY, X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3)
                         )
end

#######################################################################################################################
# 3D Constructors

# Constructor 3D
function ResidualBlock(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false)

    # Initialize weights
    W1 = Parameter(glorot_uniform(k1, k1, k1, n_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2, k2, k2, n_hidden, n_hidden))
    W3 = Parameter(glorot_uniform(k1, k1, k1, 2*n_in, n_hidden))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    # Dimensions for convolutions
    cdims1 = DenseConvDims((nx, ny, nz, n_in, batchsize), (k1, k1, k1, n_in, n_hidden); 
        stride=(s1, s1, s1), padding=(p1, p1, p1))
    cdims2 = DenseConvDims((Int(nx/s1), Int(ny/s1), Int(nz/s1), n_hidden, batchsize), 
        (k2, k2, k2, n_hidden, n_hidden); stride=(s2, s2, s2), padding=(p2, p2, p2))
    cdims3 = DenseConvDims((nx, ny, nz, 2*n_in, batchsize), (k1, k1, k1, 2*n_in, n_hidden); 
        stride=(s1, s1, s1), padding=(p1, p1, p1))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3,
                         X -> residual_forward(X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3),
                         (ΔY, X) -> residual_backward(ΔY, X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3)
                         )
end

# Constructor for given weights 3D
function ResidualBlock(W1, W2, W3, b1, b2, nx, ny, nz, batchsize; p1=1, p2=1, s1=1, s2=1, fan=false)

    # Make weights parameters
    W1 = Parameter(W1)
    W2 = Parameter(W2)
    W3 = Parameter(W3)
    b1 = Parameter(b1)
    b2 = Parameter(b2)

    # Dimensions for convolutions
    k1, n_in, n_hidden = size(W1)[3:5]
    k2 = size(W2)[1]
    cdims1 = DenseConvDims((nx, ny, nz, n_in, batchsize), (k1, k1, n_in, n_hidden); 
        stride=(s1, s1, s1), padding=(p1, p1, p1))
    cdims2 = DenseConvDims((Int(nx/s1), Int(ny/s1), Int(nz/s1), n_hidden, batchsize), 
        (k2, k2, k2, n_hidden, n_hidden); stride=(s2, s2, s2), padding=(p2, p2, p2))
    cdims3 = DenseConvDims((nx, ny, nz, 2*n_in, batchsize), (k1, k1, k1, 2*n_in, n_hidden); 
        stride=(s1, s1, s1), padding=(p1, p1, p1))

    return ResidualBlock(W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3,
                         X -> residual_forward(X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3),
                         (ΔY, X) -> residual_backward(ΔY, X, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3)
                         )
end

#######################################################################################################################
# Functions

# Forward 2D
function residual_forward(X1::AbstractArray{Float32, 4}, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3; save=false)

    Y1 = conv(X1, W1.data, cdims1) .+ reshape(b1.data, 1, 1, :, 1)
    X2 = ReLU(Y1)

    Y2 = X2 + conv(X2, W2.data, cdims2) .+ reshape(b2.data, 1, 1, :, 1)
    X3 = ReLU(Y2)
    
    Y3 = ∇conv_data(X3, W3.data, cdims3)
    fan == true ? (X4 = ReLU(Y3)) : (X4 = GaLU(Y3))

    if save == false
        return X4
    else
        return Y1, Y2, Y3, X2, X3
    end
end

# Forward 3D
function residual_forward(X1::AbstractArray{Float32, 5}, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3; save=false)

    Y1 = conv(X1, W1.data, cdims1) .+ reshape(b1.data, 1, 1, 1, :, 1)
    X2 = ReLU(Y1)

    Y2 = X2 + conv(X2, W2.data, cdims2) .+ reshape(b2.data, 1, 1, 1, :, 1)
    X3 = ReLU(Y2)
    
    Y3 = ∇conv_data(X3, W3.data, cdims3)
    fan == true ? (X4 = ReLU(Y3)) : (X4 = GaLU(Y3))

    if save == false
        return X4
    else
        return Y1, Y2, Y3, X2, X3
    end
end

# Backward 2D
function residual_backward(ΔX4::AbstractArray{Float32, 4}, X1::AbstractArray{Float32, 4}, W1, W2, W3, b1, b2, 
    fan, cdims1, cdims2, cdims3)

    # Recompute forward states from input X
    Y1, Y2, Y3, X2, X3 = residual_forward(X1, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3; save=true)

    # Backpropagate residual ΔX4 and compute gradients
    fan == true ? (ΔY3 = ReLUgrad(ΔX4, Y3)) : (ΔY3 = GaLUgrad(ΔX4, Y3))
    ΔX3 = conv(ΔY3, W3.data, cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, cdims3)

    ΔY2 = ReLUgrad(ΔX3, Y2)
    ΔX2 = ∇conv_data(ΔY2, W2.data, cdims2) + ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, cdims2)
    Δb2 = sum(ΔY2, dims=(1,2,4))[1,1,:,1]

    ΔY1 = ReLUgrad(ΔX2, Y1)
    ΔX1 = ∇conv_data(ΔY1, W1.data, cdims1)
    ΔW1 = ∇conv_filter(X1, ΔY1, cdims1)
    Δb1 = sum(ΔY1, dims=(1,2,4))[1,1,:,1]

    # Set gradients
    W1.grad = ΔW1
    W2.grad = ΔW2
    W3.grad = ΔW3
    b1.grad = Δb1
    b2.grad = Δb2

    return ΔX1
end

# Backward 3D
function residual_backward(ΔX4::AbstractArray{Float32, 5}, X1::AbstractArray{Float32, 5}, W1, W2, W3, b1, b2, 
    fan, cdims1, cdims2, cdims3)

    # Recompute forward states from input X
    Y1, Y2, Y3, X2, X3 = residual_forward(X1, W1, W2, W3, b1, b2, fan, cdims1, cdims2, cdims3; save=true)

    # Backpropagate residual ΔX4 and compute gradients
    fan == true ? (ΔY3 = ReLUgrad(ΔX4, Y3)) : (ΔY3 = GaLUgrad(ΔX4, Y3))
    ΔX3 = conv(ΔY3, W3.data, cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, cdims3)

    ΔY2 = ReLUgrad(ΔX3, Y2)
    ΔX2 = ∇conv_data(ΔY2, W2.data, cdims2) + ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, cdims2)
    Δb2 = sum(ΔY2, dims=(1,2,3,5))[1,1,1,:,1]

    ΔY1 = ReLUgrad(ΔX2, Y1)
    ΔX1 = ∇conv_data(ΔY1, W1.data, cdims1)
    ΔW1 = ∇conv_filter(X1, ΔY1, cdims1)
    Δb1 = sum(ΔY1, dims=(1,2,3,5))[1,1,1,:,1]

    # Set gradients
    W1.grad = ΔW1
    W2.grad = ΔW2
    W3.grad = ΔW3
    b1.grad = Δb1
    b2.grad = Δb2

    return ΔX1
end

# Clear gradients
function clear_grad!(RB::ResidualBlock)
    RB.W1.grad = nothing
    RB.W2.grad = nothing
    RB.W3.grad = nothing
    RB.b1.grad = nothing
    RB.b2.grad = nothing
end

"""
    P = get_params(NL::NeuralNetLayer)

 Returns a cell array of all parameters in the network layer. Each cell
 entry contains a reference to the original parameter; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
get_params(RB::ResidualBlock) = [RB.W1, RB.W2, RB.W3, RB.b1, RB.b2]