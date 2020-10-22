# Hyperbolic network layer from Lensink et al. (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export HyperbolicLayer

"""
    HyperbolicLayer(nx, ny, n_in, batchsize, kernel, stride, pad; action="same", α=1f0, hidden_factor=1)

or

    HyperbolicLayer(W, b, nx, ny, batchsize, stride, pad; action="same", α=1f0)

Create an invertible hyperbolic coupling layer.

*Input*:

 - `nx`, `ny`, `n_in`, `batchsize`: Dimensions of input tensor

 - `kernel`, `stride`, `pad`: Kernel size, stride and padding of the convolutional operator

 - `action`: String that defines whether layer keeps the number of channels fixed (`"same"`),
    increases it by a factor of 4 (`"up"`) or decreased it by a factor of 4 (`"down"`)

 - `W`, `b`: Convolutional weight and bias. `W` has dimensions of `(kernel, kernel, n_in, n_in)`.
   `b` has dimensions of `n_in`.

 - `α`: Step size for second time derivative. Default is 1.

 - `hidden_factor`: Increase the no. of channels by `hidden_factor` in the forward convolution.
    After applying the transpose convolution, the dimensions are back to the input dimensions.

*Output*:

 - `HL`: Invertible hyperbolic coupling layer

 *Usage:*

 - Forward mode: `X_curr, X_new = HL.forward(X_prev, X_curr)`

 - Inverse mode: `X_prev, X_curr = HL.inverse(X_curr, X_new)`

 - Backward mode: `ΔX_prev, ΔX_curr, X_prev, X_curr = HL.backward(ΔX_curr, ΔX_new, X_curr, X_new)`

 *Trainable parameters:*

 - `HL.W`: Convolutional kernel

 - `HL.b`: Bias

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct HyperbolicLayer <: NeuralNetLayer
    W::Parameter
    b::Parameter
    α::Float32
    cdims::DenseConvDims
    action::String
end

@Flux.functor HyperbolicLayer

# Constructor 2D
function HyperbolicLayer(nx::Int64, ny::Int64, n_in::Int64, batchsize::Int64, kernel::Int64,
    stride::Int64, pad::Int64; action="same", α=1f0, hidden_factor=1)

    # Set ouput/hidden dimensions
    if action == "same"
        n_out = n_in
    elseif action == "up"
        n_out = Int(n_in/4)
        nx = Int(nx*2)
        ny = Int(ny*2)
    elseif action == "down"
        n_out = Int(n_in*4)
        nx = Int(nx/2)
        ny = Int(ny/2)
    end
    n_hidden = n_out*hidden_factor

    W = Parameter(glorot_uniform(kernel, kernel, n_out, n_hidden))
    b = Parameter(zeros(Float32, n_hidden))

    cdims = DenseConvDims((nx, ny, n_out, batchsize), (kernel, kernel, n_out, n_hidden);
        stride=(stride, stride), padding=(pad, pad))

    return HyperbolicLayer(W, b, α, cdims, action)
end

# Constructor 3D
function HyperbolicLayer(nx::Int64, ny::Int64, nz::Int64, n_in::Int64, batchsize::Int64, kernel::Int64,
    stride::Int64, pad::Int64; action="same", α=1f0, hidden_factor=1)

    # Set ouput/hidden dimensions
    if action == "same"
        n_out = n_in
    elseif action == "up"
        n_out = Int(n_in/8)
        nx = Int(nx*2)
        ny = Int(ny*2)
        nz = Int(nz*2)
    elseif action == "down"
        n_out = Int(n_in*8)
        nx = Int(nx/2)
        ny = Int(ny/2)
        nz = Int(nz/2)
    end
    n_hidden = n_out*hidden_factor

    W = Parameter(glorot_uniform(kernel, kernel, kernel, n_out, n_hidden))
    b = Parameter(zeros(Float32, n_hidden))

    cdims = DenseConvDims((nx, ny, nz, n_out, batchsize), (kernel, kernel, kernel, n_out, n_hidden);
        stride=(stride, stride, stride), padding=(pad, pad, pad))

    return HyperbolicLayer(W, b, α, cdims, action)
end

# Constructor for given weights 2D
function HyperbolicLayer(W::AbstractArray{Float32, 4}, b::AbstractArray{Float32, 1}, nx::Int64, ny::Int64,
    batchsize::Int64, stride::Int64, pad::Int64; action="same", α=1f0)

    kernel, n_in, n_hidden = size(W)[2:4]

    # Set ouput/hidden dimensions
    if action == "same"
        n_out = n_in
    elseif action == "up"
        n_out = Int(n_in/4)
        nx = Int(nx*2)
        ny = Int(ny*2)
    elseif action == "down"
        n_out = Int(n_in*4)
        nx = Int(nx/2)
        ny = Int(ny/2)
    end

    W = Parameter(W)
    b = Parameter(b)

    cdims = DenseConvDims((nx, ny, n_out, batchsize), (kernel, kernel, n_out, n_hidden);
        stride=(stride, stride), padding=(pad, pad))

    return HyperbolicLayer(W, b, α, cdims, action)
end

# Constructor for given weights 3D
function HyperbolicLayer(W::AbstractArray{Float32, 5}, b::AbstractArray{Float32, 1}, nx::Int64, ny::Int64, nz::Int64,
    batchsize::Int64, stride::Int64, pad::Int64; action="same", α=1f0)

    kernel, n_in, n_hidden = size(W)[3:5]

    # Set ouput/hidden dimensions
    if action == "same"
        n_out = n_in
    elseif action == "up"
        n_out = Int(n_in/8)
        nx = Int(nx*2)
        ny = Int(ny*2)
        nz = Int(nz*2)
    elseif action == "down"
        n_out = Int(n_in*8)
        nx = Int(nx/2)
        ny = Int(ny/2)
        nz = Int(nz/2)
    end

    W = Parameter(W)
    b = Parameter(b)

    cdims = DenseConvDims((nx, ny, nz, n_out, batchsize), (kernel, kernel, kernel, n_out, n_hidden);
        stride=(stride, stride, stride), padding=(pad, pad, pad))

    return HyperbolicLayer(W, b, α, cdims, action)
end

# Forward pass
function forward(X_prev_in, X_curr_in, HL::HyperbolicLayer)

    # Change dimensions
    if HL.action == "same"
        X_prev = identity(X_prev_in)
        X_curr = identity(X_curr_in)
    elseif HL.action == "up"
        X_prev = wavelet_unsqueeze(X_prev_in)
        X_curr = wavelet_unsqueeze(X_curr_in)
    elseif HL.action == "down"
        X_prev = wavelet_squeeze(X_prev_in)
        X_curr = wavelet_squeeze(X_curr_in)
    else
        throw("Specified operation not defined.")
    end

    # Symmetric convolution w/ relu activation
    if length(size(X_curr)) == 4
        X_conv = conv(X_curr, HL.W.data, HL.cdims) .+ reshape(HL.b.data, 1, 1, :, 1)
    else
        X_conv = conv(X_curr, HL.W.data, HL.cdims) .+ reshape(HL.b.data, 1, 1, 1, :, 1)
    end
    X_relu = ReLU(X_conv)
    X_convT = -∇conv_data(X_relu, HL.W.data, HL.cdims)

    # Update
    X_new = 2f0*X_curr - X_prev + HL.α*X_convT

    return X_curr, X_new
end

# Inverse pass
function inverse(X_curr, X_new, HL::HyperbolicLayer; save=false)

    # Symmetric convolution w/ relu activation
    if length(size(X_curr)) == 4
        X_conv = conv(X_curr, HL.W.data, HL.cdims) .+ reshape(HL.b.data, 1, 1, :, 1)
    else
        X_conv = conv(X_curr, HL.W.data, HL.cdims) .+ reshape(HL.b.data, 1, 1, 1, :, 1)
    end
    X_relu = ReLU(X_conv)
    X_convT = -∇conv_data(X_relu, HL.W.data, HL.cdims)

    # Update
    X_prev = 2*X_curr - X_new + HL.α*X_convT

    # Change dimensions
    if HL.action == "same"
        X_prev_in = identity(X_prev)
        X_curr_in = identity(X_curr)
    elseif HL.action == "down"
        X_prev_in = wavelet_unsqueeze(X_prev)
        X_curr_in = wavelet_unsqueeze(X_curr)
    elseif HL.action == "up"
        X_prev_in = wavelet_squeeze(X_prev)
        X_curr_in = wavelet_squeeze(X_curr)
    else
        throw("Specified operation not defined.")
    end

    if save == false
        return X_prev_in, X_curr_in
    else
        return X_prev_in, X_curr_in, X_conv, X_relu
    end
end

# Backward pass
function backward(ΔX_curr, ΔX_new, X_curr, X_new, HL::HyperbolicLayer; set_grad::Bool=true)

    # Recompute forward states
    X_prev_in, X_curr_in, X_conv, X_relu = inverse(X_curr, X_new, HL; save=true)

    # Backpropagate data residual and compute gradients
    ΔX_convT = copy(ΔX_new)
    ΔX_relu = -HL.α*conv(ΔX_convT, HL.W.data, HL.cdims)
    ΔW = -HL.α*∇conv_filter(ΔX_convT, X_relu, HL.cdims)

    ΔX_conv = ReLUgrad(ΔX_relu, X_conv)
    ΔX_curr += ∇conv_data(ΔX_conv, HL.W.data, HL.cdims)
    ΔW += ∇conv_filter(X_curr, ΔX_conv, HL.cdims)
    if length(size(X_curr)) == 4
        Δb = sum(ΔX_conv; dims=[1,2,4])[1,1,:,1]
    else
        Δb = sum(ΔX_conv; dims=[1,2,3,5])[1,1,1,:,1]
    end
    ΔX_curr += 2f0*ΔX_new
    ΔX_prev = -ΔX_new

    # Set gradients
    if set_grad
        HL.W.grad = ΔW
        HL.b.grad = Δb
    else
        Δθ = [Parameter(ΔW), Parameter(Δb)]
    end

    # Change dimensions
    if HL.action == "same"
        ΔX_prev_in = identity(ΔX_prev)
        ΔX_curr_in = identity(ΔX_curr)
    elseif HL.action == "down"
        ΔX_prev_in = wavelet_unsqueeze(ΔX_prev)
        ΔX_curr_in = wavelet_unsqueeze(ΔX_curr)
    elseif HL.action == "up"
        ΔX_prev_in = wavelet_squeeze(ΔX_prev)
        ΔX_curr_in = wavelet_squeeze(ΔX_curr)
    else
        throw("Specified operation not defined.")
    end

    set_grad ? (return ΔX_prev_in, ΔX_curr_in, X_prev_in, X_curr_in) : (return ΔX_prev_in, ΔX_curr_in, Δθ, X_prev_in, X_curr_in)
end


## Jacobian utilities

# 2D
function jacobian(ΔX_prev_in, ΔX_curr_in, Δθ, X_prev_in, X_curr_in, HL::HyperbolicLayer)

    # Change dimensions
    if HL.action == "same"
        X_prev = identity(X_prev_in)
        X_curr = identity(X_curr_in)
        ΔX_prev = identity(ΔX_prev_in)
        ΔX_curr = identity(ΔX_curr_in)
    elseif HL.action == "up"
        X_prev = wavelet_unsqueeze(X_prev_in)
        X_curr = wavelet_unsqueeze(X_curr_in)
        ΔX_prev = wavelet_unsqueeze(ΔX_prev_in)
        ΔX_curr = wavelet_unsqueeze(ΔX_curr_in)
    elseif HL.action == "down"
        X_prev = wavelet_squeeze(X_prev_in)
        X_curr = wavelet_squeeze(X_curr_in)
        ΔX_prev = wavelet_squeeze(ΔX_prev_in)
        ΔX_curr = wavelet_squeeze(ΔX_curr_in)
    else
        throw("Specified operation not defined.")
    end

    # Symmetric convolution w/ relu activation
    if length(size(X_curr)) == 4
        X_conv = conv(X_curr, HL.W.data, HL.cdims) .+ reshape(HL.b.data, 1, 1, :, 1)
    else
        X_conv = conv(X_curr, HL.W.data, HL.cdims) .+ reshape(HL.b.data, 1, 1, 1, :, 1)
    end
    ΔX_conv = conv(ΔX_curr, HL.W.data, HL.cdims) .+ conv(X_curr, Δθ[1].data, HL.cdims) .+ reshape(Δθ[2].data, 1, 1, :, 1)
    X_relu = ReLU(X_conv)
    ΔX_relu = ReLUgrad(ΔX_conv, X_conv)
    X_convT = -∇conv_data(X_relu, HL.W.data, HL.cdims)
    ΔX_convT = -∇conv_data(ΔX_relu, HL.W.data, HL.cdims)-∇conv_data(X_relu, Δθ[1].data, HL.cdims)

    # Update
    X_new = 2f0*X_curr - X_prev + HL.α*X_convT
    ΔX_new = 2f0*ΔX_curr - ΔX_prev + HL.α*ΔX_convT

    return ΔX_curr, ΔX_new, X_curr, X_new
end

function adjointJacobian(ΔX_curr, ΔX_new, X_curr, X_new, HL::HyperbolicLayer)
    return backward(ΔX_curr, ΔX_new, X_curr, X_new, HL; set_grad=false)
end


## Other utils

# Clear gradients
function clear_grad!(HL::HyperbolicLayer)
    HL.W.grad = nothing
    HL.b.grad = nothing
end

# Get parameters
get_params(HL::HyperbolicLayer) = [HL.W, HL.b]
