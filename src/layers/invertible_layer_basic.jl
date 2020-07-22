# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerBasic


"""
    CL = CouplingLayerBasic(RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerBasic(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1,
            s1=1, s2=1, logdet=false) (2D)

    CL = CouplingLayerBasic(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1,
            s1=1, s2=1, logdet=false) (3D)

 Create a Real NVP-style invertible coupling layer with a residual block.

 *Input*:

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with
    ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `nx`, `ny`, `nz`: spatial dimensions of input

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the
    first and third operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second
    convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second
    convolution (`s1`)

 *Output*:

 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y1, Y2, logdet = CL.forward(X1, X2)`    (if constructed with 
    `logdet=true`)

 - Inverse mode: `X1, X2 = CL.inverse(Y1, Y2)`

 - Backward mode: `ΔX1, ΔX2, X1, X2 = CL.backward(ΔY1, ΔY2, Y1, Y2)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB`

 See also: [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct CouplingLayerBasic <: NeuralNetLayer
    RB::Union{ResidualBlock, FluxBlock}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor CouplingLayerBasic

# Constructor from 1x1 convolution and residual block
function CouplingLayerBasic(RB::ResidualBlock; logdet=false)
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerBasic(RB, logdet, false)
end

CouplingLayerBasic(RB::FluxBlock; logdet=false) = CouplingLayerBasic(RB, logdet, false)

# 2D Constructor from input dimensions
function CouplingLayerBasic(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64,
            batchsize::Int64; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false)

    # 1x1 Convolution and residual block for invertible layer
    RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2,
            s1=s1, s2=s2, fan=true)

    return CouplingLayerBasic(RB, logdet, false)
end

# 3D Constructor from input dimensions
function CouplingLayerBasic(nx::Int64, ny::Int64, nz::Int64, n_in::Int64, n_hidden::Int64,
            batchsize::Int64; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false)

    # 1x1 Convolution and residual block for invertible layer
    RB = ResidualBlock(nx, ny, nz, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2,
            s1=s1, s2=s2, fan=true)

    return CouplingLayerBasic(RB, logdet, false)
end

# 2D Forward pass: Input X, Output Y
function forward(X1::AbstractArray{Float32, 4}, X2::AbstractArray{Float32, 4},
            L::CouplingLayerBasic; save::Bool=false, logdet=nothing)
    isnothing(logdet) ? logdet = (L.logdet && ~L.is_reversed) : logdet = logdet

    # Coupling layer
    k = size(X1, 3)
    Y1 = copy(X1)
    logS_T = L.RB.forward(X1)
    S = Sigmoid(logS_T[:, :, 1:k, :])
    T = logS_T[:, :, k+1:end, :]
    Y2 = S.*X2 + T

    if logdet
        save ? (return Y1, Y2, coupling_logdet_forward(S), S) : (return Y1, Y2,
            coupling_logdet_forward(S))
    else
        save ? (return Y1, Y2, S) : (return Y1, Y2)
    end
end

# 3D Forward pass: Input X, Output Y
function forward(X1::AbstractArray{Float32, 5}, X2::AbstractArray{Float32, 5},
            L::CouplingLayerBasic; save::Bool=false, logdet=nothing)
    isnothing(logdet) ? logdet = (L.logdet && ~L.is_reversed) : logdet = logdet
    
    # Coupling layer
    k = size(X1, 4)
    Y1 = copy(X1)
    logS_T = L.RB.forward(X1)
    S = Sigmoid(logS_T[:, :, :, 1:k,: ])
    T = logS_T[:, :, :, k+1:end, :]
    Y2 = S.*X2 + T

    if logdet
        save ? (return Y1, Y2, coupling_logdet_forward(S), S) : (return Y1, Y2,
            coupling_logdet_forward(S))
    else
        save ? (return Y1, Y2, S) : (return Y1, Y2)
    end
end

# 2D Inverse pass: Input Y, Output X
function inverse(Y1::AbstractArray{Float32, 4}, Y2::AbstractArray{Float32, 4},
            L::CouplingLayerBasic; save::Bool=false, logdet=nothing)
    isnothing(logdet) ? logdet = (L.logdet && L.is_reversed) : logdet = logdet

    # Inverse layer
    k = size(Y1, 3)
    X1 = copy(Y1)
    logS_T = L.RB.forward(X1)
    S = Sigmoid(logS_T[:, :, 1:k, :])
    T = logS_T[:, :, k+1:end, :]
    X2 = (Y2 - T) ./ (S + randn(Float32, size(S))*eps(1f0)) # avoid division by 0

    if logdet
        save == true ? (return X1, X2, -coupling_logdet_forward(S), S) : (return X1, X2,
            -coupling_logdet_forward(S))
    else
        save == true ? (return X1, X2, S) : (return X1, X2)
    end
end

# 3D Inverse pass: Input Y, Output X
function inverse(Y1::AbstractArray{Float32, 5}, Y2::AbstractArray{Float32, 5},
            L::CouplingLayerBasic; save=false, logdet=true)
    isnothing(logdet) ? logdet = (L.logdet && L.is_reversed) : logdet = logdet

    # Inverse layer
    k = size(Y1, 4)
    X1 = copy(Y1)
    logS_T = L.RB.forward(X1)
    S = Sigmoid(logS_T[:, :, :, 1:k, :])
    T = logS_T[:, :, :, k+1:end, :]
    X2 = (Y2 - T) ./ (S + randn(Float32, size(S))*eps(1f0)) # avoid division by 0

    if logdet
        save == true ? (return X1, X2, -coupling_logdet_forward(S), S) : (return X1, X2,
            -coupling_logdet_forward(S))
    else
        save == true ? (return X1, X2, S) : (return X1, X2)
    end
end

# 2D/3D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY1, ΔY2, Y1, Y2, L::CouplingLayerBasic)

    # Recompute forward state
    X1, X2, S = inverse(Y1, Y2, L; save=true, logdet=false)

    # Backpropagate residual
    ΔT = copy(ΔY2)
    ΔS = ΔY2 .* X2
    L.logdet == true && (ΔS -= coupling_logdet_backward(S))
    ΔX2 = ΔY2 .* S
    ΔX1 = L.RB.backward(tensor_cat(SigmoidGrad(ΔS, S), ΔT), X1) + ΔY1

    return ΔX1, ΔX2, X1, X2
end

# 2D/3D Reverse backward pass: Input (ΔX, X), Output (ΔY, Y)
function backward_inv(ΔX1, ΔX2, X1, X2, L::CouplingLayerBasic)

    # Recompute inverse state
    Y1, Y2, S = forward(X1, X2, L; save=true, logdet=false)

    # Backpropagate residual
    ΔT = -ΔX2 ./ S
    ΔS = X2 .* ΔT
    L.logdet == true && (ΔS += coupling_logdet_backward(S))
    ΔY1 = L.RB.backward(tensor_cat(SigmoidGrad(ΔS, S), ΔT), Y1) + ΔX1
    ΔY2 = - ΔT

    return ΔY1, ΔY2, Y1, Y2
end

# Clear gradients
clear_grad!(L::CouplingLayerBasic) = clear_grad!(L.RB)

# Get parameters
get_params(L::CouplingLayerBasic) = get_params(L.RB)

# Set is_reversed flag
function tag_as_reversed!(L::CouplingLayerBasic, tag::Bool)
    L.is_reversed = tag
    return L
end

# Logdet (correct?)
coupling_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
coupling_logdet_backward(S) = 1f0./ S / size(S, 4)
