# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerGlow


"""
    CL = CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerGlow(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false)

 Create a Real NVP-style invertible coupling layer based on 1x1 convolutions and a residual block. 

 *Input*: 
 
 - `C::Conv1x1`: 1x1 convolution layer
 
 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `nx, ny`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 *Output*:
 
 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y, logdet = CL.forward(X)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CL.inverse(Y)`

 - Backward mode: `ΔX, X = CL.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerGlow <: NeuralNetLayer
    C::Conv1x1
    RB::ResidualBlock
    logdet::Bool
    forward::Function
    inverse::Function
    backward::Function
end

@Flux.functor CouplingLayerGlow

# Constructor from 1x1 convolution and residual block
function CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerGlow(C, RB, logdet,
        X -> coupling_layer_forward(X, C, RB, logdet),
        Y -> coupling_layer_inverse(Y, C, RB),
        (ΔY, Y) -> coupling_layer_backward(ΔY, Y, C, RB, logdet)
        )
end

# Constructor from input dimensions
function CouplingLayerGlow(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in)
    RB = ResidualBlock(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true)

    return CouplingLayerGlow(C, RB, logdet,
        X -> coupling_layer_forward(X, C, RB, logdet),
        Y -> coupling_layer_inverse(Y, C, RB),
        (ΔY, Y) -> coupling_layer_backward(ΔY, Y, C, RB, logdet)
        )
end

# Forward pass: Input X, Output Y
function coupling_layer_forward(X::AbstractArray{Float32, 4}, C, RB, logdet)

    # Get dimensions
    k = Int(C.k/2)
    
    X_ = C.forward(X)
    X1, X2 = tensor_split(X_)

    Y2 = copy(X2)
    logS_T = RB.forward(X2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    Y1 = S.*X1 + T
    Y = tensor_cat(Y1, Y2)
    
    logdet == true ? (return Y, glow_logdet_forward(S)) : (return Y)
end

# Inverse pass: Input Y, Output X
function coupling_layer_inverse(Y::AbstractArray{Float32, 4}, C, RB; save=false)

    # Get dimensions
    k = Int(C.k/2)
    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = RB.forward(X2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    X1 = (Y1 - T) ./ (S + randn(Float32, size(S))*eps(1f0)) # add epsilon to avoid division by 0
    X_ = tensor_cat(X1, X2)
    X = C.inverse(X_)

    save == true ? (return X, X1, X2, S) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function coupling_layer_backward(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, C, RB, logdet)

    # Recompute forward state
    k = Int(C.k/2)
    X, X1, X2, S = coupling_layer_inverse(Y, C, RB; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1
    logdet == true && (ΔS -= glow_logdet_backward(S))
    ΔX1 = ΔY1 .* S
    ΔX2 = RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), X2) + ΔY2
    ΔX_ = tensor_cat(ΔX1, ΔX2)
    ΔX = C.inverse((ΔX_, tensor_cat(X1, X2)))[1]

    return ΔX, X
end

# Clear gradients
function clear_grad!(L::CouplingLayerGlow)
    clear_grad!(L.C)
    clear_grad!(L.RB)
end

# Get parameters
function get_params(L::CouplingLayerGlow)
    p1 = get_params(L.C)
    p2 = get_params(L.RB)
    return cat(p1, p2; dims=1)
end

# Logdet (correct?)
glow_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
glow_logdet_backward(S) = 1f0./ S / size(S, 4)
