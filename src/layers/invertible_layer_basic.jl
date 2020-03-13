# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerBasic


"""
    CL = CouplingLayerBasic(RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerBasic(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=3, p1=0, p2=1, logdet=false)

 Create a Real NVP-style invertible coupling layer with a residual block. 

 *Input*: 
  
 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `nx, ny`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 *Output*:
 
 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y1, Y2, logdet = CL.forward(X1, X2)`    (if constructed with `logdet=true`)

 - Inverse mode: `X1, X2 = CL.inverse(Y1, Y2)`

 - Backward mode: `ΔX1, ΔX2, X1, X2 = CL.backward(ΔY1, ΔY2, Y1, Y2)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB`

 See also: [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerBasic <: NeuralNetLayer
    RB::ResidualBlock
    logdet::Bool
    forward::Function
    inverse::Function
    backward::Function
end

# Constructor from 1x1 convolution and residual block
function CouplingLayerBasic(RB::ResidualBlock; logdet=false)
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerBasic(RB, logdet,
        (X1, X2) -> coupling_layer_forward(X1, X2, RB, logdet),
        (Y1, Y2) -> coupling_layer_inverse(Y1, Y2, RB),
        (ΔY1, ΔY2, Y1, Y2) -> coupling_layer_backward(ΔY1, ΔY2, Y1, Y2, RB, logdet)
        )
end

# Constructor from input dimensions
function CouplingLayerBasic(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=3, k2=1, p1=1, p2=0, logdet=false)

    # 1x1 Convolution and residual block for invertible layer
    RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, fan=true)

    return CouplingLayerBasic(RB, logdet,
        (X1, X2) -> coupling_layer_forward(X1, X2, RB, logdet),
        (Y1, Y2) -> coupling_layer_inverse(Y1, Y2, RB),
        (ΔY1, ΔY2, Y1, Y2) -> coupling_layer_backward(ΔY1, ΔY2, Y1, Y2, RB, logdet)
        )
end

# Forward pass: Input X, Output Y
function coupling_layer_forward(X1::Array{Float32, 4}, X2::Array{Float32, 4}, RB, logdet)

    # Coupling layer
    k = size(X1, 3)  
    Y2 = copy(X2)
    logS_T = RB.forward(Y2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    Y1 = S.*X1 + T
    
    logdet == true ? (return Y1, Y2, coupling_logdet_forward(S)) : (return Y1, Y2)
end

# Inverse pass: Input Y, Output X
function coupling_layer_inverse(Y1::Array{Float32, 4}, Y2::Array{Float32, 4}, RB; save=false)

    # Inverse layer  
    k = size(Y1, 3)  
    X2 = copy(Y2)
    logS_T = RB.forward(X2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    X1 = (Y1 - T) ./ (S + randn(Float32, size(S))*eps(1f0)) # add epsilon to avoid division by 0
 
    save == true ? (return X1, X2, S) : (return X1, X2)

end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function coupling_layer_backward(ΔY1::Array{Float32, 4}, ΔY2::Array{Float32, 4}, 
    Y1::Array{Float32, 4}, Y2::Array{Float32, 4}, RB, logdet)

    # Recompute forward state
    X1, X2, S = coupling_layer_inverse(Y1, Y2, RB; save=true)

    # Backpropagate residual
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1
    logdet == true && (ΔS -= coupling_logdet_backward(S))
    ΔX1 = ΔY1 .* S
    ΔX2 = RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), X2) + ΔY2

    return ΔX1, ΔX2, X1, X2
end

# Clear gradients
clear_grad!(L::CouplingLayerBasic) = clear_grad!(L.RB)

# Get parameters
get_params(L::CouplingLayerBasic) = get_params(L.RB)

# Logdet (correct?)
coupling_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
coupling_logdet_backward(S) = 1f0./ S / size(S, 4)
