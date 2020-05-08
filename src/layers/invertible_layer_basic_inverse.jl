# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
#
# Note: This is the inverse of CouplingLayerBasic. The inverse version is made necessary by the fact that the inverse of CouplingLayerBasic is not of the same type.

export CouplingLayerBasic_inverse


"""
    CL = CouplingLayerBasic_inverse(RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerBasic_inverse(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false)

 Create a Real NVP-style invertible coupling layer with a residual block (inverse version).

 *Input*:

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `nx, ny`: spatial dimensions of input

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s1`)

 *Output*:

 - `CL`: Invertible Real NVP coupling layer (inverse version).

 *Usage:*

 - Forward mode: `Y1, Y2, logdet = CL.forward(X1, X2)`    (if constructed with `logdet=true`)

 - Inverse mode: `X1, X2 = CL.inverse(Y1, Y2)`

 - Backward mode: `ΔX1, ΔX2, X1, X2 = CL.backward(ΔY1, ΔY2, Y1, Y2)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB`

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerBasic_inverse <: NeuralNetLayer
    RB::ResidualBlock
    logdet::Bool
    forward::Function
    inverse::Function
    backward::Function
end

# Constructor from 1x1 convolution and residual block
function CouplingLayerBasic_inverse(RB::ResidualBlock; logdet=false)
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerBasic_inverse(RB, logdet,
        (Y1, Y2) -> coupling_layer_inv_forward(Y1, Y2, RB, logdet),
        (X1, X2) -> coupling_layer_inv_inverse(X1, X2, RB),
        (ΔX1, ΔX2, X1, X2) -> coupling_layer_inv_backward(ΔX1, ΔX2, X1, X2, RB, logdet)
        )
end

# Constructor from input dimensions
function CouplingLayerBasic_inverse(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false)

    # 1x1 Convolution and residual block for invertible layer
    RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true)

    return CouplingLayerBasic_inverse(RB, logdet,
        (Y1, Y2) -> coupling_layer_inv_forward(Y1, Y2, RB, logdet),
        (X1, X2) -> coupling_layer_inv_inverse(X1, X2, RB),
        (ΔX1, ΔX2, X1, X2) -> coupling_layer_inv_backward(ΔX1, ΔX2, X1, X2, RB, logdet)
        )
end

# Forward pass: Input Y, Output X
function coupling_layer_inv_forward(Y1::Array{Float32, 4}, Y2::Array{Float32, 4}, RB, logdet)
    X1, X2, S = coupling_layer_inverse(Y1, Y2, RB; save=true)
    logdet == true ? (return X1, X2, -coupling_logdet_forward(S)) : (return X1, X2)
end

# Inverse pass: Input X, Output Y
function coupling_layer_inv_inverse(X1::Array{Float32, 4}, X2::Array{Float32, 4}, RB)
    return coupling_layer_forward(X1, X2, RB, false)
end

# Backward pass: Input (ΔX, X), Output (ΔY, Y)
function coupling_layer_inv_backward(ΔX1::Array{Float32, 4}, ΔX2::Array{Float32, 4},
    X1::Array{Float32, 4}, X2::Array{Float32, 4}, RB, logdet)

    # Recompute inverse state
    Y1, Y2, S = coupling_layer_forward(X1, X2, RB, false; save=true)

    # Backpropagate residual
    ΔT = -ΔX2 ./ S
    ΔS = X2 .* ΔT
    logdet == true && (ΔS += coupling_logdet_backward(S))
    ΔY1 = RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), Y1) + ΔX1
    ΔY2 = - ΔT

    return ΔY1, ΔY2, Y1, Y2
end

# Clear gradients
clear_grad!(L::CouplingLayerBasic_inverse) = clear_grad!(L.RB)

# Get parameters
get_params(L::CouplingLayerBasic_inverse) = get_params(L.RB)

# Inverse network
function inverse(L::CouplingLayerBasic_inverse; copy::Bool = true)
    copy ? (RB_ = deepcopy(L.RB)) : (RB_ = L.RB)
    return CouplingLayerBasic(RB_; logdet=L.logdet)
end
