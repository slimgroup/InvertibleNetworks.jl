# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CondCouplingLayerGlowAdditive


"""
    CL = CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerGlow(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=2) (2D)

    CL = CouplingLayerGlow(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=3) (3D)
    
    CL = CouplingLayerGlow3D(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false) (3D)

 Create a Real NVP-style invertible coupling layer based on 1x1 convolutions and a residual block.

 *Input*:

 - `C::Conv1x1`: 1x1 convolution layer

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

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
struct CondCouplingLayerGlowAdditive <: NeuralNetLayer
    RB::Union{ResidualBlock, FluxBlock}
    logdet::Bool
    activation::ActivationFunction
end

@Flux.functor CondCouplingLayerGlowAdditive


# Constructor from input dimensions
function CondCouplingLayerGlowAdditive(n_in::Int64, n_c::Int64, n_hidden::Int64; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    #RB = ResidualBlock(Int(n_in/2), n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)
    RB = ResidualBlock(Int(n_in/2)+n_c, Int(n_in/2), n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)
    #tensor cat the condition which has equal amount of channels

    return CondCouplingLayerGlowAdditive(RB, logdet, activation)
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, 4}, C::AbstractArray{T, 4}, L::CondCouplingLayerGlowAdditive) where T

    X1, X2 = tensor_split(X)

    Y2 = copy(X2)
    log_T =  L.RB.forward(tensor_cat(X2,C))
    Tm = log_T
    Y1 = X1 + Tm

    Y = tensor_cat(Y1, Y2)

    L.logdet == true ? (return Y, 0) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, 4}, C::AbstractArray{T, 4}, L::CondCouplingLayerGlowAdditive; save=false) where T

    # Get dimensions
    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = L.RB.forward(tensor_cat(X2,C))
    Tm = logS_T
    X1 = Y1 - Tm # add epsilon to avoid division by 0

    X = tensor_cat(X1, X2)
    
    save == true ? (return X, X1, X2) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, 4}, Y::AbstractArray{T, 4}, ΔC::AbstractArray{T, 4}, C::AbstractArray{T, 4}, L::CondCouplingLayerGlowAdditive; set_grad::Bool=true) where T

    # Recompute forward state
    k = size(ΔY)[3]
    X, X1, X2 = inverse(Y,C,L; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)

    rb = L.RB.backward(ΔT, tensor_cat(X2,C)) #have to grab the gradient only wrt x not c
    ΔX2 = rb[:,:,1:size(ΔY2)[3],:] + ΔY2
    ΔC  = rb[:,:,size(ΔY2)[3]+1:end,:] + ΔC
   
    ΔX = tensor_cat(ΔY1, ΔX2)
   
    return ΔX, X, ΔC
end





## Other utils

# Clear gradients
function clear_grad!(L::CondCouplingLayerGlowAdditive)

    clear_grad!(L.RB)
end

# Get parameters
function get_params(L::CondCouplingLayerGlowAdditive)
 
    p2 = get_params(L.RB)
    return p2
end

# Logdet (correct?)
glow_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
glow_logdet_backward(S) = 1f0./ S / size(S, 4)
