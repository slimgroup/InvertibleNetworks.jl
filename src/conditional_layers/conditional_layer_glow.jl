# Conditional coupling layer based on GLOW and cIIN
# Date: January 2022

export ConditionalLayerGlow, ConditionalLayerGlow3D


"""
    CL = ConditionalLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)

or

    CL = ConditionalLayerGlow(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=2) (2D)

    CL = ConditionalLayerGlow(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=3) (3D)
    
    CL = ConditionalLayerGlowGlow3D(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false) (3D)

 Create a Real NVP-style invertible conditional coupling layer based on 1x1 convolutions and a residual block.

 *Input*:

 - `C::Conv1x1`: 1x1 convolution layer

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `n_in`,`n_out`, `n_hidden`: number of channels for: passive input, conditioned input and hidden layer

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 *Output*:

 - `CL`: Invertible Real NVP conditional coupling layer.

 *Usage:*

 - Forward mode: `Y, logdet = CL.forward(X, C)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CL.inverse(Y, C)`

 - Backward mode: `ΔX, X = CL.backward(ΔY, Y, C)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalLayerGlow <: NeuralNetLayer
    C::Conv1x1
    RB::ResidualBlock
    logdet::Bool
    activation::ActivationFunction
end

@Flux.functor ConditionalLayerGlow

# Constructor from 1x1 convolution and residual block
function ConditionalLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false, activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return ConditionalLayerGlow(C, RB, logdet, activation)
end

# Constructor from input dimensions
function ConditionalLayerGlow(n_in::Int64, n_cond::Int64, n_hidden::Int64;freeze_conv=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), rb_activation::ActivationFunction=RELUlayer(), ndims=2)

    # 1x1 Convolution and residual block for invertible layers
    C  = Conv1x1(n_in; freeze=freeze_conv)
    RB = ResidualBlock(Int(n_in/2)+n_cond, n_hidden; n_out=n_in, activation=rb_activation, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)

    return ConditionalLayerGlow(C, RB, logdet, activation)
end

ConditionalLayerGlow3D(args...;kw...) = ConditionalLayerGlow(args...; kw..., ndims=3)

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerGlow) where {T,N}

    X_ = L.C.forward(X)
    X1, X2 = tensor_split(X_)

    Y2 = copy(X2)

    # Cat conditioning variable C into network input
    logS_T = L.RB.forward(tensor_cat(X2,C))
    logS, log_T = tensor_split(logS_T)

    Sm = L.activation.forward(logS)
    Tm = log_T
    Y1 = Sm.*X1 + Tm

    Y = tensor_cat(Y1, Y2)

    L.logdet == true ? (return Y, glow_logdet_forward(Sm)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerGlow; save=false) where {T,N}

    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = L.RB.forward(tensor_cat(X2,C))
    logS, log_T = tensor_split(logS_T)

    Sm = L.activation.forward(logS)
    Tm = log_T
    X1 = (Y1 - Tm) ./ (Sm .+ eps(T)) # add epsilon to avoid division by 0

    X_ = tensor_cat(X1, X2)
    X = L.C.inverse(X_)

    save == true ? (return X, X1, X2, Sm) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerGlow;) where {T,N}

    # Recompute forward state
    X, X1, X2, S = inverse(Y, C, L; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1
    ΔX1 = ΔY1 .* S

    if L.logdet
        ΔS -= glow_logdet_backward(S)
    end

    # Backpropagate RB
    ΔX2_ΔC = L.RB.backward(tensor_cat(L.activation.backward(ΔS, S), ΔT), (tensor_cat(X2, C)))
    ΔX2, ΔC = tensor_split(ΔX2_ΔC; split_index=Int(size(ΔY)[N-1]/2))
    ΔX2 += ΔY2

    # Backpropagate 1x1 conv
    ΔX = L.C.inverse((tensor_cat(ΔX1, ΔX2), tensor_cat(X1, X2)))[1]

    return ΔX, X, ΔC
end
