# Conditional coupling layer based on GLOW and cIIN
# Date: January 2022
#using UNet

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
    RB::Union{ResidualBlock, NetworkUNET, FluxBlock}
    logdet::Bool
    activation::ActivationFunction
    spade::Bool
end

@Flux.functor ConditionalLayerGlow

# Constructor from 1x1 convolution and residual block
function ConditionalLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false, activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return ConditionalLayerGlow(C, RB, logdet, activation)
end

# Constructor from input dimensions
function ConditionalLayerGlow(n_in::Int64, n_cond::Int64, n_hidden::Int64;rb_activation::ActivationFunction=RELUlayer(), rb="RB", L_i=1, spade=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)
    rb_in = Int(n_in/2)+n_cond
    spade && (rb_in = n_cond)
    rb_out = n_in
    # 1x1 Convolution and residual block for invertible layer
    C  = Conv1x1(n_in)

    #n_hiddens = [8,32,128,32,8]
    #ds = [1,2,4,2,1]
    #RB = NetworkUNET(n_in, n_hiddens, ds; n_grad = n_in, ndims=2);
    #RB = FluxBlock(Chain( UNet(n_in=rb_in, n_out=rb_out), MaxPool((2^L_i,2^L_i))))
    
    #if rb = "unet"
    #    RB = FluxBlock(Chain(Unet(in_channels = rb_in,
    #    out_channels = rb_out, num_fmaps = 16, downsample_factors = [(2,2),(2,2),(2,2)]), MaxPool((2^L_i,2^L_i)))) 
    #else
    RB = ResidualBlock(rb_in, n_hidden; activation=rb_activation, n_out=n_in, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)
    #end


    return ConditionalLayerGlow(C, RB, logdet, activation,spade)
end

ConditionalLayerGlow3D(args...;kw...) = ConditionalLayerGlow(args...; kw..., ndims=3)

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerGlow) where {T,N}

    X_ = L.C.forward(X)
    X1, X2 = tensor_split(X_)

    Y2 = copy(X2)

    # Cat conditioning variable C into network input
    rb_input = C
    (!L.spade) && (rb_input = tensor_cat(X2,rb_input))
 

    #println(size(rb_input))
    #println(size(X))
    logS_T = L.RB.forward(rb_input)
    logS, log_T = tensor_split(logS_T)

    Sm = L.activation.forward(logS)
    Tm = log_T
    
    
    #println(norm(Sm.*X1 + Tm)^2 / prod(size(X1)))
    Y1 = Sm.*X1 + Tm

    Y = tensor_cat(Y1, Y2)
    #println(norm(Tm)^2 / prod(size(Tm)))
    #println(norm(X)^2 / prod(size(X)))
    #println(norm(Y)^2 / prod(size(X)))
    #println((norm(Y)^2-norm(X)^2) / prod(size(X)))

    #return Y, glow_logdet_forward(Sm)
    L.logdet ? (return Y, glow_logdet_forward(Sm)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerGlow; save=false) where {T,N}

    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    # RB
    rb_input = C
    (!L.spade) && (rb_input = tensor_cat(X2,rb_input))
   

    logS_T = L.RB.forward(rb_input)
    logS, log_T = tensor_split(logS_T)

    Sm = L.activation.forward(logS)
    Tm = log_T
    X1 = (Y1 - Tm) ./ (Sm .+ eps(T)) # add epsilon to avoid division by 0

    X_ = tensor_cat(X1, X2)
    X = L.C.inverse(X_)

    save && (return X, X1, X2, Sm,Tm)
    L.logdet ? (return X, glow_logdet_forward(Sm)) : (return X) 
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerGlow;) where {T,N}

    # Recompute forward state
    X, X1, X2, S,Tm = inverse(Y, C, L; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1

    if L.logdet
        ΔS -= glow_logdet_backward(S)
    end

    ΔX1 = ΔY1 .* S

    # RB
    rb_input = C
    (!L.spade) && (rb_input = tensor_cat(X2,rb_input))


    ΔX2_ΔC = L.RB.backward(cat(L.activation.backward(ΔS, S), ΔT; dims=3), rb_input)
    #ΔX2_ΔC = L.RB.backward(cat(L.activation.backward(ΔS, S), ΔT; dims=3), cat(S, Tm; dims=3))[1]

    if !L.spade
        ΔX2, ΔC = tensor_split(ΔX2_ΔC; split_index=Int(size(ΔY)[N-1]/2))
        ΔX2 += ΔY2
    else
        ΔC = ΔX2_ΔC;
        ΔX2 = ΔY2
    end

  
    ΔX = L.C.inverse((tensor_cat(ΔX1, ΔX2), tensor_cat(X1, X2)))[1]

    return ΔX, X, ΔC
end

function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N},C::AbstractArray{T, N}, L::ConditionalLayerGlow; ) where {T, N}

    ΔX, X = L.C.forward((ΔX, X))
    X1, X2 = tensor_split(X)
    ΔX1, ΔX2 = tensor_split(ΔX)

    # Recompute forward state
    rb_input = tensor_cat(X2,C)
    logS_T = L.RB.forward(rb_input)
    logSm, Tm = tensor_split(logS_T)
    Sm = L.activation.forward(logSm)
    Y1 = Sm.*X1 + Tm

    # Backpropagate residual
    ΔT = -ΔX1 ./ Sm
    ΔS =  X1 .* ΔT 
    if L.logdet == true
        ΔS += coupling_logdet_backward(Sm)
    end

    ΔY2_ΔC = L.RB.backward(tensor_cat(L.activation.backward(ΔS, Sm), ΔT), rb_input) 
    ΔY2, ΔC = tensor_split(ΔY2_ΔC; split_index=Int(size(ΔX)[N-1]/2))
    ΔY2 += ΔX2

    ΔY1 = -ΔT

    ΔY = tensor_cat(ΔY1, ΔY2)
    Y  = tensor_cat(Y1, X2)

    return ΔY, Y, ΔC
end
