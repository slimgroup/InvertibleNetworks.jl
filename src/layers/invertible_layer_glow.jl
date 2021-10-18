# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerGlow, CouplingLayerGlow3D


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
struct CouplingLayerGlow <: NeuralNetLayer
    C::Union{Conv1x1, Conv1x1gen}
    RB::Union{ResidualBlock,ConvolutionalBlock, FluxBlock}
    logdet::Bool
    activation::ActivationFunction
end

@Flux.functor CouplingLayerGlow

# Constructor from 1x1 convolution and residual block
function CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false, activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerGlow(C, RB, logdet, activation)
end

# Constructor from 1x1 convolution and residual Flux block
CouplingLayerGlow(C::Conv1x1, RB::FluxBlock; logdet=false, activation::ActivationFunction=SigmoidLayer()) = CouplingLayerGlow(C, RB, logdet, activation)

# Constructor from input dimensions
function CouplingLayerGlow(n_in::Int64, n_hidden::Int64;gab_1x1=gab_1x1, gab_rb=gab_rb,init_id=true, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # 1x1 Convolution and residual block for invertible layer
    
    if gab_1x1
        C = Conv1x1gen(n_in; logdet=false, orthogonal=true, init_id=init_id, T=Float32)
    else
        C = Conv1x1(n_in)
    end

    if gab_rb
        RB = ConvolutionalBlock(Int(n_in/2), n_in, n_hidden;k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, T=Float32, init_zero=true)
    else
        RB = ResidualBlock(Int(n_in/2), n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)
    end
    return CouplingLayerGlow(C, RB, logdet, activation)
end

CouplingLayerGlow3D(args...;kw...) = CouplingLayerGlow(args...; kw..., ndims=3)

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, 4}, L::CouplingLayerGlow) where T

    # Get dimensions
    X_ = L.C.forward(X)
    #X1, X2 = tensor_split(X_)
    X2, X1 = tensor_split(X_)

    Y2 = copy(X2)
    logS_T = L.RB.forward(X2)
    logS, Tm = tensor_split(logS_T)

    Sm = L.activation.forward(logS)
    Y1 = Sm.*X1 + Tm

    #Y = tensor_cat(Y1, Y2)
    Y = tensor_cat(Y2, Y1)

    L.logdet == true ? (return Y, glow_logdet_forward(Sm)) : (return Y)
end

# function forward(X::AbstractArray{T,4}, L::CouplingLayerGlow) where T

#     X_ = L.C.forward(X)

#     X1, X2 = tensor_split(X_)

#     Y1 = X1
#     t = L.RB.forward(X1)
    
#     logs, t = tensor_split(t)
#     s = L.activation.forward(logs)
#     Y2 = X2.*s+t
#     L.logdet && (lgdt = logdet(L, s))
    
#     Y = tensor_cat(Y1, Y2)
#     L.logdet ? (return Y, lgdt) : (return Y)
# end

# function forward(X::AbstractArray{T,4}, L::CouplingLayerGlow) where T

#     X_ = L.C.forward(X)

#     X1, X2 = tensor_split(X_)

#     Y2 = X2
#     t = L.RB.forward(X2)
    
#     logs, t = tensor_split(t)
#     s = L.activation.forward(logs)
#     Y1 = X1.*s+t
#     L.logdet && (lgdt = logdet(L, s))
    
#     Y = tensor_cat(Y1, Y2)
#     L.logdet ? (return Y, lgdt) : (return Y)
# end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, 4}, L::CouplingLayerGlow; save=false) where T

    # Get dimensions
    #Y1, Y2 = tensor_split(Y)
    Y2, Y1 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = L.RB.forward(X2)
    logS, Tm = tensor_split(logS_T)
    Sm = L.activation.forward(logS)
    #X1 = (Y1 - Tm) ./ (Sm .+ eps(T)) # add epsilon to avoid division by 0
    X1 = (Y1 - Tm) ./ (Sm ) # add epsilon to avoid division by 0

    #X_ = tensor_cat(X1, X2)
    X_ = tensor_cat(X2, X1)
    X = L.C.inverse(X_)

    save == true ? (return X, X1, X2, Sm) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
# function backward(ΔY::AbstractArray{T, 4}, Y::AbstractArray{T, 4}, L::CouplingLayerGlow; set_grad::Bool=true) where T

#     # Recompute forward state
#     X, X1, X2, S = inverse(Y, L; save=true)

#     # Backpropagate residual
#     #ΔY1, ΔY2 = tensor_split(ΔY)
#     ΔY2, ΔY1 = tensor_split(ΔY)
#     ΔT = copy(ΔY1)
#     ΔS = ΔY1 .* X1
#     if L.logdet
#         set_grad ? (ΔS -= glow_logdet_backward(S)) : (ΔS_ = glow_logdet_backward(S))
#     end

#     ΔX1 = ΔY1 .* S
#     if set_grad
#         ΔX2 = L.RB.backward(cat(L.activation.backward(ΔS, S, nothing), ΔT; dims=3), X2) + ΔY2
#     else
#         ΔX2, Δθrb = L.RB.backward(cat(L.activation.backward(ΔS, S, nothing), ΔT; dims=3), X2; set_grad=set_grad)
#         _, ∇logdet = L.RB.backward(cat(L.activation.backward(ΔS_, S, nothing), 0f0.*ΔT; dims=3), X2; set_grad=set_grad)
#         ΔX2 += ΔY2
#     end
#     #ΔX_ = tensor_cat(ΔX1, ΔX2)
#     ΔX_ = tensor_cat(ΔX2, ΔX1)
#     if set_grad
#         #ΔX = L.C.backward(ΔX_, tensor_cat(X1, X2))[1]
#         ΔX = L.C.backward(ΔX_, tensor_cat(X2, X1))[1]
#     else
#         ΔX, Δθc = L.C.backward(ΔX_, tensor_cat(X1, X2); set_grad=set_grad)[1:2]
#         Δθ = cat(Δθc, Δθrb; dims=1)
#     end

#     if set_grad
#         return ΔX, X
#     else
#         L.logdet ? (return ΔX, Δθ, X, cat(0*Δθ[1:3], ∇logdet; dims=1)) : (return ΔX, Δθ, X)
#     end
# end
# function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, CL::CouplingLayerGlow) where T

#     #ΔY1, ΔY2 = tensor_split(ΔY); 
#     #Y1, Y2 = tensor_split(Y);
#     ΔY2, ΔY1 = tensor_split(ΔY); 
#     Y2, Y1 = tensor_split(Y);

#     ΔX1 = ΔY1; X1 = Y1
#     t = CL.RB.forward(X1)
#     Δt = ΔY2
    
#     logs, t = tensor_split(t)
#     s = CL.activation.forward(logs)
#     X2 = (Y2-t)./s
#     ΔX2 = ΔY2.*s
#     Δs = X2.*ΔY2
#     CL.logdet && (Δs .-= dlogdet(CL, s))
#     Δlogs = CL.activation.backward(Δs,  nothing, logs)
#     ΔX1 .+= CL.RB.backward(tensor_cat(Δlogs, Δt), X1)

#     #ΔX_ = tensor_cat(ΔX1, ΔX2)
#     #ΔX = CL.C.backward(ΔX_, tensor_cat(X1, X2))[1]
#     #X = CL.C.inverse(tensor_cat(X1, X2))
#     ΔX_ = tensor_cat(ΔX2, ΔX1)
#     ΔX = CL.C.backward(ΔX_, tensor_cat(X2, X1))[1]
#     X = CL.C.inverse(tensor_cat(X2, X1))

#     return ΔX, X

# end

function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, CL::CouplingLayerGlow) where T

#     #ΔY1, ΔY2 = tensor_split(ΔY); 
#     #Y1, Y2 = tensor_split(Y);
     
     ΔY2, ΔY1 = tensor_split(ΔY); 
     Y2, Y1 = tensor_split(Y);

    ΔX2 = ΔY2; 
    X2 = Y2
    t = CL.RB.forward(X2)
    Δt = ΔY1
    
    logs, t = tensor_split(t)
    s = CL.activation.forward(logs)
    X1 = (Y1-t)./s
    ΔX1 = ΔY1.*s
    Δs = X1.*ΔY1
    CL.logdet && (Δs .-= dlogdet(CL, s))
    Δlogs = CL.activation.backward(Δs,  nothing, logs)
    ΔX2 .+= CL.RB.backward(tensor_cat(Δlogs, Δt), X2)

#     #ΔX_ = tensor_cat(ΔX1, ΔX2)
#     #ΔX = CL.C.backward(ΔX_, tensor_cat(X1, X2))[1]
#     #X = CL.C.inverse(tensor_cat(X1, X2))
     ΔX_ = tensor_cat(ΔX2, ΔX1)
     ΔX = CL.C.backward(ΔX_, tensor_cat(X2, X1))[1]
     X = CL.C.inverse(tensor_cat(X2, X1))

    return ΔX, X

end

## Jacobian-related functions

function jacobian(ΔX::AbstractArray{T, 4}, Δθ::Array{Parameter, 1}, X, L::CouplingLayerGlow) where T

    # Get dimensions
    k = Int(L.C.k/2)

    ΔX_, X_ = L.C.jacobian(ΔX, Δθ[1:3], X)
    X1, X2 = tensor_split(X_)
    ΔX1, ΔX2 = tensor_split(ΔX_)

    Y2 = copy(X2)
    ΔY2 = copy(ΔX2)
    ΔlogS_T, logS_T = L.RB.jacobian(ΔX2, Δθ[4:end], X2)
    Sm = L.activation.forward(logS_T[:,:,1:k,:])
    ΔS = L.activation.backward(ΔlogS_T[:,:,1:k,:], nothing, logS_T[:,:,1:k,:])
    Tm = logS_T[:, :, k+1:end, :]
    ΔT = ΔlogS_T[:, :, k+1:end, :]
    Y1 = Sm.*X1 + Tm
    ΔY1 = ΔS.*X1 + Sm.*ΔX1 + ΔT
    Y = tensor_cat(Y1, Y2)
    ΔY = tensor_cat(ΔY1, ΔY2)

    # Gauss-Newton approximation of logdet terms
    JΔθ = L.RB.jacobian(cuzeros(ΔX2, size(ΔX2)), Δθ[4:end], X2)[1][:, :, 1:k, :]
    GNΔθ = cat(0f0*Δθ[1:3], -L.RB.adjointJacobian(tensor_cat(L.activation.backward(JΔθ, Sm, nothing), zeros(Float32, size(Sm))), X2)[2]; dims=1)

    L.logdet ? (return ΔY, Y, glow_logdet_forward(Sm), GNΔθ) : (return ΔY, Y)
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerGlow) where {T, N}
    return backward(ΔY, Y, L; set_grad=false)
end


## Other utils

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

 logdet(::CouplingLayerGlow, s::AbstractArray{T,4}) where T = sum(log.(abs.(s)))/size(s,4)
dlogdet(::CouplingLayerGlow, s::AbstractArray{T,4}) where T = T(1)./(s*size(s,4))