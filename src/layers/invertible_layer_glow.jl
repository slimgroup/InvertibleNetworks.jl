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
    RB::Union{ResidualBlock, FluxBlock}
    logdet::Bool
end

@Flux.functor CouplingLayerGlow

# Constructor from 1x1 convolution and residual block
function CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerGlow(C, RB, logdet)
end

# Constructor from 1x1 convolution and residual Flux block
CouplingLayerGlow(C::Conv1x1, RB::FluxBlock; logdet=false) = CouplingLayerGlow(C, RB, logdet)

# Constructor from input dimensions
function CouplingLayerGlow(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in)
    RB = ResidualBlock(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true)

    return CouplingLayerGlow(C, RB, logdet)
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{Float32, 4}, L::CouplingLayerGlow)

    # Get dimensions
    k = Int(L.C.k/2)

    X_ = L.C.forward(X)
    X1, X2 = tensor_split(X_)

    Y2 = copy(X2)
    logS_T = L.RB.forward(X2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    Y1 = S.*X1 + T
    Y = tensor_cat(Y1, Y2)

    L.logdet == true ? (return Y, glow_logdet_forward(S)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{Float32, 4}, L::CouplingLayerGlow; save=false)

    # Get dimensions
    k = Int(L.C.k/2)
    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = L.RB.forward(X2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    X1 = (Y1 - T) ./ (S .+ eps(1f0)) # add epsilon to avoid division by 0
    X_ = tensor_cat(X1, X2)
    X = L.C.inverse(X_)

    save == true ? (return X, X1, X2, S) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, L::CouplingLayerGlow; set_grad::Bool=true)

    # Recompute forward state
    k = Int(L.C.k/2)
    X, X1, X2, S = inverse(Y, L; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1
    if L.logdet
        set_grad ? (ΔS -= glow_logdet_backward(S)) : (ΔS_ = glow_logdet_backward(S))
    end

    ΔX1 = ΔY1 .* S
    if set_grad
        ΔX2 = L.RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), X2) + ΔY2
    else
        ΔX2, Δθrb = L.RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), X2; set_grad=set_grad)
        _, ∇logdet = L.RB.backward(cat(SigmoidGrad(ΔS_, S), 0f0.*ΔT; dims=3), X2; set_grad=set_grad)
        ΔX2 += ΔY2
    end
    ΔX_ = tensor_cat(ΔX1, ΔX2)
    if set_grad
        ΔX = L.C.inverse((ΔX_, tensor_cat(X1, X2)))[1]
    else
        ΔX, Δθc = L.C.inverse((ΔX_, tensor_cat(X1, X2)); set_grad=set_grad)[1:2]
        Δθ = cat(Δθc, Δθrb; dims=1)
    end

    if set_grad
        return ΔX, X
    else
        L.logdet ? (return ΔX, Δθ, X, cat(0f0*Δθ[1:3], ∇logdet; dims=1)) : (return ΔX, Δθ, X)
    end
end


## Jacobian-related functions

function jacobian(ΔX::AbstractArray{Float32, 4}, Δθ::Array{Parameter, 1}, X, L::CouplingLayerGlow)

    # Get dimensions
    k = Int(L.C.k/2)

    ΔX_, X_ = L.C.jacobian(ΔX, Δθ[1:3], X)
    X1, X2 = tensor_split(X_)
    ΔX1, ΔX2 = tensor_split(ΔX_)

    Y2 = copy(X2)
    ΔY2 = copy(ΔX2)
    ΔlogS_T, logS_T = L.RB.jacobian(ΔX2, Δθ[4:end], X2)
    S = Sigmoid(logS_T[:,:,1:k,:])
    ΔS = SigmoidGrad(ΔlogS_T[:,:,1:k,:], nothing; x=logS_T[:,:,1:k,:])
    T = logS_T[:, :, k+1:end, :]
    ΔT = ΔlogS_T[:, :, k+1:end, :]
    Y1 = S.*X1 + T
    ΔY1 = ΔS.*X1 + S.*ΔX1 + ΔT
    Y = tensor_cat(Y1, Y2)
    ΔY = tensor_cat(ΔY1, ΔY2)

    # Gauss-Newton approximation of logdet terms
    JΔθ = L.RB.jacobian(zeros(Float32, size(ΔX2)), Δθ[4:end], X2)[1][:, :, 1:k, :]
    GNΔθ = cat(0f0*Δθ[1:3], -L.RB.adjointJacobian(tensor_cat(SigmoidGrad(JΔθ, S), zeros(Float32, size(S))), X2)[2]; dims=1)

    L.logdet ? (return ΔY, Y, glow_logdet_forward(S), GNΔθ) : (return ΔY, Y)
end

function adjointJacobian(ΔY, Y, L::CouplingLayerGlow)
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
