# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerBasic, CouplingLayerBasic3D


"""
    CL = CouplingLayerBasic(RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerBasic(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false, ndims=2) (2D)

    CL = CouplingLayerBasic(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false, ndims=3) (3D)

    CL = CouplingLayerBasic3D(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false) (3D)

 Create a Real NVP-style invertible coupling layer with a residual block.

 *Input*:

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s1`)

 - `ndims` : Number of dimensions

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
mutable struct CouplingLayerBasic <: NeuralNetLayer
    RB::Union{ResidualBlock, FluxBlock}
    logdet::Bool
    activation::ActivationFunction
    is_reversed::Bool
end

@Flux.functor CouplingLayerBasic

# Constructor from 1x1 convolution and residual block
function CouplingLayerBasic(RB::ResidualBlock; logdet=false, activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerBasic(RB, logdet, activation, false)
end

CouplingLayerBasic(RB::FluxBlock; logdet=false, activation::ActivationFunction=SigmoidLayer()) = CouplingLayerBasic(RB, logdet, activation, false)

# 2D Constructor from input dimensions
function CouplingLayerBasic(n_in::Int64, n_hidden::Int64; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # 1x1 Convolution and residual block for invertible layer
    RB = ResidualBlock(n_in, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)

    return CouplingLayerBasic(RB, logdet, activation, false)
end

CouplingLayerBasic3D(args...;kw...) = CouplingLayerBasic(args...; kw..., ndims=3)

# 2D/3D Forward pass: Input X, Output Y
function forward(X1::AbstractArray{T, N}, X2::AbstractArray{T, N}, L::CouplingLayerBasic; save::Bool=false, logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (L.logdet && ~L.is_reversed) : logdet = logdet

    # Coupling layer
    logS_T1, logS_T2 = tensor_split(L.RB.forward(X1))
    S = L.activation.forward(logS_T1)
    Y2 = S.*X2 + logS_T2

    if logdet
        save ? (return X1, Y2, coupling_logdet_forward(S), S) : (return X1, Y2, coupling_logdet_forward(S))
    else
        save ? (return X1, Y2, S) : (return X1, Y2)
    end
end

# 2D/3D Inverse pass: Input Y, Output X
function inverse(Y1::AbstractArray{T, N}, Y2::AbstractArray{T, N}, L::CouplingLayerBasic; save::Bool=false, logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (L.logdet && L.is_reversed) : logdet = logdet

    # Inverse layer
    logS_T1, logS_T2 = tensor_split(L.RB.forward(Y1))
    S = L.activation.forward(logS_T1)
    X2 = (Y2 - logS_T2) ./ (S .+ eps(T)) # add epsilon to avoid division by 0

    if logdet
        save == true ? (return Y1, X2, -coupling_logdet_forward(S), S) : (return Y1, X2, -coupling_logdet_forward(S))
    else
        save == true ? (return Y1, X2, S) : (return Y1, X2)
    end
end

# 2D/3D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY1::AbstractArray{T, N}, ΔY2::AbstractArray{T, N}, Y1::AbstractArray{T, N}, Y2::AbstractArray{T, N}, L::CouplingLayerBasic; set_grad::Bool=true) where {T, N}

    # Recompute forward state
    X1, X2, S = inverse(Y1, Y2, L; save=true, logdet=false)

    # Backpropagate residual
    ΔT = copy(ΔY2)
    ΔS = ΔY2 .* X2
    if L.logdet
        set_grad && (ΔS -= coupling_logdet_backward(S))
    end
    ΔX2 = ΔY2 .* S
    if set_grad
        ΔX1 = L.RB.backward(tensor_cat(L.activation.backward(ΔS, S), ΔT), X1) + ΔY1
    else
        ΔX1, Δθ = L.RB.backward(tensor_cat(L.activation.backward(ΔS, S), ΔT), X1; set_grad=set_grad)
        if L.logdet
            _, ∇logdet = L.RB.backward(tensor_cat(L.activation.backward(coupling_logdet_backward(S), S), 0 .*ΔT), X1; set_grad=set_grad)
        end
        ΔX1 += ΔY1
    end

    if set_grad
        return ΔX1, ΔX2, X1, X2
    else
        L.logdet ? (return ΔX1, ΔX2, Δθ, X1, X2, ∇logdet) : (return ΔX1, ΔX2, Δθ, X1, X2)
    end
end

# 2D/3D Reverse backward pass: Input (ΔX, X), Output (ΔY, Y)
function backward_inv(ΔX1::AbstractArray{T, N}, ΔX2::AbstractArray{T, N}, X1::AbstractArray{T, N}, X2::AbstractArray{T, N}, L::CouplingLayerBasic; set_grad::Bool=true) where {T, N}

    # Recompute inverse state
    Y1, Y2, S = forward(X1, X2, L; save=true, logdet=false)

    # Backpropagate residual
    ΔT = -ΔX2 ./ S
    ΔS = X2 .* ΔT
    if L.logdet == true
        set_grad ? (ΔS += coupling_logdet_backward(S)) : (∇logdet = -coupling_logdet_backward(S))
    end
    if set_grad
        ΔY1 = L.RB.backward(tensor_cat(L.activation.backward(ΔS, S), ΔT), Y1) + ΔX1
    else
        ΔY1, Δθ = L.RB.backward(tensor_cat(L.activation.backward(ΔS, S), ΔT), Y1; set_grad=set_grad)
        ΔY1 += ΔX1
    end
    ΔY2 = - ΔT

    if set_grad
        return ΔY1, ΔY2, Y1, Y2
    else
        L.logdet ? (return ΔY1, ΔY2, Δθ, Y1, Y2, ∇logdet) : (return ΔY1, ΔY2, Δθ, Y1, Y2)
    end
end


## Jacobian-related functions

# 2D
function jacobian(ΔX1::AbstractArray{T, N}, ΔX2::AbstractArray{T, N}, Δθ::AbstractArray{Parameter, 1},
                  X1::AbstractArray{T, N}, X2::AbstractArray{T, N}, L::CouplingLayerBasic;
                  save=false, logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (L.logdet && ~L.is_reversed) : logdet = logdet

    logS_T1, logS_T2 = tensor_split(L.RB.forward(X1))
    ΔlogS_T1, ΔlogS_T2 = tensor_split(jacobian(ΔX1, Δθ, X1, L.RB)[1])
    S = L.activation.forward(logS_T1)
    ΔS = L.activation.backward(ΔlogS_T1, S)
    Y2 = S.*X2 + logS_T2
    ΔY2 = ΔS.*X2 + S.*ΔX2 + ΔlogS_T2

    if logdet
        # Gauss-Newton approximation of logdet terms
        JΔθ = tensor_split(L.RB.jacobian(zeros(Float32, size(ΔX1)), Δθ, X1)[1])[1]
        GNΔθ = -L.RB.adjointJacobian(tensor_cat(L.activation.backward(JΔθ, S), zeros(Float32, size(S))), X1)[2]
        
        save ? (return ΔX1, ΔY2, X1, Y2, coupling_logdet_forward(S), GNΔθ, S) : (return ΔX1, ΔY2, X1, Y2, coupling_logdet_forward(S), GNΔθ)
    else
        save ? (return ΔX1, ΔY2, X1, Y2, S) : (return ΔX1, ΔY2, X1, Y2)
    end
end

# 2D/3D
function adjointJacobian(ΔY1::AbstractArray{T, N}, ΔY2::AbstractArray{T, N}, Y1::AbstractArray{T, N}, Y2::AbstractArray{T, N}, L::CouplingLayerBasic) where {T, N}
    return backward(ΔY1, ΔY2, Y1, Y2, L; set_grad=false)
end


## Logdet utils
coupling_logdet_forward(S) = sum(log.(abs.(S))) / size(S)[end]
coupling_logdet_backward(S) = 1f0./ S / size(S)[end]

# Set is_reversed flag
function tag_as_reversed!(L::CouplingLayerBasic, tag::Bool)
    L.is_reversed = tag
    return L
end