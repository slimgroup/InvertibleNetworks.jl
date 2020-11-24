# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export LearnedCouplingLayerSLIM


"""
    CS = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize; 
        logdet::Bool=false, permute::Bool=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create an invertible SLIM coupling layer with a learned data-to-image-space map.

 *Input*: 

 - `nx1`, `nx2`, `nx_in`: spatial dimensions and no. of channels of input image
 
 - `ny1`, `ny2`, `ny_in`: spatial dimensions and no. of channels of input data

 - `n_hidden`: number of hidden units in conditional residual block

 - `loget`: bool to indicate whether to return the logdet (default is `false`)

 - `permute`: bool to indicate whether to apply a channel permutation (default is `false`)

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 *Output*:
 
 - `CS`: Invertible SLIM coupling layer with learned data-to-image map

 *Usage:*

 - Forward mode: `Y, logdet = CS.forward(X, D, A)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CS.inverse(Y, D, A)`

 - Backward mode: `ΔX, X = CS.backward(ΔY, Y, D, A)`

 - where `A` is a linear forward modeling operator and `D` is the observed data.

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct LearnedCouplingLayerSLIM <: NeuralNetLayer
    C::Union{Conv1x1, Nothing}
    RB::ConditionalResidualBlock
    logdet::Bool
end

@Flux.functor LearnedCouplingLayerSLIM

# Constructor from input dimensions
function LearnedCouplingLayerSLIM(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize; 
    k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet::Bool=false, permute::Bool=false)

    # 1x1 Convolution and residual block for invertible layer
    permute == true ? (C = Conv1x1(nx_in)) : (C = nothing)
    RB = ConditionalResidualBlock(nx1, nx2, Int(nx_in/2), ny1, ny2, ny_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)

    return LearnedCouplingLayerSLIM(C, RB, logdet)
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{Float32, N}, D, CS::LearnedCouplingLayerSLIM) where N
    # Permute and split
    isnothing(CS.C) ? (X_ = copy(X)) : (X_ = CS.C.forward(X))
    X1_, X2_ = tensor_split(X_)

    # Coupling layer
    Y2_ = X2_ + CS.RB.forward(X1_, D)[1]
    Y_ = tensor_cat(X1_, Y2_)

    isnothing(CS.C) ? (Y = copy(Y_)) : (Y = CS.C.inverse(Y_))
    CS.logdet == true ? (return Y, 0f0) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{Float32, N}, D, CS::LearnedCouplingLayerSLIM; save=false) where N
    # Permute and split
    isnothing(CS.C) ?  (Y_ = copy(Y)) : (Y_ = CS.C.forward(Y))
    Y1_, Y2_ = tensor_split(Y_)

    # Coupling layer
    X2_ = Y2_ - CS.RB.forward(Y1_, D)[1]
    X_ = tensor_cat(Y1_, X2_)

    isnothing(CS.C) ? (X = copy(X_)) : (X = CS.C.inverse(X_))
    save == true ? (return X, X_) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{Float32, N}, Y::AbstractArray{Float32, N}, D, CS::LearnedCouplingLayerSLIM; set_grad::Bool=true) where N
    # Recompute forward states
    X, X_ = inverse(Y, D, CS; save=true)

    # Backpropagation
    if isnothing(CS.C)
        ΔY_ = copy(ΔY)
    else
        set_grad ? (ΔY_ = CS.C.forward((ΔY, Y))[1]) : ((ΔY_, Δθ_C1) = CS.C.forward((ΔY, Y); set_grad=set_grad)[1:2])
    end
    ΔY1_, ΔY2_ = tensor_split(ΔY_)

    if set_grad
        ΔX1_, ΔD = CS.RB.backward(ΔY2_, D.*0f0, tensor_split(X_)[1], D)[1:2]
    else
        ΔX1_, ΔD, Δθ_RB = CS.RB.backward(ΔY2_, D.*0f0, tensor_split(X_)[1], D; set_grad=set_grad)[1:3]
    end
    ΔX1_ += ΔY1_
    
    ΔX_ = tensor_cat(ΔX1_, ΔY2_)
    if isnothing(CS.C)
        ΔX = copy(ΔX_)
    else
        set_grad ? (ΔX = CS.C.inverse((ΔX_, X_))[1]) : ((ΔX, Δθ_C2) = CS.C.inverse((ΔX_, X_); set_grad=set_grad)[1:2])
    end

    set_grad ? (return ΔX, ΔD, X) : (return ΔX, ΔD, cat(Δθ_RB, Δθ_C1+Δθ_C2; dims=1), X)
end


## Jacobian-related utils
function jacobian(ΔX::AbstractArray{Float32, N}, ΔD, Δθ::Array{Parameter, 1}, X::AbstractArray{Float32, N}, D, CS::LearnedCouplingLayerSLIM) where N
    # Permute and split
    if isnothing(CS.C)
        X_ = copy(X)
        ΔX_ = copy(ΔX)
    else
        ΔX_, X_ = CS.C.jacobian(ΔX, Δθ[8:end], X)
    end
    X1_, X2_ = tensor_split(X_)
    ΔX1_, ΔX2_ = tensor_split(ΔX_)

    # Coupling layer
    ΔX2__, _, X2__, _ = CS.RB.jacobian(ΔX1_, ΔD, Δθ[1:7], X1_, D)
    Y2_ = X2_ + X2__
    ΔY2_ = ΔX2_ + ΔX2__
    Y_ = tensor_cat(X1_, Y2_)
    ΔY_ = tensor_cat(ΔX1_, ΔY2_)

    if isnothing(CS.C)
        Y = copy(Y_)
        ΔY = copy(ΔY_)
    else
        ΔY, Y = CS.C.jacobianInverse(ΔY_, Δθ[8:end], Y_)
    end
    if CS.logdet
        return ΔY, Y, 0f0
    else
        return ΔY, Y
    end
end

adjointJacobian(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, D, CS::LearnedCouplingLayerSLIM) = backward(ΔY, Y, D, CS; set_grad=false)


## Other utils

# Clear gradients
function clear_grad!(CS::LearnedCouplingLayerSLIM)
    ~isnothing(CS.C) && clear_grad!(CS.C)
    clear_grad!(CS.RB)
end

# Get parameters
function get_params(CS::LearnedCouplingLayerSLIM)
    p = get_params(CS.RB)
    ~isnothing(CS.C) && (p = cat(p, get_params(CS.C); dims=1))
    return p
end
