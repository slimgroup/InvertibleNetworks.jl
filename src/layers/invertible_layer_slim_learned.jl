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
function forward(X::AbstractArray{Float32, 4}, D, CS::LearnedCouplingLayerSLIM)

    # Get dimensions
    nx, ny, n_s, batchsize = size(X)

    # Permute and split
    isnothing(CS.C) ? (X_ = copy(X)) : (X_ = CS.C.forward(X))
    X1_, X2_ = tensor_split(X_)

    # Coupling layer
    Y1_ = copy(X1_)
    Y2_ = X2_ + CS.RB.forward(X1_, D)[1]
    Y_ = tensor_cat(Y1_, Y2_)

    isnothing(CS.C) ? (Y = copy(Y_)) : (Y = CS.C.inverse(Y_))
    CS.logdet == true ? (return Y, 0f0) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{Float32, 4}, D, CS::LearnedCouplingLayerSLIM; save=false)

    # Get dimensions
    nx, ny, n_s, batchsize = size(Y)

    # Permute and split
    isnothing(CS.C) ?  (Y_ = copy(Y)) : (Y_ = CS.C.forward(Y))
    Y1_, Y2_ = tensor_split(Y_)

    # Coupling layer
    X1_ = copy(Y1_)
    X2_ = Y2_ - CS.RB.forward(X1_, D)[1]
    X_ = tensor_cat(X1_, X2_)

    isnothing(CS.C) ? (X = copy(X_)) : (X = CS.C.inverse(X_))
    save == true ? (return X, X_) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, D, CS::LearnedCouplingLayerSLIM)

    # Recompute forward states
    X, X_ = inverse(Y, D, CS; save=true)
    nx1, nx2, nx_in, batchsize = size(X)

    # Backpropagation
    isnothing(CS.C) ? (ΔY_ = copy(ΔY)) : (ΔY_ = CS.C.forward((ΔY, Y))[1])
    ΔY1_, ΔY2_ = tensor_split(ΔY_)
    ΔX2_ = copy(ΔY2_)
    ΔX1_, ΔD = CS.RB.backward(ΔY2_, D.*0f0, tensor_split(X_)[1], D)[1:2]
    ΔX1_ += ΔY1_
    
    ΔX_ = tensor_cat(ΔX1_, ΔX2_)
    isnothing(CS.C) ? (ΔX = copy(ΔX_)) : (ΔX = CS.C.inverse((ΔX_, X_))[1])

    return ΔX, ΔD, X
end

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
