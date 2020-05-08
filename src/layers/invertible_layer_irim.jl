# Invertible network layer from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerIRIM

"""
    IL = CouplingLayerIRIM(C::Conv1x1, RB::ResidualBlock)

or

    IL = CouplingLayerIRIM(nx, ny, n_in, n_hidden, batchsize; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false) (2D)

    IL = CouplingLayerIRIM(nx, ny, nz, n_in, n_hidden, batchsize; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false) (3D)


 Create an i-RIM invertible coupling layer based on 1x1 convolutions and a residual block. 

 *Input*: 
 
 - `C::Conv1x1`: 1x1 convolution layer
 
 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 or

 - `nx`, `ny`, `nz`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 *Output*:
 
 - `IL`: Invertible i-RIM coupling layer.

 *Usage:*

 - Forward mode: `Y = IL.forward(X)`

 - Inverse mode: `X = IL.inverse(Y)`

 - Backward mode: `ΔX, X = IL.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `IL` itself

 - Trainable parameters in residual block `IL.RB` and 1x1 convolution layer `IL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerIRIM <: NeuralNetLayer
    C::Conv1x1
    RB::ResidualBlock
    forward::Function
    inverse::Function
    backward::Function
end

# Constructor from 1x1 convolution and residual block
function CouplingLayerIRIM(C::Conv1x1, RB::ResidualBlock)
    return CouplingLayerIRIM(C, RB, 
        X -> inv_layer_forward(X, C, RB),
        Y -> inv_layer_inverse(Y, C, RB),
        (ΔY, Y) -> inv_layer_backward(ΔY, Y, C, RB)
        )
end

# 2D Constructor from input dimensions
function CouplingLayerIRIM(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; 
    k1=4, k2=3, p1=0, p2=1, s1=4, s2=1)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in)
    RB = ResidualBlock(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)

    return CouplingLayerIRIM(C, RB, 
        X -> inv_layer_forward(X, C, RB),
        Y -> inv_layer_inverse(Y, C, RB),
        (ΔY, Y) -> inv_layer_backward(ΔY, Y, C, RB)
        )
end

# 3D Constructor from input dimensions
function CouplingLayerIRIM(nx::Int64, ny::Int64, nz::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; 
    k1=4, k2=3, p1=0, p2=1, s1=4, s2=1)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in)
    RB = ResidualBlock(nx, ny, nz, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)

    return CouplingLayerIRIM(C, RB, 
        X -> inv_layer_forward(X, C, RB),
        Y -> inv_layer_inverse(Y, C, RB),
        (ΔY, Y) -> inv_layer_backward(ΔY, Y, C, RB)
        )
end

# 2D Forward pass: Input X, Output Y
function inv_layer_forward(X::AbstractArray{Float32, 4}, C, RB)

    # Get dimensions
    k = Int(C.k/2)
    
    X_ = C.forward(X)
    X1_ = X_[:, :, 1:k, :]
    X2_ = X_[:, :, k+1:end, :]

    Y1_ = X1_
    Y2_ = X2_ + RB.forward(Y1_)
    
    Y_ = cat(Y1_, Y2_, dims=3)
    Y = C.inverse(Y_)
    
    return Y
end

# 3D Forward pass: Input X, Output Y
function inv_layer_forward(X::AbstractArray{Float32, 5}, C, RB)

    # Get dimensions
    k = Int(C.k/2)
    
    X_ = C.forward(X)
    X1_ = X_[:, :, :, 1:k, :]
    X2_ = X_[:, :, :, k+1:end, :]

    Y1_ = X1_
    Y2_ = X2_ + RB.forward(Y1_)
    
    Y_ = cat(Y1_, Y2_, dims=4)
    Y = C.inverse(Y_)
    
    return Y
end

# 2D Inverse pass: Input Y, Output X
function inv_layer_inverse(Y::AbstractArray{Float32, 4}, C, RB; save=false)

    # Get dimensions
    k = Int(C.k/2)

    Y_ = C.forward(Y)
    Y1_ = Y_[:, :, 1:k, :]
    Y2_ = Y_[:, :, k+1:end, :]
    
    X1_ = Y1_
    X2_ = Y2_ - RB.forward(Y1_)
    
    X_ = cat(X1_, X2_, dims=3)
    X = C.inverse(X_)
    
    if save == false
        return X
    else
        return X, X_, Y1_
    end
end

# 3D Inverse pass: Input Y, Output X
function inv_layer_inverse(Y::AbstractArray{Float32, 5}, C, RB; save=false)

    # Get dimensions
    k = Int(C.k/2)

    Y_ = C.forward(Y)
    Y1_ = Y_[:, :, :, 1:k, :]
    Y2_ = Y_[:, :, :, k+1:end, :]
    
    X1_ = Y1_
    X2_ = Y2_ - RB.forward(Y1_)
    
    X_ = cat(X1_, X2_, dims=4)
    X = C.inverse(X_)
    
    if save == false
        return X
    else
        return X, X_, Y1_
    end
end

# 2D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function inv_layer_backward(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, C, RB)

    # Recompute forward state
    k = Int(C.k/2)
    X, X_, Y1_ = inv_layer_inverse(Y, C, RB; save=true)

    # Backpropagate residual
    ΔY_ = C.forward((ΔY, Y))[1]
    ΔY2_ = ΔY_[:, :, k+1:end, :]
    ΔY1_ = RB.backward(ΔY2_, Y1_) + ΔY_[:, :, 1:k, :]
    
    ΔX_ = cat(ΔY1_, ΔY2_, dims=3)
    ΔX = C.inverse((ΔX_, X_))[1]
    
    return ΔX, X
end

# 3D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function inv_layer_backward(ΔY::AbstractArray{Float32, 5}, Y::AbstractArray{Float32, 5}, C, RB)

    # Recompute forward state
    k = Int(C.k/2)
    X, X_, Y1_ = inv_layer_inverse(Y, C, RB; save=true)

    # Backpropagate residual
    ΔY_ = C.forward((ΔY, Y))[1]
    ΔY2_ = ΔY_[:, :, :, k+1:end, :]
    ΔY1_ = RB.backward(ΔY2_, Y1_) + ΔY_[:, :, :, 1:k, :]
    
    ΔX_ = cat(ΔY1_, ΔY2_, dims=4)
    ΔX = C.inverse((ΔX_, X_))[1]
    
    return ΔX, X
end

# Clear gradients
function clear_grad!(L::CouplingLayerIRIM)
    clear_grad!(L.U)
    clear_grad!(L.RB)
end

# Get parameters
function get_params(L::CouplingLayerIRIM)
    p1 = get_params(L.C)
    p2 = get_params(L.RB)
    return cat(p1, p2; dims=1)
end