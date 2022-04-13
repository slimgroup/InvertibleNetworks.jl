# Invertible network layer from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerIRIM, CouplingLayerIRIM3D

"""
    IL = CouplingLayerIRIM(C::Conv1x1, RB::ResidualBlock)

or

    IL = CouplingLayerIRIM(n_in, n_hidden; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false, ndims=2) (2D)

    IL = CouplingLayerIRIM(n_in, n_hidden; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false, ndims=3) (3D)

    IL = CouplingLayerIRIM3D(n_in, n_hidden; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false) (3D)


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
    RB::Union{ResidualBlock, FluxBlock}
end

@Flux.functor CouplingLayerIRIM

# 2D Constructor from input dimensions
function CouplingLayerIRIM(n_in::Int64, n_hidden::Int64; 
                           k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in)
    RB = ResidualBlock(n_in÷2, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, ndims=ndims)

    return CouplingLayerIRIM(C, RB)
end

CouplingLayerIRIM3D(args...;kw...) = CouplingLayerIRIM(args...; kw..., ndims=3)

# 2D Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, L::CouplingLayerIRIM) where {T, N}
    X_ = L.C.forward(X)
    X1_, X2_ = tensor_split(X_)

    Y1_ = X1_
    Y2_ = X2_ + L.RB.forward(Y1_)

    Y_ = tensor_cat(Y1_, Y2_)
    Y = L.C.inverse(Y_)
    
    return Y
end

# 2D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, L::CouplingLayerIRIM; save=false) where {T, N}
    Y_ = L.C.forward(Y)
    Y1_, Y2_ = tensor_split(Y_)

    X1_ = Y1_
    X2_ = Y2_ - L.RB.forward(Y1_)

    X_ = tensor_cat(X1_, X2_)
    X = L.C.inverse(X_)

    if save == false
        return X
    else
        return X, X_, Y1_
    end
end

# 2D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerIRIM; set_grad::Bool=true) where {T, N}

    # Recompute forward state
    k = Int(L.C.k/2)
    X, X_, Y1_ = inverse(Y, L; save=true)

    # Backpropagate residual
    if set_grad
        ΔY_ = L.C.forward((ΔY, Y))[1]
    else
        ΔY_, Δθ_C1 = L.C.forward((ΔY, Y); set_grad=set_grad)[1:2]
    end
    ΔYl_, ΔYr_ = tensor_split(ΔY_)
    if set_grad
        ΔY1_ = L.RB.backward(ΔYr_, Y1_) + ΔYl_
    else
        ΔY1_, Δθ_RB = L.RB.backward(ΔYr_, Y1_; set_grad=set_grad)
        ΔY1_ = ΔY1_ + ΔYl_
    end
    
    ΔX_ = tensor_cat(ΔY1_, ΔYr_)
    if set_grad
        ΔX = L.C.inverse((ΔX_, X_))[1]
    else
        ΔX, Δθ_C2 = L.C.inverse((ΔX_, X_); set_grad=set_grad)[1:2]
    end
    
    set_grad ? (return ΔX, X) : (return ΔX, cat(Δθ_C1+Δθ_C2, Δθ_RB; dims=1), X)
end

## Jacobian utilities

# 2D
function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N}, L::CouplingLayerIRIM) where {T, N}

    # Get dimensions
    k = Int(L.C.k/2)
    
    ΔX_, X_ = L.C.jacobian(ΔX, Δθ[1:3], X)
    X1_, X2_ = tensor_split(X_)
    ΔX1_, ΔX2_ = tensor_split(ΔX_)

    ΔY1_, Y1__ = L.RB.jacobian(ΔX1_, Δθ[4:end], X1_)
    Y2_ = X2_ + Y1__
    ΔY2_ = ΔX2_ + ΔY1_
    
    Y_ = tensor_cat(X1_, Y2_)
    ΔY_ = tensor_cat(ΔX1_, ΔY2_)
    ΔY, Y = L.C.jacobianInverse(ΔY_, Δθ[1:3], Y_)
    
    return ΔY, Y

end

# 2D/3D
function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerIRIM) where {T, N}
    return backward(ΔY, Y, L; set_grad=false)
end
