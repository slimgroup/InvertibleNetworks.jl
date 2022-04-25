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
    C::AbstractArray{Conv1x1, 1}
    RB::AbstractArray{ResidualBlock, 1}
end

@Flux.functor CouplingLayerIRIM

# 2D Constructor from input dimensions
function CouplingLayerIRIM(n_in::Int64, n_hiddens::Array{Int64,1}, ds::Array{Int64,1}; 
                           k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2)
    if length(n_hiddens) != length(ds)
        throw("Number of downsampling factors in ds must be the same defined hidden channels in n_hidden")
    end 

    num_downsamp = length(n_hiddens)  
    C = Array{Conv1x1}(undef, num_downsamp)
    RB = Array{ResidualBlock}(undef, num_downsamp)
        
    for j=1:num_downsamp
        C[j]  = Conv1x1(n_in)
        RB[j] = ResidualBlock(n_in÷2, n_hiddens[j]; d=ds[j], k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=false, ndims=ndims)
    end

    return CouplingLayerIRIM(C, RB)
end

CouplingLayerIRIM3D(args...;kw...) = CouplingLayerIRIM(args...; kw..., ndims=3)

# 2D Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, L::CouplingLayerIRIM) where {T, N}

    # Init tensors to avoid reallocation
    Y_ = similar(X)

    num_downsamp = length(L.C)
    for j=1:num_downsamp
        X_ = L.C[j].forward(X)
        X1_, X2_ = tensor_split(X_)

        Y1_ = X1_
        Y2_ = X2_ + L.RB[j].forward(Y1_)

        tensor_cat!(Y_, Y1_, Y2_)
        X = L.C[j].inverse(Y_)
    end
    
    return X
end

# 2D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, L::CouplingLayerIRIM; save=false) where {T, N}

    # Init tensors to avoid reallocation
    X_ = similar(Y)

    num_downsamp = length(L.C)
    for j=num_downsamp:-1:1
        Y_ = L.C[j].forward(Y)
        Y1_, Y2_ = tensor_split(Y_)

        X1_ = Y1_
        X2_ = Y2_ - L.RB[j].forward(Y1_)

        tensor_cat!(X_, X1_, X2_)
        Y = L.C[j].inverse(X_)
    end

    return Y
end

# 2D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerIRIM; set_grad::Bool=true) where {T, N}

    # Initialize layer parameters
    !set_grad && (p1 = Array{Parameter, 1}(undef, 0))
    !set_grad && (p2 = Array{Parameter, 1}(undef, 0))

    # Init tensors to avoid reallocation
    ΔY_ = similar(ΔY)
    Y_  = similar(Y)

    num_downsamp = length(L.C)
    for j=num_downsamp:-1:1
        if set_grad
            ΔY_, Y_ = L.C[j].forward((ΔY, Y))
        else
            ΔY_, Δθ_C1, Y_  = L.C[j].forward((ΔY, Y); set_grad=set_grad)  
        end
  
        ΔYl_, ΔYr_ = tensor_split(ΔY_)
        Y1_,  Y2_  = tensor_split(Y_)
  
        if set_grad
            ΔYl_ .= L.RB[j].backward(ΔYr_, Y1_) + ΔYl_ 
        else
            ΔY_RB, Δθ_RB = L.RB[j].backward(ΔYr_, Y1_; set_grad=set_grad)
            ΔYl_ .= ΔY_RB + ΔYl_ 
        end

        Y2_ .-= L.RB[j].forward(Y1_)

        tensor_cat!(ΔY_, ΔYl_, ΔYr_)
        tensor_cat!(Y_, Y1_,  Y2_)

        if set_grad
            ΔY, Y = L.C[j].inverse((ΔY_, Y_))
        else
            ΔY, Δθ_C2, Y = L.C[j].inverse((ΔY_, Y_); set_grad=set_grad)  
            p1 = cat(p1, Δθ_C1+Δθ_C2; dims=1)
            p2 = cat(p2, Δθ_RB; dims=1)
        end
    end
    
    set_grad ? (return ΔY, Y) : (ΔY, cat(p1, p2; dims=1), Y)
end

## Jacobian utilities

# 2D
function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N}, L::CouplingLayerIRIM) where {T, N}
    num_downsamp = length(L.C)
    num_rb = 5
    num_1x1c = 3
    for j=1:num_downsamp
        idx_conv = (j-1)*num_1x1c+1:j*num_1x1c
        idx_rb = (j-1)*num_rb+1+num_downsamp*num_1x1c:(j)*num_rb+num_downsamp*num_1x1c

        ΔX_, X_ = L.C[j].jacobian(ΔX, Δθ[idx_conv], X)
        X1_, X2_ = tensor_split(X_)
        ΔX1_, ΔX2_ = tensor_split(ΔX_)

        ΔY1_, Y1__ = L.RB[j].jacobian(ΔX1_, Δθ[idx_rb], X1_)
        Y2_ = X2_ + Y1__
        ΔY2_ = ΔX2_ + ΔY1_
        
        Y_ = tensor_cat(X1_, Y2_)
        ΔY_ = tensor_cat(ΔX1_, ΔY2_)
        ΔX, X = L.C[j].jacobianInverse(ΔY_, Δθ[idx_conv], Y_)
    end
    
    return ΔX, X
end

# 2D/3D
function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerIRIM) where {T, N}
    return backward(ΔY, Y, L; set_grad=false)
end