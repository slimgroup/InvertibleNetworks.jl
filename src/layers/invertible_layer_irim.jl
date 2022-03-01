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
function CouplingLayerIRIM(n_in::Int64;n_hiddens=nothing, ds=nothing,
                           k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2)


    num_downsamp = length(ds)
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

    num_downsamp = length(L.C)
    for j=1:num_downsamp
        X_ = L.C[j].forward(X)
        X1_, X2_ = tensor_split(X_)

        Y1_ = X1_
        Y2_ = X2_ + L.RB[j].forward(Y1_)

        Y_ = tensor_cat(Y1_, Y2_)
        X = L.C[j].inverse(Y_)
    end
    
    return X
end

# 2D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, L::CouplingLayerIRIM; save=false) where {T, N}
   
    num_downsamp = length(L.C)
    for j=num_downsamp:-1:1
        Y_ = L.C[j].forward(Y)
        Y1_, Y2_ = tensor_split(Y_)

        X1_ = Y1_
        X2_ = Y2_ - L.RB[j].forward(Y1_)

        X_ = tensor_cat(X1_, X2_)
        X = L.C[j].inverse(X_)
    end

    if save == false
        return X
    else
        return X, X_, Y1_
    end
end

# 2D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerIRIM; set_grad::Bool=true) where {T, N}

    # Recompute forward state
    #X, X_, Y1_ = inverse(Y, L; save=true)

    num_downsamp = length(L.C)
    for j=num_downsamp:-1:1

        # Recompute forward state
        Y_ = L.C[j].forward(Y)
        Y1_, Y2_ = tensor_split(Y_)

        X1_ = Y1_
        X2_ = Y2_ - L.RB[j].forward(Y1_)
        X_ = tensor_cat(X1_, X2_)
        X = L.C[j].inverse(X_)


        # Backpropagate residual
        ΔY_, Y_ = L.C[j].forward((ΔY, Y))
        ΔYl_, ΔYr_ = tensor_split(ΔY_)
        
        ΔY1_ = L.RB[j].backward(ΔYr_, Y1_) + ΔYl_
        ΔX_ = tensor_cat(ΔY1_, ΔYr_)

        ΔY, Y = L.C[j].inverse((ΔX_, X_))
    end
    
   return ΔY, Y
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


## Other utils

# Clear gradients
function clear_grad!(L::CouplingLayerIRIM)

    maxiter = length(L.C)

    for j=1:maxiter
        clear_grad!(L.C[j])
        clear_grad!(L.RB[j])
    end
end

# Get parameters
function get_params(L::CouplingLayerIRIM)
    maxiter = length(L.C)

    p1 = get_params(L.C[1])
    p2 = get_params(L.RB[1])
    if maxiter > 1
        for j=2:maxiter
            p1 = cat(p1, get_params(L.C[j]); dims=1)
            p2 = cat(p2, get_params(L.RB[j]); dims=1)
        end
    end

    return cat(p1, p2; dims=1)
end