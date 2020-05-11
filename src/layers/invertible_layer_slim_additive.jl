# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export AdditiveCouplingLayerSLIM


"""
    CS = AdditiveCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=false, permute=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create an invertible additive SLIM coupling layer.

 *Input*: 

 - `nx, ny`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `Ψ`: link function

 - `loget`: bool to indicate whether to return the logdet (default is `false`)

 - `permute`: bool to indicate whether to apply a channel permutation (default is `false`)

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 *Output*:
 
 - `CS`: Invertible SLIM coupling layer

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
struct AdditiveCouplingLayerSLIM <: NeuralNetLayer
    C::Union{Conv1x1, Nothing}
    RB::ResidualBlock
    AN::ActNorm
    Ψ::Function
    logdet::Bool
end

@Flux.functor AdditiveCouplingLayerSLIM

# Constructor from input dimensions
function AdditiveCouplingLayerSLIM(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64, Ψ::Function; 
    k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet::Bool=false, permute::Bool=false)

    # 1x1 Convolution and residual block for invertible layer
    permute == true ? (C = Conv1x1(n_in)) : (C = nothing)
    RB = ResidualBlock(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=false)
    AN = ActNorm(1)

    return AdditiveCouplingLayerSLIM(C, RB, AN, Ψ, logdet)
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{Float32, 4}, J, D, CS::AdditiveCouplingLayerSLIM; logdet=false)

    # Get dimensions
    nx, ny, n_s, batchsize = size(X)

    # Permute and split
    isnothing(CS.C) ? (X_ = copy(X)) : (X_ = CS.C.forward(X))
    X1_, X2_ = tensor_split(X_)

    # Gradient
    g = J'*(J*reshape(CS.Ψ(X1_[:,:,1:1,:]), :, batchsize) - D)
    g = reshape(g, nx, ny, 1, batchsize)
    gn = CS.AN.forward(g)
    gs = tensor_cat(gn, X1_[:,:,2:end,:])

    # Coupling layer
    Y1_ = copy(X1_)
    Y2_ = X2_ + CS.RB.forward(gs)
    Y_ = tensor_cat(Y1_, Y2_)

    isnothing(C) ? (Y = copy(Y_)) : (Y = C.inverse(Y_))
    logdet == true ? (return Y, 0f0) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{Float32, 4}, J, D, CS::AdditiveCouplingLayerSLIM; logdet=false, save=false)

    # Get dimensions
    nx, ny, n_s, batchsize = size(Y)
    isnothing(CS.C) ?  (Y_ = copy(Y)) : (Y_ = CS.C.forward(Y))
 
    # Coupling layer
    Y1_, Y2_ = tensor_split(Y_)
    X1_ = copy(Y1_)

    # Gradient
    g = J'*(J*reshape(CS.Ψ(X1_[:,:,1:1,:]), :, batchsize) - D)
    g = reshape(g, nx, ny, 1, batchsize)
    gn = CS.AN.forward(g)
    gs = tensor_cat(gn, X1_[:,:,2:end,:])

    X2_ = Y2_ - CS.RB.forward(gs)
    X_ = tensor_cat(X1_, X2_)

    isnothing(CS.C) ? (X = copy(X_)) : (X = CS.C.inverse(X_))
    save == true ? (return X, X_, g, gn, gs) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, J, D, CS::AdditiveCouplingLayerSLIM;
                 logdet=false, permute=false)

    # Recompute forward states
    X, X_, g, gn, gs = inverse(Y, J, D, CS; logdet=logdet, save=true)
    nx1, nx2, nx_in, batchsize = size(X)

    # Backpropagation
    isnothing(CS.C) ? (ΔY_ = copy(ΔY)) : (ΔY_ = CS.C.forward((ΔY, Y))[1])
    ΔY1_, ΔY2_ = tensor_split(ΔY_)
    ΔX2_ = copy(ΔY2_)

    Δgs = CS.RB.backward(ΔY2_, gs)
    Δgn = Δgs[:,:,1:1,:]
    Δg = CS.AN.backward(Δgn, gn)[1]
    Jg = J*reshape(Δg, :, batchsize)
    ΔX1_= tensor_cat(reshape(J'*Jg, nx1, nx2, 1, batchsize), Δgs[:,:,2:end,:])
    ΔD = -Jg
    ΔX1_ += ΔY1_
    
    ΔX_ = tensor_cat(ΔX1_, ΔX2_)
    isnothing(CS.   C) ? (ΔX = copy(ΔX_)) : (ΔX = CS.C.inverse((ΔX_, X_))[1])

    return ΔX, ΔD, X
end

# Clear gradients
function clear_grad!(CS::AdditiveCouplingLayerSLIM)
    ~isnothing(CS.C) && clear_grad!(CS.C)
    clear_grad!(CS.RB)
    clear_grad!(CS.AN)
    CS.AN.s.data = nothing
    CS.AN.b.data = nothing
end

# Get parameters
function get_params(CS::AdditiveCouplingLayerSLIM)
    p = get_params(CS.RB)
    ~isnothing(CS.C) && (p = cat(p, get_params(CS.C); dims=1))
    return p
end
