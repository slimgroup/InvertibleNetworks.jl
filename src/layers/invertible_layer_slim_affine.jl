# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export AffineCouplingLayerSLIM


"""
    CS = AffineCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=false, permute=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create an invertible affine SLIM coupling layer.

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
struct AffineCouplingLayerSLIM <: NeuralNetLayer
    C::Union{Conv1x1, Nothing}
    RB::Union{ResidualBlock, FluxBlock}
    AN::ActNorm
    Ψ::Function
    logdet::Bool
end

@Flux.functor AffineCouplingLayerSLIM

# Constructor from input dimensions
function AffineCouplingLayerSLIM(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64, Ψ::Function;
    k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet::Bool=false, permute::Bool=false)

    # 1x1 Convolution and residual block for invertible layer
    permute == true ? (C = Conv1x1(n_in)) : (C = nothing)
    RB = ResidualBlock(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true)
    AN = ActNorm(1)

    return AffineCouplingLayerSLIM(C, RB, AN, Ψ, logdet)
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{Float32, N}, D, J, CS::AffineCouplingLayerSLIM) where N

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
    logS_T = CS.RB.forward(gs)
    logS, T = tensor_split(logS_T)
    S = Sigmoid(logS)
    Y2_ = S.*X2_ + T
    Y_ = tensor_cat(Y1_, Y2_)

    isnothing(CS.C) ? (Y = copy(Y_)) : (Y = CS.C.inverse(Y_))
    CS.logdet == true ? (return Y, slim_logdet_forward(S)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{Float32, 4}, D, J, CS::AffineCouplingLayerSLIM; save=false)

    # Get dimensions
    nx, ny, n_s, batchsize = size(Y)
    isnothing(CS.C) ?  (Y_ = copy(Y)) : (Y_ = CS.C.forward(Y))

    # Gradient
    Y1_, Y2_ = tensor_split(Y_)
    X1_ = copy(Y1_)
    g = J'*(J*reshape(CS.Ψ(X1_[:,:,1:1,:]), :, batchsize) - D)
    g = reshape(g, nx, ny, 1, batchsize)
    gn = CS.AN.forward(g)
    gs = tensor_cat(gn, X1_[:,:,2:end,:])

    # Coupling layer
    logS_T = CS.RB.forward(gs)
    logS, T = tensor_split(logS_T)
    S = Sigmoid(logS)
    X2_ = (Y2_ - T) ./ (S .+ eps(1f0)) # add epsilon to avoid
    X_ = tensor_cat(X1_, X2_)

    isnothing(CS.C) ? (X = copy(X_)) : (X = CS.C.inverse(X_))
    save == true ? (return X, X1_, X2_, gn, gs, S) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, D, J, CS::AffineCouplingLayerSLIM; permute=false, set_grad::Bool=true)

    # Recompute forward states
    X, X1_, X2_, gn, gs, S  = inverse(Y, D, J, CS; save=true)
    nx, ny, n_s, batchsize = size(Y)

    # Backpropagation
    if isnothing(CS.C)
        ΔY_ = copy(ΔY)
    else
        set_grad ? (ΔY_ = CS.C.forward((ΔY, Y))[1]) : ((ΔY_, Δθ_C1) = CS.C.forward((ΔY, Y); set_grad=set_grad)[1:2])
    end
    ΔY1_, ΔY2_ = tensor_split(ΔY_)
    ΔT = copy(ΔY2_)
    ΔS = ΔY2_ .* X2_
    if CS.logdet
        set_grad ? (ΔS -= slim_logdet_backward(S)) : (ΔS_ = slim_logdet_backward(S))
    end
    ΔX2_ = ΔY2_ .* S
    if set_grad
        Δgs = CS.RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), gs)
    else
        Δgs, Δθ_RB = CS.RB.backward(cat(SigmoidGrad(ΔS, S), ΔT; dims=3), gs; set_grad=set_grad)
        _, ∇logdet_RB = CS.RB.backward(cat(SigmoidGrad(ΔS_, S), 0f0.*ΔT; dims=3), gs; set_grad=set_grad)
    end
    Δgn = Δgs[:,:,1:1,:]
    set_grad ? (Δg = CS.AN.backward(Δgn, gn)[1]) : ((Δg, Δθ_AN, ∇logdet_AN) = CS.AN.backward(Δgn, gn; set_grad=set_grad)[1:2])
    Jg = J*reshape(Δg, :, batchsize)
    ΔD = -Jg
    ΔX1_ = tensor_cat(reshape(J'*Jg, nx, ny, 1, batchsize), Δgs[:,:,2:end,:]) + ΔY1_
    ΔX_ = tensor_cat(ΔX1_, ΔX2_)
    if isnothing(CS.C)
        ΔX = copy(ΔX_)
    else
        set_grad ? (ΔX = CS.C.inverse((ΔX_, tensor_cat(X1_, X2_)))[1]) : ((ΔX, Δθ_C2) = CS.C.inverse((ΔX_, tensor_cat(X1_, X2_)); set_grad=set_grad)[1:2])
    end

    set_grad ? (return ΔX, ΔD, X) : (return ΔX, ΔD, cat(Δθ_C1+Δθ_C2, Δθ_RB, Δθ_AN; dims=1), X, cat(0f0.*Δθ_C1, ∇logdet_RB, ∇logdet_AN; dims=1))
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{Float32, 4}, ΔD, X::AbstractArray{Float32, 4}, D, J, CS::AffineCouplingLayerSLIM)
    throw(ArgumentError("Jacobian for AffineCouplingLayerSLIM not yet implemented"))
end

adjointJacobian(ΔY::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4}, D, J, CS::AffineCouplingLayerSLIM; permute=false) = backward(ΔY, Y, D, J, CS; permute=permute, set_grad=false)


# Other utils

# Clear gradients
function clear_grad!(CS::AffineCouplingLayerSLIM)
    ~isnothing(CS.C) && clear_grad!(CS.C)
    clear_grad!(CS.RB)
    clear_grad!(CS.AN)
    CS.AN.s.data = nothing
    CS.AN.b.data = nothing
end

# Get parameters
function get_params(CS::AffineCouplingLayerSLIM)
    isnothing(CS.C) ? (p_C = Array{Parameter, 1}(undef, 0)) : (p_C = get_params(CS.C))
    p_RB = get_params(CS.RB)
    p_AN = get_params(CS.AN)
    return cat(p_C, p_RB, p_AN; dims=1)
end

# Logdet
slim_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
slim_logdet_backward(S) = 1f0./ S / size(S, 4)
