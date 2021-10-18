# Invertible conditional HINT network from Kruse et. al (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020


import InvertibleNetworks: clear_grad!, get_params, tag_as_reversed!, forward, backward, inverse, backward_inv, jacobian, adjointJacobian

export NetworkHINT, NetworkHINT3D

"""
    CH = NetworkHINT(n_in, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    CH = NetworkHINT3D(n_in, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create a conditional HINT network for data-driven generative modeling based
 on the change of variables formula.

 *Input*:

 - 'n_in': number of input channels

 - `n_hidden`: number of hidden units in residual blocks

 - `depth`: number network layers

 - `k1`, `k2`: kernel size for first and third residual layer (`k1`) and second layer (`k2`)

 - `p1`, `p2`: respective padding sizes for residual block layers

 - `s1`, `s2`: respective strides for residual block layers

 *Output*:

 - `CH`: HINT network

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CH.forward(X, Y)`

 - Inverse mode: `X, Y = CH.inverse(Zx, Zy)`

 - Backward mode: `ΔX, X = CH.backward(ΔZx, ΔZy, Zx, Zy)`

 *Trainable parameters:*

 - None in `CH` itself

 - Trainable parameters in activation normalizations `CH.AN_X[i]` and `CH.AN_Y[i]`,
 and in coupling layers `CH.CL[i]`, where `i` ranges from `1` to `depth`.

 See also: [`ActNorm`](@ref), [`CouplingLayerHINT!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct NetworkHINT <: InvertibleNetwork
    AN_X::AbstractArray{ActNorm, 1}
    CL::AbstractArray{CouplingLayerHINT, 1}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor NetworkHINT

# Constructor
function NetworkHINT(n_in, n_hidden, depth;gab_rb=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=true, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    AN_X = Array{ActNorm}(undef, depth)
    CL = Array{CouplingLayerHINT}(undef, depth)

    # Create layers
    for j=1:depth
        AN_X[j] = ActNorm(n_in; logdet=logdet)
        CL[j] = CouplingLayerHINT(n_in, n_hidden;gab_rb=gab_rb, permute="full", k1=k1, k2=k2, p1=p1,
                                  p2=p2, s1=s1, s2=s2, logdet=logdet, activation=activation,ndims=ndims)
    end

    return NetworkHINT(AN_X, CL, logdet, false)
end

NetworkHINT3D(args...;kw...) = NetworkHINT(args...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X, CH::NetworkHINT; logdet=nothing)
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0f0
    for j=1:depth
        logdet ? (X_, logdet1) = CH.AN_X[j].forward(X) : X_ = CH.AN_X[j].forward(X)
        logdet ? (X, logdet3) = CH.CL[j].forward(X_) : X = CH.CL[j].forward(X_)
        logdet && (logdet_ += (logdet1 + logdet3))
    end
    logdet ? (return X, logdet_) : (return X)
end

# Inverse pass and compute gradients
function inverse(Zx, CH::NetworkHINT; logdet=nothing)
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0f0
    for j=depth:-1:1
        logdet ? (Zx_, logdet1) = CH.CL[j].inverse(Zx; logdet=true) : Zx_ = CH.CL[j].inverse(Zx; logdet=false)
        logdet ? (Zx, logdet3) = CH.AN_X[j].inverse(Zx_; logdet=true) : Zx = CH.AN_X[j].inverse(Zx_; logdet=false)
        logdet && (logdet_ += (logdet1 + logdet3))
    end
    logdet ? (return Zx, logdet_) : (return Zx)
end

# Backward pass and compute gradients
function backward(ΔZx, Zx, CH::NetworkHINT; set_grad::Bool=true)
    depth = length(CH.CL)
    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 0)
        CH.logdet && (∇logdet = Array{Parameter, 1}(undef, 0))
    end
    for j=depth:-1:1
        if set_grad
            ΔZx_, Zx_ = CH.CL[j].backward(ΔZx, Zx)
            ΔZx, Zx = CH.AN_X[j].backward(ΔZx_, Zx_)
        else
            if CH.logdet
                ΔZx_, Δθcl, Zx_, ∇logdetcl = CH.CL[j].backward(ΔZx, Zx; set_grad=set_grad)
                ΔZx, Δθx, Zx, ∇logdetx = CH.AN_X[j].backward(ΔZx_, Zx_; set_grad=set_grad)
                ∇logdet = cat(∇logdetx, ∇logdetcl, ∇logdet; dims=1)
            else
                ΔZx_, Δθcl, Zx_ = CH.CL[j].backward(ΔZx, Zx; set_grad=set_grad)
                ΔZx, Δθx, Zx = CH.AN_X[j].backward(ΔZx_, Zx_; set_grad=set_grad)
            end
            Δθ = cat(Δθx, Δθcl, Δθ; dims=1)
        end
    end
    if set_grad
        return ΔZx, Zx
    else
        CH.logdet ? (return ΔZx, Δθ, Zx, ∇logdet) : (return ΔZx, Δθ, Zx)
    end
end

# Backward reverse pass and compute gradients
function backward_inv(ΔX, X, CH::NetworkHINT)
    depth = length(CH.CL)
    for j=1:depth
        ΔX_, X_ = backward_inv(ΔX, X, CH.AN_X[j])
        ΔX, X = backward_inv(ΔX_, X_, CH.CL[j])
    end
    return ΔX, X
end

## Jacobian-related utils

function jacobian(ΔX, Δθ, X, CH::NetworkHINT; logdet=nothing)
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0f0
    logdet && (GNΔθ = Array{Parameter, 1}(undef, 0))
    n = Int64(length(Δθ)/depth)
    for j=1:depth
        Δθj = Δθ[n*(j-1)+1:n*j]
        if logdet
            ΔX_, X_, logdet1, GNΔθ1 = CH.AN_X[j].jacobian(ΔX, Δθj[1:2], X)
            ΔX, X, logdet3, GNΔθ3 = CH.CL[j].jacobian(ΔX_, Δθj[5:end], X_)
            logdet_ += (logdet1 + logdet3)
            GNΔθ = cat(GNΔθ, GNΔθ1, GNΔθ3; dims=1)
        else
            ΔX_, X_ = CH.AN_X[j].jacobian(ΔX, Δθj[1:2], X)
            ΔX, X = CH.CL[j].jacobian(ΔX_, Δθj[5:end], X_)
        end
    end
    logdet ? (return ΔX, X, logdet_, GNΔθ) : (return ΔX, X)
end

adjointJacobian(ΔZx, Zx, CH::NetworkHINT) = backward(ΔZx, Zx, CH; set_grad=false)

## Other utils

# Clear gradients
function clear_grad!(CH::NetworkHINT)
    depth = length(CH.CL)
    for j=1:depth
        clear_grad!(CH.AN_X[j])
        clear_grad!(CH.CL[j])
    end
end

# Get parameters
function get_params(CH::NetworkHINT)
    depth = length(CH.CL)
    p = Array{Parameter, 1}(undef, 0)
    for j=1:depth
        p = cat(p, get_params(CH.AN_X[j]); dims=1)
        p = cat(p, get_params(CH.CL[j]); dims=1)
    end
    return p
end

# Set is_reversed flag in full network tree
function tag_as_reversed!(CH::NetworkHINT, tag::Bool)
    depth = length(CH.CL)
    CH.is_reversed = tag
    for j=1:depth
        tag_as_reversed!(CH.AN_X[j], tag)
        tag_as_reversed!(CH.CL[j], tag)
    end
    return CH
end
