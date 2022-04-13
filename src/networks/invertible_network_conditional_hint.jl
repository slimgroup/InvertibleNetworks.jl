# Invertible conditional HINT network from Kruse et. al (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkConditionalHINT, NetworkConditionalHINT3D

"""
    CH = NetworkConditionalHINT(n_in, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    CH = NetworkConditionalHINT3D(n_in, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

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

 - `CH`: conditioinal HINT network

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CH.forward(X, Y)`

 - Inverse mode: `X, Y = CH.inverse(Zx, Zy)`

 - Backward mode: `ΔX, X = CH.backward(ΔZx, ΔZy, Zx, Zy)`

 *Trainable parameters:*

 - None in `CH` itself

 - Trainable parameters in activation normalizations `CH.AN_X[i]` and `CH.AN_Y[i]`,
 and in coupling layers `CH.CL[i]`, where `i` ranges from `1` to `depth`.

 See also: [`ActNorm`](@ref), [`ConditionalLayerHINT!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct NetworkConditionalHINT <: InvertibleNetwork
    AN_X::AbstractArray{ActNorm, 1}
    AN_Y::AbstractArray{ActNorm, 1}
    CL::AbstractArray{ConditionalLayerHINT, 1}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor NetworkConditionalHINT

# Constructor
function NetworkConditionalHINT(n_in, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=true, ndims=2)

    AN_X = Array{ActNorm}(undef, depth)
    AN_Y = Array{ActNorm}(undef, depth)
    CL = Array{ConditionalLayerHINT}(undef, depth)

    # Create layers
    for j=1:depth
        AN_X[j] = ActNorm(n_in; logdet=logdet)
        AN_Y[j] = ActNorm(n_in; logdet=logdet)
        CL[j] = ConditionalLayerHINT(n_in, n_hidden; permute=true, k1=k1, k2=k2, p1=p1, p2=p2,
                                     s1=s1, s2=s2, logdet=logdet, ndims=ndims)
    end

    return NetworkConditionalHINT(AN_X, AN_Y, CL, logdet, false)
end

NetworkConditionalHINT3D(args...;kw...) = NetworkConditionalHINT(args...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::NetworkConditionalHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0
    for j=1:depth
        logdet ? (X_, logdet1) = CH.AN_X[j].forward(X) : X_ = CH.AN_X[j].forward(X)
        logdet ? (Y_, logdet2) = CH.AN_Y[j].forward(Y) : Y_ = CH.AN_Y[j].forward(Y)
        logdet ? (X, Y, logdet3) = CH.CL[j].forward(X_, Y_) : (X, Y) = CH.CL[j].forward(X_, Y_)
        logdet && (logdet_ += (logdet1 + logdet2 + logdet3))
    end
    logdet ? (return X, Y, logdet_) : (return X, Y)
end

# Inverse pass and compute gradients
function inverse(Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::NetworkConditionalHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0
    for j=depth:-1:1
        logdet ? (Zx_, Zy_, logdet1) = CH.CL[j].inverse(Zx, Zy; logdet=true) : (Zx_, Zy_) = CH.CL[j].inverse(Zx, Zy; logdet=false)
        logdet ? (Zy, logdet2) = CH.AN_Y[j].inverse(Zy_; logdet=true) : Zy = CH.AN_Y[j].inverse(Zy_; logdet=false)
        logdet ? (Zx, logdet3) = CH.AN_X[j].inverse(Zx_; logdet=true) : Zx = CH.AN_X[j].inverse(Zx_; logdet=false)
        logdet && (logdet_ += (logdet1 + logdet2 + logdet3))
    end
    logdet ? (return Zx, Zy, logdet_) : (return Zx, Zy)
end

# Backward pass and compute gradients
function backward(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, N}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::NetworkConditionalHINT; set_grad::Bool=true) where {T, N}
    depth = length(CH.CL)
    if ~set_grad
        ΔθANX, ΔθANY, ΔθCL = Vector{Parameter}(undef, 0), Vector{Parameter}(undef, 0), Vector{Parameter}(undef, 0)
        if CH.logdet
            ∇logdetANX, ∇logdetANY, ∇logdetCL = Vector{Parameter}(undef, 0), Vector{Parameter}(undef, 0), Vector{Parameter}(undef, 0)
        end
    end
    for j=depth:-1:1
        if set_grad
            ΔZx_, ΔZy_, Zx_, Zy_ = CH.CL[j].backward(ΔZx, ΔZy, Zx, Zy)
            ΔZx, Zx = CH.AN_X[j].backward(ΔZx_, Zx_)
            ΔZy, Zy = CH.AN_Y[j].backward(ΔZy_, Zy_)
        else
            if CH.logdet
                ΔZx_, ΔZy_, Δθcl, Zx_, Zy_, ∇logdetcl = CH.CL[j].backward(ΔZx, ΔZy, Zx, Zy; set_grad=set_grad)
                ΔZx, Δθx, Zx, ∇logdetx = CH.AN_X[j].backward(ΔZx_, Zx_; set_grad=set_grad)
                ΔZy, Δθy, Zy, ∇logdety = CH.AN_Y[j].backward(ΔZy_, Zy_; set_grad=set_grad)
                prepend!(∇logdetANX, ∇logdetx);prepend!(∇logdetANY, ∇logdety);prepend!(∇logdetCL, ∇logdetcl)
            else
                ΔZx_, ΔZy_, Δθcl, Zx_, Zy_ = CH.CL[j].backward(ΔZx, ΔZy, Zx, Zy; set_grad=set_grad)
                ΔZx, Δθx, Zx = CH.AN_X[j].backward(ΔZx_, Zx_; set_grad=set_grad)
                ΔZy, Δθy, Zy = CH.AN_Y[j].backward(ΔZy_, Zy_; set_grad=set_grad)
            end
            prepend!(ΔθANX, Δθx);prepend!(ΔθANY, Δθy);prepend!(ΔθCL, Δθcl)
        end
    end
    if set_grad
        return ΔZx, ΔZy, Zx, Zy
    else
        CH.logdet && (∇logdet = vcat(∇logdetANX, ∇logdetANY, ∇logdetCL))
        Δθ = vcat(ΔθANX, ΔθANY, ΔθCL)
        CH.logdet ? (return ΔZx, ΔZy, Δθ, Zx, Zy, ∇logdet) : (return ΔZx, ΔZy, Δθ, Zx, Zy)
    end
end

# Backward reverse pass and compute gradients
function backward_inv(ΔX::AbstractArray{T, N}, ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::NetworkConditionalHINT) where {T, N}
    depth = length(CH.CL)
    for j=1:depth
        ΔX_, X_ = backward_inv(ΔX, X, CH.AN_X[j])
        ΔY_, Y_ = backward_inv(ΔY, Y, CH.AN_Y[j])
        ΔX, ΔY, X, Y = backward_inv(ΔX_, ΔY_, X_, Y_, CH.CL[j])
    end
    return ΔX, ΔY, X, Y
end

# Forward pass and compute logdet
function forward_Y(Y::AbstractArray{T, N}, CH::NetworkConditionalHINT) where {T, N}
    depth = length(CH.CL)
    for j=1:depth
        Y_ = CH.AN_Y[j].forward(Y; logdet=false)
        Y = CH.CL[j].forward_Y(Y_)
    end
    return Y
end

# Inverse pass and compute gradients
function inverse_Y(Zy::AbstractArray{T, N}, CH::NetworkConditionalHINT) where {T, N}
    depth = length(CH.CL)
    for j=depth:-1:1
        Zy_ = CH.CL[j].inverse_Y(Zy)
        Zy = CH.AN_Y[j].inverse(Zy_; logdet=false)
    end
    return Zy
end

## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, ΔY::AbstractArray{T, N}, Δθ::Array{Parameter}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::NetworkConditionalHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0
    if logdet
        cls = 4*depth
        ays = 2*depth
        ΔθANX = Vector{Parameter}(undef, 0)
        ΔθANY = Vector{Parameter}(undef, 0)
        ΔθCL = Vector{Parameter}(undef, 0)
    end

    n = Int64(length(Δθ)/depth)
    for j=1:depth
        asx = length(ΔθANX)+1
        asy = ays + length(ΔθANY) + 1
        cs = cls + length(ΔθCL) + 1
        ce = cs + length(get_params(CH.CL[j])) - 1
        if logdet
            ΔX_, X_, logdet1, GNΔθ1 = CH.AN_X[j].jacobian(ΔX, Δθ[asx:asx+1], X)
            ΔY_, Y_, logdet2, GNΔθ2 = CH.AN_Y[j].jacobian(ΔY, Δθ[asy:asy+1], Y)
            ΔX, ΔY, X, Y, logdet3, GNΔθ3 = CH.CL[j].jacobian(ΔX_, ΔY_, Δθ[cs:ce], X_, Y_)
            logdet_ += (logdet1 + logdet2 + logdet3)
            append!(ΔθANX, GNΔθ1)
            append!(ΔθANY, GNΔθ2)
            append!(ΔθCL, GNΔθ3)
        else 
            ΔX_, X_ = CH.AN_X[j].jacobian(ΔX, Δθij[asx:asx+1], X)
            ΔY_, Y_ = CH.AN_Y[j].jacobian(ΔY, Δθij[asx:asx+1], Y)
            ΔX, ΔY, X, Y = CH.CL[j].jacobian(ΔX_, ΔY_, Δθj[cs:ce], X_, Y_)
        end
    end
    logdet ? (return ΔX, ΔY, X, Y, logdet_, vcat(ΔθANX, ΔθANY, ΔθCL)) : (return ΔX, ΔY, X, Y)
end

adjointJacobian(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, N}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::NetworkConditionalHINT) where {T, N} = backward(ΔZx, ΔZy, Zx, Zy, CH; set_grad=false)

# Set is_reversed flag in full network tree
function tag_as_reversed!(CH::NetworkConditionalHINT, tag::Bool)
    depth = length(CH.CL)
    CH.is_reversed = tag
    for j=1:depth
        tag_as_reversed!(CH.AN_X[j], tag)
        tag_as_reversed!(CH.AN_Y[j], tag)
        tag_as_reversed!(CH.CL[j], tag)
    end
    return CH
end