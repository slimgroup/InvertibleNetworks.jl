# Invertible conditional HINT network from Kruse et. al (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkConditionalHINT

"""
    CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create a conditional HINT network for data-driven generative modeling based
 on the change of variables formula.

 *Input*:

 - `nx`, `ny`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensors `X` and `Y`

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
    CL::AbstractArray{ConditionalLayerHINT, 1}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor NetworkConditionalHINT

# Constructor
function NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=true)

    CL = Array{ConditionalLayerHINT}(undef, depth)

    # Create layers
    for j=1:depth
        CL[j] = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize;
                                     permute=true, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet)
    end

    return NetworkConditionalHINT(CL, logdet, false)
end

# Forward pass and compute logdet
function forward(X, Y, CH::NetworkConditionalHINT; logdet=nothing)
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0f0
    for j=1:depth
        logdet ? (X, Y, logdet3) = CH.CL[j].forward(X, Y) : (X, Y) = CH.CL[j].forward(X, Y)
        logdet && (logdet_ += (logdet3))
    end
    logdet ? (return X, Y, logdet_) : (return X, Y)
end

# Inverse pass and compute gradients
function inverse(Zx, Zy, CH::NetworkConditionalHINT; logdet=nothing)
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    depth = length(CH.CL)
    logdet_ = 0f0
    for j=depth:-1:1
        logdet ? (Zx, Zy, logdet1) = CH.CL[j].inverse(Zx, Zy; logdet=true) : (Zx, Zy) = CH.CL[j].inverse(Zx, Zy; logdet=false)
        logdet && (logdet_ += (logdet1))
    end
    logdet ? (return Zx, Zy, logdet_) : (return Zx, Zy)
end

# Backward pass and compute gradients
function backward(ΔZx, ΔZy, Zx, Zy, CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    for j=depth:-1:1
        ΔZx, ΔZy, Zx, Zy = CH.CL[j].backward(ΔZx, ΔZy, Zx, Zy)
    end
    return ΔZx, ΔZy, Zx, Zy
end

# Backward reverse pass and compute gradients
function backward_inv(ΔX, ΔY, X, Y, CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    for j=1:depth
        ΔX, ΔY, X, Y = backward_inv(ΔX, ΔY, X, Y, CH.CL[j])
    end
    return ΔX, ΔY, X, Y
end

# Forward pass and compute logdet
function forward_Y(Y, CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    for j=1:depth
        Y_ = CH.AN_Y[j].forward(Y; logdet=false)
        Y = CH.CL[j].forward_Y(Y_)
    end
    return Y
end

# Inverse pass and compute gradients
function inverse_Y(Zy, CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    for j=depth:-1:1
        Zy_ = CH.CL[j].inverse_Y(Zy)
        Zy = CH.AN_Y[j].inverse(Zy_; logdet=false)
    end
    return Zy
end

# Clear gradients
function clear_grad!(CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    for j=1:depth
        clear_grad!(CH.CL[j])
    end
end

# Get parameters
function get_params(CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    p = []
    for j=1:depth
        p = cat(p, get_params(CH.CL[j]); dims=1)
    end
    return p
end

# Set is_reversed flag in full network tree
function tag_as_reversed!(CH::NetworkConditionalHINT, tag::Bool)
    depth = length(CH.CL)
    CH.is_reversed = tag
    for j=1:depth
        tag_as_reversed!(CH.CL[j], tag)
    end
    return CH
end
