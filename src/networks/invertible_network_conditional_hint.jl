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
struct NetworkConditionalHINT <: InvertibleNetwork
    AN_X::AbstractArray{ActNorm, 1}
    AN_Y::AbstractArray{ActNorm, 1}
    CL::AbstractArray{ConditionalLayerHINT, 1}
    forward::Function
    inverse::Function
    backward::Function
    forward_Y::Function
    inverse_Y::Function
end

# Constructor
function NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    AN_X = Array{ActNorm}(undef, depth)
    AN_Y = Array{ActNorm}(undef, depth)
    CL = Array{ConditionalLayerHINT}(undef, depth)

    # Create layers
    for j=1:depth
        AN_X[j] = ActNorm(n_in; logdet=true)
        AN_Y[j] = ActNorm(n_in; logdet=true)
        CL[j] = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; permute=true, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)
    end

    return NetworkConditionalHINT(AN_X, AN_Y, CL, 
        (X, Y) -> cond_hint_forward(X, Y, AN_X, AN_Y, CL),
        (Zx, Zy) -> cond_hint_inverse(Zx, Zy, AN_X, AN_Y, CL),
        (ΔZx, ΔZy, Zx, Zy) -> cond_hint_backward(ΔZx, ΔZy, Zx, Zy, AN_X, AN_Y, CL),
        Y -> cond_hint_forward_Y(Y, AN_Y, CL),
        Zy -> cond_hint_inverse_Y(Zy, AN_Y, CL)
    )
end

# Forward pass and compute logdet
function cond_hint_forward(X, Y, AN_X, AN_Y, CL)
    depth = length(CL)
    logdet = 0f0
    for j=1:depth
        X_, logdet1 = AN_X[j].forward(X)
        Y_, logdet2 = AN_Y[j].forward(Y)
        X, Y, logdet3 = CL[j].forward(X_, Y_)
        logdet += (logdet1 + logdet2 + logdet3)
    end
    return X, Y, logdet
end

# Inverse pass and compute gradients
function cond_hint_inverse(Zx, Zy, AN_X, AN_Y, CL)
    depth = length(CL)
    for j=depth:-1:1
        Zx_, Zy_ = CL[j].inverse(Zx, Zy)
        Zy = AN_Y[j].inverse(Zy_)
        Zx = AN_X[j].inverse(Zx_)
    end
    return Zx, Zy
end

# Backward pass and compute gradients
function cond_hint_backward(ΔZx, ΔZy, Zx, Zy, AN_X, AN_Y, CL)
    depth = length(CL)
    for j=depth:-1:1
        ΔZx_, ΔZy_, Zx_, Zy_ = CL[j].backward(ΔZx, ΔZy, Zx, Zy)
        ΔZx, Zx = AN_X[j].backward(ΔZx_, Zx_)
        ΔZy, Zy = AN_Y[j].backward(ΔZy_, Zy_)
    end
    return ΔZx, ΔZy, Zx, Zy
end

# Forward pass and compute logdet
function cond_hint_forward_Y(Y, AN_Y, CL)
    depth = length(CL)
    for j=1:depth
        Y_ = AN_Y[j].forward(Y)[1]
        Y = CL[j].forward_Y(Y_)
    end
    return Y
end

# Inverse pass and compute gradients
function cond_hint_inverse_Y(Zy, AN_Y, CL)
    depth = length(CL)
    for j=depth:-1:1
        Zy_ = CL[j].inverse_Y(Zy)
        Zy = AN_Y[j].inverse(Zy_)
    end
    return Zy
end

# Clear gradients
function clear_grad!(CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    for j=1:depth
        clear_grad!(CH.AN_X[j])
        clear_grad!(CH.AN_Y[j])
        clear_grad!(CH.CL[j])
    end
end

# Get parameters
function get_params(CH::NetworkConditionalHINT)
    depth = length(CH.CL)
    p = []
    for j=1:depth
        p = cat(p, get_params(CH.AN_X[j]); dims=1)
        p = cat(p, get_params(CH.AN_Y[j]); dims=1)
        p = cat(p, get_params(CH.CL[j]); dims=1)
    end
    return p
end
