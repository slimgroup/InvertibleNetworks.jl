# Invertible conditional HINT layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalLayerHINT

"""
    CH = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=3, p1=1, p2=0, permute=true)

 Create a conditional HINT layer based on coupling blocks and 1 level recursion. 

 *Input*: 

 - `nx, ny`: spatial dimensions of both `X` and `Y`.
 
 - `n_in`, `n_hidden`: number of input and hidden channels of both `X` and `Y`

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `permute`: bool to indicate whether to permute `X` and `Y`. Default is `true`

 *Output*:
 
 - `CH`: Conditional HINT coupling layer.

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CH.forward_X(X, Y)`

 - Inverse mode: `X, Y = CH.inverse(Zx, Zy)`

 - Backward mode: `ΔX, ΔY, X, Y = CH.backward(ΔZx, ΔZy, Zx, Zy)`

 - Forward mode Y: `Zy = CH.forward_Y(Y)`

 - Inverse mode Y: `Y = CH.inverse(Zy)`

 *Trainable parameters:*

 - None in `CH` itself

 - Trainable parameters in coupling layers `CH.CL_X`, `CH.CL_Y`, `CH.CL_YX` and in
   permutation layers `CH.C_X` and `CH.C_Y`.

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalLayerHINT <: NeuralNetLayer
    CL_X::CouplingLayerHINT
    CL_Y::CouplingLayerHINT
    CL_YX::CouplingLayerBasic
    C_X::Union{Conv1x1, Nothing}
    C_Y::Union{Conv1x1, Nothing}
    forward::Function
    inverse::Function
    backward::Function
    forward_Y::Function
    inverse_Y::Function
end

# Constructor from input dimensions
function ConditionalLayerHINT(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=4, k2=3, p1=0, p2=1, permute=true)

    # Create basic coupling layers
    CL_X = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true, permute="none")
    CL_Y = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true, permute="none")
    CL_YX = CouplingLayerBasic(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true)

    # Permutation using 1x1 convolution
    permute == true ? (C_X = Conv1x1(n_in)) : (C_X = nothing)
    permute == true ? (C_Y = Conv1x1(n_in)) : (C_Y = nothing)

    return ConditionalLayerHINT(CL_X, CL_Y, CL_YX, C_X, C_Y,
        (X, Y) -> forward_hint(X, Y, CL_X, CL_Y, CL_YX, C_X, C_Y),
        (Zx, Zy) -> inverse_hint(Zx, Zy, CL_X, CL_Y, CL_YX, C_X, C_Y),
        (ΔZx, ΔZy, Zx, Zy) -> backward_hint(ΔZx, ΔZy, Zx, Zy, CL_X, CL_Y, CL_YX, C_X, C_Y),
        Y -> forward_hint_Y(Y, CL_Y, C_Y),
        Zy -> inverse_hint_Y(Zy, CL_Y, C_Y)
        )
end

function forward_hint(X, Y, CL_X, CL_Y, CL_YX, C_X, C_Y)

    # Y-lane
    ~isnothing(C_Y) ? (Yp = C_Y.forward(Y)) : (Yp = copy(Y))
    Zy, logdet2 = CL_Y.forward(Yp)

    # X-lane: coupling layer
    ~isnothing(C_X) ? (Xp = C_X.forward(X)) : (Xp = copy(X))
    X, logdet1 = CL_X.forward(Xp)

    # X-lane: conditional layer
    Zx, logdet3 = CL_YX.forward(Yp, X)[2:3]
    logdet = logdet1 + logdet2 + logdet3

    return Zx, Zy, logdet
end

function inverse_hint(Zx, Zy, CL_X, CL_Y, CL_YX, C_X, C_Y)

    # Y-lane
    Yp = CL_Y.inverse(Zy)
    ~isnothing(C_Y) ? (Y = C_Y.inverse(Yp)) : (Y = copy(Yp))

    # X-lane: conditional layer
    YZ = tensor_cat(Yp, Zx)
    X = CL_YX.inverse(Yp, Zx)[2]

    # X-lane: coupling layer
    Xp = CL_X.inverse(X)
    ~isnothing(C_X) ? (X = C_X.inverse(Xp)) : (X = copy(Xp))

    return X, Y
end

function backward_hint(ΔZx, ΔZy, Zx, Zy, CL_X, CL_Y, CL_YX, C_X, C_Y)

    # Y-lane
    ΔYp, Yp = CL_Y.backward(ΔZy, Zy)

    # X-lane: conditional layer
    ΔYp_, ΔX, X = CL_YX.backward(ΔYp.*0f0, ΔZx, Yp, Zx)[[1,2,4]]
    ΔYp += ΔYp_

    # X-lane: coupling layer
    ΔXp, Xp = CL_X.backward(ΔX, X)

    # 1x1 Convolutions
    if isnothing(C_X) || isnothing(C_Y)
        ΔX = copy(ΔXp); X = copy(Xp)
        ΔY = copy(ΔYp); Y = copy(Yp)
    else
        ΔX, X = C_X.inverse((ΔXp, Xp))
        ΔY, Y = C_Y.inverse((ΔYp, Yp))
    end
    return ΔX, ΔY, X, Y
end

function forward_hint_Y(Y, CL_Y, C_Y)
    ~isnothing(C_Y) ? (Yp = C_Y.forward(Y)) : (Yp = copy(Y))
    Zy = CL_Y.forward(Yp)[1]
    return Zy

end

function inverse_hint_Y(Zy, CL_Y, C_Y)
    Yp = CL_Y.inverse(Zy)
    ~isnothing(C_Y) ? (Y = C_Y.inverse(Yp)) : (Y = copy(Yp))
    return Y
end

# Clear gradients
function clear_grad!(CH::ConditionalLayerHINT)
    clear_grad!(CH.CL_X)
    clear_grad!(CH.CL_Y)
    clear_grad!(CH.CL_YX)
    ~isnothing(CH.C_X) && clear_grad!(CH.C_X)
    ~isnothing(CH.C_Y) && clear_grad!(CH.C_Y)
end

# Get parameters
function get_params(CH::ConditionalLayerHINT)
    p = get_params(CH.CL_X)
    p = cat(p, get_params(CH.CL_Y); dims=1)
    p = cat(p, get_params(CH.CL_YX); dims=1)
    ~isnothing(CH.C_X) && (p = cat(p, get_params(CH.C_X); dims=1))
    ~isnothing(CH.C_Y) && (p = cat(p, get_params(CH.C_Y); dims=1))
    return p
end