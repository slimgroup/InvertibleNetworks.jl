# Invertible conditional HINT layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalLayerHINT

"""
    CH = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=3, p1=1, p2=0)

 Create a conditional HINT layer based on coupling blocks and 1 level recursion. 

 *Input*: 

 - `nx, ny`: spatial dimensions of both `X` and `Y`.
 
 - `n_in`, `n_hidden`: number of input and hidden channels of both `X` and `Y`

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

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

 - Trainable parameters in coupling layers `CH.CL_X`, `CH.CL_Y`, `CH.CL_XY` and in
   permutation layers `CH.C_X` and `CH.C_Y`.

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalLayerHINT <: NeuralNetLayer
    CL_X::CouplingLayerBasic
    CL_Y::CouplingLayerBasic
    CL_XY::CouplingLayerBasic
    C_X::Conv1x1
    C_Y::Conv1x1
    forward::Function
    inverse::Function
    backward::Function
    forward_Y::Function
    inverse_Y::Function
end

# Constructor from input dimensions
function ConditionalLayerHINT(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=4, k2=3, p1=0, p2=1)

    # Create basic coupling layers
    CL_X = CouplingLayerBasic(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true)
    CL_Y = CouplingLayerBasic(nx, ny, Int(n_in/2), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true)
    CL_XY = CouplingLayerBasic(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true)

    # Permutation using 1x1 convolution
    C_X = Conv1x1(n_in)
    C_Y = Conv1x1(n_in)

    return ConditionalLayerHINT(CL_X, CL_Y, CL_XY, C_X, C_Y,
        (X, Y) -> forward_hint(X, Y, CL_X, CL_Y, CL_XY, C_X, C_Y),
        (Zx, Zy) -> inverse_hint(Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y),
        (ΔZx, ΔZy, Zx, Zy) -> backward_hint(ΔZx, ΔZy, Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y),
        Y -> forward_hint_Y(Y, CL_Y, C_Y),
        Zy -> inverse_hint_Y(Zy, CL_Y, C_Y)
        )
end

function forward_hint(X, Y, CL_X, CL_Y, CL_XY, C_X, C_Y)

    # Permute X and Y
    Xp = C_X.forward(X)
    Yp = C_Y.forward(Y)

    # Split
    Xa, Xb = tensor_split(Xp)
    Ya, Yb = tensor_split(Yp)

    # Coupling layers
    Xa, Xb, logdet1 = CL_X.forward(Xa, Xb)
    Ya, Yb, logdet2 = CL_Y.forward(Ya, Yb)

    # Cat
    X = tensor_cat(Xa, Xb)
    Zy = tensor_cat(Ya, Yb)

    # Conditional layer
    Zx, logdet3 = CL_XY.forward(X, Yp)[[1,3]]
    logdet = logdet1 + logdet2 + logdet3

    return Zx, Zy, logdet

end

function inverse_hint(Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y)

    # Y-lane
    Ya, Yb = tensor_split(Zy)
    Ya, Yb = CL_Y.inverse(Ya, Yb)
    Yp = tensor_cat(Ya, Yb)

    # X-lane
    X = CL_XY.inverse(Zx, Yp)[1]
    Xa, Xb = tensor_split(X)
    Xa, Xb = CL_X.inverse(Xa, Xb)
    Xp = tensor_cat(Xa, Xb)

    # Undo permutation
    Y = C_Y.inverse(Yp)
    X = C_X.inverse(Xp)

    return X, Y
end

function backward_hint(ΔZx, ΔZy, Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y)

    # Y-lane
    ΔYa, ΔYb = tensor_split(ΔZy)
    Ya, Yb = tensor_split(Zy)
    ΔYa, ΔYb, Ya, Yb = CL_Y.backward(ΔYa, ΔYb, Ya, Yb)
    ΔYp = tensor_cat(ΔYa, ΔYb)
    Yp = tensor_cat(Ya, Yb)

    # X-lane
    ΔX, X = CL_XY.backward(ΔZx, ΔYp, Zx, Yp)[[1,3]]
    ΔXa, ΔXb = tensor_split(ΔX)
    Xa, Xb = tensor_split(X)
    ΔXa, ΔXb, Xa, Xb = CL_X.backward(ΔXa, ΔXb, Xa, Xb)
    ΔXp = tensor_cat(ΔXa, ΔXb)
    Xp = tensor_cat(Xa, Xb)

    # Undo permutation
    ΔY, Y = C_Y.inverse((ΔYp, Yp))
    ΔX, X = C_X.inverse((ΔXp, Xp))

    return ΔX, ΔY, X, Y
end

function forward_hint_Y(Y, CL_Y, C_Y)
    Yp = C_Y.forward(Y)
    Ya, Yb = tensor_split(Yp)
    Ya, Yb, logdet2 = CL_Y.forward(Ya, Yb)
    Zy = tensor_cat(Ya, Yb)
    return Zy

end

function inverse_hint_Y(Zy, CL_Y, C_Y)
    Ya, Yb = tensor_split(Zy)
    Ya, Yb = CL_Y.inverse(Ya, Yb)
    Yp = tensor_cat(Ya, Yb)
    Y = C_Y.inverse(Yp)
    return Y
end

# Clear gradients
function clear_grad!(CH::ConditionalLayerHINT)
    clear_grad!(CH.CL_X)
    clear_grad!(CH.CL_Y)
    clear_grad!(CH.CL_XY)
    clear_grad!(CH.C_X)
    clear_grad!(CH.C_Y)
end

# Get parameters
function get_params(CH::ConditionalLayerHINT)
    p = get_params(CH.CL_X)
    p = cat(p, get_params(CH.CL_Y); dims=1)
    p = cat(p, get_params(CH.CL_XY); dims=1)
    p = cat(p, get_params(CH.C_X); dims=1)
    p = cat(p, get_params(CH.C_Y); dims=1)
end