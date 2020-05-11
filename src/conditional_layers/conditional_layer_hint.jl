# Invertible conditional HINT layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalLayerHINT

"""
    CH = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true) (2D)

    CH = ConditionalLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true) (3D)

 Create a conditional HINT layer based on coupling blocks and 1 level recursion. 

 *Input*: 

 - `nx`, `ny`, `nz`: spatial dimensions of both `X` and `Y`.
 
 - `n_in`, `n_hidden`: number of input and hidden channels of both `X` and `Y`

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

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
end

# 2D Constructor from input dimensions
function ConditionalLayerHINT(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; 
    k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true)

    # Create basic coupling layers
    CL_X = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute="none")
    CL_Y = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute="none")
    CL_YX = CouplingLayerBasic(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)

    # Permutation using 1x1 convolution
    permute == true ? (C_X = Conv1x1(n_in)) : (C_X = nothing)
    permute == true ? (C_Y = Conv1x1(n_in)) : (C_Y = nothing)

    return ConditionalLayerHINT(CL_X, CL_Y, CL_YX, C_X, C_Y)
end

# 3D Constructor from input dimensions
function ConditionalLayerHINT(nx::Int64, ny::Int64, nz:: Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; 
    k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true)

    # Create basic coupling layers
    CL_X = CouplingLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute="none")
    CL_Y = CouplingLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute="none")
    CL_YX = CouplingLayerBasic(nx, ny, nz, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)

    # Permutation using 1x1 convolution
    permute == true ? (C_X = Conv1x1(n_in)) : (C_X = nothing)
    permute == true ? (C_Y = Conv1x1(n_in)) : (C_Y = nothing)

    return ConditionalLayerHINT(CL_X, CL_Y, CL_YX, C_X, C_Y)
end

function forward(X, Y, CH::ConditionalLayerHINT)

    # Y-lane
    ~isnothing(CH.C_Y) ? (Yp = CH.C_Y.forward(Y)) : (Yp = copy(Y))
    Zy, logdet2 = CH.CL_Y.forward(Yp)

    # X-lane: coupling layer
    ~isnothing(CH.C_X) ? (Xp =CH. C_X.forward(X)) : (Xp = copy(X))
    X, logdet1 = CH.CL_X.forward(Xp)

    # X-lane: conditional layer
    Zx, logdet3 = CH.CL_YX.forward(Yp, X)[2:3]
    logdet = logdet1 + logdet2 + logdet3

    return Zx, Zy, logdet
end

function inverse(Zx, Zy, CH::ConditionalLayerHINT)

    # Y-lane
    Yp = CH.CL_Y.inverse(Zy)
    ~isnothing(CH.C_Y) ? (Y = CH.C_Y.inverse(Yp)) : (Y = copy(Yp))

    # X-lane: conditional layer
    YZ = tensor_cat(Yp, Zx)
    X = CH.CL_YX.inverse(Yp, Zx)[2]

    # X-lane: coupling layer
    Xp = CH.CL_X.inverse(X)
    ~isnothing(CH.C_X) ? (X = CH.C_X.inverse(Xp)) : (X = copy(Xp))

    return X, Y
end

function backward(ΔZx, ΔZy, Zx, Zy, CH::ConditionalLayerHINT)

    # Y-lane
    ΔYp, Yp = CH.CL_Y.backward(ΔZy, Zy)

    # X-lane: conditional layer
    ΔYp_, ΔX, X = CH.CL_YX.backward(ΔYp.*0f0, ΔZx, Yp, Zx)[[1,2,4]]
    ΔYp += ΔYp_

    # X-lane: coupling layer
    ΔXp, Xp = CH.CL_X.backward(ΔX, X)

    # 1x1 Convolutions
    if isnothing(CH.C_X) || isnothing(CH.C_Y)
        ΔX = copy(ΔXp); X = copy(Xp)
        ΔY = copy(ΔYp); Y = copy(Yp)
    else
        ΔX, X = CH.C_X.inverse((ΔXp, Xp))
        ΔY, Y = CH.C_Y.inverse((ΔYp, Yp))
    end
    return ΔX, ΔY, X, Y
end

function forward_Y(Y, CH::ConditionalLayerHINT)
    ~isnothing(CH.C_Y) ? (Yp = CH.C_Y.forward(Y)) : (Yp = copy(Y))
    Zy = CH.CL_Y.forward(Yp)[1]
    return Zy

end

function inverse_Y(Zy, CH::ConditionalLayerHINT)
    Yp = CH.CL_Y.inverse(Zy)
    ~isnothing(CH.C_Y) ? (Y = CH.C_Y.inverse(Yp)) : (Y = copy(Yp))
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