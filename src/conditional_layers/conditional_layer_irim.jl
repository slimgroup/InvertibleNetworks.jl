# Invertible conditional HINT layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalLayerIRIM

"""
    CI = ConditionalLayerIRIM(nx1, nx2, nx_in, nx_hidden, ny1, ny2, ny_in, ny_hidden, batchsize, Op; k1=1, k2=3, p1=1, p2=0)

 Create a conditional i-RIM layer based on the HINT architecture.
 
 *Input*: 

 - `nx1`, `nx2`: spatial dimensions of both `X`

 - `nx_in`, `nx_hidden`: number of input and hidden channels of `X`

 - `ny1`, `ny2`: spatial dimensions of both `Y`
 
 - `ny_in`, `ny_hidden`: number of input and hidden channels of `Y`

 - `Op`: Linear forward modeling operator

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 *Output*:
 
 - `CI`: Conditional HINT coupling layer.

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CI.forward_X(X, Y)`

 - Inverse mode: `X, Y = CI.inverse(Zx, Zy)`

 - Backward mode: `ΔX, ΔY, X, Y = CI.backward(ΔZx, ΔZy, Zx, Zy)`

 - Forward mode Y: `Zy = CI.forward_Y(Y)`

 - Inverse mode Y: `Y = CI.inverse(Zy)`

 *Trainable parameters:*

 - None in `CI` itself

 - Trainable parameters in coupling layers `CI.CL_X`, `CI.CL_Y`, `CI.CL_XY` and in
   permutation layers `CI.C_X` and `CI.C_Y`.

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalLayerIRIM <: NeuralNetLayer
    CL_X::CouplingLayerBasic
    CL_Y::CouplingLayerBasic
    CL_XY::NetworkLoop
    C_X::Conv1x1
    C_Y::Conv1x1
    Op::Union{AbstractMatrix, Any}
    forward::Function
    inverse::Function
    backward::Function
    forward_Y::Function
    inverse_Y::Function
end

# Constructor from input dimensions
function ConditionalLayerIRIM(nx1::Int64, nx2::Int64, nx_in::Int64, nx_hidden::Int64, ny1::Int64, ny2::Int64, ny_in::Int64, ny_hidden::Int64,
    batchsize::Int64, Op::Union{AbstractMatrix, Any}; k1=4, k2=3, p1=0, p2=1)

    # Create basic coupling layers
    CL_X = CouplingLayerBasic(nx1, nx2, Int(nx_in/2), nx_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true)
    CL_Y = CouplingLayerBasic(Int(ny1/2), Int(ny2/2), Int(ny_in*2), ny_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, logdet=true)
    CL_XY = NetworkLoop(nx1, nx2, nx_in, nx_hidden, batchsize, 1, identity; k1=k1, k2=k2)
    
    # Permutation using 1x1 convolution
    C_X = Conv1x1(nx_in)
    C_Y = Conv1x1(Int(ny_in*4))

    return ConditionalLayerIRIM(CL_X, CL_Y, CL_XY, C_X, C_Y, Op,
        (X, Y) -> forward_cond_irim(X, Y, CL_X, CL_Y, CL_XY, C_X, C_Y, Op),
        (Zx, Zy) -> inverse_cond_irim(Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y, Op),
        (ΔZx, ΔZy, Zx, Zy) -> backward_cond_irim(ΔZx, ΔZy, Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y, Op),
        Y -> forward_cond_irim_Y(Y, CL_Y, C_Y),
        Zy -> inverse_cond_irim_Y(Zy, CL_Y, C_Y)
        )
end

function forward_cond_irim(X, Y, CL_X, CL_Y, CL_XY, C_X, C_Y, Op)

    # Y-lane: coupling
    Ys = wavelet_squeeze(Y)
    Yp = C_Y.forward(Ys)
    Ya, Yb = tensor_split(Yp)
    Ya, Yb, logdet2 = CL_Y.forward(Ya, Yb)
    Zy = tensor_cat(Ya, Yb)
    Zy = wavelet_unsqueeze(Zy)

    # X-lane: coupling
    Xp = C_X.forward(X)
    Xa, Xb = tensor_split(Xp)
    Xa, Xb, logdet1 = CL_X.forward(Xa, Xb)
    X = tensor_cat(Xa, Xb)

    # X-lane: i-RIM iteration
    η_in = X[:, :, 1:1, :]
    s_in = X[:, :, 2:end, :]
    η_out, s_out = CL_XY.forward(η_in, s_in, Op, reshape(Y, :, size(Y, 4)))
    Zx = tensor_cat(η_out, s_out)

    logdet = logdet1 + logdet2
    return Zx, Zy, logdet
end

function inverse_cond_irim(Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y, Op)

    # Y-lane
    Zy = wavelet_squeeze(Zy)
    Ya, Yb = tensor_split(Zy)
    Ya, Yb = CL_Y.inverse(Ya, Yb)
    Yp = tensor_cat(Ya, Yb)
    Ys = C_Y.inverse(Yp)
    Y = wavelet_unsqueeze(Ys)

    # X-lane: i-RIM
    η_out = Zx[:, :, 1:1, :]
    s_out = Zx[:, :, 2:end, :]
    η_in, s_in = CL_XY.inverse(η_out, s_out, Op, reshape(Y, :, size(Y, 4)))
    X = tensor_cat(η_in, s_in)

    # X-lane: coupling layer
    Xa, Xb = tensor_split(X)
    Xa, Xb = CL_X.inverse(Xa, Xb)
    Xp = tensor_cat(Xa, Xb)
    X = C_X.inverse(Xp)

    return X, Y
end

function backward_cond_irim(ΔZx, ΔZy, Zx, Zy, CL_X, CL_Y, CL_XY, C_X, C_Y, Op)

    # Y-lane
    ΔZy = wavelet_squeeze(ΔZy)
    Zy = wavelet_squeeze(Zy)
    ΔYa, ΔYb = tensor_split(ΔZy)
    Ya, Yb = tensor_split(Zy)
    ΔYa, ΔYb, Ya, Yb = CL_Y.backward(ΔYa, ΔYb, Ya, Yb)
    ΔYp = tensor_cat(ΔYa, ΔYb)
    Yp = tensor_cat(Ya, Yb)
    ΔYs, Ys = C_Y.inverse((ΔYp, Yp))
    Y = wavelet_unsqueeze(Ys)
    ΔY = wavelet_unsqueeze(ΔYs)

    # X-lane: i-RIM
    Δη_out = ΔZx[:, :, 1:1, :]
    Δs_out = ΔZx[:, :, 2:end, :]
    η_out = Zx[:, :, 1:1, :]
    s_out = Zx[:, :, 2:end, :]
    Δη_in, Δs_in, η_in, s_in = CL_XY.backward(Δη_out, Δs_out, η_out, s_out,Op, reshape(Y, :, size(Y, 4)))
    ΔX = tensor_cat(Δη_in, Δs_in)
    X = tensor_cat(η_in, s_in)

    # X-lane: coupling layer
    ΔXa, ΔXb = tensor_split(ΔX)
    Xa, Xb = tensor_split(X)
    ΔXa, ΔXb, Xa, Xb = CL_X.backward(ΔXa, ΔXb, Xa, Xb)
    ΔXp = tensor_cat(ΔXa, ΔXb)
    Xp = tensor_cat(Xa, Xb)
    ΔX, X = C_X.inverse((ΔXp, Xp))

    return ΔX, ΔY, X, Y
end

function forward_cond_irim_Y(Y, CL_Y, C_Y)
    Ys = wavelet_squeeze(Y)
    Yp = C_Y.forward(Ys)
    Ya, Yb = tensor_split(Yp)
    Ya, Yb, logdet2 = CL_Y.forward(Ya, Yb)
    Zy = tensor_cat(Ya, Yb)
    Zy = wavelet_unsqueeze(Zy)
    return Zy
end

function inverse_cond_irim_Y(Zy, CL_Y, C_Y)
    Zy = wavelet_squeeze(Zy)
    Ya, Yb = tensor_split(Zy)
    Ya, Yb = CL_Y.inverse(Ya, Yb)
    Yp = tensor_cat(Ya, Yb)
    Ys = C_Y.inverse(Yp)
    Y = wavelet_unsqueeze(Ys)
    return Y
end

# Clear gradients
function clear_grad!(CI::ConditionalLayerIRIM)
    clear_grad!(CI.CL_X)
    clear_grad!(CI.CL_Y)
    clear_grad!(CI.CL_XY)
    clear_grad!(CI.C_X)
    clear_grad!(CI.C_Y)
end

# Get parameters
function get_params(CI::ConditionalLayerIRIM)
    p = get_params(CI.CL_X)
    p = cat(p, get_params(CI.CL_Y); dims=1)
    p = cat(p, get_params(CI.CL_XY); dims=1)
    p = cat(p, get_params(CI.C_X); dims=1)
    p = cat(p, get_params(CI.C_Y); dims=1)
end
