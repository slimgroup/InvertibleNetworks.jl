# Invertible HINT coupling layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerHINT

"""
    CL = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=3, p1=1, p2=0)

 Create a recursive HINT-style invertible layer based on coupling blocks. 

 *Input*: 

 - `nx, ny`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 *Output*:
 
 - `H`: Recursive invertible HINT coupling layer.

 *Usage:*

 - Forward mode: `Y = H.forward(X)`

 - Inverse mode: `X = H.inverse(Y)`

 - Backward mode: `ΔX, X = H.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `H` itself

 - Trainable parameters in coupling layers `H.CL`

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerHINT <: NeuralNetLayer
    CL::Array{CouplingLayerBasic, 1}
    forward::Function
    inverse::Function
    backward::Function
end

# Get network depth
function get_depth(n_in)
    count = 0
    nc = n_in
    while nc > 4
        nc /= 2
        count += 1
    end
    return count +1
end

# Constructor from input dimensions
function CouplingLayerHINT(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; k1=4, k2=3, p1=0, p2=1)

    # Create basic coupling layers
    n = get_depth(n_in)
    CL = Array{CouplingLayerBasic}(undef, n) 
    for j=1:n
        CL[j] = CouplingLayerBasic(nx, ny, Int(n_in/2^j), n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2)
    end
    
    return CouplingLayerHINT(CL,
        X -> forward_hint(X, CL),
        Y -> inverse_hint(Y, CL),
        (ΔY, Y) -> backward_hint(ΔY, Y, CL)
        )
end

function forward_hint(X, CL; scale=1)
    Xa, Xb = tensor_split(X)
    if size(X, 3) > 4
        Ya = forward_hint(Xa, CL; scale=scale+1)
        Yb = CL[scale].forward(forward_hint(Xb, CL; scale=scale+1), Xa)[1]
    else
        Ya = copy(Xa)
        Yb = CL[scale].forward(Xb, Xa)[1]
    end
    Y = tensor_cat(Ya, Yb)
    return Y
end

function inverse_hint(Y, CL; scale=1)
    Ya, Yb = tensor_split(Y)
    if size(Y, 3) > 4
        Xa = inverse_hint(Ya, CL; scale=scale+1)
        Xb = inverse_hint(CL[scale].inverse(Yb, Xa)[1], CL; scale=scale+1)
    else
        Xa = copy(Ya)
        Xb = CL[scale].inverse(Yb, Ya)[1]
    end
    X = tensor_cat(Xa, Xb)
    return X
end

backward_hint(Y_tuple::Tuple, CL; scale=1) = backward_hint(Y_tuple[1], Y_tuple[2], CL; scale=scale)

function backward_hint(ΔY, Y, CL; scale=1)
    Ya, Yb = tensor_split(Y)
    ΔYa, ΔYb = tensor_split(ΔY)
    if size(Y, 3) > 4
        ΔXa, Xa = backward_hint(ΔYa, Ya, CL; scale=scale+1)
        ΔXb, Xb = backward_hint(CL[scale].backward(ΔYb, ΔXa, Yb, Xa)[[1,3]], CL; scale=scale+1)
    else
        Xa = copy(Ya)
        ΔXa = copy(ΔYa)
        ΔXb, Xb = CL[scale].backward(ΔYb, ΔYa, Yb, Ya)[[1,3]]
    end
    ΔX = tensor_cat(ΔXa, ΔXb)
    X = tensor_cat(Xa, Xb)
    return ΔX, X
end

# Clear gradients
function clear_grad!(H::CouplingLayerHINT)
    for j=1:length(H.CL)
        clear_grad!(H.CL[j])
    end
end

# Get parameters
function get_params(H::CouplingLayerHINT)
    nlayers = length(H.CL)
    p = get_params(H.CL[1])
    if nlayers > 1
        for j=2:nlayers
            p = cat(p, get_params(H.CL[j]); dims=1)
        end
    end
    return p
end

# TO DO: logdet
