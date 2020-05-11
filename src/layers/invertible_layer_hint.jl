# Invertible HINT coupling layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerHINT

"""
    H = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; 
        logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (2D)

    H = CouplingLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize; 
        logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (3D)

 Create a recursive HINT-style invertible layer based on coupling blocks. 

 *Input*: 

 - `nx`, `ny`, `nz`: spatial dimensions of input
 
 - `n_in`, `n_hidden`: number of input and hidden channels

 - `logdet`: bool to indicate whether to return the log determinant. Default is `false`.

 - `permute`: string to specify permutation. Options are `"none"`, `"lower"`, `"both"` or `"full"`.

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

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
    CL::AbstractArray{CouplingLayerBasic, 1}
    C::Union{Conv1x1, Nothing}
    logdet::Bool
    forward::Function
    inverse::Function
    backward::Function
end

# Get layer depth for recursion
function get_depth(n_in)
    count = 0
    nc = n_in
    while nc > 4
        nc /= 2
        count += 1
    end
    return count +1
end

# 2D Constructor from input dimensions
function CouplingLayerHINT(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; 
    logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    # Create basic coupling layers
    n = get_depth(n_in)
    CL = Array{CouplingLayerBasic}(undef, n) 
    for j=1:n
        CL[j] = CouplingLayerBasic(nx, ny, Int(n_in/2^j), n_hidden, batchsize; 
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet)
    end

    # Permutation using 1x1 convolution
    if permute == "full" || permute == "both"
        C = Conv1x1(n_in)
    elseif permute == "lower"
        C = Conv1x1(Int(n_in/2))
    else
        C = nothing
    end

    return CouplingLayerHINT(CL, C, logdet,
        X -> forward_hint(X, CL, C; logdet=logdet, permute=permute),
        Y -> inverse_hint(Y, CL, C, permute=permute),
        (ΔY, Y) -> backward_hint(ΔY, Y, CL, C, permute=permute)
        )
end

# 3D Constructor from input dimensions
function CouplingLayerHINT(nx::Int64, ny::Int64, nz::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64; 
    logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    # Create basic coupling layers
    n = get_depth(n_in)
    CL = Array{CouplingLayerBasic}(undef, n) 
    for j=1:n
        CL[j] = CouplingLayerBasic(nx, ny, nz, Int(n_in/2^j), n_hidden, batchsize; 
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet)
    end

    # Permutation using 1x1 convolution
    if permute == "full" || permute == "both"
        C = Conv1x1(n_in)
    elseif permute == "lower"
        C = Conv1x1(Int(n_in/2))
    else
        C = nothing
    end

    return CouplingLayerHINT(CL, C, logdet,
        X -> forward_hint(X, CL, C; logdet=logdet, permute=permute),
        Y -> inverse_hint(Y, CL, C, permute=permute),
        (ΔY, Y) -> backward_hint(ΔY, Y, CL, C, permute=permute)
        )
end

# Input is tensor X
function forward_hint(X, CL, C; scale=1, logdet=false, permute="none")
    if permute == "full" || permute == "both"
        X = C.forward(X)
    end
    Xa, Xb = tensor_split(X)
    permute == "lower" && (Xb = C.forward(Xb))

    recursive = false
    if typeof(X) == Array{Float32, 4} && size(X, 3) > 4
        recursive = true
    elseif typeof(X) == Array{Float32, 5} && size(X, 4) > 4
        recursive = true
    end
    
    if recursive
        # Call function recursively
        Ya, logdet1 = forward_hint(Xa, CL, C; scale=scale+1, logdet=logdet)
        Y_temp, logdet2 = forward_hint(Xb, CL, C; scale=scale+1, logdet=logdet)
        if logdet==false
            Yb = CL[scale].forward(Xa, Y_temp)[2]
            logdet3 = 0f0
        else
            Yb, logdet3 = CL[scale].forward(Xa, Y_temp)[[2,3]]
        end
        logdet_full = logdet1 + logdet2 + logdet3
    else
        # Finest layer
        Ya = copy(Xa)
        if logdet==false
            Yb = CL[scale].forward(Xa, Xb)[2]
            logdet_full = 0f0
        else
            Yb, logdet_full = CL[scale].forward(Xa, Xb)[[2,3]]
        end
    end
    Y = tensor_cat(Ya, Yb)
    permute == "both" && (Y = C.inverse(Y))
    if scale==1 && logdet==false
        return Y
    else
        return Y, logdet_full
    end
end

# Input is tensor Y
function inverse_hint(Y, CL, C; scale=1, permute="none")
    permute == "both" && (Y = C.forward(Y))
    Ya, Yb = tensor_split(Y)
    recursive = false
    if typeof(Y) == Array{Float32, 4} && size(Y, 3) > 4
        recursive = true
    elseif typeof(Y) == Array{Float32, 5} && size(Y, 4) > 4
        recursive = true
    end
    if recursive
        Xa = inverse_hint(Ya, CL, C; scale=scale+1)
        Xb = inverse_hint(CL[scale].inverse(Xa, Yb)[2], CL, C; scale=scale+1)
    else
        Xa = copy(Ya)
        Xb = CL[scale].inverse(Ya, Yb)[2]
    end
    permute == "lower" && (Xb = C.inverse(Xb))
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        X = C.inverse(X)
    end
    return X
end

# Input are two tensors ΔY, Y
function backward_hint(ΔY, Y, CL, C; scale=1, permute="none")
    permute == "both" && ((ΔY, Y) = C.forward((ΔY, Y)))
    Ya, Yb = tensor_split(Y)
    ΔYa, ΔYb = tensor_split(ΔY)
    recursive = false
    if typeof(Y) == Array{Float32, 4} && size(Y, 3) > 4
        recursive = true
    elseif typeof(Y) == Array{Float32, 5} && size(Y, 4) > 4
        recursive = true
    end
    if recursive
        ΔXa, Xa = backward_hint(ΔYa, Ya, CL, C; scale=scale+1)
        ΔXa_temp, ΔXb_temp, X_temp = CL[scale].backward(ΔXa.*0f0, ΔYb, Xa, Yb)[[1,2,4]]
        ΔXb, Xb = backward_hint(ΔXb_temp, X_temp, CL, C; scale=scale+1)
        ΔXa += ΔXa_temp
    else
        Xa = copy(Ya)
        ΔXa = copy(ΔYa)
        ΔXa_, ΔXb, Xb = CL[scale].backward(ΔYa.*0f0, ΔYb, Ya, Yb)[[1,2,4]]
        ΔXa += ΔXa_
    end
    permute == "lower" && ((ΔXb, Xb) = C.inverse((ΔXb, Xb)))
    ΔX = tensor_cat(ΔXa, ΔXb)
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        (ΔX, X) = C.inverse((ΔX, X))
    end
    return ΔX, X
end

# Clear gradients
function clear_grad!(H::CouplingLayerHINT)
    for j=1:length(H.CL)
        clear_grad!(H.CL[j])
    end
    ~isnothing(H.C) && clear_grad!(H.C)
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
    ~isnothing(H.C) && (p = cat(p, get_params(H.C); dims=1))
    return p
end