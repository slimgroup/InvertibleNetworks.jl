# Invertible network layer from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export NetworkUNET, NetworkUNET3D

"""
    L = NetworkUNET(n_in, n_hidden, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2) (2D)

    L = NetworkUNET3D(n_in, n_hidden, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1) (3D)

 Create an invertibel recurrent inference machine (i-RIM) consisting of an unrooled loop
 for a given number of iterations.

 *Input*: 
 
 - 'n_in': number of input channels

 - `n_hidden`: number of hidden units in residual blocks

 - `maxiter`: number unrolled loop iterations

 - `Ψ`: link function

 - `k1`, `k2`: stencil sizes for convolutions in the residual blocks. The first convolution 
   uses a stencil of size and stride `k1`, thereby downsampling the input. The second 
   convolutions uses a stencil of size `k2`. The last layer uses a stencil of size and stride `k1`,
   but performs the transpose operation of the first convolution, thus upsampling the output to 
   the original input size.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`) in
   residual block

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`) in
   residual block

 - `ndims` : number of dimensions

 *Output*:
 
 - `L`: invertible i-RIM network.

 *Usage:*

 - Forward mode: `η_out, s_out = L.forward(η_in, s_in, d, A)`

 - Inverse mode: `η_in, s_in = L.inverse(η_out, s_out, d, A)`

 - Backward mode: `Δη_in, Δs_in, η_in, s_in = L.backward(Δη_out, Δs_out, η_out, s_out, d, A)`

 *Trainable parameters:*

 - None in `L` itself

 - Trainable parameters in the invertible coupling layers `L.L[i]`, and actnorm layers
   `L.AN[i]`, where `i` ranges from `1` to the number of loop iterations.

 See also: [`CouplingLayerIRIM`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkUNET <: InvertibleNetwork
    L::CouplingLayerIRIM
    AN::ActNorm
    n_mem::Int64
    n_grad::Int64
    early_squeeze
end

@Flux.functor NetworkUNET

# 2D Constructor
function NetworkUNET(n_in::Int64, n_hiddens::Array{Int64,1}, ds::Array{Int64,1}; early_squeeze=false, n_grad=1, k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2)

    n_mem = n_in 
    if early_squeeze
        n_in = 4*n_in
    end
    L = CouplingLayerIRIM(n_in, n_hiddens, ds; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, ndims=ndims)
    AN = ActNorm(n_grad) # Only for 1 channel gradient # try turning off logdet
    
    return NetworkUNET(L, AN, n_mem, n_grad, early_squeeze)
end

# 3D Constructor
NetworkUNET3D(args...; kw...) = NetworkUNET(args...; kw..., ndims=3)

# 2D Forward loop: Input (η), Output (η)
function forward(g::AbstractArray{T, N}, UL::NetworkUNET) where {T, N}

    # Dimensions
    batchsize = size(g)[end]
    nn = size(g)[1:N-2]
   
    # Forward pass
    gs = cuzeros(g, nn..., UL.n_mem, batchsize)
    gn = UL.AN.forward(g)[1]   # normalize
   
    gs[:,:,1:UL.n_grad,:] = gn # gradient in first channel

    if UL.early_squeeze
        gs = squeeze(gs; pattern="checkerboard")
    end
    gs = UL.L.forward(gs)

    if UL.early_squeeze
        gs = unsqueeze(gs; pattern="checkerboard")
    end

    return gs
end

# 2D Inverse loop: Input (η), Output (η)
function inverse(y::AbstractArray{T, N}, UL::NetworkUNET) where {T, N}

    UL.early_squeeze && (y  = squeeze(y; pattern="checkerboard"))
    x = UL.L.inverse(y)
    UL.early_squeeze && (x  = unsqueeze(x; pattern="checkerboard"))

    x, _ = tensor_split(x; split_index=UL.n_grad)

    x  = UL.AN.inverse(x)[1]   # normalize
    return x
end

# 2D Backward loop: Input (Δη, Δs, η, s), Output (Δη, Δs, η, s)
function backward(Δy::AbstractArray{T, N}, 
    y::AbstractArray{T, N}, UL::NetworkUNET; set_grad::Bool=true) where {T, N}

    if UL.early_squeeze
        Δy = squeeze(Δy; pattern="checkerboard")
        y  = squeeze(y; pattern="checkerboard")
    end
    Δx, x = UL.L.backward(Δy, y)
    if UL.early_squeeze
        Δx = unsqueeze(Δx; pattern="checkerboard")
        x  = unsqueeze(x; pattern="checkerboard")
    end

    x,  _ = tensor_split(x;  split_index=UL.n_grad)
    Δx, _ = tensor_split(Δx; split_index=UL.n_grad)

    Δx, x   = UL.AN.backward(Δx, x)

    return Δx, x
end

## Jacobian-related utils
jacobian(::AbstractArray{T, 5}, ::AbstractArray{T, 5},  UL::NetworkUNET) where T = throw(ArgumentError("Jacobian for NetworkUNET not yet implemented"))

adjointJacobian(Δη::AbstractArray{T, N}, η::AbstractArray{T, N}, s::AbstractArray{T, N}, UL::NetworkUNET;
                set_grad::Bool=true) where {T, N} = throw(ArgumentError("Jacobian for NetworkUNET not yet implemented"))
