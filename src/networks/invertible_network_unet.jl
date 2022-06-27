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
end

@Flux.functor NetworkUNET

# 2D Constructor
function NetworkUNET(n_in::Int64, n_hiddens::Array{Int64,1}, ds::Array{Int64,1}; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2)

    L = CouplingLayerIRIM(n_in, n_hiddens, ds; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, ndims=ndims)
    AN = ActNorm(1) # Only for 1 channel gradient
    n_mem = n_in 
    return NetworkUNET(L, AN, n_mem)
end

# 3D Constructor
NetworkUNET3D(args...; kw...) = NetworkUNET(args...; kw..., ndims=3)

# 2D Forward loop: Input (η), Output (η)
function forward(g::AbstractArray{T, N}, UL::NetworkUNET) where {T, N}

    # Dimensions
    batchsize = size(g)[end]
    nn = size(g)[1:N-2]
    inds_c = [i!=(N-1) ? Colon() : 1 for i=1:N]

    # Forward pass
    sg = cuzeros(g, nn..., UL.n_mem, batchsize)
    gn = UL.AN.forward(g)   # normalize
    sg[inds_c...] = gn # gradient in first channel

    sg_ = UL.L.forward(sg)

    return sg_
end

# 2D Inverse loop: Input (η), Output (η)
function inverse(η::AbstractArray{T, N}, s::AbstractArray{T, N}, g::AbstractArray{T, N}, UL::NetworkUNET) where {T, N}

    # Inverse pass
    ηs_ = UL.L.inverse(tensor_cat(η, s))
    η, s_ = tensor_split(ηs_; split_index=1)
    
    return η
end

# 2D Backward loop: Input (Δη, Δs, η, s), Output (Δη, Δs, η, s)
function backward(Δsg::AbstractArray{T, N}, 
    sg::AbstractArray{T, N}, UL::NetworkUNET; set_grad::Bool=true) where {T, N}

    # Backwards pass
    Δηs_, ηs_ = UL.L.backward(Δsg, sg)

    s, g  = tensor_split(ηs_; split_index=1)
    Δs, Δg = tensor_split(Δηs_; split_index=1)

    gn  = UL.AN.forward(g)   # normalize
    Δgn = tensor_split(Δs; split_index=1)[1]
    Δg  = UL.AN.backward(Δgn, gn)[1]

    return Δg, g
end

## Jacobian-related utils
jacobian(::AbstractArray{T, 5}, ::AbstractArray{T, 5},  UL::NetworkUNET) where T = throw(ArgumentError("Jacobian for NetworkUNET not yet implemented"))

adjointJacobian(Δη::AbstractArray{T, N}, η::AbstractArray{T, N}, s::AbstractArray{T, N}, UL::NetworkUNET;
                set_grad::Bool=true) where {T, N} = throw(ArgumentError("Jacobian for NetworkUNET not yet implemented"))

