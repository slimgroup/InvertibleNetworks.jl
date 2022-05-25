# Invertible network layer from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export NetworkPhinj

"""
    L = NetworkLoop(n_in, n_hidden, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2) (2D)

    L = NetworkLoop3D(n_in, n_hidden, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1) (3D)

 Create an invertibel recurrent inference machine (i-RIM) consisting of an unrooled loop
 for a given number of iterations.

 *Input*: 
 
 - 'n_in': number of total channels in state variable. Needs to be even for couplying layer splits and also at least 2

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
struct NetworkPhinj <: InvertibleNetwork
    L::AbstractArray{CouplingLayerIRIM, 1}
    AN::AbstractArray{ActNorm, 1}
    n_chan::Int64
    n_pad::Int64
    squeezer::Squeezer
end

@Flux.functor NetworkPhinj

# 2D Constructor
function NetworkPhinj(n_pad::Int64, n_hidden::Int64, maxiter::Int64, n_chan::Int64; squeezer::Squeezer=ShuffleLayer(), GALU=true, k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2)
    #n_add
    #how many channels to add at each layer
    n_in = n_chan
    L = Array{CouplingLayerIRIM}(undef, maxiter)
    AN = Array{ActNorm}(undef, maxiter)
    for j=1:maxiter
        #n_in *= 4
        L[j] = CouplingLayerIRIM(n_in*4, n_hidden; GALU=true, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, ndims=ndims)
        AN[j] = ActNorm(n_chan)
        n_in += n_pad
    end
    
    return NetworkPhinj(L, AN,  n_chan, n_pad,squeezer)
end



# 2D Forward loop: Input (η, s), Output (η, s)
function forward(d::AbstractArray, J, nn, UL::NetworkPhinj) where {T, N}

    batchsize = size(d)[end]
    maxiter = length(UL.L)

    gs = cuzeros(d, nn..., 0, batchsize) 
    η  = cuzeros(d, nn..., 0, batchsize) #initial reconstruction η_0 is NOTHING 

    for j=1:maxiter
        #println("\n NOW AT j=$(j)")
        #*reshape(η, :, batchsize)

        if j == 1
            g = J'*(-reshape(d, :, batchsize))
        else
            g = J'*(J*reshape(η, :, batchsize) - reshape(d, :, batchsize))
        end

        g = reshape(g, nn..., UL.n_chan, batchsize)
        gn = UL.AN[j].forward(g)   # normalize

        if j == 1
            gs = tensor_cat(gs, gn) # pad the gradient at each iteration
        else
            gs = tensor_cat(gs, repeat(gn,1,1,UL.n_pad,1)) # pad the gradient at each iteration
        end

        ηs = tensor_cat(η, gs)
        ηs = UL.squeezer.forward(ηs) #most images will have 1 channel in gradient so increase channel length
        #println("total size of state is $(size(ηs))")
        ηs = UL.L[j].forward(ηs)
        ηs = UL.squeezer.inverse(ηs)
        #println("total size of state is $(size(ηs))")

        η, gs = tensor_split(ηs; split_index=UL.n_chan)
    end
    return η, gs
end

# 2D Inverse loop: Input (η, s), Output (η, s)
function inverse(η::AbstractArray{T, N}, gs::AbstractArray{T, N}, UL::NetworkPhinj) where {T, N}

    batchsize = size(η)[end]
    maxiter = length(UL.L)

    ηs = tensor_cat(η, gs) 

    for j=maxiter:-1:1
        #println("\n NOW AT j=$(j)")
        
        #println("total size of state is $(size(ηs))")
        ηs = UL.squeezer.forward(ηs)
        ηs = UL.L[j].inverse(ηs)
        ηs = UL.squeezer.inverse(ηs)
        if(j > 1)
            ηs, _ = tensor_split(ηs; split_index=size(ηs)[end-1]-UL.n_pad)
        end
    end
    ηs = UL.AN[1].inverse(ηs) 

    #not really important but returns the first gradient
    # Just wrote this to get proper structure of backward
    return ηs
end

# 2D Backward loop: Input (Δη, Δs, η, s), Output (Δη, Δs, η, s)
function backward(Δη::AbstractArray{T, N}, η::AbstractArray{T, N}, gs::AbstractArray{T, N}, UL::NetworkPhinj;) where {T, N}

    batchsize = size(Δη)[end]
    maxiter   = length(UL.L)

    Δgs = 0f0 .* gs;
    Δηs = tensor_cat(Δη, Δgs) ;
    ηs = tensor_cat(η, gs) ;

    for j=maxiter:-1:1
        #println("\n NOW AT j=$(j)")
        #*reshape(η, :, batchsize)

        #println("total size of state is $(size(ηs))")
        Δηs = UL.squeezer.forward(Δηs)
        ηs = UL.squeezer.forward(ηs)

        Δηs, ηs = UL.L[j].backward(Δηs, ηs)

        Δηs = UL.squeezer.inverse(Δηs)
        ηs = UL.squeezer.inverse(ηs)
        if(j > 1)
            Δηs, _ = tensor_split(Δηs; split_index=size(Δηs)[end-1]-UL.n_pad)
            ηs, _ = tensor_split(ηs; split_index=size(ηs)[end-1]-UL.n_pad)
        end
    end
    Δηs = UL.AN[1].backward(Δηs, ηs) 
end

## Jacobian-related utils
jacobian(::AbstractArray{T, 5}, ::AbstractArray{T, 5}, d::AbstractArray, J, UL::NetworkPhinj) where T = throw(ArgumentError("Jacobian for NetworkLoop not yet implemented"))

adjointJacobian(Δη::AbstractArray{T, N}, Δs::AbstractArray{T, N}, 
                η::AbstractArray{T, N}, s::AbstractArray{T, N}, d::AbstractArray, J, UL::NetworkPhinj;
                set_grad::Bool=true) where {T, N} =
            backward(Δη, Δs, η, s, d, J, UL; set_grad=false)

# Get parameters (do not update actnorm weights)
function get_params(UL::NetworkPhinj)
    maxiter = length(UL.L)
    p = get_params(UL.L[1])
    if maxiter > 1
        for j=2:maxiter
            p = cat(p, get_params(UL.L[j]); dims=1)
        end
    end
    return p
end

function clear_grad!(UL::NetworkPhinj)
    maxiter = length(UL.L)
    for j=1:maxiter
        clear_grad!(UL.L[j])
        clear_grad!(UL.AN[j])
        UL.AN[j].s.data = nothing
        UL.AN[j].b.data = nothing
    end
end