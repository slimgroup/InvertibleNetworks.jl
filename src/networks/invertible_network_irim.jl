# Invertible network layer from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export NetworkLoop

"""
    L = NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1) (2D)

    L = NetworkLoop(nx, ny, nz, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1) (3D)

 Create an invertibel recurrent inference machine (i-RIM) consisting of an unrooled loop
 for a given number of iterations.

 *Input*: 
 
 - `nx`, `ny`, `nz`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensor
 
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
struct NetworkLoop <: InvertibleNetwork
    L::Union{AbstractArray{CouplingLayerIRIM, 1}, AbstractArray{CouplingLayerHINT}}
    AN::AbstractArray{ActNorm, 1}
    Ψ::Function
end

@Flux.functor NetworkLoop

# 2D Constructor
function NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, type="additive")
    
    if type == "additive"
        L = Array{CouplingLayerIRIM}(undef, maxiter)
    elseif type == "HINT"
        L = Array{CouplingLayerHINT}(undef, maxiter)
    end

    AN = Array{ActNorm}(undef, maxiter)
    for j=1:maxiter
        if type == "additive"
            L[j] = CouplingLayerIRIM(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)
        elseif type == "HINT"
            L[j] = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; logdet=false, permute="both", k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)
        end
        AN[j] = ActNorm(1)
    end
    
    return NetworkLoop(L, AN, Ψ)
end

# 3D Constructor
function NetworkLoop(nx, ny, nz, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, type="additive")
    
    if type == "additive"
        L = Array{CouplingLayerIRIM}(undef, maxiter)
    elseif type == "HINT"
        L = Array{CouplingLayerHINT}(undef, maxiter)
    end
    AN = Array{ActNorm}(undef, maxiter)
    for j=1:maxiter
        if type == "additive"
            L[j] = CouplingLayerIRIM(nx, ny, nz, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)
        elseif type == "HINT"
            L[j] = CouplingLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize; logdet=false, 
                permute="both", k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)
        end
        AN[j] = ActNorm(1)
    end
    
    return NetworkLoop(L, AN, Ψ)
end

# 2D Forward loop: Input (η, s), Output (η, s)
function forward(η::AbstractArray{Float32, N}, s::AbstractArray{Float32, N}, d, J, UL::NetworkLoop) where N

    # Dimensions
    n_in = size(s, N-1) + 1
    batchsize = size(s)[end]
    nn = size(s)[1:N-2]
    maxiter = length(UL.L)
    N0 = cuzeros(η, nn..., n_in-2, batchsize)

    for j=1:maxiter
        g = J'*(J*reshape(UL.Ψ(η), :, batchsize) - reshape(d, :, batchsize))
        g = reshape(g, nn..., 1, batchsize)
        gn = UL.AN[j].forward(g)   # normalize
        s_ = s + tensor_cat(gn, N0)

        ηs = UL.L[j].forward(tensor_cat(η, s_))
        η, s = tensor_split(ηs; split_index=1)
    end
    return η, s
end

# 2D Inverse loop: Input (η, s), Output (η, s)
function inverse(η::AbstractArray{Float32, N}, s::AbstractArray{Float32, N}, d, J, UL::NetworkLoop) where N

    # Dimensions
    n_in = size(s, N-1) + 1
    batchsize = size(s)[end]
    nn = size(s)[1:N-2]
    maxiter = length(UL.L)

    N0 = cuzeros(η, nn..., n_in-2, batchsize)

    for j=maxiter:-1:1
        ηs_ = UL.L[j].inverse(tensor_cat(η, s))
        η, s_ = tensor_split(ηs_; split_index=1)

        g = J'*(J*reshape(UL.Ψ(η), :, batchsize) - reshape(d, :, batchsize))
        g = reshape(g, nn..., 1, batchsize)
        gn = UL.AN[j].forward(g)   # normalize
        s = s_ - tensor_cat(gn, N0)
    end
    return η, s
end

# 2D Backward loop: Input (Δη, Δs, η, s), Output (Δη, Δs, η, s)
function backward(Δη::AbstractArray{Float32, N}, Δs::AbstractArray{Float32, N}, 
    η::AbstractArray{Float32, N}, s::AbstractArray{Float32, N}, d, J, UL::NetworkLoop; set_grad::Bool=true) where N

    # Dimensions
    n_in = size(s, N-1) + 1
    batchsize = size(s)[end]
    nn = size(s)[1:N-2]
    maxiter = length(UL.L)

    N0 = cuzeros(Δη, nn..., n_in-2, batchsize)
    typeof(Δs) == Float32 && (Δs = 0f0.*s)  # make Δs zero tensor

    # Initialize net parameters
    set_grad && (Δθ = Array{Parameter, 1}(undef, 0))

    for j = maxiter:-1:1
        if set_grad
            Δηs_, ηs_ = UL.L[j].backward(tensor_cat(Δη, Δs), tensor_cat(η, s))
        else
            Δηs_, Δθ_L, ηs_ = UL.L[j].backward(tensor_cat(Δη, Δs), tensor_cat(η, s); set_grad=set_grad)
            push!(Δθ, Δθ_L)
        end

        # Inverse pass
        η, s_ = tensor_split(ηs_; split_index=1)
        g = J'*(J*reshape(UL.Ψ(η), :, batchsize) - reshape(d, :, batchsize))
        g = reshape(g, nn..., 1, batchsize)
        gn = UL.AN[j].forward(g)   # normalize
        s = s_ - tensor_cat(gn, N0)

        # Gradients
        Δs2, Δs = tensor_split(Δηs_; split_index=1)
        Δgn = tensor_split(Δs; split_index=1)[1]
        Δg = UL.AN[j].backward(Δgn, gn)[1]
        Δη = reshape(J'*J*reshape(Δg, :, batchsize), nn..., 1, batchsize) + Δs2
    end
    set_grad ? (return Δη, Δs, η, s) : (Δη, Δs, Δθ, η, s)
end

## Jacobian-related utils
function jacobian(η::AbstractArray{Float32, 5}, s::AbstractArray{Float32, 5}, d, J, UL::NetworkLoop)
    throw(ArgumentError("Jacobian for NetworkLoop not yet implemented"))
end

adjointJacobian(Δη::AbstractArray{Float32, N}, Δs::AbstractArray{Float32, N}, 
η::AbstractArray{Float32, N}, s::AbstractArray{Float32, N}, d, J, UL::NetworkLoop; set_grad::Bool=true) where N = backward(Δη, Δs, η, s, d, J, UL; set_grad=false)


## Other utils
# Clear gradients
function clear_grad!(UL::NetworkLoop)
    maxiter = length(UL.L)
    for j=1:maxiter
        clear_grad!(UL.L[j])
        clear_grad!(UL.AN[j])
        UL.AN[j].s.data = nothing
        UL.AN[j].b.data = nothing
    end
end

# Get parameters (do not update actnorm weights)
function get_params(UL::NetworkLoop)
    maxiter = length(UL.L)
    p = get_params(UL.L[1])
    if maxiter > 1
        for j=2:maxiter
            p = cat(p, get_params(UL.L[j]); dims=1)
        end
    end
    return p
end
