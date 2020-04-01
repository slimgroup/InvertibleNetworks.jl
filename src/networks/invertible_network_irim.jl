# Invertible network layer from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export NetworkLoop

"""
    L = NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3)

 Create an invertibel recurrent inference machine (i-RIM) consisting of an unrooled loop
 for a given number of iterations.

 *Input*: 
 
 - `nx`, `ny`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensor
 
 - `n_hidden`: number of hidden units in residual blocks

 - `maxiter`: number unrolled loop iterations

 - `Ψ`: link function

 - `k1`, `k2`: stencil sizes for convolutions in the residual blocks. The first convolution 
   uses a stencil of size and stride `k1`, thereby downsampling the input. The second 
   convolutions uses a stencil of size `k2`. The last layer uses a stencil of size and stride `k1`,
   but performs the transpose operation of the first convolution, thus upsampling the output to 
   the original input size.

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
    L::Array{CouplingLayerIRIM, 1}
    AN::Array{ActNorm, 1}
    Ψ::Function
    forward::Function
    inverse::Function
    backward::Function
end

function NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1)
    
    L = Array{CouplingLayerIRIM}(undef, maxiter)
    AN = Array{ActNorm}(undef, maxiter)
    for j=1:maxiter
        L[j] = CouplingLayerIRIM(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1)
        AN[j] = ActNorm(1)
    end
    
    return NetworkLoop(L, AN, Ψ,
        (η, s, d, J) -> loop_forward(η, s, d, L, AN, J, Ψ),
        (η, s, d, J) -> loop_inverse(η, s, d, L, AN, J, Ψ),
        (Δη, Δs, η, s, d, J) -> loop_backward(Δη, Δs, η, s, d, L, AN, J, Ψ)
        )
end

# Forward loop: Input (η, s), Output (η, s)
function loop_forward(η, s, d, L, AN, J, Ψ)

    # Dimensions
    nx, ny, n_s, batchsize = size(s)
    n_in = n_s + 1
    maxiter = length(L)
    N = zeros(Float32, nx, ny, n_in-2, batchsize)

    for j=1:maxiter
        g = J'*(J*reshape(Ψ(η), :, batchsize) - reshape(d, :, batchsize))
        g = reshape(g, nx, ny, 1, batchsize)
        gn = AN[j].forward(g)   # normalize
        s_ = s + tensor_cat(gn, N)

        ηs = L[j].forward(cat(η, s_; dims=3))
        η = ηs[:, :, 1:1, :]
        s = ηs[:, :, 2:end, :]
    end
    return η, s
end

# Inverse loop: Input (η, s), Output (η, s)
function loop_inverse(η, s, d, L, AN, J, Ψ)

    # Dimensions
    nx, ny, n_s, batchsize = size(s)
    n_in = n_s + 1
    maxiter = length(L)
    N = zeros(Float32, nx, ny, n_in-2, batchsize)

    for j=maxiter:-1:1
        ηs_ = L[j].inverse(tensor_cat(η, s))
        η = ηs_[:, :, 1:1, :]
        s_ = ηs_[:, :, 2:end, :]

        g = J'*(J*reshape(Ψ(η), :, batchsize) - reshape(d, :, batchsize))
        g = reshape(g, nx, ny, 1, batchsize)
        gn = AN[j].forward(g)   # normalize
        s = s_ - tensor_cat(gn, N)
    end
    return η, s
end

# Backward loop: Input (Δη, Δs, η, s), Output (Δη, Δs, η, s)
function loop_backward(Δη, Δs, η, s, d, L, AN, J, Ψ)

    # Dimensions
    nx, ny, n_s, batchsize = size(s)
    n_in = n_s + 1
    maxiter = length(L)
    N = zeros(Float32, nx, ny, n_in-2, batchsize)
    typeof(Δs) == Float32 && (Δs = 0f0.*s)  # make Δs zero tensor

    for j=maxiter:-1:1
        Δηs_, ηs_ = L[j].backward(tensor_cat(Δη, Δs), tensor_cat(η, s))

        # Inverse pass
        η = ηs_[:, :, 1:1, :]
        s_ = ηs_[:, :, 2:end, :]
        g = J'*(J*reshape(Ψ(η), :, batchsize) - reshape(d, :, batchsize))
        g = reshape(g, nx, ny, 1, batchsize)
        gn = AN[j].forward(g)   # normalize
        s = s_ - tensor_cat(gn, N)

        # Gradients
        Δs = Δηs_[:, :, 2:end, :]
        Δgn = Δs[:, :, 1:1, :]
        Δg = AN[j].backward(Δgn, gn)[1]
        Δη = reshape(J'*J*reshape(Δg, :, batchsize), nx, ny, 1, batchsize) + Δηs_[:, :, 1:1, :]
    end
    return Δη, Δs, η, s
end

# Clear gradients
function clear_grad!(UL::NetworkLoop)
    maxiter = length(UL.L)
    for j=1:maxiter
        clear_grad!(UL.L[j].C)
        clear_grad!(UL.L[j].RB)
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
