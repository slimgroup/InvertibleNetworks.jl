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

 - Forward mode: `η_out, s_out = L.forward(η_in, s_in, A, d)`

 - Inverse mode: `η_in, s_in = L.inverse(η_out, s_out, A, d)`

 - Backward mode: `Δη_in, Δs_in, η_in, s_in = L.backward(Δη_out, Δs_out, η_out, s_out, A, d)`

 *Trainable parameters:*

 - None in `L` itself

 - Trainable parameters in the invertible coupling layers `L.L[i]`, 
   where `i` ranges from `1` to the number of loop iterations.

 See also: [`CouplingLayerIRIM`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkLoop <: InvertibleNetwork
    L::Array{CouplingLayerIRIM, 1}
    Ψ::Function
    forward::Function
    inverse::Function
    backward::Function
end

function NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1)
    
    L = Array{CouplingLayerIRIM}(undef, maxiter)
    for j=1:maxiter
        L[j] = CouplingLayerIRIM(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1)
    end
    
    return NetworkLoop(L, Ψ,
        (η, s, J, d) -> loop_forward(η, s, d, L, J, Ψ),
        (η, s, J, d) -> loop_inverse(η, s, d, L, J, Ψ),
        (Δη, Δs, η, s, J, d) -> loop_backward(Δη, Δs, η, s, d, L, J, Ψ)
        )
end

# Forward loop: Input (η, s), Output (η, s)
function loop_forward(η, s, d, L, J, Ψ)

    # Dimensions
    nx, ny, n_s, batchsize = size(s)
    n_in = n_s + 1
    maxiter = length(L)
    N = zeros(Float32, nx, ny, n_in-2, batchsize)

    for j=1:maxiter
        g = J'*(J*reshape(Ψ(η), :, batchsize) - d)
        g = reshape(g/norm(g, Inf), nx, ny, 1, batchsize)
        s_ = s + cat(g, N; dims=3)

        ηs = L[j].forward(cat(η, s_; dims=3))
        η = ηs[:, :, 1:1, :]
        s = ηs[:, :, 2:end, :]
    end
    return η, s
end

# Inverse loop: Input (η, s), Output (η, s)
function loop_inverse(η, s, d, L, J, Ψ)

    # Dimensions
    nx, ny, n_s, batchsize = size(s)
    n_in = n_s + 1
    maxiter = length(L)
    N = zeros(Float32, nx, ny, n_in-2, batchsize)

    for j=maxiter:-1:1
        ηs_ = L[j].inverse(cat(η, s; dims=3))
        s_ = ηs_[:, :, 2:end, :]
        η = ηs_[:, :, 1:1, :]

        g = J'*(J*reshape(Ψ(η), :, batchsize) - d)
        g = reshape(g/norm(g, Inf), nx, ny, 1, batchsize)

        s = s_ - cat(g, N; dims=3)
    end
    return η, s
end

# Backward loop: Input (Δη, Δs, η, s), Output (Δη, Δs, η, s)
function loop_backward(Δη, Δs, η, s, d, L, J, Ψ)

    # Dimensions
    nx, ny, n_s, batchsize = size(s)
    n_in = n_s + 1
    maxiter = length(L)
    N = zeros(Float32, nx, ny, n_in-2, batchsize)
    typeof(Δs) == Float32 && (Δs = 0f0.*s)  # make Δs zero tensor

    for j=maxiter:-1:1
        Δηs_, ηs_ = L[j].backward(cat(Δη, Δs; dims=3), cat(η, s; dims=3))

        # Inverse pass
        η = ηs_[:, :, 1:1, :]
        s_ = ηs_[:, :, 2:end, :]
        g = J'*(J*reshape(Ψ(η), :, batchsize) - d)
        g_norm = norm(g, Inf)

        # Gradients
        Δs = Δηs_[:, :, 2:end, :]
        Δη = (J'*J*reshape(Ψ(Δs[:, :, 1, :]), :, batchsize)*g_norm - g.*sign.(g)) / g_norm^2
        Δη = reshape(Δη, nx, ny, 1, batchsize) + Δηs_[:, :, 1:1, :]

        # Recompute original input        
        g = reshape(g / g_norm, nx, ny, 1, batchsize)
        s = s_ - cat(g, N; dims=3)
    end
    return Δη, Δs, η, s
end

# Clear gradients
function clear_grad!(UL::NetworkLoop)
    maxiter = length(UL.L)
    for j=1:maxiter
        clear_grad!(UL.L[j].C)
        clear_grad!(UL.L[j].RB)
    end
end

# Get parameters
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
