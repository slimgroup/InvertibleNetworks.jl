# Invertible network based on Glow (Kingma and Dhariwal, 2018)
# Includes 1x1 convolution and residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkGlow

"""
    G = NetworkGlow(nx, ny, n_in, batchsize, n_hidden, L, K)

 Create an invertible network based on the Glow architecture. Each flow step in the inner loop 
 consists of an activation normalization layer, followed by an invertible coupling layer with
 1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior 
 to the inner loop, and a splitting operation afterwards.

 *Input*: 
 
 - `nx`, `ny`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensor
 
 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 *Output*:
 
 - `G`: invertible Glow network.

 *Usage:*

 - Forward mode: `Y, logdet = G.forward(X)`

 - Backward mode: `ΔX, X = G.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `G` itself

 - Trainable parameters in activation normalizations `G.AN[i,j]` and coupling layers `G.C[i,j]`,
   where `i` and `j` range from `1` to `L` and `K` respectively.

 See also: [`ActNorm`](@ref), [`CouplingLayer!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkGlow <: InvertibleNetwork
    AN::Array{ActNorm, 2}
    CL::Array{CouplingLayer, 2}
    Z_dims::Array{Tuple, 1}
    forward::Function
    inverse::Function
    backward::Function
end

# Constructor
function NetworkGlow(nx, ny, n_in, batchsize, n_hidden, L, K)

    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayer}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    Z_dims = Array{Tuple}(undef, L-1)   # save dimensions for inverse/backward pass

    for i=1:L
        for j=1:K
            AN[i, j] = ActNorm(n_in; logdet=true)
            CL[i, j] = CouplingLayer(Int(nx/2^i), Int(ny/2^i), n_in*4, n_hidden, batchsize; k1=1, k2=3, p1=0, p2=1, logdet=true)
        end
        n_in *= 2
    end

    return NetworkGlow(AN, CL, Z_dims, 
        X -> glow_forward(X, AN, CL, Z_dims, L, K),
        Y -> glow_inverse(Y, AN, CL, Z_dims, L, K),
        (ΔY, Y) -> glow_backward(ΔY, Y, AN, CL, Z_dims, L, K)
    )
end

# Concatenate states Zi and final output
function cat_states(Z_save, X)
    Y = []
    for j=1:length(Z_save)
        Y = cat(Y, vec(Z_save[j]); dims=1)
    end
    Y = cat(Y, vec(X); dims=1)
    return Float32.(Y)  # convert to Array{Float32, 1}
end

# Split 1D vector in latent space back to states Zi
function split_states(Y, Z_dims)
    L = length(Z_dims) + 1
    Z_save = Array{Array}(undef, L-1)
    count = 1
    for j=1:L-1
        Z_save[j] = reshape(Y[count: count + prod(Z_dims[j])-1], Z_dims[j])
        count += prod(Z_dims[j])
    end
    X = reshape(Y[count: count + prod(Z_dims[end])-1], Int.(Z_dims[end].*(.5, .5, 4, 1)))
    return Z_save, X
end

# Forward pass and compute logdet
function glow_forward(X, AN, CL, Z_dims, L, K)
    Z_save = Array{Array}(undef, L-1)
    logdet = 0f0
    for i=1:L
        X = squeeze(X; pattern="checkerboard")
        for j=1:K
            X, logdet1 = AN[i, j].forward(X)
            X, logdet2 = CL[i, j].forward(X)
            logdet += (logdet1 + logdet2)
        end
        if i < L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            Z_dims[i] = size(Z)
        end
    end
    X = cat_states(Z_save, X)
    return X, logdet
end

# Inverse pass and compute gradients
function glow_inverse(X, AN, CL, Z_dims, L, K)
    Z_save, X = split_states(X, Z_dims)
    for i=L:-1:1
        if i < L
            X = tensor_cat(X, Z_save[i])
        end
        for j=K:-1:1
            X = CL[i, j].inverse(X)
            X = AN[i, j].inverse(X)
        end
        X = unsqueeze(X; pattern="checkerboard")
    end
    return X
end

# Backward pass and compute gradients
function glow_backward(ΔX, X, AN, CL, Z_dims, L, K)
    ΔZ_save, ΔX = split_states(ΔX, Z_dims)
    Z_save, X = split_states(X, Z_dims)
    for i=L:-1:1
        if i < L
            X = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=K:-1:1
            ΔX, X = CL[i, j].backward(ΔX, X)
            ΔX, X = AN[i, j].backward(ΔX, X)
        end
        X = unsqueeze(X; pattern="checkerboard")
        ΔX = unsqueeze(ΔX; pattern="checkerboard")
    end
    return ΔX, X
end

# Clear gradients
function clear_grad!(G::NetworkGlow)
    L, K = size(G.AN)
    for i=1:L
        for j=1:K
            clear_grad!(G.AN[i, j])
            clear_grad!(G.CL[i, j])
        end
    end
end

# Get parameters
function get_params(G::NetworkGlow)
    L, K = size(G.AN)
    p = []
    for i=1:L
        for j=1:K
            p = cat(p, get_params(G.AN[i, j]); dims=1)
            p = cat(p, get_params(G.CL[i, j]); dims=1)
        end
    end
    return p
end
