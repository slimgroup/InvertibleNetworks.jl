# Invertible network based on Glow (Kingma and Dhariwal, 2018)
# Includes 1x1 convolution and residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkGlow

"""
    G = NetworkGlow(nx, ny, n_in, batchsize, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

 Create an invertible network based on the Glow architecture. Each flow step in the inner loop 
 consists of an activation normalization layer, followed by an invertible coupling layer with
 1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior 
 to the inner loop, and a splitting operation afterwards.

 *Input*: 
 
 - `nx`, `ny`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensor
 
 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
 operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 *Output*:
 
 - `G`: invertible Glow network.

 *Usage:*

 - Forward mode: `Y, logdet = G.forward(X)`

 - Backward mode: `ΔX, X = G.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `G` itself

 - Trainable parameters in activation normalizations `G.AN[i,j]` and coupling layers `G.C[i,j]`,
   where `i` and `j` range from `1` to `L` and `K` respectively.

 See also: [`ActNorm`](@ref), [`CouplingLayerGlow!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkGlow <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 2}
    CL::AbstractArray{CouplingLayerGlow, 2}
    Z_dims::AbstractArray{Tuple, 1}
    L::Int64
    K::Int64
end

@Flux.functor NetworkGlow

# Constructor
function NetworkGlow(nx, ny, n_in, batchsize, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayerGlow}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    Z_dims = Array{Tuple}(undef, L-1)   # save dimensions for inverse/backward pass

    for i=1:L
        n_in *= 4 # squeeze
        for j=1:K
            AN[i, j] = ActNorm(n_in; logdet=true)
            CL[i, j] = CouplingLayerGlow(Int(nx/2^i), Int(ny/2^i), n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)
        end
        (i < L) && (n_in = Int64(n_in/2)) # split
    end

    return NetworkGlow(AN, CL, Z_dims, L, K)
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
function forward(X, G::NetworkGlow)
    Z_save = Array{Array}(undef, G.L-1)
    logdet = 0f0
    for i=1:G.L
        X = squeeze(X; pattern="checkerboard")
        for j=1:G.K            
            X, logdet1 = G.AN[i, j].forward(X)
            X, logdet2 = G.CL[i, j].forward(X)
            logdet += (logdet1 + logdet2)
        end
        if i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = size(Z)
        end
    end
    X = cat_states(Z_save, X)
    return X, logdet
end

# Inverse pass and compute gradients
function inverse(X, G::NetworkGlow)
    Z_save, X = split_states(X, G.Z_dims)
    for i=G.L:-1:1
        if i < G.L
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            X = G.CL[i, j].inverse(X)
            X = G.AN[i, j].inverse(X)
        end
        X = unsqueeze(X; pattern="checkerboard")
    end
    return X
end

# Backward pass and compute gradients
function backward(ΔX, X, G::NetworkGlow; set_grad::Bool=true)
    ΔZ_save, ΔX = split_states(ΔX, G.Z_dims)
    Z_save, X = split_states(X, G.Z_dims)
    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 10*G.L*G.K)
        ∇logdet = Array{Parameter, 1}(undef, 10*G.L*G.K)
    end
    blkidx = 10*G.L*G.K
    for i=G.L:-1:1
        if i < G.L
            X = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=G.K:-1:1
            if set_grad
                ΔX, X = G.CL[i, j].backward(ΔX, X)
                ΔX, X = G.AN[i, j].backward(ΔX, X)
            else
                ΔX, Δθcl_ij, X, ∇logdetcl_ij = G.CL[i, j].backward(ΔX, X; set_grad=set_grad)
                ΔX, Δθan_ij, X, ∇logdetan_ij = G.AN[i, j].backward(ΔX, X; set_grad=set_grad)
                Δθ[blkidx-9:blkidx] = cat(Δθan_ij, Δθcl_ij; dims=1)
                ∇logdet[blkidx-9:blkidx] = cat(∇logdetan_ij, ∇logdetcl_ij; dims=1)
            end
            blkidx -= 10
        end
        X = unsqueeze(X; pattern="checkerboard")
        ΔX = unsqueeze(ΔX; pattern="checkerboard")
    end
    set_grad ? (return ΔX, X) : (return ΔX, Δθ, X, ∇logdet)
end


## Jacobian-related utils

function jacobian(ΔX, Δθ::Array{Parameter, 1}, X, G::NetworkGlow)
    Z_save = Array{Array}(undef, G.L-1)
    ΔZ_save = Array{Array}(undef, G.L-1)
    logdet = 0f0
    GNΔθ = Array{Parameter, 1}(undef, 10*G.L*G.K)
    blkidx = 0
    for i=1:G.L
        X = squeeze(X; pattern="checkerboard")
        ΔX = squeeze(ΔX; pattern="checkerboard")
        for j=1:G.K
            Δθ_ij = Δθ[blkidx+1:blkidx+10]
            ΔX, X, logdet1, GNΔθ1 = G.AN[i, j].jacobian(ΔX, Δθ_ij[1:2], X)
            ΔX, X, logdet2, GNΔθ2 = G.CL[i, j].jacobian(ΔX, Δθ_ij[3:end], X)
            logdet += (logdet1 + logdet2)
            GNΔθ[blkidx+1:blkidx+10] = cat(GNΔθ1,GNΔθ2; dims=1)
            blkidx += 10
        end
        if i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            ΔX, ΔZ = tensor_split(ΔX)
            Z_save[i] = Z
            ΔZ_save[i] = ΔZ
            G.Z_dims[i] = size(Z)
        end
    end
    X = cat_states(Z_save, X)
    ΔX = cat_states(ΔZ_save, ΔX)
    return ΔX, X, logdet, GNΔθ
end

adjointJacobian(ΔX, X, G::NetworkGlow) = backward(ΔX, X, G; set_grad=false)


## Other utils

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
    p = Array{Parameter, 1}(undef, 0)
    for i=1:L
        for j=1:K
            p = cat(p, get_params(G.AN[i, j]); dims=1)
            p = cat(p, get_params(G.CL[i, j]); dims=1)
        end
    end
    return p
end
