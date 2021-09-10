# Invertible multiscale HINT network from Kruse et. al (2020)
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

export NetworkMultiScaleHINT, NetworkMultiScaleHINT3D

"""
    H = NetworkMultiScaleHINT(n_in, n_hidden, L, K; split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=2)

    H = NetworkMultiScaleHINT3D(n_in, n_hidden, L, K; split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create a multiscale HINT network for data-driven generative modeling based
 on the change of variables formula.

 *Input*: 
 
 - 'n_in': number of input channels
 
 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 - `split_scales`: if true, split output in half along channel dimension after each scale. Feed one half through the next layers,
    while saving the remaining channels for the output.

 - `k1`, `k2`: kernel size for first and third residual layer (`k1`) and second layer (`k2`)

 - `p1`, `p2`: respective padding sizes for residual block layers
 
 - `s1`, `s2`: respective strides for residual block layers

 - `ndims` : number of dimensions

 *Output*:
 
 - `H`: multiscale HINT network

 *Usage:*

 - Forward mode: `Z, logdet = H.forward(X)`

 - Inverse mode: `X = H.inverse(Z)`

 - Backward mode: `ΔX, X = H.backward(ΔZ, Z)`

 *Trainable parameters:*

 - None in `H` itself

 - Trainable parameters in activation normalizations `H.AN[i]`, 
 and in coupling layers `H.CL[i]`, where `i` ranges from `1` to `depth`.

 See also: [`ActNorm`](@ref), [`CouplingLayerHINT!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkMultiScaleHINT <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 2}
    CL::AbstractArray{CouplingLayerHINT, 2}
    X_dims::Union{Array{Tuple, 1}, Nothing}
    L::Int64
    K::Int64
    split_scales::Bool
end

@Flux.functor NetworkMultiScaleHINT

# Constructor
function NetworkMultiScaleHINT(n_in::Int64, n_hidden::Int64, L::Int64, K::Int64;
                               split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=2)

    AN = Array{ActNorm}(undef, L, K)
    CL = Array{CouplingLayerHINT}(undef, L, K)
    if split_scales
        X_dims = Array{Tuple}(undef, L-1)
        channel_factor = 2
    else
        X_dims = nothing
        channel_factor = 4
    end

    # Create layers
    for i=1:L
        for j=1:K
            AN[i, j] = ActNorm(n_in*4; logdet=true)
            CL[i, j] = CouplingLayerHINT(n_in*4, n_hidden; permute="full", k1=k1, k2=k2, p1=p1, p2=p2,
                                         s1=s1, s2=s2, logdet=true, ndims=ndims)
        end
        n_in *= channel_factor
    end

    return NetworkMultiScaleHINT(AN, CL, X_dims, L, K, split_scales)
end

NetworkMultiScaleHINT3D(args...; kw...) = NetworkMultiScaleHINT(args...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, H::NetworkMultiScaleHINT) where {T, N}
    H.split_scales && (X_save = Array{Array}(undef, H.L-1))
    logdet = 0
    for i=1:H.L
        X = wavelet_squeeze(X)
        for j=1:H.K
            X_, logdet1 = H.AN[i, j].forward(X)
            X, logdet2 = H.CL[i, j].forward(X_)
            logdet += (logdet1 + logdet2)
        end
        if H.split_scales && i < H.L    # don't split after last iteration
            X, Z = tensor_split(X)
            X_save[i] = Z
            H.X_dims[i] = size(Z)
        end
    end
    H.split_scales && (X = cat_states(X_save, X))
    return X, logdet
end

# Inverse pass and compute gradients
function inverse(Z::AbstractArray{T, N}, H::NetworkMultiScaleHINT) where {T, N}
    H.split_scales && ((X_save, Z) = split_states(H.X_dims, Z))
    for i=H.L:-1:1
        if H.split_scales && i < H.L
            Z = tensor_cat(Z, X_save[i])
        end
        for j=H.K:-1:1
            Z_ = H.CL[i, j].inverse(Z)
            Z = H.AN[i, j].inverse(Z_; logdet=false)
        end
        Z = wavelet_unsqueeze(Z)
    end
    return Z
end

# Backward pass and compute gradients
function backward(ΔZ::AbstractArray{T, N}, Z::AbstractArray{T, N}, H::NetworkMultiScaleHINT; set_grad::Bool=true) where {T, N}

    # Split data and gradients
    if H.split_scales
        ΔX_save, ΔZ = split_states(H.X_dims, ΔZ)
        X_save, Z = split_states(H.X_dims, Z)
    end

    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 0)
        ∇logdet = Array{Parameter, 1}(undef, 0)
    end

    for i=H.L:-1:1
        if H.split_scales && i < H.L
            ΔZ = tensor_cat(ΔZ, ΔX_save[i])
            Z = tensor_cat(Z, X_save[i])
        end
        for j=H.K:-1:1
            if set_grad
                ΔZ_, Z_ = H.CL[i, j].backward(ΔZ, Z)
                ΔZ, Z = H.AN[i, j].backward(ΔZ_, Z_)
            else
                ΔZ_, Δθcl, Z_, ∇logdet_cl = H.CL[i, j].backward(ΔZ, Z; set_grad=set_grad)
                ΔZ, Δθx, Z, ∇logdet_x = H.AN[i, j].backward(ΔZ_, Z_; set_grad=set_grad)
                Δθ = cat(Δθx, Δθcl, Δθ; dims=1)
                ∇logdet = cat(∇logdet_x, ∇logdet_cl, ∇logdet; dims=1)
            end
        end
        ΔZ = wavelet_unsqueeze(ΔZ)
        Z = wavelet_unsqueeze(Z)
    end
    set_grad ? (return ΔZ, Z) : (return ΔZ, Δθ, Z, ∇logdet)
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X, H::NetworkMultiScaleHINT) where {T, N}
    if H.split_scales
        X_save = array_of_arry(ΔX, H.L-1, 2)
        ΔX_save = array_of_arry(ΔX, H.L-1, 2)
    end
    logdet = 0
    GNΔθ = Array{Parameter, 1}(undef, 0)
    idxblk = 0
    for i=1:H.L
        X = wavelet_squeeze(X)
        ΔX = wavelet_squeeze(ΔX)
        for j=1:H.K
            npars_ij = 2+length(get_params(H.CL[i, j]))
            Δθij = Δθ[idxblk+1:idxblk+npars_ij]
            ΔX_, X_, logdet1, GNΔθ1 = H.AN[i, j].jacobian(ΔX, Δθij[1:2], X)
            ΔX, X, logdet2, GNΔθ2 = H.CL[i, j].jacobian(ΔX_, Δθij[3:end], X_)
            logdet += (logdet1 + logdet2)
            GNΔθ = cat(GNΔθ, GNΔθ1, GNΔθ2; dims=1)
            idxblk += npars_ij
        end
        if H.split_scales && i < H.L    # don't split after last iteration
            X, Z = tensor_split(X)
            ΔX, ΔZ = tensor_split(ΔX)
            X_save[i] = Z
            ΔX_save[i] = ΔZ
            H.X_dims[i] = size(Z)
        end
    end
    if H.split_scales
        X = cat_states(X_save, X)
        ΔX = cat_states(ΔX_save, ΔX)
    end
    return ΔX, X, logdet, GNΔθ
end

adjointJacobian(ΔZ::AbstractArray{T, N}, Z::AbstractArray{T, N}, H::NetworkMultiScaleHINT) where {T, N} = backward(ΔZ, Z, H; set_grad=false)


## Other utils

# Clear gradients
function clear_grad!(H::NetworkMultiScaleHINT)
    depth = length(H.CL)
    for j=1:depth
        clear_grad!(H.AN[j])
        clear_grad!(H.CL[j])
    end
end

# Get parameters
function get_params(H::NetworkMultiScaleHINT)
    L, K = size(H.CL)
    p = Array{Parameter, 1}(undef, 0)
    for i = 1:L
        for j = 1:K
            p = cat(p, get_params(H.AN[i, j]); dims=1)
            p = cat(p, get_params(H.CL[i, j]); dims=1)
        end
    end
    return p
end
