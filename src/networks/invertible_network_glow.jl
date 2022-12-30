# Invertible network based on Glow (Kingma and Dhariwal, 2018)
# Includes 1x1 convolution and residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkGlow, NetworkGlow3D

"""
    G = NetworkGlow(n_in, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

    G = NetworkGlow3D(n_in, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

 Create an invertible network based on the Glow architecture. Each flow step in the inner loop 
 consists of an activation normalization layer, followed by an invertible coupling layer with
 1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior 
 to the inner loop, and a splitting operation afterwards.

 *Input*: 

 - 'n_in': number of input channels

 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 - `split_scales`: if true, perform squeeze operation which halves spatial dimensions and duplicates channel dimensions
    then split output in half along channel dimension after each scale. Feed one half through the next layers,
    while saving the remaining channels for the output.

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
 operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 - `squeeze_type` : squeeze type that happens at each multiscale level

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
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    squeezer::Squeezer
    split_scales::Bool
    logdet::Bool
end

@Flux.functor NetworkGlow

# Constructor
function NetworkGlow(n_in, n_hidden, L, K; logdet=true,freeze_conv=false, split_scales=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayerGlow}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
 
    if split_scales
        Z_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        channel_factor = 2^(ndims)
    else
        Z_dims = nothing
        channel_factor = 1
    end

    for i=1:L
        n_in *= channel_factor # squeeze if split_scales is turned on
        for j=1:K
            AN[i, j] = ActNorm(n_in; logdet=logdet)
            CL[i, j] = CouplingLayerGlow(n_in, n_hidden; logdet=logdet, freeze_conv=freeze_conv, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, activation=activation, ndims=ndims)
        end
        (i < L && split_scales) && (n_in = Int64(n_in/2)) # split
    end

    return NetworkGlow(AN, CL, Z_dims, L, K, squeezer, split_scales, logdet)
end

NetworkGlow3D(args; kw...) = NetworkGlow(args...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, G::NetworkGlow) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, G.L-1))
    orig_shape = size(X)
    logdet = 0
    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        for j=1:G.K            
            G.logdet ? (X, logdet1) = G.AN[i, j].forward(X) : X = G.AN[i, j].forward(X)
            G.logdet ? (X, logdet2) = G.CL[i, j].forward(X) : X = G.CL[i, j].forward(X)
            G.logdet && (logdet += (logdet1 + logdet2))
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = reshape(cat_states(Z_save, X),orig_shape))
    G.logdet ? (return X, logdet) : (return X)
end

# Inverse pass 
function inverse(X::AbstractArray{T, N}, G::NetworkGlow) where {T, N}
    G.split_scales && ((Z_save, X) = split_states(X[:], G.Z_dims))
    logdet = 0
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            G.logdet ? (X, logdet1) = G.CL[i, j].inverse(X) : X = G.CL[i, j].inverse(X)
            G.logdet ? (X, logdet2) = G.AN[i, j].inverse(X) : X = G.AN[i, j].inverse(X)
            G.logdet && (logdet += (logdet1 + logdet2))
        end

        (G.split_scales) && (X = G.squeezer.inverse(X))
    end
    G.logdet ? (return X, logdet) : (return X)
end

# Backward pass and compute gradients
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, G::NetworkGlow; set_grad::Bool=true) where {T, N}
    
    # Split data and gradients
    if G.split_scales
        ΔZ_save, ΔX = split_states(ΔX[:], G.Z_dims)
        Z_save, X = split_states(X[:], G.Z_dims)
    end

    if ~set_grad
        ΔθAN = Vector{Parameter}(undef, 0)
        ΔθCL = Vector{Parameter}(undef, 0)
        ∇logdetAN = Vector{Parameter}(undef, 0)
        ∇logdetCL = Vector{Parameter}(undef, 0)
    end

    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X  = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=G.K:-1:1
            if set_grad
                ΔX, X = G.CL[i, j].backward(ΔX, X)
                ΔX, X = G.AN[i, j].backward(ΔX, X)
            else
                ΔX, Δθcl_ij, X, ∇logdetcl_ij = G.CL[i, j].backward(ΔX, X; set_grad=set_grad)
                ΔX, Δθan_ij, X, ∇logdetan_ij = G.AN[i, j].backward(ΔX, X; set_grad=set_grad)
                prepend!(ΔθAN, Δθan_ij)
                prepend!(ΔθCL, Δθcl_ij)
                prepend!(∇logdetAN, ∇logdetan_ij)
                prepend!(∇logdetCL, ∇logdetcl_ij)
            end
        end

        if G.split_scales 
          X = G.squeezer.inverse(X)
          ΔX = G.squeezer.inverse(ΔX)
        end
    end
    set_grad ? (return ΔX, X) : (return ΔX, vcat(ΔθAN, ΔθCL), X, vcat(∇logdetAN, ∇logdetCL))
end

# Backward reverse pass and compute gradients
function backward_inv(ΔX::AbstractArray{T, N},  X::AbstractArray{T, N}, G::NetworkGlow) where {T, N}
    G.split_scales && (X_save = array_of_array(X, G.L-1))
    G.split_scales && (ΔX_save = array_of_array(ΔX, G.L-1))
    orig_shape = size(X)

    for i=1:G.L
        G.split_scales && (ΔX = G.squeezer.forward(ΔX))
        G.split_scales && (X  = G.squeezer.forward(X))
        for j=1:G.K
            ΔX_, X_ = backward_inv(ΔX, X, G.AN[i, j])
            ΔX,  X  = backward_inv(ΔX_, X_, G.CL[i, j])
        end

        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            ΔX, ΔZx = tensor_split(ΔX)

            X_save[i] = Z
            ΔX_save[i] = ΔZx

            G.Z_dims[i] = collect(size(X))
        end
    end

    G.split_scales && (X = reshape(cat_states(X_save, X), orig_shape))
    G.split_scales && (ΔX = reshape(cat_states(ΔX_save, ΔX), orig_shape))
    return ΔX, X
end

## Jacobian-related utils
function jacobian(ΔX::AbstractArray{T, N}, Δθ::Vector{Parameter}, X, G::NetworkGlow) where {T, N}
    if G.split_scales 
        Z_save = array_of_array(ΔX, G.L-1)
        ΔZ_save = array_of_array(ΔX, G.L-1)
    end
    orig_shape = size(X)

    logdet = 0
    cls = 2*G.K*G.L
    ΔθAN = Vector{Parameter}(undef, 0)
    ΔθCL = Vector{Parameter}(undef, 0)

    for i=1:G.L
        if G.split_scales 
            X = G.squeezer.forward(X) 
            ΔX = G.squeezer.forward(ΔX)
        end
        
        for j=1:G.K
            as = length(ΔθAN)+1
            cs = cls + length(ΔθCL) + 1
            ΔX, X, logdet1, GNΔθ1 = G.AN[i, j].jacobian(ΔX, Δθ[as:as+1], X)
            ΔX, X, logdet2, GNΔθ2 = G.CL[i, j].jacobian(ΔX, Δθ[cs:cs+7], X)
            logdet += (logdet1 + logdet2)
            append!(ΔθAN, GNΔθ1)
            append!(ΔθCL, GNΔθ2)
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            ΔX, ΔZ = tensor_split(ΔX)
            Z_save[i] = Z
            ΔZ_save[i] = ΔZ
            G.Z_dims[i] = collect(size(Z))
        end
    end
    if G.split_scales 
        X = reshape(cat_states(Z_save, X), orig_shape)
        ΔX = reshape(cat_states(ΔZ_save, ΔX), orig_shape)
    end
    
    return ΔX, X, logdet, vcat(ΔθAN, ΔθCL)
end

adjointJacobian(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, G::NetworkGlow) where {T, N} = backward(ΔX, X, G; set_grad=false)
