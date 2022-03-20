# Invertible network based on Glow (Kingma and Dhariwal, 2018)
# Includes 1x1 convolution and residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkCIIN, NetworkCIIN3D

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
struct NetworkCIIN <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 2}
    AN_c::AbstractArray{ActNorm, 2}
    CL::Union{AbstractArray{CondCouplingLayerGlow, 2}, AbstractArray{CondCouplingLayerSpade, 2}, AbstractArray{CondCouplingLayerSpade_additive, 2}}
    Z_dims::Union{Array{Array, 1}, Nothing}
    final_dim::Array
    L::Int64
    K::Int64
    squeezer::Squeezer
    split_scales::Bool
end

@Flux.functor NetworkCIIN

# Constructor
#function NetworkCIIN(n_in, n_cond, n_hidden, L, K;n_hiddens=nothing, ds=nothing, coupling_type ="glow", split_scales=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
function NetworkCIIN(n_in, n_cond, n_hidden, L, K;n_hiddens=nothing, ds=nothing, coupling_type ="glow", split_scales=false, k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
       
    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    AN_c = Array{ActNorm}(undef, L, K)    # activation normalization
    
    if coupling_type == "spade"
        CL = Array{CondCouplingLayerSpade}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    elseif coupling_type == "spade_additive"
        CL = Array{CondCouplingLayerSpade_additive}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    else
        CL = Array{CondCouplingLayerGlow}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    end
 
    final_dim = [1,1,1,1]
    if split_scales
        Z_dims = fill!(Array{Array}(undef, max(L-1,1)), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        channel_factor = 4
    else
        Z_dims = nothing
        channel_factor = 1
    end

    for i=1:L
        n_in *= channel_factor # squeeze if split_scales is turned on
        n_cond *= channel_factor # squeeze if split_scales is turned on
        for j=1:K
            AN[i, j] = ActNorm(n_in; logdet=true)
            AN_c[i, j] = ActNorm(n_cond; logdet=true)
            if coupling_type == "spade"
                #CL[i, j] = CondCouplingLayerSpade_additive(n_in, n_cond; n_hiddens=n_hiddens, ds=ds, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, activation=activation, ndims=ndims)
                CL[i, j] = CondCouplingLayerSpade(n_in, n_cond, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, activation=activation, ndims=ndims)
            
            elseif coupling_type == "spade_additive"
                CL[i, j] = CondCouplingLayerSpade_additive(n_in, n_cond, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, activation=activation, ndims=ndims)
            else
                CL[i, j] = CondCouplingLayerGlow(n_in, n_cond, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, activation=activation, ndims=ndims)
            end
        end
        (i < L && split_scales) && (n_in = Int64(n_in/2)) # split
    end

    return NetworkCIIN(AN,AN_c, CL, Z_dims,final_dim, L, K, squeezer, split_scales)
end

NetworkCIIN3D(args; kw...) = NetworkCIIN(args...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkCIIN) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, max(G.L-1,1)))

    logdet = 0
    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        (G.split_scales) && (C = G.squeezer.forward(C))
        #X = G.squeezer.forward(X)
        #C = G.squeezer.forward(C)
        for j=1:G.K          
            X, logdet1 = G.AN[i, j].forward(X)
            C, _       = G.AN_c[i, j].forward(C)
            

            X, logdet2 = G.CL[i, j].forward(X,C)
            logdet += (logdet1 + logdet2)
        end
        if G.split_scales && (i < G.L || G.L == 1)   # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = cat_states(Z_save, X))
    #G.final_dim = collect(size(X))
    #X = X[:] #always output as vector for consitency. 

    return X, C, logdet
end

# Inverse pass 
function inverse(X::AbstractArray{T, N}, C::AbstractArray{T, M}, G::NetworkCIIN) where {T, N, M}
    X# = reshape(X, G.final_dim...)
    G.split_scales && ((Z_save, X) = split_states(X, G.Z_dims))
    for i=G.L:-1:1
        if G.split_scales && (i < G.L || G.L == 1)
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            X = G.CL[i, j].inverse(X, C)
            X = G.AN[i, j].inverse(X)
            C = G.AN_c[i, j].inverse(C)
        end

        (G.split_scales) && (X = G.squeezer.inverse(X))
        (G.split_scales) && (C = G.squeezer.inverse(C))
        #X = G.squeezer.inverse(X)
        #C = G.squeezer.inverse(C)
    end
    return X
end

# Backward pass and compute gradients
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, ΔC::AbstractArray{T, M}, C::AbstractArray{T, M}, G::NetworkCIIN; set_grad::Bool=true) where {T, N, M}
    #X = reshape(X, G.final_dim...)
    #ΔX = reshape(ΔX, G.final_dim...)
    # Split data and gradients
    if G.split_scales
        ΔZ_save, ΔX = split_states(ΔX, G.Z_dims)
        Z_save, X = split_states(X, G.Z_dims)
    end

    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 10*G.L*G.K)
        ∇logdet = Array{Parameter, 1}(undef, 10*G.L*G.K)
    end
    blkidx = 10*G.L*G.K
    for i=G.L:-1:1
        if G.split_scales && (i < G.L || G.L == 1)
            X  = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=G.K:-1:1
            if set_grad
                ΔX, X, ΔC  = G.CL[i, j].backward(ΔX, X, ΔC, C)
                ΔX, X      = G.AN[i, j].backward(ΔX, X)
                ΔC, C      = G.AN_c[i, j].backward(ΔC, C)
                #C         = G.AN_c[i, j].inverse(C)
            else
                ΔX, Δθcl_ij, X, ∇logdetcl_ij = G.CL[i, j].backward(ΔX, X, C; set_grad=set_grad)
                ΔX, Δθan_ij, X, ∇logdetan_ij = G.AN[i, j].backward(ΔX, X; set_grad=set_grad)
                Δθ[blkidx-9:blkidx] = cat(Δθan_ij, Δθcl_ij; dims=1)
                ∇logdet[blkidx-9:blkidx] = cat(∇logdetan_ij, ∇logdetcl_ij; dims=1)
            end
            blkidx -= 10
        end

        if G.split_scales 
            X  = G.squeezer.inverse(X)
            C  = G.squeezer.inverse(C)
            ΔX = G.squeezer.inverse(ΔX)
            ΔC = G.squeezer.inverse(ΔC)
        end
    end
    set_grad ? (return ΔX, X) : (return ΔX, Δθ, X, ∇logdet)
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X, G::NetworkCIIN) where {T, N}

    if G.split_scales 
        Z_save = array_of_array(ΔX, G.L-1)
        ΔZ_save = array_of_array(ΔX, G.L-1)
    end
    logdet = 0
    GNΔθ = Array{Parameter, 1}(undef, 10*G.L*G.K)
    blkidx = 0
    for i=1:G.L
        if G.split_scales 
            X = G.squeezer.forward(X) 
            ΔX = G.squeezer.forward(ΔX) 
        end
        
        for j=1:G.K
            Δθ_ij = Δθ[blkidx+1:blkidx+10]
            ΔX, X, logdet1, GNΔθ1 = G.AN[i, j].jacobian(ΔX, Δθ_ij[1:2], X)
            ΔX, X, logdet2, GNΔθ2 = G.CL[i, j].jacobian(ΔX, Δθ_ij[3:end], X)
            logdet += (logdet1 + logdet2)
            GNΔθ[blkidx+1:blkidx+10] = cat(GNΔθ1,GNΔθ2; dims=1)
            blkidx += 10
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
        X = cat_states(Z_save, X)
        ΔX = cat_states(ΔZ_save, ΔX)
    end
    
    return ΔX, X, logdet, GNΔθ
end

adjointJacobian(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, G::NetworkCIIN) where {T, N} = backward(ΔX, X, G; set_grad=false)


## Other utils

# Clear gradients
function clear_grad!(G::NetworkCIIN)
    L, K = size(G.AN)
    for i=1:L
        for j=1:K
            clear_grad!(G.AN[i, j])
            clear_grad!(G.AN_c[i, j])
            clear_grad!(G.CL[i, j])
        end
    end
end

# Get parameters
function get_params(G::NetworkCIIN)
    L, K = size(G.AN)
    p = Array{Parameter, 1}(undef, 0)
    for i=1:L
        for j=1:K
            p = cat(p, get_params(G.AN[i, j]); dims=1)
            p = cat(p, get_params(G.AN_c[i, j]); dims=1)
            p = cat(p, get_params(G.CL[i, j]); dims=1)
        end
    end
    return p
end
