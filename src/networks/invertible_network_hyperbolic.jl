# Invertible network based hyperbolic layers (Lensink et. al, 2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkHyperbolic

"""
    H = NetworkHyperbolic(nx, ny, n_in, batchsize, n_hidden, L, K)

 Create an invertible network based on hyperbolic layers. The network consists of L downsampling
 layers (i.e. scales) using the wavelet transform and `L` upsampling layers to bring the output
 back to the original input dimensions. Each of the `L` scales consists of `K` hyperbolic layers.

 *Input*: 
 
 - `nx`, `ny`, `n_in`, `batchsize`: spatial dimensions, number of channels and batchsize of input tensor
 
 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales

 - `K`: number of time steps per scale

 *Output*:
 
 - `H`: invertible hyperbolic network.

 *Usage:*

 - Forward mode: `Y, logdet = H.forward(X)`

 - Inverse mode: `X = H.inverse(Y)`

 - Backward mode: `ΔX, X = H.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `H` itself

 - Trainable parameters in activation normalization `H.AL` and hyperbolic layers `H.HL[j]`.

 See also: [`AffineLayer`](@ref), [`CouplingLayer!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkHyperbolic <: InvertibleNetwork
    AL::AffineLayer
    HL::AbstractArray{HyperbolicLayer, 1}
    logdet::Bool
end

@Flux.functor NetworkHyperbolic

# Constructor
function NetworkHyperbolic(nx::Int64, ny::Int64, n_in::Int64, batchsize::Int64, L::Int64, K::Int64; 
        k=3, s=1, p=1, logdet=true, α=1f0, hidden_factor=1, ncenter=1)

    depth = Int(2*(L-1)*K + ncenter)
    AL = AffineLayer(Int(nx/2), Int(ny/2), Int(n_in*4); logdet=logdet)
    HL = Array{HyperbolicLayer}(undef, depth)
    nx = Int(nx/2); ny = Int(ny/2); n_in = Int(n_in*2)  # dimensions after initial wavelet transform

    # Downsampling layers
    count=1
    for i=1:L-1
        if K > 1
            for j=2:K
                HL[count] = HyperbolicLayer(Int(nx/2^(i-1)), Int(ny/2^(i-1)), Int(n_in*4^(i-1)),
                    batchsize, k, s, p; action="same", α=α, hidden_factor=hidden_factor)
                count += 1
            end
        end 
        HL[count] = HyperbolicLayer(Int(nx/2^(i-1)), Int(ny/2^(i-1)), Int(n_in*4^(i-1)), 
            batchsize, k, s, p; action="down", α=α, hidden_factor=hidden_factor)
        count += 1
    end
    
    # Middle layers at coarsest scale
    for i=1:ncenter
        HL[count] = HyperbolicLayer(Int(nx/2^(L-1)), Int(ny/2^(L-1)), Int(n_in*4^(L-1)), 
            batchsize, k, s, p; action="same", α=α, hidden_factor=hidden_factor)
        count += 1
    end

    # Upsampling layers
    for i=L-1:-1:1
        HL[count] = HyperbolicLayer(Int(nx/2^i), Int(ny/2^i), Int(n_in*4^i), 
            batchsize, k, s, p; action="up", α=α, hidden_factor=hidden_factor)
        count += 1
        if K > 1
            for j=2:K
                HL[count] = HyperbolicLayer(Int(nx/2^(i-1)), Int(ny/2^(i-1)), Int(n_in*4^(i-1)), batchsize, k, s, p;
                    action="same", α=α, hidden_factor=hidden_factor)
                count += 1
            end
        end
    end

    return NetworkHyperbolic(AL, HL, logdet)
end

# Forward pass
function forward(X, H::NetworkHyperbolic)
    X = wavelet_squeeze(X)
    X, logdet = H.AL.forward(X)
    X_prev, X_curr = tensor_split(X)
    for j=1:length(H.HL)
        X_prev, X_curr = H.HL[j].forward(X_prev, X_curr)
    end
    X = tensor_cat(X_prev, X_curr)
    X = wavelet_unsqueeze(X)
    return X, logdet
end

# Inverse pass
function inverse(Y, H::NetworkHyperbolic)
    Y = wavelet_squeeze(Y)
    Y_curr, Y_new = tensor_split(Y)
    for j=length(H.HL):-1:1
        Y_curr, Y_new = H.HL[j].inverse(Y_curr, Y_new)
    end
    Y = tensor_cat(Y_curr, Y_new)
    Y = H.AL.inverse(Y)
    Y = wavelet_unsqueeze(Y)
    return Y
end

# Backward pass
function backward(ΔY, Y, H::NetworkHyperbolic; set_grad::Bool=true)
    ΔY = wavelet_squeeze(ΔY)
    Y = wavelet_squeeze(Y)
    ΔY_curr, ΔY_new = tensor_split(ΔY)
    Y_curr, Y_new = tensor_split(Y)
    ~set_grad && (Δθ = Array{Parameter, 1}(undef, 0))
    for j=length(H.HL):-1:1
        if set_grad
            ΔY_curr, ΔY_new, Y_curr, Y_new = H.HL[j].backward(ΔY_curr, ΔY_new, Y_curr, Y_new)
        else
            ΔY_curr, ΔY_new, Δθ_HLj, Y_curr, Y_new = H.HL[j].backward(ΔY_curr, ΔY_new, Y_curr, Y_new; set_grad=set_grad)
            Δθ = cat(Δθ_HLj, Δθ; dims=1)
        end
    end
    ΔY = tensor_cat(ΔY_curr, ΔY_new)
    Y = tensor_cat(Y_curr, Y_new)
    set_grad ? ((ΔY, Y) = H.AL.backward(ΔY, Y)) : ((ΔY, Δθ_AL, Y, ∇logdet) = H.AL.backward(ΔY, Y; set_grad=set_grad))
    ΔY = wavelet_unsqueeze(ΔY)
    Y = wavelet_unsqueeze(Y)
    set_grad ? (return ΔY, Y) : (return ΔY, cat(Δθ_AL, Δθ; dims=1), Y, ∇logdet)
end


## Jacobian-related utils

function jacobian(ΔX, Δθ::Array{Parameter, 1}, X, H::NetworkHyperbolic)
    X = wavelet_squeeze(X)
    ΔX = wavelet_squeeze(ΔX)
    ΔX, X, logdet, GNΔθ = H.AL.jacobian(ΔX, Δθ[1:2], X)
    X_prev, X_curr = tensor_split(X)
    ΔX_prev, ΔX_curr = tensor_split(ΔX)
    npars_hl = Int64((length(Δθ)-2)/length(H.HL))
    for j=1:length(H.HL)
        Δθj = Δθ[3+(j-1)*npars_hl:2+j*npars_hl]
        ΔX_prev, ΔX_curr, X_prev, X_curr = H.HL[j].jacobian(ΔX_prev, ΔX_curr, Δθj, X_prev, X_curr)
    end
    X = tensor_cat(X_prev, X_curr)
    ΔX = tensor_cat(ΔX_prev, ΔX_curr)
    X = wavelet_unsqueeze(X)
    ΔX = wavelet_unsqueeze(ΔX)
    return ΔX, X, logdet, GNΔθ
end

adjointJacobian(ΔY, Y, H::NetworkHyperbolic) = backward(ΔY, Y, H; set_grad=false)

## Other utils

# Clear gradients
function clear_grad!(H::NetworkHyperbolic)
    depth = length(H.HL)
    clear_grad!(H.AL)
    for j=1:depth
        clear_grad!(H.HL[j])
    end
end

# Get parameters
function get_params(H::NetworkHyperbolic)
    depth = length(H.HL)
    p = get_params(H.AL)
    for j=1:depth
        p = cat(p, get_params(H.HL[j]); dims=1)
    end
    return p
end
