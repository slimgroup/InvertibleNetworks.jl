# Invertible network based hyperbolic layers (Lensink et. al, 2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkHyperbolic, NetworkHyperbolic3D

"""
    H = NetworkHyperbolic(n_in, architecture; k=3, s=1, p=1, logdet=true, α=1f0)
    H = NetworkHyperbolic(n_in, architecture; k=3, s=1, p=1, logdet=true, α=1f0, ndims=2)
    H = NetworkHyperbolic3D(n_in, architecture; k=3, s=1, p=1, logdet=true, α=1f0)

 Create an invertible network based on hyperbolic layers. The network architecture is specified by a tuple
 of the form ((action_1, n_hidden_1), (action_2, n_hidden_2), ... ). Each inner tuple corresonds to an additional layer. 
 The first inner tuple argument specifies whether the respective layer increases the number of channels (set to `1`), 
 decreases it (set to `-1`) or leaves it constant (set to `0`).  The second argument specifies the number of hidden 
 units for that layer.
 
 *Input*: 
 
 - `n_in`: number of channels of input tensor.
 
 - `n_hidden`: number of hidden units in residual blocks

 - `architecture`: Tuple of tuples specifying the network architecture; ((action_1, n_hidden_1), (action_2, n_hidden_2))

 - `k`, `s`, `p`: Kernel size, stride and padding of convolutional kernels
 
 - `logdet`: Bool to indicate whether to return the logdet

 - `α`: Step size in hyperbolic network. Defaults to `1`

 - `ndims`: Number of dimension

 *Output*:
 
 - `H`: invertible hyperbolic network.

 *Usage:*

 - Forward mode: `Y_prev, Y_curr, logdet = H.forward(X_prev, X_curr)`

 - Inverse mode: `X_curr, X_new = H.inverse(Y_curr, Y_new)`

 - Backward mode: `ΔX_curr, ΔX_new, X_curr, X_new = H.backward(ΔY_curr, ΔY_new, Y_curr, Y_new)`

 *Trainable parameters:*

 - None in `H` itself

 - Trainable parameters in the hyperbolic layers `H.HL[j]`.

 See also: [`CouplingLayer!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkHyperbolic <: InvertibleNetwork
    HL::AbstractArray{HyperbolicLayer, 1}
    logdet::Bool
end

@Flux.functor NetworkHyperbolic

# Constructor 2D
function NetworkHyperbolic(n_in::Int64, architecture::NTuple; 
                           k=3, s=1, p=1, logdet=true, α=1f0, ndims=2)#, affine_layer=false)

    depth = length(architecture)
    HL = Array{HyperbolicLayer}(undef, depth)

    for j=1:depth
        
        # Hyperbolic layer at level j
        HL[j] = HyperbolicLayer(n_in, k, s, p; action=architecture[j][1], n_hidden=architecture[j][2], α=α, ndims=ndims)

        # adjust dimensions
        if architecture[j][1] == 1
            n_in = Int(n_in/2^ndims)
        elseif architecture[j][1] == -1
            n_in = Int(n_in*2^ndims)
        end
    end

    return NetworkHyperbolic(HL, logdet)
end

NetworkHyperbolic3D(args...; kw...) = NetworkHyperbolic(args...; kw..., ndims=3)

# Forward pass
function forward(X_prev::AbstractArray{T, N}, X_curr::AbstractArray{T, N}, H::NetworkHyperbolic) where {T, N}
    for j=1:length(H.HL)
        X_prev, X_curr = H.HL[j].forward(X_prev, X_curr)
    end
    return X_prev, X_curr, T(1)  # logdet is always 1
end

# Inverse pass
function inverse(Y_curr::AbstractArray{T, N}, Y_new::AbstractArray{T, N}, H::NetworkHyperbolic) where {T, N}
    for j=length(H.HL):-1:1
        Y_curr, Y_new = H.HL[j].inverse(Y_curr, Y_new)
    end
    return Y_curr, Y_new
end

# Backward pass
function backward(ΔY_curr::AbstractArray{T, N}, ΔY_new::AbstractArray{T, N}, Y_curr::AbstractArray{T, N}, Y_new::AbstractArray{T, N}, H::NetworkHyperbolic; set_grad::Bool=true) where {T, N}
    ~set_grad && (Δθ = Array{Parameter, 1}(undef, 0))
    for j=length(H.HL):-1:1
        if set_grad
            ΔY_curr, ΔY_new, Y_curr, Y_new = H.HL[j].backward(ΔY_curr, ΔY_new, Y_curr, Y_new)
        else
            ΔY_curr, ΔY_new, Δθ_HLj, Y_curr, Y_new = H.HL[j].backward(ΔY_curr, ΔY_new, Y_curr, Y_new; set_grad=set_grad)
            Δθ = cat(Δθ_HLj, Δθ; dims=1)
        end
    end
    set_grad ? (return ΔY_curr, ΔY_new, Y_curr, Y_new) : (return ΔY_curr, ΔY_new, Δθ, Y_curr, Y_new)
end

# Jacobian-related utils
function jacobian(ΔX_prev::AbstractArray{T, N}, ΔX_curr::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X_prev::AbstractArray{T, N}, X_curr::AbstractArray{T, N}, H::NetworkHyperbolic) where {T, N}
    npars_hl = Int64(length(Δθ)/length(H.HL))
    for j=1:length(H.HL)
        Δθj = Δθ[1+(j-1)*npars_hl:j*npars_hl]
        ΔX_prev, ΔX_curr, X_prev, X_curr = H.HL[j].jacobian(ΔX_prev, ΔX_curr, Δθj, X_prev, X_curr)
    end
    return ΔX_prev, ΔX_curr, X_prev, X_curr
end

adjointJacobian(ΔY_curr::AbstractArray{T, N}, ΔY_new::AbstractArray{T, N}, Y_curr::AbstractArray{T, N}, Y_new::AbstractArray{T, N}, H::NetworkHyperbolic) where {T, N} = backward(ΔY_curr, ΔY_new, Y_curr, Y_new, H; set_grad=false)
