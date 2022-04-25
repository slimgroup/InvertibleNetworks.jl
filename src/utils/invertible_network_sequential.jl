# Invertible network obtained from stringing invertible networks together
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020

export ComposedInvertibleNetwork, Composition
import Base.length, Base.∘

struct ComposedInvertibleNetwork <: InvertibleNetwork
    layers::Array{T, 1} where {T <: Union{NeuralNetLayer, InvertibleNetwork}}
    logdet_array::Array{Bool, 1}
    logdet::Bool
    npars::Array{Int64, 1}
end

@Flux.functor ComposedInvertibleNetwork


## Constructors

function Composition(layer...)

    # Initializing output
    depth = length(layer)
    net_array = Array{Union{NeuralNetLayer, InvertibleNetwork}, 1}(undef, depth)
    logdet_array = Array{Bool, 1}(undef, depth)
    logdet = false
    npars = Array{Int64, 1}(undef, depth)

    # Loop over layers
    for i = 1:depth

        # Selecting layer (last-to-first)
        layer_i = layer[depth-i+1]

        # Setting layer/logdets/# of parameters
        net_array[i] = layer_i
        if hasproperty(layer_i, :logdet) && layer_i.logdet
            logdet_array[i] = true
            logdet = true
        else
            logdet_array[i] = false
        end
        npars[i] = length(get_params(layer_i))
    end

    return ComposedInvertibleNetwork(net_array, logdet_array, logdet, npars)

end


## Composition utilities

function ∘(net1::ComposedInvertibleNetwork, net2::ComposedInvertibleNetwork)
    return Composition(cat(net1.layers[end:-1:1], net2.layers[end:-1:1]; dims=1)...)
end

function ∘(net1::Union{NeuralNetLayer, InvertibleNetwork}, net2::Union{NeuralNetLayer, InvertibleNetwork})
    return Composition(cat(net1, net2; dims=1)...)
end

function ∘(net1::Union{NeuralNetLayer, InvertibleNetwork}, net2::ComposedInvertibleNetwork)
    return Composition(cat(net1, net2.layers[end:-1:1]; dims=1)...)
end

function ∘(net1::ComposedInvertibleNetwork, net2::Union{NeuralNetLayer, InvertibleNetwork})
    return Composition(cat(net1.layers[end:-1:1], net2; dims=1)...)
end

function length(N::ComposedInvertibleNetwork)
    return length(N.layers)
end


## Forward/inverse/backward

function forward(X::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    N.logdet && (logdet = 0)
    for i = 1:length(N)
        if N.logdet_array[i]        
            X, logdet_ = N.layers[i].forward(X)
            logdet += logdet_
        else
            X = N.layers[i].forward(X)
        end
    end
    N.logdet ? (return X, logdet) : (return X)
end

function inverse(Y::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    for i = length(N):-1:1
        Y = N.layers[i].inverse(Y)
    end
    return Y
end

function backward(ΔY::AbstractArray{T, N1}, Y::AbstractArray{T, N1}, N::ComposedInvertibleNetwork; set_grad::Bool = true) where {T, N1}
    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 0)
        N.logdet && (∇logdet = Array{Parameter, 1}(undef, 0))
    end
    for i = length(N):-1:1
        if set_grad
            ΔY, Y = N.layers[i].backward(ΔY, Y)
        else
            if N.logdet_array[i]
                ΔY, Δθi, Y, ∇logdet_i = N.layers[i].backward(ΔY, Y; set_grad=set_grad)
                ∇logdet = cat(∇logdet_i, ∇logdet; dims=1)
            else
                ΔY, Δθi, Y = N.layers[i].backward(ΔY, Y; set_grad=set_grad)
            end
            Δθ = cat(Δθi, Δθ; dims=1)
        end
    end
    if set_grad
        return ΔY, Y
    else
        N.logdet ? (return ΔY, Δθ, Y, ∇logdet) : (return ΔY, Δθ, Y)
    end
end


## Jacobian-related utilities

function jacobian(ΔX::AbstractArray{T, N1}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    N.logdet && (l = 0; GNΔθ = Array{Parameter, 1}(undef, 0))
    idx_pars = 0
    for i = 1:length(N)
        npars_i = N.npars[i]
        Δθi = Δθ[idx_pars+1:idx_pars+npars_i]
        if N.logdet_array[i]
            ΔX, X, li, GNΔθi = N.layers[i].jacobian(ΔX, Δθi, X)
            l += li
            GNΔθ = cat(GNΔθ, GNΔθi; dims=1)
        else
            ΔX, X = N.layers[i].jacobian(ΔX, Δθi, X)
        end
        idx_pars += npars_i
    end
    N.logdet ? (return ΔX, X, l, GNΔθ) : (return ΔX, X)
end

function adjointJacobian(ΔY::AbstractArray{T, N1}, Y::AbstractArray{T, N1}, N::ComposedInvertibleNetwork) where {T, N1}
    return backward(ΔY, Y, N; set_grad = false)
end
