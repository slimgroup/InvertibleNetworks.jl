# Residual block from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export FluxBlock

"""
    FB = FluxBlock(model::Chain)

 Create a (non-invertible) neural network block from a Flux network.

 *Input*: 

 - `model`: Flux neural network of type `Chain`

 *Output*:
 
 - `FB`: residual block layer

 *Usage:*

 - Forward mode: `Y = FB.forward(X)`

 - Backward mode: `ΔX = FB.backward(ΔY, X)`

 *Trainable parameters:*

 - Network parameters given by `Flux.parameters(model)`

 See also:  [`Chain`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct FluxBlock <: NeuralNetLayer
    model::Chain
    params::Array{Parameter, 1}
end

@Flux.functor FluxBlock

#######################################################################################################################
# Constructor

function FluxBlock(model::Chain)

    # Collect Flux parameters
    model_params = Flux.params(model)
    nparam = length(model_params.order)
    params = Array{Parameter}(undef, nparam)

    # Create InvertibleNetworks parameter
    for j=1:nparam
        params[j] = Parameter(model_params.order[j])
    end
    return FluxBlock(model, params)
end


#######################################################################################################################
# Functions

# Forward 
forward(X::AbstractArray{T, N}, FB::FluxBlock) where {T, N} = FB.model(X)


# Backward 2D
function backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, FB::FluxBlock; set_grad::Bool=true) where {T, N}
    
    # Backprop using Zygote
    θ = Flux.params(X, FB.model)
    back = Flux.Zygote.pullback(() -> FB.model(X), θ)[2]
    grad = back(ΔY)

    # Set gradients
    ΔX = grad[θ[1]]
    if set_grad
        for j=1:length(FB.params)
            FB.params[j].grad = grad[θ[j+1]]
        end
    else
        Δθ = Array{Parameter, 1}(undef, length(FB.params))
        for j=1:length(FB.params)
            Δθ[j] = grad[θ[j+1]]
        end
    end

    set_grad ? (return ΔX) : (ΔX, Δθ)

end


## Jacobian utilities

function jacobian(::AbstractArray{T, N}, ::Array{Parameter, 1}, ::AbstractArray{T, N}, ::FluxBlock) where {T, N}
    throw(ArgumentError("Jacobian for Flux block not yet implemented"))
end

function adjointJacobian(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, FB::FluxBlock) where {T, N}
    return backward(ΔY, X, FB; set_grad=false)
end


## Other utils

# Clear gradients
function clear_grad!(FB::FluxBlock)
    nparams = length(FB.params)
    for j=1:nparams
        FB.params[j].grad = nothing
    end
end

"""
    P = get_params(NL::FluxBlock)

 Returns directly the Flux params array
"""
function get_params(FB::FluxBlock)
    return FB.params
end

function set_params!(FB::FluxBlock, θ::Array{Parameter, 1})
    model_params = Flux.params(FB.model)
    nparams = length(model_params.order)
    for j=1:nparams
        model_params.order[j] .= θ[j].data
        FB.params[j].data = θ[j].data
        FB.params[j].grad = θ[j].grad
    end
end