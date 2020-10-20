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
struct FluxBlock <: NeuralNetLayer
    model::Chain
    params::Array{Parameter}
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
forward(X::AbstractArray{Float32, 4}, FB::FluxBlock) = FB.model(X)


# Backward 2D
function backward(ΔY::AbstractArray{Float32, 4}, X::AbstractArray{Float32, 4}, FB::FluxBlock)
    
    # Backprop using Zygote
    θ = Flux.params(X, FB.model)
    back = Zygote.pullback(() -> FB.model(X), θ)[2]
    grad = back(ΔY)

    # Set gradients
    ΔX = grad[θ[1]]
    for j=1:length(FB.params)
        FB.params[j].grad = grad[θ[j+1]]
    end
    return ΔX
end


## Jacobian utilities

function jacobian(ΔX::AbstractArray{Float32, 4}, Δθ::Array{Parameter, 1}, X::AbstractArray{Float32, 4}, FB::FluxBlock)
    throw(ArgumentError("Jacobian for Flux block not yet implemented, sorry :("))
end

function adjointJacobian(ΔY::AbstractArray{Float32, 4}, X::AbstractArray{Float32, 4}, FB::FluxBlock)

    # Backprop using Zygote
    θ = Flux.params(X, FB.model)
    back = Zygote.pullback(() -> FB.model(X), θ)[2]
    grad = back(ΔY)

    # Set gradients
    ΔX = grad[θ[1]]
    Δθ = Array{Params, 1}(undef, length(FB.params))
    for j = 1:length(FB.params)
        Δθ[j] = grad[θ[j+1]]
    end
    return ΔX, Δθ

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
    P = get_params(NL::NeuralNetLayer)

 Returns a cell array of all parameters in the network layer. Each cell
 entry contains a reference to the original parameter; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
function get_params(FB::FluxBlock)
    params = Array{Parameter, 1}(undef, length(FB.params))
    for j=1:length(FB.params)
        params[j] = FB.params[j]
    end
    return params
end