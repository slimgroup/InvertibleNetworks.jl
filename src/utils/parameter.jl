# Parameter of neural network
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export NeuralNetLayer, Parameter

mutable struct Parameter
    data
    grad
end


"""
    Class for trainable network parameters.

 *Fields:*

 - `Parameter.data`: weights

 - `Parameter.grad`: gradient
    
"""
Parameter(x) = Parameter(x, nothing)

size(x::Parameter) = size(x.data)

@Flux.functor Parameter

"""
    clear_grad!(NL::NeuralNetLayer)

or

    clear_grad!(P::AbstractArray{Parameter, 1})

 Set gradients of each `Parameter` in the network layer to `nothing`. 
"""
function clear_grad!(P::AbstractArray{Parameter, 1})
    for j=1:length(P)
        P[j].grad = nothing
    end
end

abstract type NeuralNetLayer end

abstract type InvertibleNetwork end
