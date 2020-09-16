# Parameter of neural network
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export Parameter
import LinearAlgebra.dot, LinearAlgebra.norm, Base.+, Base.*, Base.-, Base./

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

function set_params!(pold::Parameter, pnew::Parameter)
    pold.data = pnew.data
    pold.grad = pnew.grad
end


# Algebraic utilities for parameters
function dot(p1::Parameter, p2::Parameter)
    return dot(p1.data, p2.data)
end

function norm(p::Parameter)
    return norm(p.data)
end

function +(p1::Parameter, p2::Parameter)
    return Parameter(p1.data+p2.data)
end

function -(p1::Parameter, p2::Parameter)
    return Parameter(p1.data-p2.data)
end

function *(p1::Parameter, p2::Float32)
    return Parameter(p1.data*p2)
end

function *(p1::Float32, p2::Parameter)
    return Parameter(p1*p2.data)
end

function /(p1::Parameter, p2::Real)
    return Parameter(p1.data/p2)
end

function /(p1::Real, p2::Parameter)
    return Parameter(p1/p2.data)
end
