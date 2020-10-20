# Parameter of neural network
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export Parameter, get_params, get_grads, set_params!, par2vec, vec2par

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

# Size and length for parameter types
size(x::Parameter) = size(x.data)
length(x::Parameter) = length(x.data)


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

function get_grads(p::Parameter)
    return Parameter(p.grad)
end

function get_grads(pvec::Array{Parameter, 1})
    g = Array{Parameter, 1}(undef, length(pvec))
    for i = 1:length(pvec)
        g[i] = get_grads(pvec[i])
    end
    return g
end

function set_params!(pold::Parameter, pnew::Parameter)
    pold.data = pnew.data
    pold.grad = pnew.grad
end

function set_params!(pold::Array{Parameter, 1}, pnew::Array{Parameter, 1})
    for i = 1:length(pold)
        set_params!(pold[i], pnew[i])
    end
end


## Algebraic utilities for parameters

function dot(p1::Parameter, p2::Parameter)
    return dot(p1.data, p2.data)
end

function norm(p::Parameter)
    return norm(p.data)
end

function +(p1::Parameter, p2::Parameter)
    return Parameter(p1.data+p2.data)
end

function +(p1::Parameter, p2::Float32)
    return Parameter(p1.data+p2)
end

function +(p1::Float32, p2::Parameter)
    return p2+p1
end

function -(p1::Parameter, p2::Parameter)
    return Parameter(p1.data-p2.data)
end

function -(p1::Parameter, p2::Float32)
    return Parameter(p1.data-p2)
end

function -(p1::Float32, p2::Parameter)
    return -(p2-p1)
end

function -(p::Parameter)
    return Parameter(-p.data)
end

function *(p1::Parameter, p2::Float32)
    return Parameter(p1.data*p2)
end

function *(p1::Float32, p2::Parameter)
    return p2*p1
end

function /(p1::Parameter, p2::Float32)
    return Parameter(p1.data/p2)
end

function /(p1::Float32, p2::Parameter)
    return Parameter(p1/p2.data)
end

# Shape manipulation

function par2vec(x::Parameter)
    return vec(x.data), size(x.data)
end

function vec2par(x::Array{Float32, 1}, s::NTuple{N, Int64}) where {N}
    return Parameter(reshape(x, s))
end

function par2vec(x::Array{Parameter, 1})
    v = Array{Float32, 1}(undef, 0)
    s = Array{Any, 1}(undef, 0)
    for i = 1:length(x)
        vi, si = par2vec(x[i])
        v = cat(v, vi; dims=1)
        s = cat(s, si; dims=1)
    end
    return v, s
end

function vec2par(x::Array{Float32, 1}, s::Array{Any, 1})
    xpar = Array{Parameter, 1}(undef, length(s))
    idx_i = 0
    for i = 1:length(s)
        xpar[i] = vec2par(x[idx_i+1:idx_i+prod(s[i])], s[i])
        idx_i += prod(s[i])
    end
    return xpar
end