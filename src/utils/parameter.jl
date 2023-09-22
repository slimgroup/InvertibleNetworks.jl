# Parameter of neural network
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export Parameter, get_params, get_grads, set_params!, par2vec, vec2par

mutable struct Parameter
    data
    grad
end

convert_data!(::Type{T}, ::Parameter, ::AbstractArray{T, N}) where {T, N} = nothing
convert_data!(::Type{T}, p::Parameter, data::AbstractArray{T2, N}) where {T, T2, N} = (p.data = convert(AbstractArray{T, N}, data))
convert_data!(::Type{T}, ::Parameter, ::Nothing) where T = nothing

convert_grad!(::Type{T}, ::Parameter, ::AbstractArray{T, N}) where {T, N} = nothing
convert_grad!(::Type{T}, p::Parameter, data::AbstractArray{T2, N}) where {T, T2, N} = (p.grad = convert(AbstractArray{T, N}, data))
convert_grad!(::Type{T}, ::Parameter, ::Nothing) where T = nothing

function convert_param!(::Type{T}, p::Parameter) where T
    convert_data!(T, p, p.data)
    convert_grad!(T, p, p.grad)
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
    clear_grad!(NL::InvertibleNetwork)

or

    clear_grad!(P::AbstractAbstractVector{<:Parameter})

 Set gradients of each `Parameter` in the network layer to `nothing`.
"""
function clear_grad!(P::AbstractVector{<:Parameter})
    for j=1:length(P)
        P[j].grad = nothing
    end
end

function get_grads(p::Parameter)
    return Parameter(p.grad)
end

function get_grads(pvec::AbstractVector{<:Parameter})
    g = Vector{Parameter}(undef, length(pvec))
    for i = 1:length(pvec)
        g[i] = get_grads(pvec[i])
    end
    return g
end

get_params(p::Parameter) = p

function set_params!(pold::Parameter, pnew::Parameter)
    pold.data = pnew.data
    pold.grad = pnew.grad
end

function set_params!(pold::AbstractVector{<:Parameter}, pnew::AbstractVector{<:Parameter})
    for i = 1:length(pold)
        set_params!(pold[i], pnew[i])
    end
end

function set_params!(pold::AbstractVector{<:Parameter}, pnew::AbstractVector{<:Any})
    for i = 1:length(pold)
        set_params!(pold[i], pnew[i])
    end
end



## Algebraic utilities for parameters

dot(p1::Parameter, p2::Parameter) = dot(p1.data, p2.data)
norm(p::Parameter) = norm(p.data)
+(p1::Parameter, p2::Parameter) = Parameter(p1.data+p2.data)
+(p1::Parameter, p2::T) where {T<:Real} = Parameter(p1.data+p2)
+(p1::T, p2::Parameter) where {T<:Real} = p2+p1
-(p1::Parameter, p2::Parameter) = Parameter(p1.data-p2.data)
-(p1::Parameter, p2::T) where {T<:Real} = Parameter(p1.data-p2)
-(p1::T, p2::Parameter) where {T<:Real} = -(p2-p1)
-(p::Parameter) = Parameter(-p.data)
*(p1::Parameter, p2::T) where {T<:Real} = Parameter(p1.data*p2) 
*(p1::T, p2::Parameter) where {T<:Real} = p2*p1
/(p1::Parameter, p2::T) where {T<:Real} = Parameter(p1.data/p2)
/(p1::T, p2::Parameter) where {T<:Real} = Parameter(p1/p2.data)

# Shape manipulation

par2vec(x::Parameter) = vec(x.data), size(x.data)


function vec2par(x::AbstractVector{T}, s::NTuple{N, Integer}) where {T, N}
    return Parameter(reshape(x, s))
end

function par2vec(x::AbstractVector{<:Parameter})
    v = cat([vec(x[i].data) for i=1:length(x)]..., dims=1)
    s = cat([size(x[i].data) for i=1:length(x)]..., dims=1)
    return v, s
end

function vec2par(x::AbstractVector{T}, s::AbstractVector) where T
    xpar = AbstractVector{<:Parameter}(undef, length(s))
    idx_i = 0
    for i = 1:length(s)
        xpar[i] = vec2par(x[idx_i+1:idx_i+prod(s[i])], s[i])
        idx_i += prod(s[i])
    end
    return xpar
end