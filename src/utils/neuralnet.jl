export InvertibleNetwork
export get_grads

# Invertible network abstract type
abstract type NeuralNetwork end
abstract type InvertibleNetwork <: NeuralNetwork end

# Reversed invertible network type (concrete)
struct ReversedNetwork <: InvertibleNetwork
    I::InvertibleNetwork
end

@Flux.functor ReversedNetwork

# Simple display
Base.show(io::IO, m::NeuralNetwork) = print(io, typeof(m))
Base.display(m::NeuralNetwork) = println(typeof(m))

# Propagation modes
_INet_modes = [:forward, :inverse, :backward, :backward_inv, :inverse_Y, :forward_Y,
               :jacobian, :jacobianInverse, :adjointJacobian, :adjointJacobianInverse]

_RNet_modes = Dict(:forward=>:inverse, :inverse=>:forward, :backward=>:backward_inv,
                   :inverse_Y=>:forward_Y, :forward_Y=>:inverse_Y)

# Actual call to propagation function
function _predefined_mode(obj, sym::Symbol, args...; kwargs...)
    convert_params!(input_type(args[1]), obj)
    eval(sym)(args..., obj; kwargs...)
end

# Base getproperty
getproperty(I::NeuralNetwork, s::Symbol) = _get_property(I, Val{s}())

_get_property(I::NeuralNetwork, ::Val{s}) where {s} = getfield(I, s)
_get_property(R::ReversedNetwork, ::Val{:I}) = getfield(R, :I)
_get_property(R::ReversedNetwork, ::Val{s}) where s = _get_property(R.I, Val{s}())

for m ∈ _INet_modes
    @eval _get_property(I::NeuralNetwork, ::Val{$(Meta.quot(m))}) = (args...; kwargs...) -> _predefined_mode(I, $(Meta.quot(m)), args...; kwargs...)
end

for (m, k) ∈ _RNet_modes
    @eval _get_property(R::ReversedNetwork, ::Val{$(Meta.quot(m))}) = _get_property(R.I, Val{$(Meta.quot(k))}())
end

# Type conversions
function convert_params!(::Type{T}, obj::NeuralNetwork) where T
    for p ∈ get_params(obj)
        convert_param!(T, p)
    end
end

input_type(x::AbstractArray) = eltype(x)
input_type(x::Tuple) = eltype(x[1])

# Reverse
# For networks and layers not needing the tag
tag_as_reversed!(I::InvertibleNetwork, ::Bool) = I

reverse(N::InvertibleNetwork) = ReversedNetwork(tag_as_reversed!(deepcopy(N), true))
reverse(RL::ReversedNetwork) = tag_as_reversed!(deepcopy(RL.I), false)

"""
    P = get_params(NL::NeuralNetwork)

 Returns a cell array of all parameters in the network or layer. Each cell
 entry contains a reference to the original parameter; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
function get_params(I::NeuralNetwork)
    params = Vector{Parameter}(undef, 0)
    for (f, tp) ∈ zip(fieldnames(typeof(I)), typeof(I).types)
        p = getfield(I, f)
        if tp == Parameter
            append!(params, [p])
        else
            append!(params, get_params(p))
        end
    end
    params
end

get_params(::Any) = Array{Parameter}(undef, 0)
get_params(A::AbstractArray{T}) where {T <: Union{NeuralNetwork, Nothing}} = vcat([get_params(A[i]) for i in 1:length(A)]...)
get_params(A::AbstractMatrix{T}) where {T <: Union{NeuralNetwork, Nothing}} = vcat([get_params(A[i, j]) for i=1:size(A, 1) for j in 1:size(A, 2)]...)
get_params(RN::ReversedNetwork) = get_params(RN.I)

# reset! parameters
"""
    P = reset!(NL::NeuralNetwork)

 Resets the data of all the parameters in NL
"""
function reset!(I::NeuralNetwork)
    for p ∈ get_params(I)
        p.data = nothing
    end
end

reset!(AI::AbstractArray{<:NeuralNetwork}) = for I ∈ AI reset!(I) end

# Clear grad functionality for reversed layers/networks
"""
    P = clear_grad!(NL::NeuralNetwork)

 Resets the gradient of all the parameters in NL
"""
clear_grad!(I::NeuralNetwork) = clear_grad!(get_params(I))

# Get gradients
"""
    P = get_grads(NL::NeuralNetwork)

 Returns a cell array of all parameters gradients in the network or layer. Each cell
 entry contains a reference to the original parameter's gradient; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
get_grads(I::NeuralNetwork) = [Parameter(p.grad) for p ∈ get_params(I)]
get_grads(A::AbstractArray{T}) where {T <: Union{NeuralNetwork, Nothing}} = vcat([get_grads(A[i]) for i in 1:length(A)]...)
get_grads(RL::ReversedNetwork)= get_grads(RL.I)
get_grads(::Nothing) = []

# Set parameters
function set_params!(N::NeuralNetwork, θnew::AbstractVector{<:Parameter})
    set_params!(get_params(N), θnew)
end

# Set parameters with BSON loaded params
function set_params!(N::NeuralNetwork, θnew::AbstractVector{<:Any})
    set_params!(get_params(N), θnew)
end

# Set parameters for reversed networks
set_params!(Nrev::ReversedNetwork, θnew::AbstractVector{<:Parameter}) = set_params!(Nrev.I, θnew)

# Make Invertible nets callable objects
(net::NeuralNetwork)(X::AbstractArray{T,N} where {T, N}) = forward_net(net, X, getfield.(get_params(net), :data))
forward_net(net::NeuralNetwork, X::AbstractArray{T,N}, ::Any) where {T, N} = net.forward(X)

# Make conditional Invertible nets callable objects
(net::NeuralNetwork)(X::AbstractArray{T,N}, Y::AbstractArray{T,N}) where {T, N} = forward_net(net, X, Y, getfield.(get_params(net), :data))
forward_net(net::NeuralNetwork, X::AbstractArray{T,N}, Y::AbstractArray{T,N}, ::Any) where {T, N} = net.forward(X,Y)