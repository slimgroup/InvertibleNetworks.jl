export NeuralNetLayer, InvertibleNetwork, ReverseLayer, ReverseNetwork
export get_grads

abstract type Invertible end

# Base Layer and network types with property getters
abstract type NeuralNetLayer <: Invertible end
abstract type InvertibleNetwork <: Invertible end

# Concrete reversed types
struct Reversed <: Invertible
    I::Invertible
end

# Simple display
Base.show(io::IO, m::Invertible) = print(io, typeof(m))
Base.display(m::Invertible) = println(typeof(m))

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
getproperty(I::Invertible, s::Symbol) = _get_property(I, Val{s}())

_get_property(I::Invertible, ::Val{s}) where {s} = getfield(I, s)
_get_property(R::Reversed, ::Val{:I}) where s = getfield(R, :I)
_get_property(R::Reversed, ::Val{s}) where s = _get_property(R.I, Val{s}())

for m ∈ _INet_modes
    @eval _get_property(I::Union{InvertibleNetwork,NeuralNetLayer}, ::Val{$(Meta.quot(m))}) = (args...; kwargs...) -> _predefined_mode(I, $(Meta.quot(m)), args...; kwargs...)
end

for (m, k) ∈ _RNet_modes
    @eval _get_property(R::Reversed, ::Val{$(Meta.quot(m))}) = _get_property(R.I, Val{$(Meta.quot(k))}())
end

# Type conversions
function convert_params!(::Type{T}, obj::Invertible) where T
    for p ∈ get_params(obj)
        convert_param!(T, p)
    end
end

input_type(x::AbstractArray) = eltype(x)
input_type(x::Tuple) = eltype(x[1])

# Reverse
# For networks and layers not needing the tag
tag_as_reversed!(I::Invertible, ::Bool) = I

reverse(L::NeuralNetLayer) = Reversed(tag_as_reversed!(deepcopy(L), true))
reverse(N::InvertibleNetwork) = Reversed(tag_as_reversed!(deepcopy(N), true))
reverse(RL::Reversed) = tag_as_reversed!(deepcopy(RL.I), false)

"""
    P = get_params(NL::Invertible)

 Returns a cell array of all parameters in the network or layer. Each cell
 entry contains a reference to the original parameter; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
function get_params(I::Invertible)
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

get_params(x) = Array{Parameter}(undef, 0)
get_params(A::Array{T}) where T<:Union{Invertible, Nothing} = vcat([get_params(A[i]) for i in 1:length(A)]...)
get_params(A::Matrix{T}) where T<:Union{Invertible, Nothing} = vcat([get_params(A[i, j]) for i=1:size(A, 1) for j in 1:size(A, 2)]...)
get_params(RN::Reversed) = get_params(RN.I)

# reset! parameters
"""
    P = reset!(NL::Invertible)

 Resets the data of all the parameters in NL
"""
function reset!(I::Invertible)
    for p ∈ get_params(I)
        p.data = nothing
    end
end

reset!(AI::Array{<:Invertible}) = for I ∈ AI reset!(I) end

# Clear grad functionality for reversed layers/networks
"""
    P = clear_grad!(NL::Invertible)

 Resets the gradient of all the parameters in NL
"""
clear_grad!(I::Invertible) = clear_grad!(get_params(I))
clear_grad!(RL::Reversed) = clear_grad!(RL.I)

# Get gradients
"""
    P = get_grads(NL::Invertible)

 Returns a cell array of all parameters gradients in the network or layer. Each cell
 entry contains a reference to the original parameter's gradient; i.e. modifying
 the paramters in `P`, modifies the parameters in `NL`.
"""
get_grads(I::Invertible) = [Parameter(p.grad) for p ∈ get_params(I)]
get_grads(A::Array{Union{Invertible, Nothing}}) = vcat([get_grads(A[i]) for i in 1:length(A)]...)
get_grads(RL::Reversed)= get_grads(RL.I)
get_grads(::Nothing) = []

# Set parameters
function set_params!(N::Union{NeuralNetLayer, InvertibleNetwork}, θnew::Array{Parameter, 1})
    set_params!(get_params(N), θnew)
end

# Set params for reversed layers/networks
set_params!(RL::Reversed, θ::Array{Parameter, 1}) = set_params!(RL.I, θ)

# Make invertible nets callable objects
(N::Invertible)(X::AbstractArray{T,N} where {T, N}) = N.forward(X)
