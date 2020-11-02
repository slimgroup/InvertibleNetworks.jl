export NeuralNetLayer, InvertibleNetwork, ReverseLayer, ReverseNetwork
export get_grads

# Base Layer and network types with property getters

abstract type NeuralNetLayer end

abstract type InvertibleNetwork end

function Base.show(io::IO, m::Union{NeuralNetLayer, InvertibleNetwork}) 
    println(typeof(m))
end


function Base.getproperty(obj::Union{InvertibleNetwork,NeuralNetLayer}, sym::Symbol)
    if sym == :forward
        return (args...; kwargs...) -> forward(args..., obj; kwargs...)
    elseif sym == :inverse
        return (args...; kwargs...) -> inverse(args..., obj; kwargs...)
    elseif sym == :backward
        return (args...; kwargs...) -> backward(args..., obj; kwargs...)
    elseif sym == :inverse_Y
        return (args...; kwargs...) -> inverse_Y(args..., obj; kwargs...)
    elseif sym == :forward_Y
        return (args...; kwargs...) -> forward_Y(args..., obj; kwargs...)
    elseif sym == :jacobian
        return (args...; kwargs...) -> jacobian(args..., obj; kwargs...)
    elseif sym == :jacobianInverse
        return (args...; kwargs...) -> jacobianInverse(args..., obj; kwargs...)
    elseif sym == :adjointJacobian
        return (args...; kwargs...) -> adjointJacobian(args..., obj; kwargs...)
    elseif sym == :adjointJacobianInverse
        return (args...; kwargs...) -> adjointJacobianInverse(args..., obj; kwargs...)
    else
         # fallback to getfield
        return getfield(obj, sym)
    end
end

abstract type ReverseLayer end

function Base.getproperty(obj::ReverseLayer, sym::Symbol)
    if sym == :forward
        return (args...; kwargs...) -> inverse(args..., obj.layer; kwargs...)
    elseif sym == :inverse
        return (args...; kwargs...) -> forward(args..., obj.layer; kwargs...)
    elseif sym == :backward
        return (args...; kwargs...) -> backward_inv(args..., obj.layer; kwargs...)
    elseif sym == :inverse_Y
        return (args...; kwargs...) -> forward_Y(args..., obj.layer; kwargs...)
    elseif sym == :forward_Y
        return (args...; kwargs...) -> inverse_Y(args..., obj.layer; kwargs...)
    elseif sym == :layer
        return getfield(obj, sym)
    else
         # fallback to getfield
        return getfield(obj.layer, sym)
    end
end


struct Reverse <: ReverseLayer
    layer::NeuralNetLayer
end

function reverse(L::NeuralNetLayer)
    L_rev = deepcopy(L)
    tag_as_reversed!(L_rev, true)
    return Reverse(L_rev)
end

function reverse(RL::ReverseLayer)
    R = deepcopy(RL)
    tag_as_reversed!(R.layer, false)
    return R.layer
end

abstract type ReverseNetwork end

function Base.getproperty(obj::ReverseNetwork, sym::Symbol)
    if sym == :forward
        return (args...; kwargs...) -> inverse(args..., obj.network; kwargs...)
    elseif sym == :inverse
        return (args...; kwargs...) -> forward(args..., obj.network; kwargs...)
    elseif sym == :backward
        return (args...; kwargs...) -> backward_inv(args..., obj.network; kwargs...)
    elseif sym == :inverse_Y
        return (args...; kwargs...) -> forward_Y(args..., obj.network; kwargs...)
    elseif sym == :forward_Y
        return (args...; kwargs...) -> inverse_Y(args..., obj.network; kwargs...)
    elseif sym == :network
        return getfield(obj, sym)
    else
         # fallback to getfield
        return getfield(obj.network, sym)
    end
end


struct ReverseNet <: ReverseNetwork
    network::InvertibleNetwork
end

function reverse(N::InvertibleNetwork)
    N_rev = deepcopy(N)
    tag_as_reversed!(N_rev, true)
    return ReverseNet(N_rev)
end

function reverse(RN::ReverseNetwork)
    R = deepcopy(RN)
    tag_as_reversed!(R.network, false)
    return R.network
end

# Clear grad functionality for reversed layers/networks

function clear_grad!(RL::ReverseLayer)
    clear_grad!(RL.layer)
end


function clear_grad!(RN::ReverseNetwork)
    clear_grad!(RN.network)
end

# Get params for reversed layers/networks

function get_params(RL::ReverseLayer)
    return get_params(RL.layer)
end

function get_params(RN::ReverseNetwork)
    return get_params(RN.network)
end

function get_grads(N::Union{NeuralNetLayer, InvertibleNetwork})
    return get_grads(get_params(N))
end

function get_grads(RL::ReverseLayer)
    return get_grads(RL.layer)
end

function get_grads(RN::ReverseNetwork)
    return get_grads(RN.network)
end

# Set parameters

function set_params!(N::Union{NeuralNetLayer, InvertibleNetwork}, θnew::Array{Parameter, 1})
    set_params!(get_params(N), θnew)
end

# Set params for reversed layers/networks

function set_params!(RL::ReverseLayer, θ::Array{Parameter, 1})
    return set_params!(RL.layer, θ)
end

function set_params!(RN::ReverseNetwork, θ::Array{Parameter, 1})
    return set_params!(RN.network, θ)
end
