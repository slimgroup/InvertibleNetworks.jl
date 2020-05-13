export NeuralNetLayer, InvertibleNetwork

# Base Layer and network types with property getters

abstract type NeuralNetLayer end

abstract type InvertibleNetwork end

function Base.getproperty(obj::Union{InvertibleNetwork,NeuralNetLayer}, sym::Symbol)
    if sym == :forward
        return (args...;kwargs...) -> forward(args..., obj;kwargs...)
    elseif sym == :inverse
        return (args...;kwargs...) -> inverse(args..., obj;kwargs...)
    elseif sym == :backward
        return (args...;kwargs...) -> backward(args..., obj;kwargs...)
    elseif sym == :inverse_Y
        return (args...;kwargs...) -> inverse_Y(args..., obj;kwargs...)
    elseif sym == :forward_Y
        return (args...;kwargs...) -> forward_Y(args..., obj;kwargs...)
    else
         # fallback to getfield
        return getfield(obj, sym)
    end
end

abstract type InverseLayer end

function Base.getproperty(obj::InverseLayer, sym::Symbol)
    if sym == :forward
        return (args...;kwargs...) -> inverse(args..., obj.layer;kwargs...)
    elseif sym == :inverse
        return (args...;kwargs...) -> forward(args..., obj.layer;kwargs...)
    elseif sym == :backward
        return (args...;kwargs...) -> backward_inv(args..., obj.layer;kwargs...)
    elseif sym == :inverse_Y
        return (args...;kwargs...) -> forward_Y(args..., obj.layer;kwargs...)
    elseif sym == :forward_Y
        return (args...;kwargs...) -> inverse_Y(args..., obj.layer;kwargs...)
    elseif sym == :layer
        return getfield(obj, sym)
    else
         # fallback to getfield
        return getfield(obj.layer, sym)
    end
end


struct Inv <: InverseLayer
    layer::NeuralNetLayer
end


function inverse(L::NeuralNetLayer)
    return Inv(L)
end

function inverse(IL::InverseLayer)
    return IL.layer
end