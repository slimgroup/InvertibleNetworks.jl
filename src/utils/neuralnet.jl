export NeuralNetLayer, InvertibleNetwork

# Base Layer and network types with property getters

abstract type NeuralNetLayer end

abstract type InvertibleNetwork end

function Base.getproperty(obj::Union{InvertibleNetwork,NeuralNetLayer}, sym::Symbol)
    if sym == :forward
        return (args...) -> forward(args..., obj)
    elseif sym == :inverse
        return (args...) -> inverse(args..., obj)
    elseif sym == :backward
        return (args...) -> backward(args..., obj)
    elseif sym == :inverse_Y
        return (args...) -> inverse_Y(args..., obj)
    elseif sym == :forward_Y
        return (args...) -> forward_Y(args..., obj)
    else
         # fallback to getfield
        return getfield(obj, sym)
    end
end

abstract type InverseLayer end

function Base.getproperty(obj::InverseLayer, sym::Symbol)
    if sym == :forward
        return (args...) -> inverse(args..., obj.layer)
    elseif sym == :inverse
        return (args...) -> forward(args..., obj.layer)
    elseif sym == :backward
        return (args...) -> backward_inv(args..., obj.layer)
    elseif sym == :inverse_Y
        return (args...) -> forward_Y(args..., obj.layer)
    elseif sym == :forward_Y
        return (args...) -> inverse_Y(args..., obj.layer)
    else
         # fallback to getfield
        return getfield(obj, sym)
    end
end


struct Inv
    layer::NeuralNetLayer
end


function inverse(L::NeuralNetLayer)
    return Inv(L)
end

function inverse(IL::Inv)
    return IL.layer
end