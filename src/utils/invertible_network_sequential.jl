# Invertible network obtained from stringing invertible networks together
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020

export Sequential
import Base.length

struct Sequential <: InvertibleNetwork
    layers::Array{T, 1} where T <: Union{NeuralNetLayer, InvertibleNetwork}
end

@Flux.functor Sequential

# Constructor
function Sequential(other_layers...)

    depth = length(other_layers)
    n_array = Array{Union{NeuralNetLayer, InvertibleNetwork}, 1}(undef, depth)
    for i=1:depth
        n_array[i] = other_layers[i]
    end
    return Sequential(n_array)

end

# length
function length(N::Sequential)
    return length(N.layers)
end

# Forward pass and compute logdet
function forward(X, N::Sequential)
    logdet = 0f0
    for i = 1:length(N)
        X, logdet_ = N.layers[i].forward(X)
        logdet += logdet_
    end
    return X, logdet
end

# Inverse pass and compute gradients
function inverse(Y, N::Sequential)
    for i = length(N):-1:1
        Y = N.layers[i].inverse(Y)
    end
    return Y
end

# Backward pass and compute gradients
function backward(ΔY, Y, N::Sequential)
    for i = length(N):-1:1
        ΔY, Y = N.layers[i].backward(ΔY, Y)
    end
    return ΔY, Y
end

# Clear gradients
function clear_grad!(N::Sequential)
    for i = 1:length(N)
        clear_grad!(N.layers[i])
    end
end

# Get parameters
function get_params(N::Sequential)
    p = []
    for i=1:length(N)
        p = cat(p, get_params(N.layers[i]); dims=1)
    end
    return p
end
