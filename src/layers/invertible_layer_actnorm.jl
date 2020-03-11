# Activation normalization layer
# Adapted from Kingma and Dhariwal (2018): https://arxiv.org/abs/1807.03039
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
#

export ActNorm

"""
    AN = ActNorm(k; logdet=false)

 Create activation normalization layer. The parameters are initialized during
 the first use, such that the output has zero mean and unit variance along
 channels for the current mini-batch size.

 *Input*: 
 
 - `k`: number of channels 
 
 - `logdet`: bool to indicate whether to compute the logdet

 *Output*:
 
 - `AN`: Network layer for activation normalization.

 *Usage:*

 - Forward mode: `Y, logdet = AN.forward(X)`

 - Inverse mode: `X = AN.inverse(Y)`

 - Backward mode: `ΔX, X = AN.backward(ΔY, Y)`

 *Trainable parameters:*

 - Scaling factor `AN.s`

 - Bias `AN.b`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ActNorm <: NeuralNetLayer
    k::Integer
    s::Parameter
    b::Parameter
    logdet::Bool
    forward::Function
    inverse::Function
    backward::Function
end

# Constructor: Initialize with nothing
function ActNorm(k; logdet=false)
    s = Parameter(nothing)
    b = Parameter(nothing)
    return ActNorm(k, s, b, logdet,
        X -> actnorm_forward(X, k, s, b, logdet),
        Y -> actnorm_inverse(Y, k, s, b),
        (ΔY, Y) -> actnorm_backward(ΔY, Y, k, s, b, logdet)
    )
end

# Foward pass: Input X, Output Y
function actnorm_forward(X, k, s, b, logdet)
    nx, ny, n_in, batchsize = size(X)

    # Initialize during first pass such that 
    # output has zero mean and unit variance
    if s.data == nothing
        μ = mean(X; dims=(1,2,4))[1,1,:,1]
        σ_sqr = var(X; dims=(1,2,4))[1,1,:,1]
        s.data = 1f0 ./ sqrt.(σ_sqr .+ eps(1f0))
        b.data = -μ ./ sqrt.(σ_sqr .+ eps(1f0))
    end
    Y = X .* reshape(s.data, 1, 1, :, 1) .+ reshape(b.data, 1, 1, :, 1)
    
    # If logdet true, return as second ouput argument
    logdet == true ? (return Y, logdet_forward(nx, ny, s)) : (return Y)
end

# Inverse pass: Input Y, Output X
function actnorm_inverse(Y, k, s, b)
    ϵ = randn(Float32, size(s.data)) .* eps(1f0)
    X = (Y .- reshape(b.data, 1, 1, :, 1)) ./ reshape(s.data + ϵ, 1, 1, :, 1)   # avoid division by 0
    return X
end

# Backward pass: Input (ΔY, Y), Output (ΔY, Y)
function actnorm_backward(ΔY, Y, k, s, b, logdet)
    nx, ny, n_in, batchsize = size(Y)
    X = actnorm_inverse(Y, k, s, b)
    ΔX = ΔY .* reshape(s.data, 1, 1, :, 1)
    Δs = sum(ΔY .* X, dims=(1,2,4))[1,1,:,1]
    logdet == true && (Δs -= logdet_backward(nx, ny, s))
    Δb = sum(ΔY, dims=(1,2,4))[1,1,:,1]
    s.grad = Δs
    b.grad = Δb
    return ΔX, X
end

# Clear gradients
function clear_grad!(AN::ActNorm)
    AN.s.grad = nothing
    AN.b.grad = nothing
end

# Get parameters
get_params(AN::ActNorm) = [AN.s, AN.b]

# Logdet
logdet_forward(nx, ny, s) = nx*ny*sum(log.(abs.(s.data))) 
logdet_backward(nx, ny, s) = nx*ny ./ s.data
