# Activation functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ReLU, ReLUgrad
export LeakyReLU, LeakyReLUinv, LeakyReLUgrad
export Sigmoid, SigmoidInv, SigmoidGrad
export GaLU, GaLUgrad
export ExpClamp, ExpClampInv, ExpClampGrad
export ReLUlayer, LeakyReLUlayer, SigmoidLayer, Sigmoid2Layer, GaLUlayer, ExpClampLayer


###############################################################################
# Custom type for activation functions

struct ActivationFunction
    forward::Function
    inverse::Union{Nothing, Function}
    backward::Function
end

function ReLUlayer()
    return ActivationFunction(ReLU, nothing, ReLUgrad)
end

function LeakyReLUlayer()
    return ActivationFunction(LeakyReLU, LeakyReLUinv, LeakyReLUgrad)
end

function SigmoidLayer(;low=0f0, high=1f0)
    fwd_a(x) = Sigmoid(x; low=low, high=high)
    inv_a(y) = SigmoidInv(y; low=low, high=high)
    grad_a(Δy, y; x=nothing) = SigmoidGrad(Δy, y; x=x, low=low, high=high)
    return ActivationFunction(fwd_a, inv_a, grad_a)
end

function Sigmoid2Layer()
    fwd_a(x) = 2f0*Sigmoid(x)
    inv_a(y) = SigmoidInv(y/2f0)
    grad_a(Δy, y; x=nothing) = SigmoidGrad(Δy*2f0, y/2f0; x=x)
    return ActivationFunction(fwd_a, inv_a, grad_a)
end

function GaLUlayer()
    return ActivationFunction(GaLU, nothing, GaLUgrad)
end

function ExpClampLayer()
    return ActivationFunction(x -> ExpClamp(x), y -> ExpClampInv(y/2f0), (Δy, y) -> ExpClampGrad(Δy*2f0, y/2f0))
end


###############################################################################
# Rectified linear unit (ReLU) (not invertible)

"""
    y = ReLU(x)

 Rectified linear unit (not invertible).

 See also: [`ReLUgrad`](@ref)
"""
ReLU(x::AbstractArray{T, N}) where {T, N} =  relu.(x)

"""
    Δx = ReLUgrad(Δy, x)

 Backpropagate data residual through ReLU function.

 *Input*:

 - `Δy`: data residual

 - `x`: original input (since not invertible)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`ReLU`](@ref)
"""
ReLUgrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N} = _relugrad.(Δy, x)

_relugrad(Δy, x) = ifelse(x < 0, zero(x), Δy)

###############################################################################
# Leaky ReLU (invertible)

"""
    y = LeakyReLU(x; slope=0.01f0)

 Leaky rectified linear unit.

 See also: [`LeakyReLUinv`](@ref), [`LeakyReLUgrad`](@ref)
"""
LeakyReLU(x::AbstractArray{T, N}; slope=T(0.01)) where {T, N} = leakyrelu.(x, slope)

"""
    x = LeakyReLUinv(y; slope=0.01f0)

 Inverse of leaky ReLU.

 See also: [`LeakyReLU`](@ref), [`LeakyReLUgrad`](@ref)
"""
LeakyReLUinv(y::AbstractArray{T, N}; slope=T(0.01)) where {T, N} = _lreluinv.(y, slope)

_lreluinv(y::T, slope=T(0.01)) where T = ifelse(y < 0, y/slope, y)

"""
    Δx = LeakyReLUgrad(Δy, x; slope=0.01f0)

 Backpropagate data residual through leaky ReLU function.

 *Input*:

 - `Δy`: data residual

 - `x`: original input (since not invertible)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`LeakyReLU`](@ref), [`LeakyReLUinv`](@ref)
"""

"""
    Δx = ReLUgrad(Δy, y; slope=0.01f0)

 Backpropagate data residual through leaky ReLU function.

 *Input*:

 - `Δy`: residual

 - `y`: original output

 - `slope`: slope of non-active part of ReLU

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`LeakyReLU`](@ref), [`LeakyReLUinv`](@ref)
"""
LeakyReLUgrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; slope=T(0.01)) where {T, N} = _lrelugrad.(Δy, y, slope)

_lrelugrad(Δy::T, y::T, slope=T(0.01)) where T = ifelse(_lreluinv(y, slope) < 0, Δy*slope, Δy)

###############################################################################
# Sigmoid (invertible if ouput nonzero)

"""
    y = Sigmoid(x; low=0, high=1)

 Sigmoid activation function. Shifted and scaled such that output is [low,high].

 See also: [`SigmoidInv`](@ref), [`SigmoidGrad`](@ref)
"""
Sigmoid(x::AbstractArray{T, N}; low=0f0, high=1f0) where {T, N} = _sigmoid.(x, low, high)

_sigmoid(x::T, low=T(0), high=T(1)) where T = high/(1+exp(-x)) + low/(1+exp(x))

"""
    x = SigmoidInv(y; low=0, high=1)

 Inverse of Sigmoid.

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""


"""
    x = SigmoidInv(y; low=0, high=1f0)

 Inverse of Sigmoid function. Shifted and scaled such that output is [low,high]

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""
_sigmoidinv(y::T, low=T(0), high=T(1)) where T = log(y - low) - log(high - y)

function SigmoidInv(y::AbstractArray{T, N}; low=0f0, high=1f0) where {T, N}
    if sum(isapprox.(y, 0f-6)) == 0
        return _sigmoidinv.(y, low, high)
    else
        throw(InputError("Input contains zeros."))
    end
end

"""
    Δx = SigmoidGrad(Δy, y; x=nothing, low=nothing, high=nothing)

 Backpropagate data residual through Sigmoid function. Can be shifted and scaled such that output is (low,high]

 *Input*:

 - `Δy`: residual

 - `y`: original output

 - `x`: original input, if y not available (in this case, set y=nothing)

 - `low`: if provided then scale and shift such that output is (low,high]

 - `high`: if provided then scale and shift such that output is (low,high]

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`Sigmoid`](@ref), [`SigmoidInv`](@ref)
"""
SigmoidGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; x=nothing, low=0f0, high=1f0) where {T, N} = _sigmoidgrad.(x, Δy, y, low, high)
SigmoidGrad(Δy::AbstractArray{T, N}, ::Nothing; x=nothing, low=0f0, high=1f0) where {T, N} = _sigmoidgrad.(x, Δy, nothing, low, high)

_sigmoidgrad(::Nothing, Δy::T, y::T, low=T(0), high=T(1)) where T = _sigmoidgrad(_sigmoidinv(y, low, high), Δy, y, low, high)
_sigmoidgrad(x::T, Δy::T, y, low=T(0), high=T(1)) where T = (high - low) * Δy * exp(-x) / (1 + exp(-x))^2

###############################################################################
# Gated linear unit (GaLU) (not invertible)
# Adapted from Dauphin et al. (2017)

"""
    y = GaLU(x)

 Gated linear activation unit (not invertible).

 See also: [`GaLUgrad`](@ref)
"""
@inline function GaLU(x::AbstractArray{T, N}) where {T, N}
    x1, x2 = tensor_split(x)
    return x1 .* Sigmoid(x2)
end

"""
    Δx = GaLUgrad(Δy, x)

 Backpropagate data residual through GaLU activation.

 *Input*:

 - `Δy`: residual

 - `x`: original input (since not invertible)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`GaLU`](@ref)
"""
function GaLUgrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    k = Int(size(x, N-1) / 2)
    x1, x2 = tensor_split(x)
    Δx = 0 .*x
    return tensor_cat(Sigmoid(x2) .* Δy, SigmoidGrad(Δy, nothing; x=x2) .* x1)
end

function GaLUjacobian(Δx::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    k = Int(size(x, 3) / 2)
    x1, x2 = tensor_split(x)
    Δx1, Δx2 = tensor_split(Δx)
    s = Sigmoid(x2)
    Δs = SigmoidGrad(Δx2, nothing; x=x2)
    y = x1 .* s
    Δy = Δx1 .* s + x1 .* Δs
    return Δy, y
end

###############################################################################
# Soft-clamped exponential function

"""
    y = ExpClamp(x)
 Soft-clamped exponential function.
 See also: [`ExpClampGrad`](@ref)
"""
ExpClamp(x::AbstractArray{T, N}; clamp=T(2)) where {T, N} = exp.(clamp * T(0.636) * atan.(x))

"""
    x = ExpClampInv(y)
 Inverse of ExpClamp function.
 See also: [`ExpClamp`](@ref), [`ExpClampGrad`](@ref)
"""
function ExpClampInv(y::AbstractArray{T, N}; clamp=T(2)) where {T, N}
    if any(y .≈ 0)
        throw(InputError("Input contains zeros."))
    else
        return tan.(log.(y) / clamp / T(0.636))
    end
end

"""
    Δx = ExpClampGrad(Δy, x; y=nothing)
 Backpropagate data residual through soft-clamped exponential function.
 *Input*:
 - `Δy`: residual
 - `x`: original input
 *Output*:
 - `Δx`: backpropagated residual
 See also: [`ExpClamp`](@ref)
"""

function ExpClampGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; x=nothing, clamp=T(2)) where {T, N}
    if isnothing(x)
        x = ExpClampInv(y)  # recompute forward state
    end
    return clamp * T(0.636) * Δy .* y ./ (1 .+ x.^2)
end

ExpClampGrad(Δy::AbstractArray{T, N}, ::Nothing; x=nothing, clamp=T(2)) where {T, N} = clamp * T(0.636) * Δy .* y ./ (1 .+ x.^2)