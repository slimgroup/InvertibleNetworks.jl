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
    return ActivationFunction(x -> Sigmoid(x; low=low, high=high), y -> SigmoidInv(y;low=low, high=high), (Δy, y) -> SigmoidGrad(Δy, y; low=low, high=high))
end

function Sigmoid2Layer()
    return ActivationFunction(x -> 2f0*Sigmoid(x), y -> SigmoidInv(y/2f0), (Δy, y) -> SigmoidGrad(Δy*2f0, y/2f0))
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
function ReLU(x)
    return max.(0f0, x)
end

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
function ReLUgrad(Δy, x)
    return Δy.*(sign.(x) .+ 1)/2
end

###############################################################################
# Leaky ReLU (invertible)

"""
    y = LeakyReLU(x; slope=0.01f0)

 Leaky rectified linear unit.

 See also: [`LeakyReLUinv`](@ref), [`LeakyReLUgrad`](@ref)
"""
function LeakyReLU(x; slope=0.01f0)
    return max.(0f0, x) + slope*min.(0f0, x)
end

"""
    x = LeakyReLUinv(y; slope=0.01f0)

 Inverse of leaky ReLU.

 See also: [`LeakyReLU`](@ref), [`LeakyReLUgrad`](@ref)
"""
function LeakyReLUinv(y; slope=0.01f0)
    return max.(0f0, y) + (1f0/slope)*min.(0f0, y)
end

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
function LeakyReLUgrad(Δy, y; slope=0.01f0)
    x = LeakyReLUinv(y; slope=slope)  # recompute forward state
    p_mask = (sign.(x) .+ 1f0)/2f0
    return Δy.*p_mask + slope*Δy.*(1f0 .- p_mask)
end


###############################################################################
# Sigmoid (invertible if ouput nonzero)

"""
    y = Sigmoid(x; low=0, high=1)

 Sigmoid activation function. Shifted and scaled such that output is [low,high].

 See also: [`SigmoidInv`](@ref), [`SigmoidGrad`](@ref)
"""

function Sigmoid(x; low=0f0, high=1f0)
    y = high .* (1f0 ./ (1f0 .+ exp.(-x))) + low .* (1f0 ./ (1f0 .+ exp.(x)))
    return y
end


"""
    x = SigmoidInv(y; low=0, high=1f0)

 Inverse of Sigmoid function. Shifted and scaled such that output is [low,high]

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""
function SigmoidInv(y; low=0f0, high=1f0)
    if sum(isapprox.(y, 0f-6)) == 0
        x = log.(y .- low) - log.(high .- y)
    else
        throw("Input contains zeros.")
    end
    return x
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
function SigmoidGrad(Δy, y; x=nothing, low=0f0, high=1f0)
    if ~isnothing(y) && isnothing(x)
        x = SigmoidInv(y; low=low, high=high)  # recompute forward state
    end
        
    ΔSig_x = exp.(-x) ./ (1f0 .+ exp.(-x)).^2f0
    Δx = (high - low) .* Δy .* ΔSig_x 

    return Δx
end

###############################################################################
# Gated linear unit (GaLU) (not invertible)
# Adapted from Dauphin et al. (2017)

"""
    y = GaLU(x)

 Gated linear activation unit (not invertible).

 See also: [`GaLUgrad`](@ref)
"""
function GaLU(x::AbstractArray{Float32, N}) where N
    x1, x2 = tensor_split(x)
    y =  x1 .* Sigmoid(x2)
    return y
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
function GaLUgrad(Δy::AbstractArray{Float32, N}, x::AbstractArray{Float32, N}) where N
    k = Int(size(x, N-1) / 2)
    x1, x2 = tensor_split(x)
    Δx = 0f0.*x
    Δx = tensor_cat(Sigmoid(x2) .* Δy, SigmoidGrad(Δy, nothing; x=x2) .* x1)
    return Δx
end

function GaLUjacobian(Δx::AbstractArray{Float32, N}, x::AbstractArray{Float32, N}) where N
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
function ExpClamp(x; clamp=2f0)
    return exp.(clamp * 0.636f0 * atan.(x))
end

"""
    x = ExpClampInv(y)
 Inverse of ExpClamp function.
 See also: [`ExpClamp`](@ref), [`ExpClampGrad`](@ref)
"""
function ExpClampInv(y; clamp=2f0)
    if sum(isapprox.(y, 0f-6)) == 0
        x = tan.(log.(y) / clamp / 0.636f0)
    else
        throw("Input contains zeros.")
    end
    return x
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

function ExpClampGrad(Δy, y; x=nothing, clamp=2f0)
    if ~isnothing(y) && isnothing(x)
        x = ExpClampInv(y)  # recompute forward state
    end
    Δx = clamp * 0.636f0 * Δy .* y ./ (1f0 .+ x.^2f0)
    return Δx
end