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
function ReLU(x::AbstractArray{T, N}) where {T, N}
    return max.(0, x)
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
function ReLUgrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    return  Δy .* (sign.(x) .+ 1) ./ 2
end

###############################################################################
# Leaky ReLU (invertible)

"""
    y = LeakyReLU(x; slope=0.01f0)

 Leaky rectified linear unit.

 See also: [`LeakyReLUinv`](@ref), [`LeakyReLUgrad`](@ref)
"""
function LeakyReLU(x::AbstractArray{T, N}; slope=T(0.01)) where {T, N}
    return max.(0, x) + slope*min.(0, x)
end

"""
    x = LeakyReLUinv(y; slope=0.01f0)

 Inverse of leaky ReLU.

 See also: [`LeakyReLU`](@ref), [`LeakyReLUgrad`](@ref)
"""
function LeakyReLUinv(y::AbstractArray{T, N}; slope=T(0.01)) where {T, N}
    return max.(0, y) + (1/slope)*min.(0, y)
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
function LeakyReLUgrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; slope=T(0.01)) where {T, N}
    x = LeakyReLUinv(y; slope=slope)  # recompute forward state
    p_mask = (sign.(x) .+ 1) ./ 2
    return Δy.*p_mask + slope*Δy.*(1 .- p_mask)
end


###############################################################################
# Sigmoid (invertible if ouput nonzero)

"""
    y = Sigmoid(x; low=0, high=1)

 Sigmoid activation function. Shifted and scaled such that output is [low,high].

 See also: [`SigmoidInv`](@ref), [`SigmoidGrad`](@ref)
"""

function Sigmoid(x::AbstractArray{T, N}; low=0f0, high=1f0) where {T, N}
    y = high .* (1f0 ./ (1f0 .+ exp.(-x))) + low .* (1f0 ./ (1f0 .+ exp.(x)))
    return y
end


"""
    x = SigmoidInv(y; low=0, high=1f0)

 Inverse of Sigmoid function. Shifted and scaled such that output is [low,high]

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""
function SigmoidInv(y::AbstractArray{T, N}; low=0f0, high=1f0) where {T, N}
    if sum(isapprox.(y, 0f-6)) == 0
        x = log.(y .- low) - log.(high .- y)
    else
        throw(InputError("Input contains zeros."))
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
function SigmoidGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; x=nothing, low=0f0, high=1f0) where {T, N}

    if isnothing(x)
        x = SigmoidInv(y; low=low, high=high)  # recompute forward state
    end
        
    ΔSig_x = exp.(-x) ./ (1f0 .+ exp.(-x)).^2f0
    Δx = (high - low) .* Δy .* ΔSig_x 

    return Δx
end

function SigmoidGrad(Δy::AbstractArray{T, N}, ::Nothing; x=nothing, low=0f0, high=1f0) where {T, N}
    if isnothing(x)
       throw(InputError("Input x must be provided with y=nothing, can't inverse recompute"))  # recompute forward state
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
function GaLU(x::AbstractArray{T, N}) where {T, N}
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
function GaLUgrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    k = Int(size(x, N-1) / 2)
    x1, x2 = tensor_split(x)
    Δx = 0 .*x
    Δx = tensor_cat(Sigmoid(x2) .* Δy, SigmoidGrad(Δy, nothing; x=x2) .* x1)
    return Δx
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
function ExpClamp(x::AbstractArray{T, N}; clamp=T(2)) where {T, N}
    return exp.(clamp * T(0.636) * atan.(x))
end

"""
    x = ExpClampInv(y)
 Inverse of ExpClamp function.
 See also: [`ExpClamp`](@ref), [`ExpClampGrad`](@ref)
"""
function ExpClampInv(y::AbstractArray{T, N}; clamp=T(2)) where {T, N}
    if any(y .≈ 0)
        throw(InputError("Input contains zeros."))
    else
        x = tan.(log.(y) / clamp / T(0.636))
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

function ExpClampGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; x=nothing, clamp=T(2)) where {T, N}
    if isnothing(x)
        x = ExpClampInv(y)  # recompute forward state
    end
    Δx = clamp * T(0.636) * Δy .* y ./ (1 .+ x.^2)
    return Δx
end


function ExpClampGrad(Δy::AbstractArray{T, N}, ::Nothing; x=nothing, clamp=T(2)) where {T, N}
    Δx = clamp * T(0.636) * Δy .* y ./ (1 .+ x.^2)
    return Δx
end