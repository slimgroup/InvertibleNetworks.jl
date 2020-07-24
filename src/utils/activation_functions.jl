# Activation functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ReLU, ReLUgrad
export LeakyReLU, LeakyReLUinv, LeakyReLUgrad
export Sigmoid, SigmoidInv, SigmoidGrad
export GaLU, GaLUgrad

###############################################################################
# Rectified linear unit (ReLU) (not invertible)

"""
    y = ReLU(x)

 Rectified linear unit (not invertible).

 See also: [`ReLUgrad`](@ref)
"""
function ReLU(x)
    y = 0f0.*x
    y[x.>=0f0] = x[x.>=0f0]
    return y
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
    Δx = 0f0.*x
    Δx[x.>=0f0] = Δy[x.>=0f0]
    return Δx
end

###############################################################################
# Leaky ReLU (invertible)

"""
    y = LeakyReLU(x; slope=0.01f0)

 Leaky rectified linear unit.

 See also: [`LeakyReLUinv`](@ref), [`LeakyReLUgrad`](@ref)
"""
function LeakyReLU(x; slope=0.01f0)
    y = 0f0.*x
    y[x.>=0f0] = x[x.>=0f0]
    y[x.<0f0] = slope*x[x.<0f0]
    return y
end

"""
    x = LeakyReLUinv(y; slope=0.01f0)

 Inverse of leaky ReLU.

 See also: [`LeakyReLU`](@ref), [`LeakyReLUgrad`](@ref)
"""
function LeakyReLUinv(y; slope=0.01f0)
    x = 0f0.*y
    x[y.>=0f0] = y[y.>=0f0]
    x[y.<0f0] = 1f0./slope*y[y.<0f0]
    return x
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
    Δx = 0f0.*y
    Δx[x.>=0f0] = Δy[x.>=0f0]
    Δx[x.<0f0] = slope*Δy[x.<0f0]
    return Δx
end


###############################################################################
# Sigmoid (invertible if ouput nonzero)

"""
    y = Sigmoid(x)

 Sigmoid activation function.

 See also: [`SigmoidInv`](@ref), [`SigmoidGrad`](@ref)
"""
function Sigmoid(x)
    y = 1f0 ./ (1f0 .+ exp.(-x))
    return y
end

"""
    x = SigmoidInv(y)

 Inverse of Sigmoid function.

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""
function SigmoidInv(y)
    if sum(isapprox.(y, 0f-6)) == 0
        x = -log.((1f0 .- y) ./ y)
    else
        throw("Input contains zeros.")
    end
    return x
end

"""
    Δx = SigmoidGrad(Δy, y; x=nothing)

 Backpropagate data residual through Sigmoid function.

 *Input*:

 - `Δy`: residual

 - `y`: original output

 - `x`: original input, if y not available (in this case, set y=nothing)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`Sigmoid`](@ref), [`SigmoidInv`](@ref)
"""
function SigmoidGrad(Δy, y; x=nothing)
    if ~isnothing(y) && isnothing(x)
        x = SigmoidInv(y)  # recompute forward state
    end
    Δx = Δy .* exp.(-x) ./ (1f0 .+ exp.(-x)).^2f0
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
function GaLU(x::AbstractArray{Float32, 4})
    k = Int(size(x, 3) / 2)
    y = x[:, :, 1:k, :] .* Sigmoid(x[:, :, k+1:end, :])
    return y
end

function GaLU(x::AbstractArray{Float32, 5})
    k = Int(size(x, 4) / 2)
    y = x[:, :, :, 1:k, :] .* Sigmoid(x[:, :, :, k+1:end, :])
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
function GaLUgrad(Δy::AbstractArray{Float32, 4}, x::AbstractArray{Float32, 4})
    k = Int(size(x, 3) / 2)
    x1 = x[:, :, 1:k, :]
    x2 = x[:, :, k+1:end, :]
    Δx = 0f0.*x
    Δx[:, :, 1:k, :] = Sigmoid(x2) .* Δy
    Δx[:, :, k+1:end, :] = SigmoidGrad(Δy, nothing; x=x2) .* x1
    return Δx
end

function GaLUgrad(Δy::AbstractArray{Float32, 5}, x::AbstractArray{Float32, 5})
    k = Int(size(x, 4) / 2)
    x1 = x[:, :, :, 1:k, :]
    x2 = x[:, :, :, k+1:end, :]
    Δx = 0f0.*x
    Δx[:, :, :, 1:k, :] = Sigmoid(x2) .* Δy
    Δx[:, :, :, k+1:end, :] = SigmoidGrad(Δy, nothing; x=x2) .* x1
    return Δx
end

###############################################################################
# Soft-clamped exponential function

"""
    y = ExpClamp(x)

 Soft-clamped exponential function.

 See also: [`ExpClampGrad`](@ref)
"""
function ExpClamp(x::Array{Float32}; clamp::Float32=2f0)
    return exp.(clamp * 0.636f0 * atan.(x))
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
function ExpClampGrad(Δy::Array{Float32}, x::Array{Float32}; y=nothing, clamp::Float32=2f0)
    y==nothing && (y=ExpClamp(x; clamp=clamp))
    return clamp * 0.636f0 * Δy .* y ./ (1f0 .+ x.^2f0)
end
