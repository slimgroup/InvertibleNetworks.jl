# Objective functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export log_likelihood, ∇log_likelihood, Hlog_likelihood, mse, ∇mse, Hmse

###################################################################################################
# Mean squared error
"""
    f = mse(X, Y)

Mean squared error between arrays/tensor X and Y

See also: [`∇mse`](@ref)
"""
mse(X::AbstractArray{Float32, N}, Y::AbstractArray{Float32, N}) where N = .5f0/size(X, N)*norm(X - Y, 2)^2


"""
    ∇f = ∇mse(X, Y)

Gradient of the MSE loss with respect to input tensors X and Y.

See also: [`mse`](@ref)
"""
∇mse(X::Array{Float32, N}, Y::Array{Float32, N}) where N = 1f0/size(X, N)*(X - Y)


"""
    Hf = Hmse(X, Y)

Hessian of the MSE loss with respect to input tensors X.

See also: [`mse`](@ref)
"""
function Hmse(X::Array{Float32, N}, Y::Array{Float32, N}) where N
    return InvertibleNetworkLinearOperator{Array{Float32, N},Array{Float32, N}}(
        ΔX -> 1f0/size(X, N)*ΔX,
        ΔX -> 1f0/size(X, N)*ΔX)
end


###################################################################################################
# Log-likelihood

"""
    f = log_likelihood(X; μ=0f0, σ=1f0)

Log-likelihood of X for a Gaussian distribution with given mean μ and variance σ. All elements of X are assumed to be iid.

See also: [`∇log_likelihood`](@ref)
"""
log_likelihood(X::AbstractArray{Float32, N}; μ=0f0, σ=1f0) where N = 1f0/size(X, N)*sum(-.5f0*((X .- μ)/σ).^2)


"""
    Hf = Hlog_likelihood(X; μ=0f0, σ=1f0)

Hessian of the log-likelihood function with respect to the input tensor X.

See also: [`log_likelihood`](@ref)
"""
∇log_likelihood(X::AbstractArray{Float32, N}; μ=0f0, σ=1f0) where N = -1f0/size(X, N)*(X .- μ)/σ^2


"""
    Hf = Hlog_likelihood(X; μ=0f0, σ=1f0)

Hessian of the log-likelihood function with respect to the input tensor X.

See also: [`log_likelihood`](@ref)
"""
function Hlog_likelihood(X::AbstractArray{Float32, N}; μ=0f0, σ=1f0) where N
    return InvertibleNetworkLinearOperator{AbstractArray{Float32, N},AbstractArray{Float32, N}}(
        ΔX -> -1f0/size(X, N)*ΔX/σ^2,
        ΔX -> -1f0/size(X, N)*ΔX/σ^2)
end
