# Objective functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export log_likelihood, ∇log_likelihood, log_likelihood_multi, ∇log_likelihood_multi, mse, ∇mse

###################################################################################################
# Mean squared error
"""
    f = mse(X, Y)

Mean squared error between arrays/tensor X and Y

See also: [`∇mse`](@ref)
"""
mse(X, Y) = .5f0/prod(size(X))*norm(X - Y, 2)^2


"""
    ∇f = ∇mse(X, Y)

Gradient of the MSE loss with respect to input tensors X and Y.

See also: [`mse`](@ref)
"""
∇mse(X, Y) = 1f0/prod(size(X))*(X - Y)


###################################################################################################
# Log-likelihood

"""
    f = log_likelihood(X; μ=0f0, σ=1f0)

Log-likelihood of X for a Gaussian distribution with given mean μ and variance σ. All elements of X are assumed to be iid.

See also: [`∇log_likelihood`](@ref)
"""
log_likelihood(X; μ=0f0, σ=1f0) = 1f0/prod(size(X))*sum(log(1f0/(σ*sqrt(2f0*pi))) .- .5f0*((X .- μ)/σ).^2)


"""
    ∇f = ∇log_likelihood(X; μ=0f0, σ=1f0)

Gradient of the log-likelihood function with respect to the input tensor X.

See also: [`log_likelihood`](@ref)
"""
∇log_likelihood(X; μ=0f0, σ=1f0) = -1f0/prod(size(X))*(X .- μ)/σ^2


# Multivariate log likelihood for zero mean and unit variance tensor
log_likelihood_multi(X) = -.5f0*dot(X, X) / size(X, 4)
∇log_likelihood_multi(X) = -X / size(X, 4)
