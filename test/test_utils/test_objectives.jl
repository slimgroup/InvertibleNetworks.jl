# Test objective functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, Test, LinearAlgebra


###################################################################################################
# MSE loss

Y = randn(Float32, 28, 28, 3, 1)
X = randn(Float32, 28, 28, 3, 1)
X0 = randn(Float32, 28, 28, 3, 1)
dX = X - X0

function loss_mse(X, Y)
    f = mse(X, Y)
    g = ∇mse(X, Y)
    return f, g
end

f0, g = loss_mse(X0, Y)
h = .1f0
maxiter = 6

err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test mse loss\n")
for j=1:maxiter
    f = loss_mse(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, g))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# (Negative) Log-likelihood

X = randn(Float32, 28, 28, 3, 1)
X0 = randn(Float32, 28, 28, 3, 1)
dX = X - X0

function loss_mle(X)
    f = log_likelihood(X; μ=1f0, σ=2f0)
    g = ∇log_likelihood(X; μ=1f0, σ=2f0)
    return f, g
end

f0, g = loss_mle(X0)
h = .1f0
maxiter = 6

err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test log likelihood\n")
for j=1:maxiter
    f = loss_mle(X0 + h*dX)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dX, g))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

