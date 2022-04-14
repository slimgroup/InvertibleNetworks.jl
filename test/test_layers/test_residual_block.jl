# Test residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, Test, Random, BenchmarkTools
Random.seed!(2)

# Input
nx = 28
ny = 28
n_in = 4
n_hidden = 8
batchsize = 2
k1 = 3
k2 = 3

# Input
X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = (X - X0)/norm(X0)

# Weights
W1 = glorot_uniform(k1, k1, n_in, n_hidden)
W2 = glorot_uniform(k2, k2, n_hidden, n_hidden)
W3 = glorot_uniform(k1, k1, 2*n_in, n_hidden)
b1 = glorot_uniform(n_hidden)
b2 = glorot_uniform(n_hidden)

W01 = glorot_uniform(k1, k1, n_in, n_hidden)
W02 = glorot_uniform(k2, k2, n_hidden, n_hidden)
W03 = glorot_uniform(k1, k1, 2*n_in, n_hidden)
b01 = glorot_uniform(n_hidden)
b02 = glorot_uniform(n_hidden)

dW1 = W1 - W01; dW1 *= norm(W01)/norm(dW1)
dW2 = W2 - W02; dW2 *= norm(W02)/norm(dW2)
dW3 = W3 - W03; dW3 *= norm(W03)/norm(dW3)
db1 = b1 - b01; db1 *= norm(b01)/norm(db1)
db2 = b2 - b02; db2 *= norm(b02)/norm(db2)

# Residual blocks
RB = ResidualBlock(W1, W2, W3, b1, b2)   # true weights

# Observed data
Y = RB.forward(X)

function loss(RB, X, Y)
    Y_ = RB.forward(X)
    ΔY = Y_ - Y
    f = .5f0*norm(ΔY)^2
    ΔX = RB.backward(ΔY, X)
    return f, ΔX, RB.W1.grad, RB.W2.grad, RB.W3.grad, RB.b1.grad, RB.b2.grad
end


###################################################################################################
# Gradient tests

# Gradient test w.r.t. input
f0, ΔX = loss(RB, X0, Y)[1:2]
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = loss(RB, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for weights
RB0 = ResidualBlock(W01, W02, W03, b1, b2)   # initial weights
f0, ΔX, ΔW1, ΔW2, ΔW3 = loss(RB0, X0, Y)[1:5]
h = 0.1f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    RB0.W1.data = W01 + h*dW1
    RB0.W2.data = W02 + h*dW2
    RB0.W3.data = W03 + h*dW3
    f = loss(RB0, X0, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for bias
RB0 = ResidualBlock(W1, W2, W3, b01, b02)
f0, ΔX, ΔW1, ΔW2, ΔW3, Δb1, Δb2 = loss(RB0, X0, Y)
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
print("\nGradient test convolutions\n")
for j=1:maxiter
    RB0.b1.data = b01 + h*db1
    RB0.b2.data = b02 + h*db2
    f = loss(RB0, X0, Y)[1]
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h*dot(db1, Δb1) - h*dot(db2, Δb2))
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
W1 = glorot_uniform(k1, k1, n_in, n_hidden)
W2 = glorot_uniform(k2, k2, n_hidden, n_hidden)
W3 = glorot_uniform(k1, k1, 2*n_in, n_hidden)
b1 = glorot_uniform(n_hidden)
b2 = glorot_uniform(n_hidden)
RB = ResidualBlock(W1, W2, W3, b1, b2; fan=true)
θ = deepcopy(get_params(RB))
X = randn(Float32, nx, ny, n_in, batchsize)

# Perturbation (normalized)
dθ = [Parameter(dW1)/norm(W1), Parameter(dW2)/norm(W2), Parameter(dW3)/norm(W3), Parameter(db1)/norm(b1), Parameter(db2)/norm(b2)]
dX = randn(Float32, nx, ny, n_in, batchsize)/norm(X)

# Jacobian eval
dY, Y = RB.jacobian(dX, dθ, X)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err7 = zeros(Float32, maxiter)
err8 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(RB, θ+h*dθ)
    Y_loc = RB.forward(X+h*dX)
    err7[j] = norm(Y_loc - Y)
    err8[j] = norm(Y_loc - Y - h*dY)
    print(err7[j], "; ", err8[j], "\n")
    global h = h/2f0
end

@test isapprox(err7[end] / (err7[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err8[end] / (err8[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(RB, θ)
dY = RB.jacobian(dX, dθ, X)[1]
dY_ = randn(Float32, size(dY))
dX_, dθ_ = RB.adjointJacobian(dY_, X)
a = dot(dY, dY_)
b = dot(dX, dX_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)
