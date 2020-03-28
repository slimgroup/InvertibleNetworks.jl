# Test residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using Test, LinearAlgebra, InvertibleNetworks

# Input
nx1 = 32
nx2 = 32
nx_in = 8
n_hidden = 16   # same for x and y
batchsize = 2

ny1 = 64
ny2 = 24
ny_in = 1

# Residual blocks
RB = ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize)

function loss(RB, X, Y)
    Zx, Zy = RB.forward(X, Y)
    f = -log_likelihood(Zx) - log_likelihood(Zy)
    ΔZx = -∇log_likelihood(Zx)
    ΔZy = -∇log_likelihood(Zy)
    ΔX, ΔY = RB.backward(ΔZx, ΔZy, X, Y)[1:2]
    return f, ΔX, ΔY, RB.W0.grad, RB.W1.grad, RB.W2.grad, RB.W3.grad, 
        RB.b0.grad, RB.b1.grad, RB.b2.grad
end


###################################################################################################
# Gradient tests

# Input image and data
X0 = glorot_uniform(nx1, nx2, nx_in, batchsize)
X = glorot_uniform(nx1, nx2, nx_in, batchsize)
Y0 = glorot_uniform(ny1, ny2, ny_in, batchsize)
Y = glorot_uniform(ny1, ny2, ny_in, batchsize)
dX = X - X0
dY = Y - Y0

# Gradient test w.r.t. input
f0, ΔX, ΔY = loss(RB, X0, Y0)[1:3]
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = loss(RB, X0 + h*dX, Y0 + h*dY)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX) - h*dot(dY, ΔY))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test for weights
RB0 = ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize)
RBini = deepcopy(RB0)
dW0 = RB.W0.data - RB0.W0.data
dW1 = RB.W1.data - RB0.W1.data
dW2 = RB.W2.data - RB0.W2.data
dW3 = RB.W3.data - RB0.W3.data
f0, ΔX, ΔY, ΔW0, ΔW1, ΔW2, ΔW3 = loss(RB0, X, Y)[1:7]
h = 0.1f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    RB0.W0.data = RBini.W0.data + h*dW0
    RB0.W1.data = RBini.W1.data + h*dW1
    RB0.W2.data = RBini.W2.data + h*dW2
    RB0.W3.data = RBini.W3.data + h*dW3
    f = loss(RB0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW0, ΔW0) - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
