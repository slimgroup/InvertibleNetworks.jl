# Tests for conditional HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test


#######################################################################################################################
# Test invertibility

# Use affine or additive coupling ConditionalLayerSLIM
type = "affine" # "affine", "additive" or "learned"

# X dimensions
nx1 = 16
nx2 = 16
nx_channel = 4
nx_hidden = 8
batchsize = 2

# Y dimensions
ny1 = 128
ny2 = 64
ny_channel = 1
ny_hidden = 32

# Linear operator
A = randn(Float32, ny1*ny2, nx1*nx2)

# Input image
X = glorot_uniform(nx1, nx2, nx_channel, batchsize)
Y = glorot_uniform(ny1, ny2, ny_channel, batchsize)

# Conditional HINT layer
CI = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type=type)

# Forward/inverse
Zx, Zy, logdet = CI.forward(X, Y, A)
X_, Y_ = CI.inverse(Zx, Zy, A)

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)

# Forward/backward
Zx, Zy, logdet = CI.forward(X, Y, A)
X_, Y_ = CI.backward(0f0*Zx, 0f0*Zy, Zx, Zy, A)[3:4]

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)

# Forward/inverse Y-lane only
Zy = CI.forward_Y(Y)
Y_ = CI.inverse_Y(Zy)

@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-6)


#######################################################################################################################
# Gradient test

# Input image
X = randn(Float32, nx1, nx2, nx_channel, batchsize)
X0 = randn(Float32, nx1, nx2, nx_channel, batchsize)
dX = X - X0

# Input data
Y = randn(Float32, ny1, ny2, ny_channel, batchsize)
Y0 = randn(Float32, ny1, ny2, ny_channel, batchsize)
dY = Y - Y0

function loss(CI, X, Y, A)
    Zx, Zy, logdet = CI.forward(X, Y, A)
    f = -log_likelihood(Zx) -log_likelihood(Zy) - logdet
    ΔZx = -∇log_likelihood(Zx)
    ΔZy = -∇log_likelihood(Zy)
    ΔX, ΔY = CI.backward(ΔZx, ΔZy, Zx, Zy, A)[1:2]
    return f, ΔX, ΔY, CI.CL_X.CL[1].RB.W1.grad, CI.C_X.v1.grad
end

# Gradient test for input X, Y
CI = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type=type)
f0, gX, gY = loss(CI, X0, Y0, A)[1:3]

maxiter = 5
h = 0.25f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("Gradient test ΔX\n")
for j=1:maxiter
    f = loss(CI, X0 + h*dX, Y0 + h*dY, A)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, gX) - h*dot(dY, gY))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Test for weights
X = randn(Float32, nx1, nx2, nx_channel, batchsize)
Y = randn(Float32, ny1, ny2, ny_channel, batchsize)
CI = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type=type)
CI0 = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type=type)

# Make weights larger (otherweise too close to zero after initialization)
CI.CL_X.CL[1].RB.W1.data .*= 4; CI0.CL_X.CL[1].RB.W1.data .*= 4
CI.C_X.v1.data .*= 4; CI0.C_X.v1.data .*= 4
CIini = deepcopy(CI0)

dW = CI.CL_X.CL[1].RB.W1.data - CI0.CL_X.CL[1].RB.W1.data
dv = CI.C_X.v1.data - CI0.C_X.v1.data

f0, gW, gv = loss(CI0, X, Y, A)[[1,4,5]]

maxiter = 5
h = 0.25f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test weights\n")
for j=1:maxiter
    CI0.CL_X.CL[1].RB.W1.data = CIini.CL_X.CL[1].RB.W1.data + h*dW
    CI0.C_X.v1.data = CIini.C_X.v1.data + h*dv
    f = loss(CI0, X, Y, A)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(gW, dW) - h*dot(gv, dv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)