# Tests for conditional HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test


#######################################################################################################################
# Test invertibility

# Input
nx = 16
ny = 16
n_channel = 8
n_hidden = 32
batchsize = 2

# Input image and data
X = randn(Float32, nx, ny, n_channel, batchsize)
Y = randn(Float32, nx, ny, n_channel, batchsize)

# HINT layer
CH = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize)

# Test inverse
Zx, Zy, logdet = CH.forward(X, Y)
X_, Y_ = CH.inverse(Zx, Zy)

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-6)

# Test backward
Zx, Zy, logdet = CH.forward(X, Y)
ΔZx = randn(Float32, size(Zx))  # random derivative
ΔZy = randn(Float32, size(Zx))
ΔX_, ΔY_, X_, Y_ = CH.backward(ΔZx, ΔZy, Zx, Zy)

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-6)

# Test inverse Y only
Zy = CH.forward_Y(Y)
Y_ = CH.inverse_Y(Zy)

@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-6)

#######################################################################################################################
# Gradient test

# Input image
X = randn(Float32, nx, ny, n_channel, batchsize)
X0 = randn(Float32, nx, ny, n_channel, batchsize)
dX = X - X0

# Input data
Y = randn(Float32, nx, ny, n_channel, batchsize)
Y0 = randn(Float32, nx, ny, n_channel, batchsize)
dY = Y - Y0

function loss(CH, X, Y)
    Zx, Zy, logdet = CH.forward(X, Y)
    Z = tensor_cat(Zx, Zy)
    f = -log_likelihood(Z) - logdet
    ΔZ = -∇log_likelihood(Z)
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    return f, ΔX, ΔY, CH.CL_X.RB.W1.grad, CH.C_X.v1.grad
end

# Gradient test for input X, Y
CH = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize)
f0, gX, gY = loss(CH, X0, Y0)[1:3]

maxiter = 5
h = 0.5f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("Gradient test ΔX\n")
for j=1:maxiter
    f = loss(CH, X0 + h*dX, Y0 + h*dY)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, gX) - h*dot(dY, gY))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Test for weights
X = glorot_uniform(nx, ny, n_channel, batchsize)
Y = glorot_uniform(nx, ny, n_channel, batchsize)
CH = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize)
CH0 = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize)
CHini = deepcopy(CH0)
dW = CH.CL_X.RB.W1.data - CH0.CL_X.RB.W1.data
dv = CH.C_X.v1.data - CH0.C_X.v1.data

f0, gW, gv = loss(CH0, X, Y)[[1,4,5]]

maxiter = 5
h = 0.5f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test weights\n")
for j=1:maxiter
    CH0.CL_X.RB.W1.data = CHini.CL_X.RB.W1.data + h*dW
    CH0.C_X.v1.data = CHini.C_X.v1.data + h*dv
    f = loss(CH0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(gW, dW) - h*dot(gv, dv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
