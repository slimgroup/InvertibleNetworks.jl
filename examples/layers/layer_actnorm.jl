# Activation normalization from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 64
ny = 64
k = 10
batchsize = 4

# Input image: nx x ny x k x batchsize
X = randn(Float32, nx, ny, k, batchsize)
Y = randn(Float32, nx, ny, k, batchsize)

# Activation normalization
AN = ActNorm(k; logdet=true)

# Test invertibility
Y_, logdet = AN.forward(X)
ΔY = Y_ - Y

# Backpropagation
ΔX, X_ = AN.backward(ΔY, Y_)

# Test invertibility
isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)
