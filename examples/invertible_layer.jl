# Invertible CNN layer from Putzky & Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 64
ny = 64
k = 20
n_in = 10
n_hidden = 20
batchsize = 2
k1 = 4
k2 = 3

# Input image
X = glorot_uniform(nx, ny, k, batchsize)
X0 = glorot_uniform(nx, ny, k, batchsize)

# 1x1 convolution and residual blocks
C = Conv1x1(k)
RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2)

# Invertible splitting layer
L = InvertibleLayer(C, RB)

# Forward + backward
Y = L.forward(X)
Y0 = L.forward(X0)
ΔY = Y0 - Y
ΔX, X0_ = L.backward(ΔY, Y0)

@test isapprox(norm(X0_ - X0)/norm(X0), 0f0, atol=1f-2)