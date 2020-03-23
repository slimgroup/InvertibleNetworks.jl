# Example for HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# X dimensions
nx1 = 32
nx2 = 32
nx_channel = 32
nx_hidden = 64
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

# Conditional i-RIM layer
CI = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize)

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