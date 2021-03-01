# Example for HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 32
ny = 32
n_channel = 32
n_hidden = 64
batchsize = 2

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)
Y = glorot_uniform(nx, ny, n_channel, batchsize)

# Conditional HINT layer
CH = ConditionalLayerHINT(n_channel, n_hidden)

# Forward/inverse
Zx, Zy, logdet = CH.forward(X, Y)
X_, Y_ = CH.inverse(Zx, Zy)

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-4)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-4)

# Forward/inverse Y-lane only
Zy = CH.forward_Y(Y)
Y_ = CH.inverse_Y(Zy)

@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-4)