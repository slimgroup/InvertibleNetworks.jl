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
k1 = 4
k2 = 3

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)

# Create HINT layer
HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize)

# Call forward and inverse  
Y = HL.forward(X)
X_ = HL.inverse(Y)

@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-6)

