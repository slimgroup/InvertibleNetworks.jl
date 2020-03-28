# Example how to use the residual block 
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
ny2 = 22
ny_in = 1

# Input image
X = glorot_uniform(nx1, nx2, nx_in, batchsize)
D = glorot_uniform(ny1, ny2, ny_in, batchsize)

# Residual blocks
RB = ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize)

# Observed data
Y, D_ = RB.forward(X, D)

# Set data residual to zero
ΔY = Y.*0f0; ΔD = D.*0f0

# Backward pass
ΔX, ΔD = RB.backward(ΔY, ΔD, X, D)
