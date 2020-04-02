# Example how to use the residual block 
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using Test, LinearAlgebra, InvertibleNetworks

# Input
nx = 64
ny = 64
n_in = 10
n_hidden = 20
batchsize = 2
k1 = 4
k2 = 3

# Input image
X = glorot_uniform(nx, ny, n_in, batchsize)

# Residual blocks
RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2)
RB0 = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2)

# Observed data
Y = RB.forward(X)

# Predicted data
Y0 = RB0.forward(X)
@test isnothing(RB0.W1.grad) # after forward pass, gradients are not set

# Residual
ΔY = Y0 - Y 

# Backward pass: need to pass data residual and original input X (as layer is not invertible)
ΔX = RB0.backward(ΔY, X)   # returns derivative w.r.t input
@test ~isnothing(RB0.W1.grad)    # after inverse pass, gradients are set
@test ~isnothing(RB0.b1.grad)
