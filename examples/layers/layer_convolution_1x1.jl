# Example for using the 1x1 convolution operator to permute an image along the channel dimension.
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test
using Flux

# Dimensions
nx = 64 # no. of pixels in x dimension
ny = 64 # no. of pixels in y dimension
k = 10  # no. of channels
batchsize = 4

# Input image: nx x ny x k x batchsize
X = glorot_uniform(nx, ny, k, batchsize) |> gpu

# 1x1 convolution operators
C = Conv1x1(k) |> gpu
C0 = Conv1x1(k) |> gpu

# Generate "true/observed" data with the same dimension as X
Y = C.forward(X)

# Predicted data
Y0 = C0.forward(X)
@test isnothing(C0.v1.grad) # after forward pass, gradients are not set

# Data residual
ΔY = Y0 - Y 

# Backward pass: Pass ΔY to compute ΔX, the gradient w.r.t the input X.
# Also pass Y0 to recompute the forward state X using the inverse mapping
# and use it to compute the derivative w.r.t. the coefficients of the 
# Householder matrix.
ΔX, X_ = C0.inverse((ΔY, Y0))
@test ~isnothing(C0.v1.grad)    # after inverse pass, gradients are set
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)    # X and X_ should be the same

