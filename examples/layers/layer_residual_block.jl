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

# Residual blocks of various networks
architecture = "Glow"

if architecture == "Glow"
    # Glow
    k1 = 3; p1 = 1; s1 = 1
    k2 = 1; p2 = 0; s2 = 1
elseif architecture == "HINT"
    # HINT
    k1 = 3; p1 = 1; s1 = 1  # standard res-net block
    k2 = 3; p2 = 1; s2 = 1
elseif architecture == "iRIM"
    # i-RIM
    k1 = 4; p1 = 0; s1 = 4  # downsampling/upsampling in first and third layer
    k2 = 3; p2 = 1; s2 = 1  # Standard res-net in center layer
end

# Input image
X = glorot_uniform(nx, ny, n_in, batchsize)

# Residual blocks
RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)
RB0 = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2)

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
