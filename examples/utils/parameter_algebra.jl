# Example of Parameter algebra operations
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra


# Input
nx = 16
ny = 16
n_channel = 16
n_hidden = 64
batchsize = 2
X1 = randn(Float32, nx, ny, n_channel, batchsize)
X2 = randn(Float32, nx, ny, n_channel, batchsize)

# Networks
logdet=true
permute="full"
N1 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=logdet); N1.forward(X1)
N2 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=logdet); N2.forward(X2)

# Collect parameters
θ1 = get_params(N1)
θ2 = get_params(N2)

# Operations
θ1+θ2
randn(Float32)*θ1
norm.(θ1)
dot.(θ1, θ2)
dot(θ1, θ2)

# Set parameters
set_params!(N1, θ2)
Y1 = N1.forward(X1)[1]
Y1_ = N2.forward(X1)[1]
err_rel = norm(Y1-Y1_)/norm(Y1)