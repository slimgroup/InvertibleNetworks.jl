using InvertibleNetworks, NNlib, LinearAlgebra, Test

# Data
nx = 128
ny = 128
nz = 128
n_in = 3
n_hidden = 6
batchsize = 4
k = 3   # kernel size
s = 1   # stride
p = 1   # padding

###################################################################################################
# Hyperbolic layer

# Data
X_prev = zeros(Float32, nx, ny, nz, n_in, batchsize)
X_curr = randn(Float32, nx, ny, nz, n_in, batchsize)

# Layer
HL = HyperbolicLayer(nx, ny, nz, n_in, batchsize, k, s, p; action=-1, α=1f-1, n_hidden=6)

Y_curr, Y_new = HL.forward(X_prev, X_curr)
X_prev_, X_curr_ = HL.inverse(Y_curr, Y_new)

# Test invertibility
@test isapprox(norm(X_prev_)/prod(size(X_prev_)), 0f0; atol=1f-6)
@test isapprox(norm(X_curr - X_curr_)/norm(X_curr_), 0f0; atol=1f-6)

