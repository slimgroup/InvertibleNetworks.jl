using InvertibleNetworks, NNlib, LinearAlgebra, Test

# Data
nx = 32
ny = 32
nz = 32
n_in = 3
batchsize = 4
k = 3   # kernel size
s = 1   # stride
p = 1   # padding

###################################################################################################
# Hyperbolic layer

# Data
X_prev = randn(Float32, nx, ny, nz, n_in, batchsize)
X_curr = randn(Float32, nx, ny, nz, n_in, batchsize)

# Network architecture ((decrease channel no?, no. of hidden units), ...)
architecture = ((0, 3), (-1, 3,), (0, 3), (1, 4), (0, 8))

# Hyperbolic network
HN = H = NetworkHyperbolic(nx, ny, nz, n_in, batchsize, architecture; 
    k=3, s=1, p=1, logdet=true, α=1f0)

# Forward/inverse pass
Y_curr_, Y_new_, lgdet = HN.forward(X_prev, X_curr)
X_prev_, X_curr_ = HN.inverse(Y_curr_, Y_new_)

# Test invertibility
@test isapprox(norm(X_prev - X_prev_)/norm(X_prev_), 0f0; atol=1f-3)
@test isapprox(norm(X_curr - X_curr_)/norm(X_prev_), 0f0; atol=1f-3)

