# Invertible CNN layer from Dinh et al. (2017)/Kingma & Dhariwal (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, JOLI

# Input
nx = 64
ny = 64
n_in = 10
n_hidden = 20
batchsize = 2
X = glorot_uniform(nx, ny, n_in, batchsize)

# Observed data
nrec = 20
nt = 50
D = randn(Float32, nt*nrec, batchsize)

# Modeling/imaging operator (can be JOLI/JUDI operator or explicit matrix)
A = joMatrix(randn(Float32, nt*nrec, nx*ny))

# Link function
Ψ(η) = identity(η)


###################################################################################################
# Additive slim coupling layer (zero logdet)

# Create layer
L1 = AdditiveCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=true, permute=false)

Y, logdet = L1.forward(X, D, A)
X_ = L1.inverse(Y, D, A)
@test iszero(logdet)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)

ΔY = Y .* 0f0
X_ = L1.backward(ΔY, Y, D, A)[3]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)


###################################################################################################
# Affine slim coupling layer (non-zero logdet)

# Create layer
L2 = AffineCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=true, permute=false)

Y, logdet = L2.forward(X, D, A)
X_ = L2.inverse(Y, D, A)
@test ~iszero(logdet)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)

ΔY = Y .* 0f0
X_ = L2.backward(ΔY, Y, D, A)[3]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)
