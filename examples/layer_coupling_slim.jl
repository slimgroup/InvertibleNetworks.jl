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

# Slim coupling layer
L = CouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=false, permute=false)

Y = L.forward(X, D, A)
X_ = L.inverse(Y, D, A)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)

ΔY = Y .* 0f0
X_ = L.backward(ΔY, Y, D, A)[2]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)