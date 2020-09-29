# Author: Gabrio Rizzuti, grizzuti33@gatech.edu
# Date: September 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics

# Input
nx = 28
ny = 28*2
nc = 4
n_hidden = 64
batchsize = 5

# Initialization network
logdet = true
X1 = rand(Float32, nx, ny, nc, batchsize)
N1 = ActNorm(nc; logdet=logdet); Y1 = N1.forward(X1)[1]
X2 = rand(Float32, nx, ny, nc, batchsize)
N2 = ActNorm(nc; logdet=logdet); Y2 = N2.forward(X2)[1]
ΔX = X1-X2
Δθ = get_params(N1)-get_params(N2)
ΔY = Y1-Y2

# Jacobian initialization
N = ActNorm(nc; logdet=logdet); N.forward(rand(Float32, nx, ny, nc, batchsize))
X = rand(Float32, nx, ny, nc, batchsize)
J = Jacobian(N, X)

# Evaluation
JΔ = J*(ΔX, Δθ)
JΔ = J*Δθ

# Adjoint
Jadj = adjoint(J)
ΔX_, Δθ_ = Jadj*ΔY

# Composition
GN = Jadj*J
GN*(ΔX, Δθ)
adjoint(GN)
