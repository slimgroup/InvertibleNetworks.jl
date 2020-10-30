# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Example of network composition
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics


# Input
nx = 28
ny = 28*2
nc = 4
n_hidden = 64
batchsize = 5
X = rand(Float32, nx, ny, nc, batchsize)

# Many layers/networks
logdet = true
N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
N3 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)

# Composition (1)
N = Composition(N1, N2, N3)

# Composition (2)
N = N1∘N2∘N3