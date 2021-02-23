# Example for HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 32
ny = 32
n_channel = 32
n_hidden = 64
batchsize = 2

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)


###################################################################################################
# Basic HINT layer

# Create HINT layer w/o logdet
HL1 = CouplingLayerHINT(n_channel, n_hidden; logdet=false, permute="none")

# Call forward and inverse  
Y = HL1.forward(X)
X_ = HL1.inverse(Y)

@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)


###################################################################################################
# Basic HINT layer w/ logdet

# HINT layer with logdet
HL2 = CouplingLayerHINT(n_channel, n_hidden; logdet=true, permute="none")

Y, logdet = HL2.forward(X)
X_ = HL2.inverse(Y)

@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)


###################################################################################################
# Basic HINT layer w/ logdet and permutation

# HINT layer with permutation: set to "none", "lower" or "full"
HL3 = CouplingLayerHINT(n_channel, n_hidden; logdet=true, permute="lower")

Y, logdet = HL3.forward(X)
X_ = HL3.inverse(Y)

@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)
