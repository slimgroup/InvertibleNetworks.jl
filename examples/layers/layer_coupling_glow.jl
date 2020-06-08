# Invertible CNN layer from Dinh et al. (2017)/Kingma & Dhariwal (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 64
ny = 64
k = 20
n_in = 10
n_hidden = 20
batchsize = 2
type = "Flux"   # Flux CNN or residual block

# Input image
X = glorot_uniform(nx, ny, k, batchsize)
X0 = glorot_uniform(nx, ny, k, batchsize)

# 1x1 convolution and residual blocks
C = Conv1x1(k)

if type == "Flux"
    # Flux residual block (needs twice the number of output as input channels)
    model = Chain(
        Conv((3,3), n_in => n_hidden; pad=1),
        BatchNorm(n_hidden, relu),
        Conv((3,3), n_hidden => n_hidden; pad=1),
        BatchNorm(n_hidden, relu),
        Conv((3,3), n_hidden => n_hidden; pad=1),
        BatchNorm(n_hidden, relu)
    )
    RB = FluxBlock(model)
else
    RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; fan=true)
end

# Invertible splitting layer
L = CouplingLayerGlow(C, RB; logdet=true)   # compute logdet

# Forward + backward
Y = L.forward(X)[1]
Y0, logdet = L.forward(X0)
ΔY = Y0 - Y
ΔX, X0_ = L.backward(ΔY, Y0)

@test isapprox(norm(X0_ - X0)/norm(X0), 0f0, atol=1f-2)