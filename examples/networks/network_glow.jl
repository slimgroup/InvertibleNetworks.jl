# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Flux
using CuArrays
#CuArrays.allowscalar(false)

import Flux.Optimise.update!

# Define network
nx = 512     # must be multiple of 2
ny = 512
n_in = 3
n_hidden = 32
batchsize = 20
L = 4   # number of scales
K = 4   # number of flow steps per scale

# Input
X = rand(Float32, nx, ny, n_in, batchsize) |> gpu

# Glow network
G = NetworkGlow(nx, ny, n_in, batchsize, n_hidden, L, K)

# Objective function
function loss(X)
    Y, logdet = G.forward(X)
    f = .5f0/batchsize*norm(Y)^2 - logdet
    ΔX, X_ = G.backward(1f0./batchsize*Y, Y)
    return f
end

# Evaluate loss
f = loss(X)

# Update weights
opt = Flux.ADAM()
Params = get_params(G)
for p in Params
    update!(opt, p.data, p.grad)
end
clear_grad!(G)
