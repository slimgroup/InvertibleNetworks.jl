# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Rafael Orozco, pwitte3@gatech.edu
# Date: march 2023

using InvertibleNetworks, LinearAlgebra, Flux

device = InvertibleNetworks.CUDA.functional() ? gpu : cpu

# Define network
nx   = 64   # if split_scale=true then nx / (2^L) needs to be a whole number
n_in = 1    # if 1 then split_scale needs to be true to increase channel for coupling layer
n_hidden = 32
batchsize = 10
L = 2   # number of scales
K = 2   # number of flow steps per scale

# Input
X = rand(Float32, nx, n_in, batchsize) |> device

# Glow network
G = NetworkGlow(n_in, n_hidden, L, K; split_scales=true, dense=true, nx=nx, ndims=1)  |> device

# Objective function
function loss(X)
    Y, logdet = G.forward(X)
    f = .5f0/batchsize*norm(Y)^2 - logdet
    ΔX, X_ = G.backward(1f0./batchsize*Y, Y)
    return f
end

# Evaluate loss
f = loss(X)
@time loss(X)

# Update weights
opt = Flux.ADAM()
Params = get_params(G)
for p in Params
    Flux.update!(opt, p.data, p.grad)
end
clear_grad!(G)
