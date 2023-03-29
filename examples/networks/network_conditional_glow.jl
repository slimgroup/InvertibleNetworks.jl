# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Network layers are made conditional with CIIN type layers 
# Author: Rafael Orozco, rorozco@gatech.edu
# Date: March 2023

using InvertibleNetworks, LinearAlgebra, Flux

device = InvertibleNetworks.CUDA.functional() ? gpu : cpu

nx = 32    # must be multiple of 2^L where L is the multiscale level of the network
ny = 32    # must be multiple of 2^L where L is the multiscale level of the network
n_in   = 4
n_cond = 4
n_hidden = 32
batchsize = 5
L = 2   # number of scales
K = 2   # number of flow steps per scale

# Input
X = rand(Float32, nx, ny, n_in, batchsize) |> device;

# Condition
Y = rand(Float32, nx, ny, n_in, batchsize) |> device;

# Glow network
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K)  |> device

# Objective function
function loss(G, X, Y)
    ZX, ZY, logdet = G.forward(X, Y)
    f = .5f0/batchsize*norm(ZX)^2 - logdet
    G.backward(1f0./batchsize*ZX, ZX, ZY)
    return f
end

# Evaluate loss
f = loss(G, X, Y)

# Update weights
opt = Flux.ADAM()
Params = get_params(G)
for p in Params
    Flux.update!(opt, p.data, p.grad)
end
clear_grad!(G)

################ 3D example: To do with 3 spatial dimensions you need to set ndims on network. 
############################## or use NetworkConditionalGlow3D
nz = 32

# 3D Input
X_3d = rand(Float32, nx, ny, nz, n_in, batchsize) |> device;

# #dCondition
Y_3d = rand(Float32, nx, ny, nz, n_in, batchsize) |> device;

# 3D Glow network
G_3d = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; ndims=3)  |> device

# Evaluate loss
f = loss(G_3d, X_3d, Y_3d) 

# Update weights
opt = Flux.ADAM()
Params = get_params(G_3d)
for p in Params
    Flux.update!(opt, p.data, p.grad)
end
clear_grad!(G_3d)