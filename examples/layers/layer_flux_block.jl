using Test, LinearAlgebra, InvertibleNetworks, Flux

# Input
nx = 64
ny = 64
n_in = 10
n_hidden = 20
batchsize = 2

X = glorot_uniform(nx, ny, n_in, batchsize)

# Flux network
model = Chain(
    Conv((3,3), n_in => n_hidden; pad=1),
    BatchNorm(n_hidden, relu),
    Conv((3,3), n_hidden => n_hidden; pad=1),
    BatchNorm(n_hidden, relu),
    Conv((3,3), n_hidden => n_in; pad=1),
    BatchNorm(n_in, relu)
)

# Create Flux CNN block
FB = FluxBlock(model)

# Evaluate forward
Y = FB.forward(X)

# Backpropagate data residual
ΔY = Y.*.1f0  # data residual
ΔX = FB.backward(ΔY, X)

# Update weights
Params = get_params(FB)
α = 1f-2
for p in Params
    p.data[:] += α*p.grad[:]    # need to set values via [:]
end
clear_grad!(FB)

