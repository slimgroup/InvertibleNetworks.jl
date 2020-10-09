# Example of training speed-up with Jacobians
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, PyPlot, Flux
import Flux.Optimise.update!
using Random; Random.seed!(123)

# Initialize HINT layer
nx = 32
ny = 32
n_channel = 2
n_hidden = 64
batchsize = 1
logdet = false
N = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute="full")
θ = get_params(N)

# Fixed input
X = randn(Float32, nx, ny, n_channel, batchsize)

# Loss function
Ŷ = randn(Float32, nx, ny, n_channel, batchsize)
loss(Y) = 0.5f0*norm(Ŷ-Y)^2f0
∇loss(Y) = Y-Ŷ

# Training
lr = 1f-3
opt = Flux.ADAM(lr)
maxiter = 1000
fval = zeros(Float32, maxiter)
for i = 1:maxiter

    # Evaluate network
    Y = N.forward(X)

    # Evaluate objective
    fval[i] = loss(Y)
    (mod(i, 10) == 0 || i == 1) && (print("Iteration: ", i, "; f = ", fval[i], "\n"))

    # Compute gradient
    ΔY = ∇loss(Y)
    JT = adjoint(Jacobian(N, X))
    _, Δθ = JT*ΔY # opt1
    # N.backward(ΔY, Y); Δθ = get_grads(N) # opt2

    # Update params
    for j = 1:length(θ)
        update!(opt, θ[j].data, Δθ[j].data)
    end

end