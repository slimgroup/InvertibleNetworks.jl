# Example of training speed-up with Jacobians
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, PyPlot
using Random; Random.seed!(123)

# Initialize HINT layer
nx = 32
ny = 32
n_channel = 2
n_hidden = 64
batchsize = 1
logdet = false
N = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute="full")

# Fixed input
X = randn(Float32, nx, ny, n_channel, batchsize)

# Loss function
Ŷ = randn(Float32, nx, ny, n_channel, batchsize)
loss(Y) = 0.5f0*norm(Ŷ-Y)^2f0
∇loss(Y) = Y-Ŷ

# Training
lr = 1f-2
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
    J = Jacobian(N, X)
    _, Δθ = adjoint(J)*ΔY

    # Update parameters
    θ = get_params(N)
    lr_ = lr*norm.(θ)./(norm.(Δθ).+1f-10)
    θ = θ-lr_.*Δθ
    set_params!(N, θ)

end