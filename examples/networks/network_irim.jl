# Example for loop unrolling using invertible networks
# Adapted from Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test
using Flux, JOLI
import Flux.Optimise.update!

# Input
nx = 32
nz = 32
n_in = 4
n_hidden = 8
batchsize = 2
maxiter = 2

# Observed data
nxrec = 20
nt = 50
d = randn(Float32, nt, nxrec, 1, batchsize)

# Modeling/imaging operator (can be JOLI/JUDI operator or explicit matrix)
J = joMatrix(randn(Float32, nt*nxrec, nx*nz))

# Link function
Ψ(η) = identity(η)

# Unrolled loop
L = NetworkLoop(nx, nz, n_in, n_hidden, batchsize, maxiter, Ψ; type="additive")

# Initializations
η_obs = randn(Float32, nx, nz, 1, batchsize)
s_obs = randn(Float32, nx, nz, n_in-1, batchsize)
η_in = randn(Float32, nx, nz, 1, batchsize)
s_in = randn(Float32, nx, nz, n_in-1, batchsize)

# Forward pass and residual
rtm = reshape(J'*reshape(d, :, batchsize), nx, nz, 1, batchsize)
η_out, s_out = L.forward(η_in, s_in, rtm, J'*J)
Δη = η_out - η_obs
Δs = s_out - s_obs  # in practice there is no observed s, so Δs=0f0

# Backward pass
Δη_inv, Δs_inv, η_inv, s_inv = L.backward(Δη, Δs, η_out, s_out, rtm, J'*J)

# Check invertibility
@test isapprox(norm(η_inv - η_in)/norm(η_inv), 0f0, atol=1e-5)
@test isapprox(norm(s_inv - s_in)/norm(s_inv), 0f0, atol=1e-5)

# Update using Flux optimizer
opt = Flux.ADAM()
P = get_params(L)
for p in P
    update!(opt, p.data, p.grad)
end
clear_grad!(L)

