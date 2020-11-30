# Multiscale HINT network
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
Random.seed!(11)

# Define network
nx = 64
ny = 64
n_in = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2

# Multi-scale and single scale network
H = NetworkMultiScaleHINT(nx, ny, n_in, batchsize, n_hidden, L, K; split_scales=false, k1=3, k2=1, p1=1, p2=0)

print("\nInvertibility test multiscale HINT network\n")

# Test layers
test_size = 10
X = randn(Float32, nx, ny, n_in, test_size)

# Forward-backward
Z, logdet = H.forward(X)
X_ = H.backward(0f0.*Z, Z)[2]
@test isapprox(X, X_; rtol=1f-3)

# Forward-inverse
Z, logdet = H.forward(X)
X_ = H.inverse(Z)
@test isapprox(X, X_; rtol=1f-3)

# Loss
function loss(H, X)
    Z, logdet = H.forward(X)
    f = -log_likelihood(Z) - logdet
    ΔZ = -∇log_likelihood(Z)
    ΔX = H.backward(ΔZ, Z)[1]
    return f, ΔX
end

test_size = 10
X = randn(Float32, nx, ny, n_in, test_size)
X0 = randn(Float32, nx, ny, n_in, test_size)
dX = X - X0

f0, ΔX = loss(H, X0)
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test multiscale HINT net: input\n")
for j=1:maxiter
    f = loss(H, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
H = NetworkMultiScaleHINT(nx, ny, n_in, batchsize, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0); H.forward(randn(Float32, nx, ny, n_in, batchsize))
θ = deepcopy(get_params(H))
H0 = NetworkMultiScaleHINT(nx, ny, n_in, batchsize, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0); H0.forward(randn(Float32, nx, ny, n_in, batchsize))
θ0 = deepcopy(get_params(H0))
X = randn(Float32, nx, ny, n_in, batchsize)

# Perturbation (normalized)
dθ = θ-θ0; dθ .*= norm.(θ0)./(norm.(dθ).+1f-10)
dX = randn(Float32, nx, ny, n_in, batchsize); dX *= norm(X)/norm(dX)

# Jacobian eval
dZ, Z, _, _ = H.jacobian(dX, dθ, X)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(H, θ+h*dθ)
    Z_, _ = H.forward(X+h*dX)
    err5[j] = norm(Z_ - Z)
    err6[j] = norm(Z_ - Z - h*dZ)
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(H, θ)
dZ, Z, _, _ = H.jacobian(dX, dθ, X)
dZ_ = randn(Float32, size(dZ))
dX_, dθ_, _ = H.adjointJacobian(dZ_, Z)
a = dot(dZ, dZ_)
b = dot(dX, dX_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-1) ####### need to check low accuracy here