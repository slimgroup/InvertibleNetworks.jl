# Author: Gabrio Rizzuti, grizzuti33@gatech.edu
# Date: September 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics


###############################################################################
# Initialization and invertibility

# Input
nx = 28
ny = 28*2
nc = 4
n_hidden = 64
batchsize = 5
X = rand(Float32, nx, ny, nc, batchsize)

# Layers and initialization
logdet = true
N1 = ActNorm(nc; logdet=logdet)
N2 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N = Sequential(N1, N2)

# Forward
Y, _ = N.forward(X)

# Test invertibility
@test isapprox(norm(X - N.inverse(N.forward(X)[1]))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - N.forward(N.inverse(X))[1])/norm(X), 0f0, atol=1f-6)


###############################################################################
# Gradient Test

function loss(N, X, Y)
    # Forward pass
    Y_, lgdet = N.forward(X)

    # Residual and function value
    ΔY = Y_ - Y
    f = 0.5f0/batchsize*norm(ΔY)^2f0-lgdet

    # Back propagation
    ΔX, _ = N.backward(ΔY./batchsize, Y_)

    return f, ΔX, deepcopy(get_grads(N))
end

# Inputs
logdet = true
N1 = ActNorm(nc; logdet=logdet)
N2 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N = Sequential(N1, N2)

X = rand(Float32, nx, ny, nc, batchsize)
X0 = rand(Float32, nx, ny, nc, batchsize)
dX = rand(Float32, nx, ny, nc, batchsize)
Y, _ = N.forward(X)
Y0, _ = N.forward(X0)

# Gradient test for X
maxiter = 10
print("\nGradient test sequential\n")
f0, ΔX = loss(N, X0, Y)[1:2]
h = 0.1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(N, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for parameters
logdet = true
N1_0 = ActNorm(nc; logdet=logdet)
N2_0 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N0 = Sequential(N1_0, N2_0)
N0.forward(randn(Float32, size(X)))
θ = deepcopy(get_params(N0))

dθ = get_params(N)-get_params(N0)
maxiter = 10
print("\nGradient test sequential\n")
f0, ΔX, Δθ = loss(N0, X, Y)
h = 0.1f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(N0, θ+h*dθ)
    f = loss(N0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
