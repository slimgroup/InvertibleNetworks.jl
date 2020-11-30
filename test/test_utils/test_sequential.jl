# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics


###############################################################################
# Initialization

# Input
nx = 28
ny = 28*2
nc = 4
n_hidden = 64
batchsize = 5
X = rand(Float32, nx, ny, nc, batchsize)

# Layers and initialization
logdet = true
N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
N3 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N = Composition(N1, N2, N3)


###############################################################################
# Test coherency of composition and manual composition

Y = copy(X)
Y, l3 = N3.forward(Y)
Y, l2 = N2.forward(Y)
Y, l1 = N1.forward(Y)
l = l1+l2+l3

Y_, l_ = N.forward(X)

@test isapprox(Y, Y_; rtol=1f-3)
@test isapprox(l, l_; rtol=1f-3)


###############################################################################
# Test coherency of composition and ∘

# Composition with ∘
N_ = N1∘N2∘N3

Y, l = N.forward(X)
Y_, l_ = N_.forward(X)

@test isapprox(Y, Y_; rtol=1f-3)
@test isapprox(l, l_; rtol=1f-3)


###############################################################################
# Test invertibility

@test isapprox(X, N.inverse(N.forward(X)[1]); rtol=1f-3)
@test isapprox(X, N.forward(N.inverse(X))[1]; rtol=1f-3)


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

# Initializing nets
N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
N3 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N0 = Composition(N1, N2, N3); N0.forward(rand(Float32, nx, ny, nc, batchsize))
θ0 = deepcopy(get_params(N0))
N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
N3 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N = Composition(N1, N2, N3); N.forward(rand(Float32, nx, ny, nc, batchsize))
θ = deepcopy(get_params(N))

# Inputs
X = rand(Float32, nx, ny, nc, batchsize)
X0 = rand(Float32, nx, ny, nc, batchsize)
dX = rand(Float32, nx, ny, nc, batchsize)
Y = rand(Float32, nx, ny, nc, batchsize)
dθ = θ-θ0
for i = 1:length(dθ)
    (norm(θ0[i].data) != 0f0) && (dθ[i].data .*= norm(θ0[i].data)/norm(dθ[i].data))
end

# Gradient test (input)
set_params!(N0, θ0)
f0, ΔX, Δθ = loss(N0, X, Y)
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
print("\nGradient test sequential network: input\n")
for j=1:maxiter
    f = loss(N0, X+h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h /= 2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test (parameters)
h = 0.1f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
print("\nGradient test sequential network: parameters\n")
for j=1:maxiter
    set_params!(N0, θ0+h*dθ)
    f = loss(N0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err3[j], "; ", err4[j], "\n")
    global h /= 2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initializing nets
N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
N3 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N0 = Composition(N1, N2, N3); N0.forward(rand(Float32, nx, ny, nc, batchsize))
θ0 = deepcopy(get_params(N0))
N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
N3 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N = Composition(N1, N2, N3); N.forward(rand(Float32, nx, ny, nc, batchsize))
θ = deepcopy(get_params(N))
X = randn(Float32, nx, ny, nc, batchsize)

# Perturbation (normalized)
dθ = θ-θ0; dθ .*= norm.(θ0)./(norm.(dθ).+1f-10)
dX = randn(Float32, nx, ny, nc, batchsize); dX *= norm(X)/norm(dX)

# Jacobian eval
dY, Y, _, _ = N.jacobian(dX, dθ, X)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(N, θ+h*dθ)
    Y_, _ = N.forward(X+h*dX)
    err5[j] = norm(Y_ - Y)
    err6[j] = norm(Y_ - Y - h*dY)
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(N, θ)
dY, Y, _, _ = N.jacobian(dX, dθ, X)
dY_ = randn(Float32, size(dY))
dX_, dθ_, _, _ = N.adjointJacobian(dY_, Y)
a = dot(dY, dY_)
b = dot(dX, dX_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)