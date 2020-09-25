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
N = ActNorm(nc; logdet=logdet); N.forward(X)

# Jacobian
J = JacobianInvNet(N, X)

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
# N1 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2 = ActNorm(nc; logdet=logdet)
# N2 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N = Sequential(N1, N2)

X = rand(Float32, nx, ny, nc, batchsize)
X0 = rand(Float32, nx, ny, nc, batchsize)
dX = rand(Float32, nx, ny, nc, batchsize)
Y, _ = N.forward(X)
Y0, _ = N.forward(X0)

# Gradient test for X
print("\nGradient test sequential\n")
acc = 4
function fd_approx(f, X, dX, h; acc = 2)
    a = 0f0
    acc == 2 && (c = [-0.5f0 0f0 0.5f0])
    acc == 4 && (c = [1f0/12f0 -2f0/3f0 0f0 2f0/3f0 -1f0/12f0])
    l = Int64((length(c)-1)/2)
    for i = -l:l
        i != 0 && (a += c[i+l+1]*f(X+i*h*dX)/h)
    end
    return a
end
f0, ΔX = loss(N, X0, Y)[1:2]
h = 1f-2
f(X) = loss(N, X, Y)[1]
a = fd_approx(f, X0, dX, h; acc = acc)
b = dot(dX, ΔX)
@test isapprox(a, b; rtol=1f-2)

# Gradient test for parameters
logdet = true
N1_0 = ActNorm(nc; logdet=logdet)
# N1_0 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N2_0 = ActNorm(nc; logdet=logdet)
# N2_0 = CouplingLayerHINT(nx, ny, nc, n_hidden, batchsize; permute="full", logdet=logdet)
N0 = Sequential(N1_0, N2_0)
N0.forward(randn(Float32, size(X)))
θ = deepcopy(get_params(N0))
dθ = get_params(N)-get_params(N0)

print("\nGradient test sequential\n")
f0, ΔX, Δθ = loss(N0, X, Y)
h = 1f-1
function f(θ)
    set_params!(N0, θ)
    return loss(N0, X, Y)[1]
end
a = fd_approx(f, θ, dθ, h; acc = acc)
b = dot(dθ, Δθ)
@test isapprox(a, b; rtol=1f-2)
