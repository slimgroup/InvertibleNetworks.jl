# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics

# Input
nx = 28
ny = 28
k = 4
batchsize = 2

# Input image: nx x ny x k x batchsize
X = rand(Float32, nx, ny, k, batchsize)

# Activation normalization
AN = ActNorm(k; logdet=false)

###############################################################################

# Test initialization
Y = AN.forward(X)
@test isapprox(mean(Y), 0f0; atol=1f-6)
@test isapprox(var(Y), 1f0; atol=1f-3)

# Test invertibility
@test isapprox(norm(X - AN.inverse(AN.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.forward(AN.inverse(X)))/norm(X), 0f0, atol=1f-6)

###############################################################################
# Gradient Test

AN = ActNorm(k; logdet=true)
X = randn(Float32, nx, ny, k, batchsize)
X0 = randn(Float32, nx, ny, k, batchsize)
dX = X - X0
Y = AN.forward(X)[1]

function loss(AN, X, Y)

    # Forward pass
    if AN.logdet == true
        Y_, lgdet = AN.forward(X)
    else
        Y_ = AN.forward(X)
    end

    # Residual and function value
    ΔY = Y_ - Y
    f = .5f0/batchsize*norm(ΔY)^2
    AN.logdet == true && (f -= lgdet)

    # Back propagation
    ΔX, X_ = AN.backward(ΔY./batchsize, Y)

    # Check invertibility
    isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)

    return f, ΔX, AN.s.grad, AN.b.grad
end

# Gradient test for X
maxiter = 6
print("\nGradient test actnorm\n")
f0, ΔX = loss(AN, X0, Y)[1:2]
h = .01f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(AN, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for parameters
AN0 = ActNorm(k; logdet=true); AN0.forward(X)
AN_ini = deepcopy(AN0)
ds = AN.s.data - AN0.s.data
db = AN.b.data - AN0.b.data
maxiter = 6
print("\nGradient test actnorm\n")
f0, ΔX, Δs, Δb = loss(AN0, X, Y)
h = .01f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    AN0.s.data = AN_ini.s.data + h*Δs
    AN0.b.data = AN_ini.b.data + h*Δb
    f = loss(AN0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(ds, Δs) - h*dot(db, Δb))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

