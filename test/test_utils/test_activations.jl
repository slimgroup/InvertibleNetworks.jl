# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, Test


###############################################################################
# Leaky ReLU

nx = 28
ny = 28
n_in = 4
batchsize = 2

X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Invertibility
err = norm(X - LeakyReLUinv(LeakyReLU(X)))
@test isapprox(err, 0f0, atol=1f-6)

err = norm(X - LeakyReLU(LeakyReLUinv(X)))
@test isapprox(err, 0f0, atol=1f-6)

# Gradient test
function objective(X, Y)
    Y0 = LeakyReLU(X)
    ΔY = Y0 - Y
    f = .5f0*norm(ΔY)^2
    ΔX = LeakyReLUgrad(ΔY, Y0)  # Use Y0 to recompute forward state
    return f, ΔX
end

# Observed data
Y = ReLU(X)

# Gradient test for X
maxiter = 5
print("\nGradient test leaky ReLU\n")
f0, ΔX = objective(X0, Y)
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = objective(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

###############################################################################
# ReLU

nx = 28
ny = 28
n_in = 4
batchsize = 2

X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Gradient test
function objective(X, Y)
    Y0 = ReLU(X)
    ΔY = Y0 - Y
    f = .5f0*norm(ΔY)^2
    ΔX = ReLUgrad(ΔY, X)    # Pass X, as ReLU not invertible
    return f, ΔX
end

# Observed data
Y = ReLU(X)

# Gradient test for X
maxiter = 5
print("\nGradient test ReLU\n")
f0, ΔX = objective(X0, Y)
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = objective(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# Sigmoid

nx = 12
ny = 12
n_in = 4
batchsize = 2

X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Invertibility
err = norm(X - SigmoidInv(Sigmoid(X))) / norm(X)
@test isapprox(err, 0f0, atol=1f-5)

# Gradient test sigmoid
function objective(X, Y)
    Y0 = Sigmoid(X)
    ΔY = Y0 - Y
    f = .5f0*norm(ΔY)^2
    ΔX = SigmoidGrad(ΔY, Y0)
    return f, ΔX
end

# Observed data
Y = Sigmoid(X)

# Gradient test for X
maxiter = 5
print("\nGradient test Sigmoid\n")
f0, ΔX = objective(X0, Y)
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = objective(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# Gated linear unit (GaLU)

nx = 12
ny = 12
n_in = 4
batchsize = 2

X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Gradient test GaLU
function objective(X, Y)
    Y0 = GaLU(X)
    ΔY = Y0 - Y
    f = .5f0*norm(ΔY)^2
    ΔX = GaLUgrad(ΔY, X)
    return f, ΔX
end

# Observed data
Y = GaLU(X)

# Gradient test for X
maxiter = 5
print("\nGradient test GaLU\n")
f0, ΔX = objective(X0, Y)
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = objective(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# Sigmoid2 (scaled version of Sigmoid)

nx = 12
ny = 12
n_in = 4
batchsize = 2

X = randn(Float32, nx, ny, n_in, batchsize)
X0 = randn(Float32, nx, ny, n_in, batchsize)
dX = X - X0

# Gradient test Sigmoid2
L = Sigmoid2Layer()
function objective(X, Y)
    Y0 = L.forward(X)
    ΔY = Y0 - Y
    f = .5f0*norm(ΔY)^2
    ΔX = L.backward(ΔY, Y0)
    return f, ΔX
end

# Observed data
Y = L.forward(X)

# Gradient test for X
maxiter = 5
print("\nGradient test Sigmoid2\n")
f0, ΔX = objective(X0, Y)
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = objective(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# ExpClamp

nx = 12
ny = 12
n_in = 4
batchsize = 3

X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# Invertibility
err = norm(X - ExpClampInv(ExpClamp(X))) / norm(X)
@test isapprox(err, 0f0, atol=1f-5)

# Gradient test sigmoid
function objective(X, Y)
    Y0 = ExpClamp(X)
    ΔY = Y0 - Y
    f = .5f0*norm(ΔY)^2
    ΔX = ExpClampGrad(ΔY, Y0)
    return f, ΔX
end

# Observed data
Y = ExpClamp(X)

# Gradient test for X
maxiter = 5
print("\nGradient test ExpClamp\n")
f0, ΔX = objective(X0, Y)
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = objective(X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)