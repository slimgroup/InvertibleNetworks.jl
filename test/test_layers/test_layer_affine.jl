# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics


###############################################################################
# Initialization and invertibility

# Input
nx = 28
ny = 28
nc = 4
batchsize = 10
X = rand(Float32, nx, ny, nc, batchsize)

# Layer and initialization
AL = AffineLayer(nx, ny, nc; logdet=false)
Y = AL.forward(X)

# Test invertibility
@test isapprox(norm(X - AL.inverse(AL.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AL.forward(AL.inverse(X)))/norm(X), 0f0, atol=1f-6)

# Test with logdet enabled
AL = AffineLayer(nx, ny, nc; logdet=true)
Y, lgdt = AL.forward(X)

# Test invertibility
@test isapprox(norm(X - AL.inverse(AL.forward(X)[1]))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AL.forward(AL.inverse(X))[1])/norm(X), 0f0, atol=1f-6)


###############################################################################
# Gradient Test

AL = AffineLayer(nx, ny, nc; logdet=true)
X = randn(Float32, nx, ny, nc, batchsize)
X0 = randn(Float32, nx, ny, nc, batchsize)
dX = X - X0; dX *= norm(X0)/norm(dX)

# Forward pass
Y = AL.forward(X)[1]

function loss(AL, X, Y)

    # Forward pass
    if AL.logdet
        Y_, lgdet = AL.forward(X)
    else
        Y_ = AL.forward(X)
    end

    # Residual and function value
    ΔY = Y_ - Y
    f = .5f0/batchsize*norm(ΔY)^2
    AL.logdet == true && (f -= lgdet)

    # Back propagation
    ΔX, X_ = AL.backward(ΔY./batchsize, Y_)

    # Check invertibility
    isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)

    return f, ΔX, get_grads(AL)
end

# Gradient test for X
maxiter = 5
print("\nGradient test affine layer\n")
f0, ΔX = loss(AL, X0, Y)[1:2]
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(AL, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for parameters
logdet=false
AL = AffineLayer(nx, ny, nc; logdet=logdet)
θ = deepcopy(get_params(AL))
θ[2].data = randn(Float32, size(θ[2].data))
AL0 = AffineLayer(nx, ny, nc; logdet=logdet)
θ0 = get_params(AL0)
θ0[2].data = randn(Float32, size(θ0[2].data))
set_params!(AL0, θ0)
θ0 = deepcopy(θ0)
dθ = θ-θ0; dθ .*= norm.(θ0)./(norm.(dθ).+1f-10)

maxiter = 5
print("\nGradient test affine layer\n")
f0, ΔX, Δθ = loss(AL0, X, Y)
h = 0.1f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(AL0, θ0+h*dθ)
    f = loss(AL0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
logdet=true
AL = AffineLayer(nx, ny, nc; logdet=logdet)
θ = deepcopy(get_params(AL))
AL0 = AffineLayer(nx, ny, nc; logdet=logdet)
θ0 = deepcopy(get_params(AL0))
X = randn(Float32, nx, ny, nc, batchsize)

# Perturbation (normalized)
dθ = θ-θ0
dθ[2].data = randn(Float32, size(dθ[2].data))
dθ .*= norm.(θ0)./(norm.(dθ).+1f-10)
dX = randn(Float32, nx, ny, nc, batchsize); dX *= norm(X)/norm(dX)

# Jacobian eval
logdet ? ((dY, Y, lgdet, GNdθ) = AL.jacobian(dX, dθ, X)) : ((dY, Y) = AL.jacobian(dX, dθ, X))

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err9 = zeros(Float32, maxiter)
err10 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(AL, θ+h*dθ)
    logdet ? ((Y_, _) = AL.forward(X+h*dX)) : (Y_ = AL.forward(X+h*dX))
    err9[j] = norm(Y_ - Y)
    err10[j] = norm(Y_ - Y - h*dY)
    print(err9[j], "; ", err10[j], "\n")
    global h = h/2f0
end

@test isapprox(err9[end] / (err9[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err10[end] / (err10[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(AL, θ)
logdet ? ((dY, Y, _, _) = AL.jacobian(dX, dθ, X)) : ((dY, Y) = AL.jacobian(dX, dθ, X))
dY_ = randn(Float32, size(dY))
logdet ? ((dX_, dθ_, _, _) = AL.adjointJacobian(dY_, Y)) : ((dX_, dθ_, _) = AL.adjointJacobian(dY_, Y))
a = dot(dY, dY_)
b = dot(dX, dX_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)