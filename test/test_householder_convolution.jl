# Test 1 x 1 convolution module using Householder matrices
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

###################################################################################################
# Initialize parameters

# Dimensions
nx = 28
ny = 28
k = 4
batchsize = 2

# Variables
v1 = glorot_uniform(k)
v10 = glorot_uniform(k)
dv1 = v1 - v10

v2 = glorot_uniform(k)
v20 = glorot_uniform(k)
dv2 = v2 - v20

v3 = glorot_uniform(k)
v30 = glorot_uniform(k)
dv3 = v3 - v30

# Input
X = glorot_uniform(nx, ny, k, batchsize)
X0 = glorot_uniform(nx, ny, k, batchsize)
dX = X - X0

# Operators
C = Conv1x1(v1, v2, v3)
C0 = Conv1x1(v10, v20, v30)


###################################################################################################
# Test invertibility

X_ = C.inverse(C.forward(X))
err1 = norm(X - X_)/norm(X)

@test isapprox(err1, 0f0; atol=1f-6)

X_ = C.forward(C.inverse(X))
err2 = norm(X - X_)/norm(X)

@test isapprox(err2, 0f0; atol=1f-6)

Y = C.forward(X)
ΔY = glorot_uniform(nx, ny, k, batchsize)
ΔX_, X_ = C.inverse((ΔY, Y))
err3 = norm(X - X_)/norm(X)

@test isapprox(err3, 0f0; atol=1f-6)

###################################################################################################
# Test gradients are set in inverse pass

# Predicted data and misfit
C0.v1.grad = nothing
Y_ = C0.forward(X)
@test isnothing(C0.v1.grad)

ΔY = Y_ - Y

# Compute gradients w.r.t. v
ΔX, X_ = C0.inverse((ΔY, Y_))
@test ~isnothing(C0.v1.grad)


###################################################################################################
# Gradient test

loss(ΔY) = .5f0*norm(ΔY)^2

function objective(C, X, Y)
    Y0 = C.forward(X)
    ΔY = Y0 - Y
    f = loss(ΔY)
    ΔX, X_ = C.inverse((ΔY, Y0))
    @test isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)
    return f, ΔX, C.v1.grad, C.v2.grad, C.v3.grad
end

# Observed data
Y = C.forward(X)

# Gradient test for X
maxiter = 5
print("Gradient test ΔX\n")
clear_grad!(C)
f0, ΔX = objective(C, X0, Y)[1:2]
h = .01f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(C.forward(X0 + h*dX) - Y)
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test for v
print("\nGradient test Δv1\n")
clear_grad!(C0)
f0, ΔX, Δv1, Δv2, Δv3 = objective(C0, X, Y)
h = .01f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    C0.v1.data = v10 + h*dv1
    C0.v2.data = v20 + h*dv2
    C0.v3.data = v30 + h*dv3
    f = loss(C0.forward(X) - Y)
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dv1, Δv1) - h*dot(dv2, Δv2) - h*dot(dv3, Δv3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Gradient test: forward-inverse in reverse order

loss(ΔY) = .5f0*norm(ΔY)^2

function objectiveT(C, X, Y)
    Y0 = C.inverse(X)
    ΔY = Y0 - Y
    f = loss(ΔY)
    ΔX, X_ = C.forward((ΔY, Y0))
    @test isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)
    return f, ΔX, C.v1.grad, C.v2.grad, C.v3.grad
end

# Observed data
Y = C.forward(X)

# Gradient test for X
maxiter = 5
print("\nGradient test ΔX\n")
C = Conv1x1(v1, v2, v3)
f0, ΔX = objectiveT(C, X0, Y)[1:2]
h = .01f0
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(C.inverse(X0 + h*dX) - Y)
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test for v
print("\nGradient test Δv1\n")
C0 = Conv1x1(v10, v20, v30)
f0, ΔX, Δv1, Δv2, Δv3 = objectiveT(C0, X, Y)
h = .01f0
err7 = zeros(Float32, maxiter)
err8 = zeros(Float32, maxiter)
for j=1:maxiter
    C0.v1.data = v10 + h*dv1
    C0.v2.data = v20 + h*dv2
    C0.v3.data = v30 + h*dv3
    f = loss(C0.inverse(X) - Y)
    err7[j] = abs(f - f0)
    err8[j] = abs(f - f0 - h*dot(dv1, Δv1) - h*dot(dv2, Δv2) - h*dot(dv3, Δv3))
    print(err7[j], "; ", err8[j], "\n")
    global h = h/2f0
end

@test isapprox(err7[end] / (err7[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err8[end] / (err8[1]/4^(maxiter-1)), 1f0; atol=1f1)
