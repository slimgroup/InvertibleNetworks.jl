# Test 1 x 1 convolution module using Householder matrices
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Flux, Random

Random.seed!(11)
###################################################################################################
# Initialize parameters

# Dimensions
nx = 28
ny = 28
k = 4
batchsize = 2

# Variables
v1 = randn(Float32, k) # |> gpu
v10 = randn(Float32, k) # |> gpu
dv1 = v1 - v10

v2 = randn(Float32, k) # |> gpu
v20 = randn(Float32, k) # |> gpu
dv2 = v2 - v20

v3 = randn(Float32, k) # |> gpu
v30 = randn(Float32, k) # |> gpu
dv3 = v3 - v30

# Input
X = randn(Float32, nx, ny, k, batchsize) # |> gpu
X0 = randn(Float32, nx, ny, k, batchsize) # |> gpu
dX = X - X0

# Operators
C = Conv1x1(v1, v2, v3) # |> gpu
C0 = Conv1x1(v10, v20, v30) # |> gpu


###################################################################################################
# Test invertibility

X_ = C.inverse(C.forward(X))
err1 = norm(X - X_)/norm(X)

@test isapprox(err1, 0f0; atol=1f-6)

X_ = C.forward(C.inverse(X))
err2 = norm(X - X_)/norm(X)

@test isapprox(err2, 0f0; atol=1f-6)

Y = C.forward(X)
ΔY = randn(Float32, nx, ny, k, batchsize) # |> gpu
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
@show dot(Δv1, Δv1), dot(Δv2, Δv2) , dot(Δv3, Δv3)
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


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
batchsize=10
v10 = randn(Float32, k)
v20 = randn(Float32, k)
v30 = randn(Float32, k)
C0 = Conv1x1(v10, v20, v30; logdet=true)
θ0 = deepcopy(get_params(C0))
v1 = randn(Float32, k)
v2 = randn(Float32, k)
v3 = randn(Float32, k)
C = Conv1x1(v1, v2, v3; logdet=true)
θ = deepcopy(get_params(C))
X = randn(Float32, nx, ny, k, batchsize)

# Perturbation (normalized)
dθ = θ-θ0
for i = 1:length(θ)
    dθ[i] = norm(θ0[i])*dθ[i]/(norm(dθ[i]).+1f-10)
end
dX = randn(Float32, nx, ny, k, batchsize); dX = norm(X)*dX/norm(dX)

# Jacobian eval
dY, Y = C.jacobian(dX, dθ, X)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 7
err9 = zeros(Float32, maxiter)
err10 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(C, θ+h*dθ)
    Y_loc, _ = C.forward(X+h*dX)
    err9[j] = norm(Y_loc - Y)
    err10[j] = norm(Y_loc - Y - h*dY)
    print(err9[j], "; ", err10[j], "\n")
    global h = h/2f0
end

@test isapprox(err9[end] / (err9[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err10[end] / (err10[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(C, θ)
dY, Y = C.jacobian(dX, 0f0*dθ, X)
dY_ = randn(Float32, size(dY))
dX_, dθ_, _ = C.adjointJacobian(dY_, Y)
a = dot(dY, dY_)
b = dot(dX, dX_)+dot(0f0*dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)

# Gradient test (inverse)

# Perturbation (normalized)
dY = randn(Float32, nx, ny, k, batchsize); dY *= norm(Y)/norm(dY)

# Jacobian (inverse) eval
dX, X = C.jacobianInverse(dY, dθ, Y)

# Test
print("\nJacobian (inverse) test\n")
h = 0.1f0
maxiter = 7
err11 = zeros(Float32, maxiter)
err12 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(C, θ+h*dθ)
    X_loc, _ = C.inverse(Y+h*dY)
    err11[j] = norm(X_loc - X)
    err12[j] = norm(X_loc - X - h*dX)
    print(err11[j], "; ", err12[j], "\n")
    global h = h/2f0
end

@test isapprox(err11[end] / (err11[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err12[end] / (err12[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Inverse test

dY = randn(Float32, nx, ny, k, batchsize)
Y = randn(Float32, nx, ny, k, batchsize)
dX = randn(Float32, nx, ny, k, batchsize)
X = randn(Float32, nx, ny, k, batchsize)
dX_, X_ = C.jacobianInverse(dY, 0f0*dθ, Y)
dY_, Y_ = C.jacobian(dX_, 0f0*dθ, X_)
@test isapprox(dY_, dY; rtol=1f-3)
@test isapprox(Y_, Y; rtol=1f-3)
dY_, Y_ = C.jacobian(dX, 0f0*dθ, X)
dX_, X_ = C.jacobianInverse(dY_, 0f0*dθ, Y_)
@test isapprox(dX_, dX; rtol=1f-3)
@test isapprox(X_, X; rtol=1f-3)

# Adjoint test (inverse)

set_params!(C, θ)
dY, Y = C.jacobianInverse(dX, dθ, X)
dY_ = randn(Float32, size(dY))
dX_, dθ_, _ = C.adjointJacobianInverse(dY_, Y)
a = dot(dY, dY_)
b = dot(dX, dX_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)