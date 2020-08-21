# Adjoint test for convolution functions provided in NNlib
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using NNlib, Test, LinearAlgebra

###################################################################################################
# Adjoint test 2D

d = 4
n_in = 10
n_out = 20

X = glorot_uniform(32, 32, n_in, 1)
Y = glorot_uniform(8, 8, n_out, 1)
W = glorot_uniform(d, d, n_in, n_out)

# Convolution using NNlib functions
cdims = DenseConvDims(X, W; stride=(d,d), padding=(1,1))

Y_ = conv(X, W, cdims)
X_ = ∇conv_data(Y, W, cdims)    # corresponds to transpose operation

a = vec(Y)'*vec(Y_)
b = vec(X)'*vec(X_)

@test isapprox(a/b - 1f0, 0f0, atol=1f-4)

###################################################################################################
# Adjoint test 3D

d = 4
n_in = 10
n_out = 20

X = glorot_uniform(32, 32, 32, n_in, 1)
Y = glorot_uniform(8, 8, 8, n_out, 1)
W = glorot_uniform(d, d, d, n_in, n_out)

# Convolution using NNlib functions
cdims = DenseConvDims(X, W; stride=(d,d,d), padding=(1,1,1))

Y_ = conv(X, W, cdims)
X_ = ∇conv_data(Y, W, cdims)    # corresponds to transpose operation

a = vec(Y)'*vec(Y_)
b = vec(X)'*vec(X_)

@test isapprox(a/b - 1f0, 0f0, atol=1f-4)

###################################################################################################
# Gradient test 2D

# Input
X = glorot_uniform(64, 64, n_in, 1)
W = glorot_uniform(d, d, n_in, n_out)
W0 = glorot_uniform(d, d, n_in, n_out)
dW = W - W0
b = glorot_uniform(n_out)
b0 = glorot_uniform(n_out)
db = b - b0

# Operators
d = 4
k = 20

cdims = DenseConvDims(X, W; stride=(d,d), padding=(1,1))

function objective(W, b, X, Y)
    ΔY = (conv(X, W, cdims) .+ reshape(b, 1, 1, :, 1)) - Y
    f = .5f0*norm(ΔY)^2f0
    gW = ∇conv_filter(X, ΔY, cdims)
    gb = sum(ΔY, dims=[1,2,4])[1, 1, :, 1]
    return f, gW, gb
end

Y = conv(X, W, cdims) .+ reshape(b, 1, 1, :, 1)

# Gradient test weights
f0, gW, gb = objective(W0, b, X, Y)
h = 0.01f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = objective(W0 + h*dW, b, X, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dW, gW))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test bias
f0, gW, gb = objective(W, b0, X, Y)
h = 0.01f0
maxiter = 6
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = objective(W, b0 + h*db, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(db, gb))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

###################################################################################################
# Gradient test 3D

# Input
X = glorot_uniform(64, 64, 64, n_in, 1)
W = glorot_uniform(d, d, d, n_in, n_out)
W0 = glorot_uniform(d, d, d, n_in, n_out)
dW = W - W0
b = glorot_uniform(n_out)
b0 = glorot_uniform(n_out)
db = b - b0

# Operators
d = 4
k = 20

cdims = DenseConvDims(X, W; stride=(d,d,d), padding=(1,1,1))

function objective(W, b, X, Y)
    ΔY = (conv(X, W, cdims) .+ reshape(b, 1, 1, 1, :, 1)) - Y
    f = .5f0*norm(ΔY)^2f0
    gW = ∇conv_filter(X, ΔY, cdims)
    gb = sum(ΔY, dims=[1,2,3,5])[1, 1, 1, :, 1]
    return f, gW, gb
end

Y = conv(X, W, cdims) .+ reshape(b, 1, 1, 1, :, 1)

# Gradient test weights
f0, gW, gb = objective(W0, b, X, Y)
h = 0.01f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = objective(W0 + h*dW, b, X, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dW, gW))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test bias
f0, gW, gb = objective(W, b0, X, Y)
h = 0.01f0
maxiter = 6
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = objective(W, b0 + h*db, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(db, gb))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
