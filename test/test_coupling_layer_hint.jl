# Tests for HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test


#######################################################################################################################
# Test invertibility

# Input
nx = 16
ny = 16
n_channel = 8
n_hidden = 32
batchsize = 2

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)

# Create HINT layer
HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize)

# Test 
Y = HL.forward(X)
X_ = HL.inverse(Y)

@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-6)


#######################################################################################################################
# Gradient test

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)
X0 = glorot_uniform(nx, ny, n_channel, batchsize)
dX = X - X0

function loss(HL, X)
    Y = HL.forward(X)
    f = .5*norm(Y)^2
    ΔY = copy(Y)
    ΔX, X_ = HL.backward(ΔY, Y)
    return f, ΔX, HL.CL[1].RB.W1.grad, X_
end

# Test for input
f0, gX, X_ = loss(HL, X0)[[1,2,4]]
@test isapprox(norm(X_ - X0)/norm(X0), 0f0; atol=1f-6)

maxiter = 5
h = 0.5f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("Gradient test ΔX\n")
for j=1:maxiter
    f = loss(HL, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, gX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Test for weights
X = glorot_uniform(nx, ny, n_channel, batchsize)
HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize)
HL0 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize)
HLini = deepcopy(HL0)
dW = HL.CL[1].RB.W1.data - HL0.CL[1].RB.W1.data

f0, gX, gW, X_ = loss(HL0, X)
@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-6)

maxiter = 5
h = 0.5f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test weights\n")
for j=1:maxiter
    HL0.CL[1].RB.W1.data = HLini.CL[1].RB.W1.data + h*dW
    f = loss(HL0, X)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(gW, dW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Test for weights and logdet

function loss_logdet(HL, X)
    Y, logdet = HL.forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = HL.backward(ΔY, Y)
    return f, ΔX, HL.CL[1].RB.W1.grad, X_
end

X = glorot_uniform(nx, ny, n_channel, batchsize)
HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=true)
HL0 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=true)
HLini = deepcopy(HL0)
dW = HL.CL[1].RB.W1.data - HL0.CL[1].RB.W1.data

f0, gX, gW, X_ = loss_logdet(HL0, X)
@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-6)

maxiter = 5
h = 0.5f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test weights\n")
for j=1:maxiter
    HL0.CL[1].RB.W1.data = HLini.CL[1].RB.W1.data + h*dW
    f = loss_logdet(HL0, X)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(gW, dW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)