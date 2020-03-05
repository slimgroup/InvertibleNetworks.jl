# Invertible CNN layer from Dinh et al. (2017)/Kingma and Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random

# Random seed
Random.seed!(11)

###################################################################################################
# Test invertibility

# Input
nx = 24
ny = 24
k = 4
n_in = 2
n_hidden = 4
batchsize = 1
k1 = 4
k2 = 3

# Input images
X = glorot_uniform(nx, ny, k, batchsize)
X0 = glorot_uniform(nx, ny, k, batchsize)
dX = X - X0

# 1x1 convolution and residual blocks
C = Conv1x1(k)
RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, fan=true)
L = CouplingLayer(C, RB; logdet=true)

X_ = L.inverse(L.forward(X)[1])
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)

X_ = L.forward(L.inverse(X))[1]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)

###################################################################################################
# Gradient tests

# Loss Function
function loss(L, X, Y)
    Y_, logdet = L.forward(X)
    f = mse(Y_, Y) - logdet
    ΔY = ∇mse(Y_, Y)
    ΔX = L.backward(ΔY, Y_)[1]

    # Pass back gradients w.r.t. input X and from the residual block and 1x1 conv. layer
    return f, ΔX, L.C.v1.grad, L.C.v2.grad, L.C.v3.grad, L.RB.W1.grad, L.RB.W2.grad, L.RB.W3.grad
end

# Invertible layers
C0 = Conv1x1(k)
RB0 = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, fan=true)
L01 = CouplingLayer(C0, RB; logdet=true)
L02 = CouplingLayer(C, RB0; logdet=true)

# Gradient test w.r.t. input X0
Y = L.forward(X)[1]
f0, ΔX = loss(L, X0, Y)[1:2]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test coupling layer\n")
for j=1:maxiter
    f = loss(L, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test w.r.t. weights of residual block
Y = L.forward(X)[1]
Lini = deepcopy(L02)
dW1 = L.RB.W1.data - L02.RB.W1.data
dW2 = L.RB.W2.data - L02.RB.W2.data
dW3 = L.RB.W3.data - L02.RB.W3.data

f0, ΔX, Δv1, Δv2, Δv3, ΔW1, ΔW2, ΔW3 = loss(L02, X, Y)
h = 0.1f0
maxiter = 6
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test coupling layer\n")
for j=1:maxiter
    L02.RB.W1.data = Lini.RB.W1.data + h*dW1
    L02.RB.W2.data = Lini.RB.W2.data + h*dW2
    L02.RB.W3.data = Lini.RB.W3.data + h*dW3
    f = loss(L02, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test w.r.t. 1x1 conv weights
Y = L.forward(X)[1]
Lini = deepcopy(L01)
dv1 = C.v1.data - C0.v1.data
dv2 = C.v2.data - C0.v2.data
dv3 = C.v3.data - C0.v3.data

f0, ΔX, Δv1, Δv2, Δv3, ΔW1, ΔW2, ΔW3 = loss(L01, X, Y)
h = 0.1f0
maxiter = 6
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)

print("\nGradient test coupling layer\n")
for j=1:maxiter
    L01.C.v1.data = Lini.C.v1.data + h*dv1
    L01.C.v2.data = Lini.C.v2.data + h*dv2
    L01.C.v3.data = Lini.C.v3.data + h*dv3
    f = loss(L01, X, Y)[1]
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h*dot(dv1, Δv1) - h*dot(dv2, Δv2) - h*dot(dv3, Δv3))
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)
