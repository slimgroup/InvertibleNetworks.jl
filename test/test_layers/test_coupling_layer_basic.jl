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
Xa = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb = randn(Float32, nx, ny, Int(k/2), batchsize)
Xa0 = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb0 = randn(Float32, nx, ny, Int(k/2), batchsize)
dXa = Xa - Xa0
dXb = Xb - Xb0

# 1x1 convolution and residual blocks
RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, fan=true)
L = CouplingLayerBasic(RB; logdet=true)

###################################################################################################
# Invertibility tests

Ya, Yb = L.forward(Xa, Xb)
Xa_, Xb_ = L.inverse(Ya, Yb)
@test isapprox(norm(Xa - Xa_)/norm(Xa), 0f0; atol=1e-2)
@test isapprox(norm(Xb - Xb_)/norm(Xb), 0f0; atol=1e-2)

Ya, Yb = L.forward(Xa, Xb)
Xa_, Xb_ = L.backward(Ya.*0f0, Yb.*0f0, Ya, Yb)[3:4]
@test isapprox(norm(Xa - Xa_)/norm(Xa), 0f0; atol=1e-2)
@test isapprox(norm(Xb - Xb_)/norm(Xb), 0f0; atol=1e-2)


Ya, Yb = L.inverse(Xa, Xb)
Xa_, Xb_ = L.forward(Ya, Yb)
@test isapprox(norm(Xa - Xa_)/norm(Xa), 0f0; atol=1e-2)
@test isapprox(norm(Xb - Xb_)/norm(Xb), 0f0; atol=1e-2)


###################################################################################################
# Gradient tests

# Loss Function
function loss(L, Xa, Xb, Ya, Yb)
    Ya_, Yb_, logdet = L.forward(Xa, Xb)
    f = mse(tensor_cat(Ya_, Yb_), tensor_cat(Ya, Yb)) - logdet
    ΔY = ∇mse(tensor_cat(Ya_, Yb_), tensor_cat(Ya, Yb))
    ΔYa, ΔYb = tensor_split(ΔY)
    ΔXa, ΔXb = L.backward(ΔYa, ΔYb, Ya_, Yb_)[1:2]

    # Pass back gradients w.r.t. input X and from the residual block and 1x1 conv. layer
    return f, ΔXa, ΔXb, L.RB.W1.grad, L.RB.W2.grad, L.RB.W3.grad
end

# Invertible layers
RB0 = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=k1, k2=k2, fan=true)
L01 = CouplingLayerBasic(RB; logdet=true)
L02 = CouplingLayerBasic(RB0; logdet=true)


# Gradient test w.r.t. input X0
Ya, Yb = L.forward(Xa, Xb)[1:2]
f0, ΔXa, ΔXb  = loss(L, Xa0, Xb0, Ya, Yb)[1:3]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test coupling layer\n")
for j=1:maxiter
    f = loss(L, Xa0 + h*dXa, Xb0 + h*dXb, Ya, Yb)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dXa, ΔXa) - h*dot(dXb, ΔXb))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test w.r.t. weights of residual block
Ya, Yb = L.forward(Xa, Xb)[1:2]
Lini = deepcopy(L02)
dW1 = L.RB.W1.data - L02.RB.W1.data
dW2 = L.RB.W2.data - L02.RB.W2.data
dW3 = L.RB.W3.data - L02.RB.W3.data

f0, ΔXa, ΔXb, ΔW1, ΔW2, ΔW3 = loss(L02, Xa, Xb, Ya, Yb)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test coupling layer\n")
for j=1:maxiter
    L02.RB.W1.data = Lini.RB.W1.data + h*dW1
    L02.RB.W2.data = Lini.RB.W2.data + h*dW2
    L02.RB.W3.data = Lini.RB.W3.data + h*dW3
    f = loss(L02, Xa, Xb, Ya, Yb)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
