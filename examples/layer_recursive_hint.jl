# Invertible CNN layer from Dinh et al. (2017)/Kingma & Dhariwal (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test


# Get network depth
function get_depth(n_channel)
    count = 0
    nc = n_channel
    while nc > 4
        nc /= 2
        count += 1
    end
    return count +1
end

function forward_hint(L, X; scale=1)
    Xa, Xb = tensor_split(X)
    if size(X, 3) > 4
        Ya = forward_hint(L, Xa; scale=scale+1)
        Yb = L[scale].forward(forward_hint(L, Xb; scale=scale+1), Xa)[1]
    else
        Ya = copy(Xa)
        Yb = L[scale].forward(Xb, Xa)[1]
    end
    Y = tensor_cat(Ya, Yb)
    return Y
end

function inverse_hint(L, Y; scale=1)
    Ya, Yb = tensor_split(Y)
    if size(Y, 3) > 4
        Xa = inverse_hint(L, Ya; scale=scale+1)
        Xb = inverse_hint(L, L[scale].inverse(Yb, Xa)[1]; scale=scale+1)
    else
        Xa = copy(Ya)
        Xb = L[scale].inverse(Yb, Ya)[1]
    end
    X = tensor_cat(Xa, Xb)
    return X
end

backward_hint(L, Y_tuple::Tuple; scale=1) = backward_hint(L, Y_tuple[1], Y_tuple[2]; scale=scale)

function backward_hint(L, ΔY, Y; scale=1)
    Ya, Yb = tensor_split(Y)
    ΔYa, ΔYb = tensor_split(ΔY)
    if size(Y, 3) > 4
        ΔXa, Xa = backward_hint(L, ΔYa, Ya; scale=scale+1)
        ΔXb, Xb = backward_hint(L, L[scale].backward(ΔYb, ΔXa, Yb, Xa)[[1,3]]; scale=scale+1)
    else
        Xa = copy(Ya)
        ΔXa = copy(ΔYa)
        ΔXb, Xb = L[scale].backward(ΔYb, ΔYa, Yb, Ya)[[1,3]]
    end
    ΔX = tensor_cat(ΔXa, ΔXb)
    X = tensor_cat(Xa, Xb)
    return ΔX, X
end

#######################################################################################################################
# Test invertibility

# Input
nx = 32
ny = 32
n_channel = 16
n_hidden = 64
batchsize = 2
k1 = 4
k2 = 3

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)

# Create network
n = get_depth(n_channel)
L = Array{CouplingLayerBasic}(undef, n) 
for j=1:n
    L[j] = CouplingLayerBasic(nx, ny, Int(n_channel/2^j), n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1)
end


# Test 
print("Forward\n")
Y = forward_hint(L, X)
ΔY = copy(Y)

print("\nInverse\n")
X_ = inverse_hint(L, Y)

@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-6)


#######################################################################################################################
# Gradient test

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)
X0 = glorot_uniform(nx, ny, n_channel, batchsize)
dX = X - X0

function loss(L, X)

    Y = forward_hint(L, X)
    f = .5*norm(Y)^2
    ΔY = copy(Y)
    ΔX, X_ = backward_hint(L, ΔY, Y)
    return f, ΔX, L[1].RB.W1.grad, X_
end

# Test for input
f0, gX, X_ = loss(L, X0)[[1,2,4]]
@test isapprox(norm(X_ - X0)/norm(X0), 0f0; atol=1f-6)

maxiter = 6
h = 0.5f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("Gradient test ΔX\n")
for j=1:maxiter
    f = loss(L, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, gX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Test for weights
X = glorot_uniform(nx, ny, n_channel, batchsize)
L = Array{CouplingLayerBasic}(undef, n)
L0 = Array{CouplingLayerBasic}(undef, n)
for j=1:n
    L[j] = CouplingLayerBasic(nx, ny, Int(n_channel/2^j), n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1)
    L0[j] = CouplingLayerBasic(nx, ny, Int(n_channel/2^j), n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1)
end
Lini = deepcopy(L0)
i = 1
dW = L[i].RB.W1.data - L0[i].RB.W1.data

f0, gX, gW, X_ = loss(L0, X)
@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-6)

maxiter = 6
h = 0.5f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test weights\n")
for j=1:maxiter
    L0[i].RB.W1.data = Lini[i].RB.W1.data + h*dW
    f = loss(L0, X)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(gW, dW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)