# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, Flux, Test
import Flux.Optimise.update!

# Target distribution
function swirl(batchsize; noise=.5f0)
    n = sqrt.(rand(Float32, batchsize)) * 1000f0 * 2f0*pi/360f0
    d1x = -cos.(n).*n + rand(Float32, batchsize) .* noise
    d1y = sin.(n).*n + rand(Float32, batchsize) .* noise
    X = cat(d1x', d1y'; dims=1)
    return reshape(X, 1, 1, 2, batchsize) / norm(X, Inf)
end

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 32
batchsize = 10
depth = 2
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayerGlow}(undef, depth)
AN0 = Array{ActNorm}(undef, depth)
L0 = Array{CouplingLayerGlow}(undef, depth)

for j=1:depth
    AN[j] = ActNorm(n_in; logdet=true)
    L[j] = CouplingLayerGlow(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=1, p1=0, p2=0, logdet=true)
    AN0[j] = ActNorm(n_in; logdet=true)
    L0[j] = CouplingLayerGlow(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=1, p1=0, p2=0, logdet=true)
end

# Forward pass
function forward(AN, L, X)
    logdet = 0f0
    for j=1:depth
        X_, logdet1 = AN[j].forward(X)
        X, logdet2 = L[j].forward(X_)
        logdet += (logdet1 + logdet2)
    end
    return X, logdet
end

# Backward pass
function backward(AN, L, ΔX, X)
    logdet = 0f0
    for j=depth:-1:1
        ΔX_, X_ = L[j].backward(ΔX, X)
        ΔX, X = AN[j].backward(ΔX_, X_)
    end
    return ΔX, X
end

###################################################################################################

# Invertibility
X = swirl(1000)
X0 = swirl(1000)
dX = X - X0
Y = forward(AN, L, X)[1]
X_ = backward(AN, L, Y, Y)[2]
@test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-3)


###################################################################################################

# Loss
function loss(AN, L, X)
    Y_, logdet = forward(AN, L, X)
    f = .5f0/batchsize*norm(Y_)^2 - logdet
    ΔX, X = backward(AN, L, 1f0/batchsize*Y_, Y_)
    return f, ΔX, L[1].RB.W1.grad, AN[1].s.grad
end

# Gradient test for input
f0, g = loss(AN, L, X0)[1:2]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    f = loss(AN, L, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, g))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test for weights
forward(AN0, L0, X) # initialize parameters
ANini = deepcopy(AN0) 
Lini  = deepcopy(L0)
dW = L[1].RB.W1.data - L0[1].RB.W1.data
ds = AN[1].s.data - AN0[1].s.data

f0, gX, gW, gs = loss(AN0, L0, X)
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    L0[1].RB.W1.data = Lini[1].RB.W1.data + h*dW
    AN0[1].s.data = ANini[1].s.data + h*ds
    f = loss(AN0, L0, X)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(ds, gs) - h*dot(dW, gW))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
