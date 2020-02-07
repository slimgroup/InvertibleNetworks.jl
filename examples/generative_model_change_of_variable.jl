# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random
import Flux.Optimise.update!


# Target distribution
function sample_banana(batchsize; c=[1f0, 4f0])
    x = randn(Float32, 2, batchsize)
    y = zeros(Float32, 1, 1, 2, batchsize)
    y[1,1,1,:] = x[1,:] ./ c[1]
    y[1,1,2,:] = x[2,:].*c[1] + c[1].*c[2].*(x[1,:].^2 .+ c[1]^2)
    return y
end

####################################################################################################

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 256
batchsize = 100
depth = 10
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayer}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
    AN[j] = ActNorm(n_in; logdet=true)
    L[j] = CouplingLayer(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=1, p1=0, p2=0, logdet=true)

    # Collect parameters
    global Params = cat(Params, get_params(AN[j]); dims=1)
    global Params = cat(Params, get_params(L[j]); dims=1)
end

# Forward pass
function forward(X)
    logdet = 0f0
    for j=1:depth
        X_, logdet1 = AN[j].forward(X)
        X, logdet2 = L[j].forward(X_)
        logdet += (logdet1 + logdet2)
    end
    return X, logdet
end

# Backward pass
function backward(ΔX, X)
    for j=depth:-1:1
        ΔX_, X_ = L[j].backward(ΔX, X)
        ΔX, X = AN[j].backward(ΔX_, X_)
    end
    return ΔX, X
end

####################################################################################################

# Loss
function loss(X)
    Y_, logdet = forward(X)
    f = .5f0/batchsize*norm(Y_)^2 - logdet
    ΔX = backward(1f0/batchsize*Y_, Y_)[1]
    return f, ΔX
end

# Training
maxiter = 5000
opt = Flux.ADAM(1f-5)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    fval[j] = loss(X)[1]
    mod(j, 1) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
    end
    clear_grad!(Params)
end

####################################################################################################

# Testing
test_size = 1000
#X = sample_banana(test_size)
Y_ = forward(X)[1]
Y = randn(Float32, 1, 1, 2, test_size)
X_ = backward(Y, Y)[2]

figure(figsize=[8,8])
subplot(2,2,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Data space: $x \sim \hat{p}_X$")
subplot(2,2,2); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Latent space: $z = f(x)$")
subplot(2,2,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Data space: $x = f^{-1}(z)$")
subplot(2,2,4); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "."); title(L"Latent space: $z \sim \hat{p}_Z$")
