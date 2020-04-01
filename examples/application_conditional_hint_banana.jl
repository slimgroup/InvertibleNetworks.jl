# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random
import Flux.Optimise.update!

# Random seed
Random.seed!(11)


####################################################################################################

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 64
batchsize = 20
depth = 4
AN_X = Array{ActNorm}(undef, depth)
AN_Y = Array{ActNorm}(undef, depth)
L = Array{ConditionalLayerHINT}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
    AN_X[j] = ActNorm(n_in; logdet=true)
    AN_Y[j] = ActNorm(n_in; logdet=true)
    L[j] = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=1, p1=0, p2=0)

    # Collect parameters
    global Params = cat(Params, get_params(AN_X[j]); dims=1)
    global Params = cat(Params, get_params(AN_Y[j]); dims=1)
    global Params = cat(Params, get_params(L[j]); dims=1)
end

# Forward pass
function forward(X, Y)
    logdet = 0f0
    for j=1:depth
        X_, logdet1 = AN_X[j].forward(X)
        Y_, logdet2 = AN_Y[j].forward(Y)
        X, Y, logdet3 = L[j].forward(X_, Y_)
        logdet += (logdet1 + logdet2 + logdet3)
    end
    return X, Y, logdet
end

# Backward pass
function backward(ΔX, ΔY, X, Y)
    for j=depth:-1:1
        ΔX_, ΔY_, X_, Y_ = L[j].backward(ΔX, ΔY, X, Y)
        ΔX, X = AN_X[j].backward(ΔX_, X_)
        ΔY, Y = AN_Y[j].backward(ΔY_, Y_)
    end
    return ΔX, ΔY, X, Y
end

####################################################################################################

# Loss
function loss(X, Y)
    Zx, Zy, logdet = forward(X, Y)
    f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    ΔZ = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    return f, ΔX, ΔY
end

# Training
maxiter = 1000
opt = Flux.ADAM(1f-3)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    Y = X + .2f0*randn(Float32, nx, ny, n_in, batchsize)
    fval[j] = loss(X, Y)[1]
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
    end
    clear_grad!(Params)
end

####################################################################################################

# Testing
test_size = 500
X = sample_banana(test_size)
Y = X + .2f0*randn(Float32, nx, ny, n_in, test_size)
Zx_, Zy_ = forward(X, Y)[1:2]
Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = backward(0f0.*Zx, 0f0.*Zy, Zx, Zy)[3:4]

# Plot
figure(figsize=[8,8])
ax1 = subplot(2,2,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Model space: $x \sim \hat{p}_x$")
ax2 = subplot(2,2,2); plot(Zx_[1, 1, 1, :], Zx_[1, 1, 2, :], "g."); title(L"Latent space: $zx = f(x|y)$")
ax3 = subplot(2,2,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Model space: $x = f^{-1}(zx|zy)$")
ax4 = subplot(2,2,4); plot(Zx[1, 1, 1, :], Zx[1, 1, 2, :], "."); title(L"Latent space: $zx \sim \hat{p}_{zx}$")

figure(figsize=[8,8])
ax1 = subplot(2,2,1); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "."); title(L"Data space: $y \sim \hat{p}_y$")
ax2 = subplot(2,2,2); plot(Zy_[1, 1, 1, :], Zy_[1, 1, 2, :], "g."); title(L"Latent space: $zy = f(y)$")
ax3 = subplot(2,2,3); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Data space: $y = f^{-1}(zy)$")
ax4 = subplot(2,2,4); plot(Zy[1, 1, 1, :], Zy[1, 1, 2, :], "."); title(L"Latent space: $zy \sim \hat{p}_{zy}$")
