# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random, Test, JLD, Statistics
import Flux.Optimise.update!

# Random seed
Random.seed!(22)


####################################################################################################

# Load original data X (size of n1 x n2 x nc x ntrain)
X_orig = load("../../data/seismic_samples_64_by_64_num_10k.jld")["X"]
n1, n2, nc, nsamples = size(X_orig)
AN = ActNorm(nsamples)
X_orig = AN.forward(X_orig) # zero mean and unit std

# Split in training - testing
ntrain = Int(nsamples*.9)
ntest = nsamples - ntrain

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(n1/2)
ny = Int(n2/2)
n_in = Int(nc*4)

# Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
X_train = zeros(Float32, nx, ny, n_in, ntrain)
for j=1:ntrain
    X_train[:, :, :, j:j] = wavelet_squeeze(X_orig[:, :, :, j:j])
end

X_test = zeros(Float32, nx, ny, n_in, ntest)
for j=1:ntest
    X_test[:, :, :, j:j] = wavelet_squeeze(X_orig[:, :, :, ntrain+j:ntrain+j])
end

# Network parameters
n_hidden = 64
batchsize = 4
depth = 8
AN_X = Array{ActNorm}(undef, depth)
AN_Y = Array{ActNorm}(undef, depth)
L = Array{ConditionalLayerHINT}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth

    # Actnorm layers
    AN_X[j] = ActNorm(n_in; logdet=true)
    AN_Y[j] = ActNorm(n_in; logdet=true)
    global Params = cat(Params, get_params(AN_X[j]); dims=1)
    global Params = cat(Params, get_params(AN_Y[j]); dims=1)

    # Conditional HINT layers
    L[j] = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; permute=true)
    global Params = cat(Params, get_params(L[j]); dims=1)
end


####################################################################################################

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

# Inverse pass
function inverse(X, Y)
    for j=depth:-1:1
        X_, Y_ = L[j].inverse(X, Y)
        Y = AN_Y[j].inverse(Y_)
        X = AN_X[j].inverse(X_)
    end
    return X, Y
end

# Forward pass Y-lane
function forward_Y(Y)
    for j=1:depth
        Y_ = AN_Y[j].forward(Y)[1]
        Y = L[j].forward_Y(Y_)
    end
    return Y
end

# Inverse pass Y-lane
function inverse_Y(Y)
    for j=depth:-1:1
        Y_ = L[j].inverse_Y(Y)
        Y = AN_Y[j].inverse(Y_)
    end
    return Y
end

####################################################################################################

# Test layers
test_size = 10
idx = randperm(ntrain)[1:test_size]
X = X_train[:, :, :, idx]
Y = X + .1f0*randn(Float32, nx, ny, n_in, test_size)

# Forward-backward
Zx, Zy, logdet = forward(X, Y)
X_, Y_ = backward(0f0.*Zx, 0f0.*Zy, Zx, Zy)[3:4]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-5)

# Forward-inverse
Zx, Zy, logdet = forward(X, Y)
X_, Y_ = inverse(Zx, Zy)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-5)

# Y-lane only
Zyy = forward_Y(Y)
Yy = inverse_Y(Zyy)
@test isapprox(norm(Y - Yy)/norm(Y), 0f0; atol=1f-5)


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
lr_step = 100
lr_decay_fn = Flux.ExpDecay(1f-3, .9, lr_step, 0.)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    idx = randperm(ntrain)[1:batchsize]
    X = X_train[:, :, :, idx]
    Y = X + .5f0*randn(Float32, nx, ny, n_in, batchsize)
    
    fval[j] = loss(X, Y)[1]
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
        update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(Params)
end

####################################################################################################
# Plotting

# Testing
test_size = 100
idx = randperm(ntest)[1:test_size]  # draw random samples from testing data
X = X_test[:, :, :, idx]
Y = X + .5f0*randn(Float32, nx, ny, n_in, test_size)
Zx_, Zy_ = forward(X, Y)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = inverse(Zx, Zy)

# Now select single fixed sample from all Ys
idx = 1
X_fixed = X[:, :, :, idx:idx]
Y_fixed = Y[:, :, :, idx:idx]
Zy_fixed = forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
X_post = inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

# Unsqueeze all tensors
X = wavelet_unsqueeze(X)
Y = wavelet_unsqueeze(Y)
Zx_ = wavelet_unsqueeze(Zx_)
Zy_ = wavelet_unsqueeze(Zy_)

X_ = wavelet_unsqueeze(X_)
Y_ = wavelet_unsqueeze(Y_)
Zx = wavelet_unsqueeze(Zx)
Zy = wavelet_unsqueeze(Zy)

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)
Zy_fixed = wavelet_unsqueeze(Zy_fixed)
X_post = wavelet_unsqueeze(X_post)

# Plot one sample from X and Y and their latent versions
figure(figsize=[16,8])
ax1 = subplot(2,4,1); imshow(X[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Model space: $x \sim \hat{p}_x$")
ax2 = subplot(2,4,2); imshow(Y[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Noisy data $y=x+n$ ")
ax3 = subplot(2,4,3); imshow(X_[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax4 = subplot(2,4,4); imshow(Y_[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Data space: $y = f(zx|zy)^{-1}$")
ax5 = subplot(2,4,5); imshow(Zx_[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Latent space: $zx = f(x|y)$")
ax6 = subplot(2,4,6); imshow(Zy_[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Latent space: $zy = f(x|y)$")
ax7 = subplot(2,4,7); imshow(Zx[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Latent space: $zx \sim \hat{p}_{zx}$")
ax8 = subplot(2,4,8); imshow(Zy[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Latent space: $zy \sim \hat{p}_{zy}$")

# Plot various samples from X and Y
figure(figsize=[16,8])
i = randperm(test_size)[1:4]
ax1 = subplot(2,4,1); imshow(X_[:, :, 1, i[1]], cmap="gray", aspect="auto"); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax2 = subplot(2,4,2); imshow(X_[:, :, 1, i[2]], cmap="gray", aspect="auto"); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax3 = subplot(2,4,3); imshow(X_[:, :, 1, i[3]], cmap="gray", aspect="auto"); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax4 = subplot(2,4,4); imshow(X_[:, :, 1, i[4]], cmap="gray", aspect="auto"); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax5 = subplot(2,4,5); imshow(X[:, :, 1, i[1]], cmap="gray", aspect="auto"); title(L"Model space: $x \sim \hat{p}_x$")
ax6 = subplot(2,4,6); imshow(X[:, :, 1, i[2]], cmap="gray", aspect="auto"); title(L"Model space: $x \sim \hat{p}_x$")
ax7 = subplot(2,4,7); imshow(X[:, :, 1, i[3]], cmap="gray", aspect="auto"); title(L"Model space: $x \sim \hat{p}_x$")
ax8 = subplot(2,4,8); imshow(X[:, :, 1, i[4]], cmap="gray", aspect="auto"); title(L"Model space: $x \sim \hat{p}_x$")

# Plot posterior samples, mean and standard deviation
figure(figsize=[16,8])
X_post_mean = mean(X_post; dims=4)
X_post_std = std(X_post; dims=4)
ax1 = subplot(2,4,1); imshow(X_fixed[:, :, 1, 1], cmap="gray", aspect="auto"); title("True x")
ax2 = subplot(2,4,2); imshow(X_post[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
ax3 = subplot(2,4,3); imshow(X_post[:, :, 1, 2], cmap="gray", aspect="auto"); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
ax4 = subplot(2,4,4); imshow(X_post_mean[:, :, 1, 1], cmap="gray", aspect="auto"); title("Posterior mean")
ax5 = subplot(2,4,5); imshow(Y_fixed[:, :, 1, 1], cmap="gray", aspect="auto");  title(L"Noisy data $y_i=x_i+n$ ")
ax6 = subplot(2,4,6); imshow(X_post[:, :, 1, 4], cmap="gray", aspect="auto"); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
ax7 = subplot(2,4,7); imshow(X_post[:, :, 1, 5], cmap="gray", aspect="auto"); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
ax8 = subplot(2,4,8); imshow(X_post_std[:, :, 1,1], cmap="binary", aspect="auto", vmin=0, vmax=0.9*maximum(X_post_std)); colorbar(); 
title("Posterior std");

