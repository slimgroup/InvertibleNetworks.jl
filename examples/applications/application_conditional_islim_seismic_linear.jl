# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, JOLI, PyPlot, Flux, Random, Test, JLD, Statistics
import Flux.Optimise.update!
using Distributions: Uniform

# Random seed
Random.seed!(111)

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
nx1 = Int(n1/2)
nx2 = Int(n2/2)
nx_in = Int(nc*4)
nx_hidden = 128

# Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
X_train = zeros(Float32, nx1, nx2, nx_in, ntrain)
for j=1:ntrain
    X_train[:, :, :, j:j] = wavelet_squeeze(X_orig[:, :, :, j:j])
end

X_test = zeros(Float32, nx1, nx2, nx_in, ntest)
for j=1:ntest
    X_test[:, :, :, j:j] = wavelet_squeeze(X_orig[:, :, :, ntrain+j:ntrain+j])
end

# Data dimensions
ny1 = nx1
ny2 = nx2
ny_in = 1
ny_hidden = 64

####################################################################################################

# Create network
batchsize = 4
depth = 8
AN_X = Array{ActNorm}(undef, depth)
AN_Y = Array{ActNorm}(undef, depth)
L = Array{ConditionalLayerSLIM}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth

    # Actnorm layers
    AN_X[j] = ActNorm(nx_in; logdet=true)
    AN_Y[j] = ActNorm(ny_in; logdet=true)
    global Params = cat(Params, get_params(AN_X[j]); dims=1)
    global Params = cat(Params, get_params(AN_Y[j]); dims=1)

    # Conditional HINT layers
    L[j] = ConditionalLayerSLIM(nx1, nx2, nx_in, nx_hidden, ny1, ny2, ny_in, ny_hidden, batchsize; type="additive")
    global Params = cat(Params, get_params(L[j]); dims=1)
end

# Forward pass
function forward(X, Y, A)
    logdet = 0f0
    for j=1:depth
        X_, logdet1 = AN_X[j].forward(X)
        Y_, logdet2 = AN_Y[j].forward(Y)
        X, Y, logdet3 = L[j].forward(X_, Y_, A)
        logdet += (logdet1 + logdet2 + logdet3)
    end
    return X, Y, logdet
end

# Backward pass
function backward(ΔX, ΔY, X, Y, A)
    for j=depth:-1:1
        ΔX_, ΔY_, X_, Y_ = L[j].backward(ΔX, ΔY, X, Y, A)
        ΔX, X = AN_X[j].backward(ΔX_, X_)
        ΔY, Y = AN_Y[j].backward(ΔY_, Y_)
    end
    return ΔX, ΔY, X, Y
end

# Inverse pass
function inverse(X, Y, A)
    for j=depth:-1:1
        X_, Y_ = L[j].inverse(X, Y, A)
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
# Model data

# Data modeling function
function model_data(X, A)
    nx, ny, nc, nb = size(X)
    Y = reshape(A*reshape(X[:,:,1:1,:], :, nb), nx, ny, 1, nb)
    return Y
end

# Generate observed data
A = randn(Float32, Int(prod((ny1, ny2))), prod((nx1, nx2)))
A = A'*A
A = A / (2*opnorm(A))

Y_train = model_data(X_train, A)
Y_test = model_data(X_test, A)

####################################################################################################

# Loss
function loss(X, Y, A)
    Zx, Zy, logdet = forward(X, Y, A)
    f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    ΔZ = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ; split_index=nx_in)
    ΔX, ΔY = backward(ΔZx, ΔZy, Zx, Zy, A)[1:2]
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
    Y = Y_train[:, :, :, idx]
    
    fval[j] = loss(X, Y, A)[1]
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
Y = Y_test[:, :, :, idx]
Zx_, Zy_ = forward(X, Y, A)[1:2]

Zx = randn(Float32, nx1, nx2, nx_in, test_size)
Zy = randn(Float32, ny1, ny2, ny_in, test_size)
X_, Y_ = inverse(Zx, Zy, A)

# Now select single fixed sample from all Ys
idx = 1
X_fixed = X[:, :, :, idx:idx]
Y_fixed = Y[:, :, :, idx:idx]
Zy_fixed = forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
X_post = inverse(Zx, Zy_fixed.*ones(Float32, ny1, ny2, ny_in, test_size), A)[1]

# Unsqueeze all tensors
X = wavelet_unsqueeze(X)
Zx_ = wavelet_unsqueeze(Zx_)

X_ = wavelet_unsqueeze(X_)
Zx = wavelet_unsqueeze(Zx)

X_fixed = wavelet_unsqueeze(X_fixed)
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
ax8 = subplot(2,4,8); imshow(X_post_std[:, :, 1,1], cmap="binary", aspect="auto", vmin=0, vmax=0.9*maximum(X_post_std)); 
colorbar(); title("Posterior std");

