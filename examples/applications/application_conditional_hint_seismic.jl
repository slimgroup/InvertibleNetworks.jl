# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random, Test, JLD, Statistics
import Flux.Optimise.update!

# Random seed
Random.seed!(66)


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

# Create network
n_hidden = 64
batchsize = 4
depth = 8
CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)
Params = get_params(CH)

####################################################################################################

# Loss
function loss(CH, X, Y)
    Zx, Zy, logdet = CH.forward(X, Y)
    f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    ΔZ = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
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
    
    fval[j] = loss(CH, X, Y)[1]
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
        update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(CH)
end

####################################################################################################
# Plotting

# Testing
test_size = 100
idx = randperm(ntest)[1:test_size]  # draw random samples from testing data
X = X_test[:, :, :, idx]
Y = X + .5f0*randn(Float32, nx, ny, n_in, test_size)
Zx_, Zy_ = CH.forward(X, Y)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = CH.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
idx = 1
X_fixed = X[:, :, :, idx:idx]
Y_fixed = Y[:, :, :, idx:idx]
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
X_post = CH.inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

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
ax8 = subplot(2,4,8); imshow(X_post_std[:, :, 1,1], cmap="binary", aspect="auto", vmin=0, vmax=0.9*maximum(X_post_std)); 
colorbar(); title("Posterior std");

