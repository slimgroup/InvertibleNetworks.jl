# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, JOLI, PyPlot, Flux, Random, Test, JLD, Statistics
import Flux.Optimise.update!
using Distributions: Uniform

# Random seed
Random.seed!(66)

####################################################################################################

# Load original data X (size of n1 x n2 x nc x ntrain)
#X_orig = load("../../data/seismic_samples_32_by_32_num_10k.jld")["X"]


# Load original data X (size of n1 x n2 x nc x ntrain)
datadir = dirname(pathof(InvertibleNetworks))*"/../data/"
filename = "seismic_samples_64_by_64_num_10k.jld"
~isfile("$(datadir)$(filename)") && run(`curl -L https://www.dropbox.com/s/mh5dv0yprestot4/seismic_samples_32_by_32_num_10k.jld\?dl\=0 --create-dirs -o $(datadir)$(filename)`)
X_orig = load("$(datadir)$(filename)")["X"]
n1, n2, nc, n_samples = size(X_orig)

AN = ActNorm(n_samples)
X_orig = AN.forward(X_orig) # zero mean and unit std

# Split in training - testing
ntrain = Int(n_samples*.9)
ntest = n_samples - ntrain

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(n1)
ny = Int(n2)
n_in = Int(nc)

# Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
X_train = zeros(Float32, nx, ny, n_in, ntrain)
for j=1:ntrain
    X_train[:, :, :, j:j] = X_orig[:, :, :, j:j]
end

X_test = zeros(Float32, nx, ny, n_in, ntest)
for j=1:ntest
    X_test[:, :, :, j:j] = X_orig[:, :, :, ntrain+j:ntrain+j]
end

# Create network
n_hidden = 32
batchsize = 4
L = 1
K = 8
CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, split_scales=false)
Params = get_params(CH)

# Data modeling function
function model_data(X, A)
    nx, ny, nc, nb = size(X)
    Y = reshape(A*reshape(X, :, nb), nx, ny, nc, nb)
    return Y
end

# Forward operator, precompute phase in ambient dimension
function phasecode(n)
	F = joDFT(n; DDT=Float32,RDT=ComplexF64)
	phase=F*(adjoint(F)*exp.(1im*2*pi*convert(Array{Float32},rand(dist,n))))
	phase = phase ./abs.(phase)
	sgn = sign.(convert(Array{Float32},randn(n)))
	# Return operator	
	return M = joDiag(sgn) * adjoint(F) * joDiag(phase)*F
end

# Generate observed data
dist = Uniform(-1, 1)
input_dim = (n1, n2)
subsamp = 2
M = phasecode(prod(input_dim))
R = joRestriction(prod(input_dim),randperm(prod(input_dim))[1:Int(round(prod(input_dim)/subsamp))]; DDT=Float32, RDT=Float32);
A_flat = R*M;
A = A_flat'*A_flat

Y_train = model_data(X_train, A)
Y_test = model_data(X_test, A)

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
maxiter = 500
opt = Flux.ADAM(1f-3)
lr_step = 100
lr_decay_fn = Flux.ExpDecay(1f-3, .9, lr_step, 0.)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    idx = randperm(ntrain)[1:batchsize]
    X = X_train[:, :, :, idx]
    Y = Y_train[:, :, :, idx]
    
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
Y = Y_test[:, :, :, idx]
Zx_, Zy_ = CH.forward(X, Y)[1:2]

Zx = randn(Float32, size(Zx_))
Zy = randn(Float32, size(Zy_))
X_, Y_ = CH.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
idx = 1
X_fixed = X[:, :, :, idx:idx]
Y_fixed = Y[:, :, :, idx:idx]
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
CH.forward_Y(X) # set X dimensions in forward pass (this needs to be fixed)
X_post = CH.inverse(Zx, Zy_fixed.*ones(Float32, size(Zx_)))[1]

# Plot one sample from X and Y and their latent versions
figure(figsize=[8,8])
ax1 = subplot(2,2,1); imshow(X[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Model space: $x \sim \hat{p}_x$")
ax2 = subplot(2,2,2); imshow(Y[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Noisy data $y=x+n$ ")
ax3 = subplot(2,2,3); imshow(X_[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax4 = subplot(2,2,4); imshow(Y_[:, :, 1, 1], cmap="gray", aspect="auto"); title(L"Data space: $y = f(zx|zy)^{-1}$")

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