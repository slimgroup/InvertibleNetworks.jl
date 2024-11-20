# Method: Amortized posterior sampler / simulation based inference / Forwad KL variational inference
# Application: sample from conditional distribution given noisy observations of the rosenbrock distribution.
# Note: we currently recommend conditional glow architectures instead of HINT, unless you need the latent space of
#       the observation. 

# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
using Pkg
Pkg.add("InvertibleNetworks"); Pkg.add("Flux"); Pkg.add("PyPlot")

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random, Test
import Flux.Optimise.update!

# Random seed
Random.seed!(11)

####################################################################################################

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 64
batchsize = 64
depth = 8

# Create network
H = NetworkConditionalHINT(n_in, n_hidden, depth; k1=1, k2=1, p1=0, p2=0)

####################################################################################################

# Loss
function loss(H, X, Y)
    Zx, Zy, logdet = H.forward(X, Y)
    f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    ΔZ = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = H.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    return f, ΔX, ΔY
end

# Training
maxiter = 1000
opt = Flux.Optimiser(Flux.ExpDecay(1f-3, .9, 100, 0.), Flux.ADAM(1f-3))
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    Y = X + .2f0*randn(Float32, nx, ny, n_in, batchsize)
    fval[j] = loss(H, X, Y)[1]
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in get_params(H)
        update!(opt, p.data, p.grad)
    end
    clear_grad!(H)
end

# ####################################################################################################

# Testing
test_size = 1000
X = sample_banana(test_size)
Y = X + .2f0*randn(Float32, nx, ny, n_in, test_size)
Zx_, Zy_ = H.forward(X, Y)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = H.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
idx = 1
Y_fixed = Y[:, :, :, idx:idx]
Zy_fixed = H.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
Zx = randn(Float32, nx, ny, n_in, test_size)
X_post = H.inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

# Plot samples from X and Y and their latent versions
figure(figsize=[16,8])
ax1 = subplot(2,4,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Model space: $x \sim \hat{p}_x$")
ax1.set_xlim([-3.5, 3.5]); ax1.set_ylim([0,50])
ax2 = subplot(2,4,5); plot(Zx_[1, 1, 1, :], Zx_[1, 1, 2, :], "g."); title(L"Latent space: $zx = f(x|y)$")
ax2.set_xlim([-3.5, 3.5]); ax2.set_ylim([-3.5, 3.5])
ax3 = subplot(2,4,2); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax3.set_xlim([-3.5,3.5]); ax3.set_ylim([0,50])
ax4 = subplot(2,4,6); plot(Zx[1, 1, 1, :], Zx[1, 1, 2, :], "."); title(L"Latent space: $zx \sim \hat{p}_{zx}$")
ax4.set_xlim([-3.5, 3.5]); ax4.set_ylim([-3.5, 3.5])
ax5 = subplot(2,4,3); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Data space: $y = f(zx|zy)^{-1}$")
ax5.set_xlim([-3.5,3.5]); ax5.set_ylim([0,50])
ax6 = subplot(2,4,7); plot(Zy[1, 1, 1, :], Zy[1, 1, 2, :], "."); title(L"Latent space: $zy \sim \hat{p}_{zy}$")
ax6.set_xlim([-3.5, 3.5]); ax6.set_ylim([-3.5, 3.5])
ax7 = subplot(2,4,4); plot(X_post[1, 1, 1, :], X_post[1, 1, 2, :], "g."); 
plot(Y_fixed[1, 1, 1, :], Y_fixed[1, 1, 2, :], "r."); title(L"Model space: $x = f(zx|zy_{fix})^{-1}$")
ax7.set_xlim([-3.5,3.5]); ax7.set_ylim([0,50])
ax8 = subplot(2,4,8); plot(Zx[1, 1, 1, :], Zx[1, 1, 2, :], "."); 
plot(Zy_fixed[1, 1, 1, :], Zy_fixed[1, 1, 2, :], "r."); title(L"Latent space: $zx \sim \hat{p}_{zx}$")
ax8.set_xlim([-3.5, 3.5]); ax8.set_ylim([-3.5, 3.5])