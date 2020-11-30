# Generative model using the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random, Test
import Flux.Optimise.update!

# Random seed
Random.seed!(99)


####################################################################################################

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 64
batchsize = 64
depth = 8

# Conditional HINT network
H = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth; k1=1, k2=1, p1=0, p2=0)

A = randn(Float32,2,2);
A = A/(2*(opnorm(A)));

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
opt = Flux.ADAM(1f-3)
lr_step = 100
lr_decay_fn = Flux.ExpDecay(1f-3, .9, lr_step, 0.)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    Y = reshape(A*reshape(X, :, batchsize), nx, ny, n_in, batchsize)
    Y += .2f0*randn(Float32, nx, ny, n_in, batchsize)

    fval[j] = loss(H, X, Y)[1]
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in get_params(H)
        update!(opt, p.data, p.grad)
        update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(H)
end

####################################################################################################

# Testing
test_size = 1000
X = sample_banana(test_size)
Y = reshape(A*reshape(X, :, test_size), nx, ny, n_in, test_size)
Y += .2f0*randn(Float32, nx, ny, n_in, test_size)

Zx_, Zy_ = H.forward(X, Y)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = H.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
X_fixed = sample_banana(1);
Y_fixed = zeros(Float32,size(X_fixed));
Y_fixed[:,:,:]=A*vec(X_fixed[:,:,:])
Y_fixed = Y_fixed + .2f-0*randn(Float32, size(X_fixed));

Zy_fixed = H.forward_Y(Y_fixed)
Zx = randn(Float32, nx, ny, n_in, test_size)

X_post = H.inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

# Model/data spaces
figure(figsize=[16,6])
ax1 = subplot(2,5,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Model space: $x \sim \hat{p}_x$")
ax1.set_xlim([-3.5, 3.5]); ax1.set_ylim([0,50])
ax2 = subplot(2,5,2); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "."); title(L"Noisy data $y=Ax+n$ ")
ax2.set_xlim([-11, 1]); ax2.set_ylim([-16, 1])
ax3 = subplot(2,5,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax3.set_xlim([-3.5, 3.5]); ax3.set_ylim([0,50])
ax4 = subplot(2,5,4); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Data space: $y = f(zx|zy)^{-1}$")
ax4.set_xlim([-11, 1]); ax4.set_ylim([-16, 1])
ax5 = subplot(2,5,5); plot(X_post[1, 1, 1, :], X_post[1, 1, 2, :], "g."); 
plot(X_fixed[1, 1, 1, :], X_fixed[1, 1, 2, :], "r."); title(L"Model space: $x = f(zx|zy_{fix})^{-1}$")
ax5.set_xlim([-3.5, 3.5]); ax5.set_ylim([0,50])

# Latent spaces
ax6 = subplot(2,5,6); plot(Zx_[1, 1, 1, :], Zx_[1, 1, 2, :], "g."); title(L"Latent space: $zx = f(x|y)$")
ax6.set_xlim([-3.5, 3.5]); ax6.set_ylim([-3.5, 3.5])
ax7 = subplot(2,5,7); plot(Zy_[1, 1, 1, :], Zy[1, 1, 2, :], "g."); title(L"Latent space: $zy \sim \hat{p}_{zy}$")
ax7.set_xlim([-3.5, 3.5]); ax7.set_ylim([-3.5, 3.5])
ax8 = subplot(2,5,9); plot(Zx[1, 1, 1, :], Zx[1, 1, 2, :], ".");  title(L"Latent space: $zx \sim \hat{p}_{zy}$")
ax8.set_xlim([-3.5, 3.5]); ax8.set_ylim([-3.5, 3.5])
ax9 = subplot(2,5,8); plot(Zy[1, 1, 1, :], Zy[1, 1, 2, :], "."); title(L"Latent space: $zy \sim \hat{p}_{zy}$")
ax9.set_xlim([-3.5, 3.5]); ax9.set_ylim([-3.5, 3.5])
ax10 = subplot(2,5,10); plot(Zx[1, 1, 1, :], Zx[1, 1, 2, :], "."); 
plot(Zy_fixed[1, 1, 1, :], Zy_fixed[1, 1, 2, :], "r."); title(L"Latent space: $zx \sim \hat{p}_{zx}$")
ax10.set_xlim([-3.5, 3.5]); ax10.set_ylim([-3.5, 3.5])