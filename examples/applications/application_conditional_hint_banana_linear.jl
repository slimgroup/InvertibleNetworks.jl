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
    L[j] = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=1, k2=1, p1=0, p2=0, permute=true)
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

A = randn(Float32,2,2)
A = A / (2*opnorm(A))

# Test layers
test_size = 10
X = sample_banana(test_size)
Y = reshape(A*reshape(X, :, test_size), nx, ny, n_in, test_size)
Y += .2f0*randn(Float32, nx, ny, n_in, test_size)

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
    X = sample_banana(batchsize)
    Y = reshape(A*reshape(X, :, batchsize), nx, ny, n_in, batchsize)
    Y += .2f0*randn(Float32, nx, ny, n_in, batchsize)

    fval[j] = loss(X, Y)[1]
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
        update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(Params)
end

# ####################################################################################################

# Testing
test_size = 1000
X = sample_banana(test_size)
Y = reshape(A*reshape(X, :, test_size), nx, ny, n_in, test_size)
Y += .2f0*randn(Float32, nx, ny, n_in, test_size)

Zx_, Zy_ = forward(X, Y)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = inverse(Zx, Zy)

# Now select single fixed sample from all Ys
X_fixed = sample_banana(1)
Y_fixed = reshape(A*vec(X_fixed), nx, ny, n_in, 1)
Y_fixed += .2f0*randn(Float32, size(X_fixed))

Zy_fixed = forward_Y(Y_fixed)
Zx = randn(Float32, nx, ny, n_in, test_size)

X_post = inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

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






