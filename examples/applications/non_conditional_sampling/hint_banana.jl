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
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayerHINT}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
    AN[j] = ActNorm(n_in; logdet=true)
    L[j] = CouplingLayerHINT(n_in, n_hidden; logdet=true, permute="lower", k1=1, k2=1, p1=0, p2=0)

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
    Y, logdet = forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX = backward(ΔY, Y)[1]
    return f, ΔX
end

# Training
maxiter = 1000
opt = Flux.ADAM(1f-3)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    fval[j] = loss(X)[1]
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
Y_ = forward(X)[1]
Y = randn(Float32, 1, 1, 2, test_size)
X_ = backward(Y, Y)[2]

# Plot
figure(figsize=[8,8])
ax1 = subplot(2,2,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Data space: $x \sim \hat{p}_X$")
#ax1.set_xlim([-3.5,3.5]); ax1.set_ylim([0,50])
ax2 = subplot(2,2,2); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Latent space: $z = f(x)$")
#ax2.set_xlim([-3.5, 3.5]); ax2.set_ylim([-3.5, 3.5])
ax3 = subplot(2,2,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Data space: $x = f^{-1}(z)$")
#ax3.set_xlim([-3.5,3.5]); ax3.set_ylim([0,50])
ax4 = subplot(2,2,4); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "."); title(L"Latent space: $z \sim \hat{p}_Z$")
#ax4.set_xlim([-3.5, 3.5]); ax4.set_ylim([-3.5, 3.5])
