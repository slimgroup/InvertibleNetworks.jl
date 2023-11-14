## Further examples

We provide usage examples for all the layers and network in our [examples](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples) subfolder. Each of the example show how to setup and use the building block for simple random variables.

## 2D Rosenbrock/banana distribution sampling w/ GLOW

```@example banana
using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random

# Random seed
Random.seed!(11)

# Define network
nx = 1; ny = 1; n_in = 2
n_hidden = 64
batchsize = 20
depth = 4
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayerGlow}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
    AN[j] = ActNorm(n_in; logdet=true)
    L[j] = CouplingLayerGlow(n_in, n_hidden; k1=1, k2=1, p1=0, p2=0, logdet=true)

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
maxiter = 2000
opt = Flux.ADAM(1f-3)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    fval[j] = loss(X)[1]

    # Update params
    for p in Params
        Flux.update!(opt, p.data, p.grad)
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
fig = figure(figsize=[8,8])
ax1 = subplot(2,2,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Data space: $x \sim \hat{p}_X$")
ax1.set_xlim([-3.5,3.5]); ax1.set_ylim([0,50])
ax2 = subplot(2,2,2); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Latent space: $z = f(x)$")
ax2.set_xlim([-3.5, 3.5]); ax2.set_ylim([-3.5, 3.5])
ax3 = subplot(2,2,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Data space: $x = f^{-1}(z)$")
ax3.set_xlim([-3.5,3.5]); ax3.set_ylim([0,50])
ax4 = subplot(2,2,4); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "."); title(L"Latent space: $z \sim \hat{p}_Z$")
ax4.set_xlim([-3.5, 3.5]); ax4.set_ylim([-3.5, 3.5])
savefig("plot_banana.svg")
nothing
```
![](plot_banana.svg)





## Conditional 2D Rosenbrock/banana distribution sampling w/ cHINT

```@example cbanana
using InvertibleNetworks
using Flux, LinearAlgebra, PyPlot

# Define network
nx = 1; ny = 1; n_in = 2
n_hidden = 64
batchsize = 64
depth = 8

# Construct HINT network
H = NetworkConditionalHINT(n_in, n_hidden, depth; k1=1, k2=1, p1=0, p2=0)

# Linear forward operator
A = randn(Float32,2,2)
A = A / (2*opnorm(A))

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
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_banana(batchsize)
    Y = reshape(A*reshape(X, :, batchsize), nx, ny, n_in, batchsize)
    Y += .2f0*randn(Float32, nx, ny, n_in, batchsize)

    fval[j] = loss(H, X, Y)[1]

    # Update params
    for p in get_params(H)
        Flux.update!(opt, p.data, p.grad)
    end
    clear_grad!(H)
end

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
X_fixed = sample_banana(1)
Y_fixed = reshape(A*vec(X_fixed), nx, ny, n_in, 1)
Y_fixed += .2f0*randn(Float32, size(X_fixed))

Zy_fixed = H.forward_Y(Y_fixed)
Zx = randn(Float32, nx, ny, n_in, test_size)

X_post = H.inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

# Model/data spaces
fig = figure(figsize=[16,6])
ax1 = subplot(2,5,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Model space: $x \sim \hat{p}_x$")
ax1.set_xlim([-3.5, 3.5]); ax1.set_ylim([0,50])
ax2 = subplot(2,5,2); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "."); title(L"Noisy data $y=Ax+n$ ")

ax3 = subplot(2,5,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax3.set_xlim([-3.5, 3.5]); ax3.set_ylim([0,50])
ax4 = subplot(2,5,4); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "g."); title(L"Data space: $y = f(zx|zy)^{-1}$")

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
savefig("plot_cbanana.svg")
nothing
```
![](plot_cbanana.svg)









## Literature applications

The following examples show the implementation of applications from the linked papers with [InvertibleNetworks.jl]:

- Invertible recurrent inference machines (Putzky and Welling, 2019) ([generic example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/networks/network_irim.jl))

- Generative models with maximum likelihood via the change of variable formula ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/applications/application_glow_banana_dist.jl))

- Glow: Generative flow with invertible 1x1 convolutions (Kingma and Dhariwal, 2018) ([generic example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/networks/network_glow.jl), [source](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/src/networks/invertible_network_glow.jl))
