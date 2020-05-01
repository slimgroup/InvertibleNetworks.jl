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
batchsize = 64
depth = 8
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayerHINT}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
    AN[j] = ActNorm(n_in; logdet=true)
    L[j] = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize; logdet=true, permute="lower", k1=1, k2=1, p1=0, p2=0)

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

# Negative log-prior for banana distribution
c = [1f0, 4f0]
function neglogprior(X::Array{Float32, 4})
    z1 = c[1]*X[:, :, 1:1, :]
    z2 = X[:, :, 2:2, :]/c[1]-c[2]*c[1]^2*(X[:, :, 1:1, :].^2 .+1)
    return sum(0.5f0*(z1.^2+z2.^2)), cat(c[1]*z1-2*c[2]*c[1]^2*z2.*X[:, :, 1:1, :], z2/c[1]; dims = 3)
end

# Negative log-likelihood (Gaussian)
σ2 = (0.2f0)^2
Ydat = sample_banana(1)+sqrt(σ2)*randn(Float32, 1, 1, 2, 1)
function negloglike(X::Array{Float32, 4}; Y::Array{Float32, 4} = Ydat, σ2::Float32 = σ2)
    return -log_likelihood(Y.-X)/σ2, ∇log_likelihood(Y.-X)/σ2
end

# Negative log-posterior
function neglogpost(X::Array{Float32, 4})
    flike,  glike  = negloglike(X)
    fprior, gprior = neglogprior(X)
    return flike+fprior, glike+gprior
end

# Loss
function loss(Z::Array{Float32, 4})
    X, logdet = forward(Z)
    fpost, gpost = neglogpost(X)
    f = fpost-logdet
    ΔX = gpost
    backward(ΔX, X)
    return f
end

# Training
maxiter = 2000
opt = Flux.ADAM(1f-3)
lr_step = 100
lr_decay_fn = Flux.ExpDecay(1f-3, .9, lr_step, 0.)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    Z = randn(Float32, 1, 1, 2, batchsize)
    fval[j] = loss(Z)
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
        update!(lr_decay_fn, p.data, p.grad)
    end
    clear_grad!(Params)

end

####################################################################################################

# Testing (posterior sampling)
test_size = 500
Z = randn(Float32, 1, 1, 2, test_size)
Xpost = forward(Z)[1]

# Prior sampling
Xprior = sample_banana(test_size)

# Plot
fig = figure(figsize = [8, 8])
title("Prior and posterior samples")
plot(Xprior[1, 1, 1, :], Xprior[1, 1, 2, :], "y.", label = "Prior")
plot(Xpost[1, 1, 1, :], Xpost[1, 1, 2, :], "b.", label = "Posterior")
plot(Ydat[1], Ydat[2], "r*", label = "Data")
legend()
ax = gca()
ax.set_xlim([-3.5, 3.5]); ax.set_ylim([0, 20])
