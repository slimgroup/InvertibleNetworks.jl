# Example of training with learning rates computed from Jacobians
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, Statistics, Flux, PyPlot, Images, TestImages
import Flux.Optimise: update!
using Random; Random.seed!(123)
flag_gpu = true
# flag_gpu = false
flag_gpu && (using CUDA)

# Load image
Y = Float32.(testimage("mandril_gray")); Y = (Y.-mean(Y))./sqrt(var(Y))
nx, ny = size(Y)
Y = reshape(Y, nx, ny, 1, 1)
for i = 1:2
    global Y = wavelet_squeeze(Y)
end
nx, ny, n_ch = size(Y)[1:3]
flag_gpu && (Y = Y |> gpu)

# Initialize resnet
n_hidden = 2*n_ch
batchsize = 1
nblocks = 2
norm_type = nothing
N = ResNet(n_ch, n_hidden, nblocks; norm=norm_type)
flag_gpu && (N = N |> gpu)

# Fixed input
X = randn(Float32, nx, ny, n_ch, batchsize)
flag_gpu && (X = X |> gpu)

# Initial normalization
Y_ = N.forward(X)
μ0 = mean(Y_)
σ0 = sqrt.(var(Y_))


# Loss function
loss(Y_) = 0.5f0*norm(Y-Y_)^2f0
∇loss(Y_) = Y_-Y

# Training
lr = 1f-3
opt = ADAM(lr)
maxiter = 3000
fval = zeros(Float32, maxiter)
for i = 1:maxiter

    # Evaluate network
    Y_ = (N.forward(X).-μ0)/σ0

    # Evaluate objective
    fval[i] = loss(Y_)
    (mod(i, 10) == 0 || i == 1) && (print("Iteration: ", i, "; err_rel = ", sqrt(2f0*fval[i]/norm(Y)^2f0), "\n"))

    # Compute gradient
    ΔY = ∇loss(Y_)/σ0
    N.backward(ΔY, X)

    # Update parameters
    for p in get_params(N)
        update!(opt, p.data, p.grad)
    end

end

Y_ = N.forward(X)
for i = 1:2
    global Y_ = wavelet_unsqueeze(Y_)
    global Y = wavelet_unsqueeze(Y)
end
flag_gpu && (Y = Y |> cpu; Y_ = Y_ |> cpu)