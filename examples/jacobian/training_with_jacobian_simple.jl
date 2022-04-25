# Example of training with learning rates computed from Jacobians
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, Statistics, Flux, PyPlot, Images, TestImages
using Random; Random.seed!(123)

device = InvertibleNetworks.CUDA.functional() ? gpu : cpu

# Load image
Y = Float32.(testimage("mandril_gray")); Y = (Y.-mean(Y))./sqrt(var(Y))
nx, ny = size(Y)
Y = reshape(Y, nx, ny, 1, 1)
for i = 1:2
    global Y = wavelet_squeeze(Y)
end
nx, ny, n_ch = size(Y)[1:3]
Y = Y |> device

# Initialize HINT layer
n_hidden = 2*n_ch
batchsize = 1
N = CouplingLayerHINT(n_ch, n_hidden; logdet=false, permute="full")∘
    CouplingLayerHINT(n_ch, n_hidden; logdet=false, permute="full")∘
    CouplingLayerHINT(n_ch, n_hidden; logdet=false, permute="full")
N = N |> device

# Fixed input
X = randn(Float32, nx, ny, n_ch, batchsize) |> device

# Loss function
loss(Y_) = 0.5f0*norm(Y-Y_)^2f0
∇loss(Y_) = Y_-Y

# Training
lr = 0.5f0
maxiter = 1000
fval = zeros(Float32, maxiter)
for i = 1:maxiter

    # Evaluate network
    Y_ = N.forward(X)

    # Evaluate objective
    fval[i] = loss(Y_)
    (mod(i, 10) == 0 || i == 1) && (print("Iteration: ", i, "; err_rel = ", sqrt(2f0*fval[i]/norm(Y)^2f0), "\n"))

    # Compute gradient
    ΔY = ∇loss(Y_)
    ΔX, Δθ, _ = N.adjointJacobian(ΔY, Y_)

    # Computing quasi-optimal step-length
    JΔθ, _ = N.jacobian(0f0.*X, Δθ, X)
    α = dot(JΔθ, ΔY)/dot(JΔθ, JΔθ)

    # Update parameters
    θ = get_params(N)
    set_params!(N, θ-lr*α.*Δθ)

end

Y_ = N.forward(X)
for i = 1:2
    global Y_ = wavelet_unsqueeze(Y_)
    global Y = wavelet_unsqueeze(Y)
end
Y = Y |> cpu
Y_ = Y_ |> cpu