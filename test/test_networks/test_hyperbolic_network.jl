using InvertibleNetworks, LinearAlgebra, Test, Random

# Data
nx = 16
ny = 16
n_in = 3
batchsize = 4
X = randn(Float32, nx, ny, n_in, batchsize)

# Network
nscales = 2
steps_per_scale = 4
nc = 2
hf = 32
α0 = .2f0
H = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=α0, hidden_factor=hf, ncenter=nc)

###################################################################################################
# Test invertibility

X_ = H.inverse(H.forward(X)[1])
@show norm(X - X_), norm(X)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)

X_ = H.forward(H.inverse(X))[1]
@show norm(X - X_), norm(X)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)


####################################################################################################
# Training

# Loss
function loss(H, X)
    Y, logdet = H.forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X = H.backward(ΔY, Y)
    return f, ΔX, H.HL[1].W.grad, H.AL.s.grad
end

# Data
X = randn(Float32, nx, ny, n_in, batchsize)
X0 = randn(Float32, nx, ny, n_in, batchsize)
dX = X - X0

# Gradient test w.r.t. input X0
Y = H.forward(X)
f0, ΔX = loss(H, X0)[1:2]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test hyperbolic network\n")
for j=1:maxiter
    f = loss(H, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

@show rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1)
@show rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1)

# Gradient test w.r.t. weights of hyperbolic network
H = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=α0, hidden_factor=hf, ncenter=nc)
H0 = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=α0, hidden_factor=hf, ncenter=nc)
H.forward(X)
H0.forward(X)   # evaluate to initialize actnorm layer

W0 = H0.HL[1].W.data
s0 = H0.AL.s.data
dW = randn(Float32, size(W0)) #H.HL[1].W.data - H0.HL[1].W.data
ds = randn(Float32, size(s0)) #H.AL.s.data - H0.AL.s.data

f0, ΔX, ΔW, Δs = loss(H0, X)
h = 0.01f0
maxiter = 6 
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test invertible layer\n")

for j=1:maxiter
    H0.HL[1].W.data = W0 + h*dW
    H0.AL.s.data = s0 + h*ds
    f = loss(H0, X)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW)- h*dot(ds, Δs))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@show rate_1 = sum(err3[1:end-1]./err3[2:end])/(maxiter - 1)
@show rate_2 = sum(err4[1:end-1]./err4[2:end])/(maxiter - 1)

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=2f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=2f1)
