using InvertibleNetworks, LinearAlgebra, Test, Random

# Random seed
Random.seed!(100)

# Data
nx = 16
ny = 16
n_in = 3
batchsize = 4
k = 3   # kernel size
s = 1   # stride
p = 1   # padding
X = randn(Float32, nx, ny, n_in, batchsize)

# Network
nscales = 2
steps_per_scale = 4
H = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=1f-1, hidden_factor=32, ncenter=2)


###################################################################################################
# Test invertibility

X_ = H.inverse(H.forward(X)[1])
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)

X_ = H.forward(H.inverse(X))[1]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)


#############################################################################################################
# Training

# Loss
function loss(H, X)
    Y, logdet = H.forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X = H.backward(ΔY, Y)
    return f, ΔX, H.HL[1].W.data, H.AN.s.data
end

# Data
X = randn(Float32, nx, ny, n_in, batchsize)
X0 = randn(Float32, nx, ny, n_in, batchsize)
dX = X - X0

# Gradient test w.r.t. input X0
Y = H.forward(X)
f0, ΔX = loss(H, X0)[1:2]
h = 0.01f0
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


# Gradient test w.r.t. weights of hyperbolic network
H = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=1f-1, hidden_factor=32, ncenter=2)
H0 = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=1f-1, hidden_factor=32, ncenter=2)
H.forward(X)
H0.forward(X)   # evaluate to initialize actnorm layer
Hini = deepcopy(H0)

dW = H.HL[1].W.data - H0.HL[1].W.data
ds = H.AN.s.data - H0.AN.s.data

f0, ΔX, ΔW, Δs = loss(H0, X)
h = 0.01f0
maxiter = 6
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test invertible layer\n")
for j=1:maxiter
    H0.HL[1].W.data = Hini.HL[1].W.data + h*dW
    H0.AN.s.data = Hini.AN.s.data + h*ds
    f = loss(H0, X)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW)- h*dot(ds, Δs))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)