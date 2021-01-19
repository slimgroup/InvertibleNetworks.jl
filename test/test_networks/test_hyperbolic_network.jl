using InvertibleNetworks, LinearAlgebra, Test, Random

# Data
nx = 16
ny = 16
n_in = 4
batchsize = 4
X_prev = randn(Float32, nx, ny, n_in, batchsize)
X_curr = randn(Float32, nx, ny, n_in, batchsize)

# Network
architecture = ((0, 6), (1, 12), (0, 24), (-1, 12), (0, 6))
α0 = .2f0
H = NetworkHyperbolic(nx, ny, n_in, batchsize, architecture; α=α0)

###################################################################################################
# Test invertibility

X_prev_, X_curr_ = H.inverse(H.forward(X_prev, X_curr)[1:2]...)
@show norm(X_curr - X_curr_), norm(X_curr_)
@test isapprox(norm(X_curr - X_curr_)/norm(X_curr), 0f0; atol=1e-2)

X_prev_, X_curr_ = H.forward(H.inverse(X_prev, X_curr)...)[1:2]
@show norm(X_curr - X_curr_), norm(X_curr_)
@test isapprox(norm(X_curr - X_curr_)/norm(X_curr), 0f0; atol=1e-2)


####################################################################################################
# Training

# Loss
function loss(H, X_prev, X_curr, ind)
    Y_prev, Y_curr, lgdet = H.forward(X_prev, X_curr)
    f = -log_likelihood(tensor_cat(Y_prev, Y_curr)) - lgdet
    ΔY = -∇log_likelihood(tensor_cat(Y_prev, Y_curr))
    ΔY_prev, ΔY_curr = tensor_split(ΔY)
    ΔX_prev, ΔX_curr, X_prev, X_curr = H.backward(ΔY_prev, ΔY_curr, Y_prev, Y_curr)
    return f, ΔX_prev, ΔX_curr, H.HL[ind].W.grad
end

# Data
X_prev = randn(Float32, nx, ny, n_in, batchsize)
X_curr = randn(Float32, nx, ny, n_in, batchsize)
X0_prev = randn(Float32, nx, ny, n_in, batchsize)
X0_curr = randn(Float32, nx, ny, n_in, batchsize)
dX_prev = X_prev - X0_prev
dX_curr = X_curr - X0_curr

# Gradient test w.r.t. input X0
Y_prev, Y_curr = H.forward(X_prev, X_curr)
f0, ΔX_prev, ΔX_curr = loss(H, X0_prev, X0_curr, 1)[1:3]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test hyperbolic network\n")
for j=1:maxiter
    f = loss(H, X0_prev + h*dX_prev,X0_curr + h*dX_curr, 1)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX_prev, ΔX_prev) - h*dot(dX_curr, ΔX_curr))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

@show rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1)
@show rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1)

# Gradient test w.r.t. weights of hyperbolic network
H = NetworkHyperbolic(nx, ny, n_in, batchsize, architecture; α=α0)
H0 = NetworkHyperbolic(nx, ny, n_in, batchsize, architecture; α=α0)
H.forward(X_prev, X_curr)
H0.forward(X_prev, X_curr)   # evaluate to initialize actnorm layer

maxiter = 6 
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

for i=1:length(H0.HL)
    h1 = 0.1f0
    print("\nGradient test invertible layer for layer $(i)\n")
    W0s = H0.HL[i].W.data
    dWg = H.HL[i].W.data - W0s; dWg *= norm(W0s)/norm(dWg)
    f01, ΔX1_prev, ΔX1_curr, ΔWg = loss(H0, X_prev, X_curr, i)
    for j=1:maxiter
        H0.HL[i].W.data = W0s + h1*dWg
        f = loss(H0, X_prev, X_curr, i)[1]
        err3[j] = abs(f - f01)
        err4[j] = abs(f - f01 - h1*dot(dWg, ΔWg))
        print(err3[j], "; ", err4[j], "\n")
        h1 = h1/2f0
    end

    @show local rate_1 = sum(err3[1:end-1]./err3[2:end])/(maxiter - 1)
    @show local rate_2 = sum(err4[1:end-1]./err4[2:end])/(maxiter - 1)
    H0.HL[i].W.data = W0s
    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=2f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=2f1)
end

###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
H = NetworkHyperbolic(nx, ny, n_in, batchsize, architecture; α=α0); H.forward(randn(Float32, nx, ny, n_in, batchsize), randn(Float32, nx, ny, n_in, batchsize))
θ = deepcopy(get_params(H))
H0 = NetworkHyperbolic(nx, ny, n_in, batchsize, architecture; α=α0); H0.forward(randn(Float32, nx, ny, n_in, batchsize), randn(Float32, nx, ny, n_in, batchsize))
θ0 = deepcopy(get_params(H0))
X_prev = randn(Float32, nx, ny, n_in, batchsize)
X_curr = randn(Float32, nx, ny, n_in, batchsize)

# Perturbation (normalized)
dθ = θ-θ0; dθ .*= norm.(θ0)./(norm.(dθ).+1f-10)
dX_prev = randn(Float32, nx, ny, n_in, batchsize); dX_prev *= norm(X_prev)/norm(dX_prev)
dX_curr = randn(Float32, nx, ny, n_in, batchsize); dX_curr *= norm(X_curr)/norm(dX_curr)

# Jacobian eval
dY_prev, dY_curr, Y_prev, Y_curr = H.jacobian(dX_prev, dX_curr, dθ, X_prev, X_curr)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(H, θ+h*dθ)
    Y_prev_, Y_curr_, _ = H.forward(X_prev + h*dX_prev, X_curr + h*dX_curr)
    err5[j] = norm(tensor_cat(Y_prev_, Y_curr_) - tensor_cat(Y_prev, Y_curr))
    err6[j] = norm(tensor_cat(Y_prev_, Y_curr_) - tensor_cat(Y_prev, Y_curr) - tensor_cat(h*dY_prev, h*dY_curr))
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test
set_params!(H, θ)
dY_prev, dY_curr, Y_prev, Y_curr = H.jacobian(dX_prev, dX_curr, dθ, X_prev, X_curr)
dY_prev_ = randn(Float32, size(dY_prev)); dY_curr_ = randn(Float32, size(dY_curr))

dX_prev_, dX_curr_, dθ_, _, _ = H.adjointJacobian(dY_prev_, dY_curr_, Y_prev, Y_curr)

a = dot(tensor_cat(dY_prev, dY_curr), tensor_cat(dY_prev_, dY_curr_))
b = dot(tensor_cat(dX_prev, dX_curr), tensor_cat(dX_prev_, dX_curr_)) + dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)