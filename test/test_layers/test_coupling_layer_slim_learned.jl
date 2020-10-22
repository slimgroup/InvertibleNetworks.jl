using InvertibleNetworks, LinearAlgebra, Test

# Input
nx1 = 16
nx2 = 16
nx_in = 2
nx_hidden = 4
batchsize = 2

# Observed data
nd1 = 14
nd2 = 8
nd_in = 1

# Additive SLIM layer with learned operator
CS = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, nd1, nd2, nd_in, nx_hidden, batchsize; logdet=false, permute=true)

# Initializations
D = randn(Float32, nd1*nd2*nd_in, batchsize)
X = randn(Float32, nx1, nx2, nx_in, batchsize)

###################################################################################################

# Test invertibility
Y = CS.forward(X, D)
X_ = CS.inverse(Y, D)
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1e-6)

Y = CS.inverse(X, D)
X_ = CS.forward(Y, D)
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1e-6)

Y = CS.forward(X, D)
X_ = CS.backward(0f0.*Y, Y, D)[3]
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1e-6)


###################################################################################################

# Initializations
CS = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, nd1, nd2, nd_in, nx_hidden, batchsize; logdet=true, permute=true)

D = randn(Float32, nd1*nd2*nd_in, batchsize) .* 10f0
D0 = randn(Float32, nd1*nd2*nd_in, batchsize) .* 10f0
dD = D - D0

X = randn(Float32, nx1, nx2, nx_in, batchsize)
X0 = randn(Float32, nx1, nx2, nx_in, batchsize)
dX = X - X0

function loss(CS, X, D)
    Y, logdet = CS.forward(X, D)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, ΔD, X_ = CS.backward(ΔY, Y, D)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-4)
    return f, ΔX, ΔD, CS.RB.W1.grad
end

# Gradient test for input
f0, gX, gD = loss(CS, X0, D0)[1:3]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test slim layer input\n")
for j=1:maxiter
    f = loss(CS, X0 + h*dX, D0 + h*dD)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0  - h*dot(gX, dX) - h*dot(gD, dD))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test for weights
CS = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, nd1, nd2, nd_in, nx_hidden, batchsize; logdet=true, permute=true)
CS0 = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, nd1, nd2, nd_in, nx_hidden, batchsize; logdet=true, permute=true)
CS.RB.W1.data *= 10; CS0.RB.W1.data *= 10     # make weights larger
CSini = deepcopy(CS0)
dW = CS.RB.W1.data - CS0.RB.W1.data   # just test for 2 parameters

f0, gX, gD, gW = loss(CS0, X, D)
h = 0.1f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test slim layer weights\n")
for j=1:maxiter
    CS0.RB.W1.data = CSini.RB.W1.data + h*dW
    f = loss(CS0, X, D)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, gW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
CS = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, nd1, nd2, nd_in, nx_hidden, batchsize; logdet=true, permute=true)
θ = deepcopy(get_params(CS))
CS0 = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, nd1, nd2, nd_in, nx_hidden, batchsize; logdet=true, permute=true)
θ0 = deepcopy(get_params(CS0))
X = randn(Float32, nx1, nx2, nx_in, batchsize)
D = randn(Float32, nd1*nd2*nd_in, batchsize).*10f0

# Perturbation
dθ = θ-θ0
dX = randn(Float32, nx1, nx2, nx_in, batchsize)
dD = randn(Float32, nd1*nd2*nd_in, batchsize).*10f0

# Jacobian eval
dY, Y, _ = CS.jacobian(dX, dD, dθ, X, D)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(CS, θ+h*dθ)
    Y_, _ = CS.forward(X+h*dX, D+h*dD)
    err5[j] = norm(Y_ - Y)
    err6[j] = norm(Y_ - Y - h*dY)
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(CS, θ)
dY, Y, _ = CS.jacobian(dX, dD, dθ, X, D)
dY_ = randn(Float32, size(dY))
dX_, dD_, dθ_, _ = CS.adjointJacobian(dY_, Y, D)
a = dot(dY, dY_)
b = dot(dX, dX_)+dot(dD, dD_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)