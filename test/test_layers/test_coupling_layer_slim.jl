using InvertibleNetworks, LinearAlgebra, Test

# Test affine or additive coupling layer
affine = true

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

# Modeling/imaging operator
A = randn(Float32, nd1*nd2*nd_in, nx1*nx2)

# Link function
Ψ(η) = identity(η)

# Unrolled loop
if affine
    CS = AffineCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=false, permute=true)
else
    CS = AdditiveCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=false, permute=true)
end

# Initializations
D = randn(Float32, nd1*nd2*nd_in, batchsize)
X = randn(Float32, nx1, nx2, nx_in, batchsize)

###################################################################################################

# Test invertibility
Y = CS.forward(X, D, A)
X_ = CS.inverse(Y, D, A)
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1e-6)

Y = CS.inverse(X, D, A)
X_ = CS.forward(Y, D, A)
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1e-6)

Y = CS.forward(X, D, A)
X_ = CS.backward(0f0.*Y, Y, D, A)[3]
@test isapprox(norm(X - X_)/norm(X), 0f0, atol=1e-6)


###################################################################################################

# Initializations
if affine
    CS = AffineCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=true, permute=true)
else
    CS = AdditiveCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=true, permute=true)
end

D = randn(Float32, nd1*nd2*nd_in, batchsize) .* 10f0
D0 = randn(Float32, nd1*nd2*nd_in, batchsize) .* 10f0
dD = D - D0

X = randn(Float32, nx1, nx2, nx_in, batchsize)
X0 = randn(Float32, nx1, nx2, nx_in, batchsize)
dX = X - X0

function loss(CS, X, D, A)
    Y, logdet = CS.forward(X, D, A)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, ΔD, X_ = CS.backward(ΔY, Y, D, A)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)
    return f, ΔX, ΔD, CS.RB.W1.grad
end

# Gradient test for input
f0, gX, gD = loss(CS, X0, D0, A)[1:3]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test slim layer input\n")
for j=1:maxiter
    f = loss(CS, X0 + h*dX, D0 + h*dD, A)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0  - h*dot(gX, dX) - h*dot(gD, dD))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Gradient test for weights
if affine
    CS = AffineCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=true, permute=false)
    CS0 = AffineCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=true, permute=false)
else
    CS = AdditiveCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=true, permute=false)
    CS0 = AdditiveCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=true, permute=false)
end
CS.RB.W1.data *= 10; CS0.RB.W1.data *= 10     # make weights larger
CSini = deepcopy(CS0)
dW = CS.RB.W1.data - CS0.RB.W1.data   # just test for 2 parameters

f0, gX, gD, gW = loss(CS0, X, D, A)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test slim layer weights\n")
for j=1:maxiter
    CS0.RB.W1.data = CSini.RB.W1.data + h*dW
    f = loss(CS0, X, D, A)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, gW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)