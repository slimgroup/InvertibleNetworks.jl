# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Statistics


###############################################################################
# Test logdet implementation

# Input
nx = 4
ny = 4
nc = 3
batchsize = 1
X = rand(Float32, nx, ny, nc, batchsize)

# Actnorm and initialize
AN = ActNorm(nc; logdet=true)
AN.forward(X)

# Explicitely compute logdet of Jacobian through probing
# for small number of dimensions
J = zeros(Float32, Int(nx*ny*nc), Int(nx*ny*nc))
for i=1:nc
    count = 1
    for j=1:nx
        for k=1:ny
            E = zeros(Float32, nx, ny, nc, 1)
            E[k, j, i] = 1f0
            Y = AN.forward(X)[1]
            J[:, (i-1)*nx*ny + count] = vec(AN.backward(E, Y)[1])
            count += 1
        end
    end
end
lgdet1 = log(abs(det(J)))
lgdet2 = AN.forward(X)[2]
@test isapprox((lgdet1 - lgdet2)/lgdet1, 0f0; atol=1f-6)


###############################################################################
# Initialization and invertibility

# Input
nx = 28
ny = 28
nc = 4
batchsize = 1
X = rand(Float32, nx, ny, nc, batchsize)

# Layer and
AN = ActNorm(nc)
Y = AN.forward(X)

# Test initialization
@test isapprox(mean(Y), 0f0; atol=1f-6)
@test isapprox(var(Y), 1f0; atol=1f-3)

# Test invertibility
@test isapprox(norm(X - AN.inverse(AN.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.forward(AN.inverse(X)))/norm(X), 0f0, atol=1f-6)

###############################################################################
# Gradient Test

AN = ActNorm(nc; logdet=true)
X = randn(Float32, nx, ny, nc, batchsize)
X0 = randn(Float32, nx, ny, nc, batchsize)
dX = X - X0
if AN.logdet == true
    Y = AN.forward(X)[1]
else
    Y = AN.forward(X)
end

function loss(AN, X, Y)

    # Forward pass
    if AN.logdet == true
        Y_, lgdet = AN.forward(X)
    else
        Y_ = AN.forward(X)
    end

    # Residual and function value
    ΔY = Y_ - Y
    f = .5f0/batchsize*norm(ΔY)^2
    AN.logdet == true && (f -= lgdet)

    # Back propagation
    ΔX, X_ = AN.backward(ΔY./batchsize, Y)

    # Check invertibility
    isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)

    return f, ΔX, AN.s.grad, AN.b.grad
end

# Gradient test for X
maxiter = 6
print("\nGradient test actnorm\n")
f0, ΔX = loss(AN, X0, Y)[1:2]
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(AN, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for parameters
AN0 = ActNorm(nc; logdet=true); AN0.forward(X)
AN_ini = deepcopy(AN0)
ds = AN.s.data - AN0.s.data
db = AN.b.data - AN0.b.data
maxiter = 6
print("\nGradient test actnorm\n")
f0, ΔX, Δs, Δb = loss(AN0, X, Y)
h = .1f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    AN0.s.data = AN_ini.s.data + h*Δs
    AN0.b.data = AN_ini.b.data + h*Δb
    f = loss(AN0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(ds, Δs) - h*dot(db, Δb))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

