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

# Layer and initialization
AN = ActNorm(nc; logdet=false)
Y = AN.forward(X)

# Test initialization
@test isapprox(mean(Y), 0f0; atol=1f-6)
@test isapprox(var(Y), 1f0; atol=1f-3)

# Test invertibility
@test isapprox(norm(X - AN.inverse(AN.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.forward(AN.inverse(X)))/norm(X), 0f0, atol=1f-6)

# Reversed layer (all combinations)
AN_rev = reverse(AN)

@test isapprox(norm(X - AN_rev.inverse(AN_rev.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN_rev.forward(AN_rev.inverse(X)))/norm(X), 0f0, atol=1f-6)

@test isapprox(norm(X - AN_rev.forward(AN.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN_rev.inverse(AN.inverse(X)))/norm(X), 0f0, atol=1f-6)

@test isapprox(norm(X - AN.forward(AN_rev.forward(X)))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.inverse(AN_rev.inverse(X)))/norm(X), 0f0, atol=1f-6)

# Test with logdet enabled
AN = ActNorm(nc; logdet=true)
Y, lgdt = AN.forward(X)

# Test initialization
@test isapprox(mean(Y), 0f0; atol=1f-6)
@test isapprox(var(Y), 1f0; atol=1f-3)

# Test invertibility
@test isapprox(norm(X - AN.inverse(AN.forward(X)[1]))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.forward(AN.inverse(X))[1])/norm(X), 0f0, atol=1f-6)

# Reversed layer (all combinations)
AN_rev = reverse(AN)

@test isapprox(norm(X - AN_rev.inverse(AN_rev.forward(X)[1]))/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN_rev.forward(AN_rev.inverse(X))[1])/norm(X), 0f0, atol=1f-6)

@test isapprox(norm(X - AN_rev.forward(AN.forward(X)[1])[1])/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN_rev.inverse(AN.inverse(X)))/norm(X), 0f0, atol=1f-6)

@test isapprox(norm(X - AN.forward(AN_rev.forward(X)[1])[1])/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.inverse(AN_rev.inverse(X)))/norm(X), 0f0, atol=1f-6)


###############################################################################
# Gradient Test

AN = ActNorm(nc; logdet=true)
X = randn(Float32, nx, ny, nc, batchsize)
X0 = randn(Float32, nx, ny, nc, batchsize)
dX = X - X0

# Forward pass
Y = AN.forward(X)[1]

function loss(AN, X, Y)

    # Forward pass
    Y_, lgdet = AN.forward(X)

    # Residual and function value
    ΔY = Y_ - Y
    f = .5f0/batchsize*norm(ΔY)^2
    AN.logdet == true && (f -= lgdet)

    # Back propagation
    ΔX, X_ = AN.backward(ΔY./batchsize, Y_)

    # Check invertibility
    isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)

    return f, ΔX, get_grads(AN)
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
AN0 = ActNorm(nc; logdet=true); AN0.forward(randn(Float32, nx, ny, nc, batchsize))
AN_ini = deepcopy(AN0)
θ = get_params(AN_ini)
dθ = get_params(AN)-get_params(AN0)
maxiter = 6
print("\nGradient test actnorm\n")
f0, ΔX, Δθ = loss(AN0, X, Y)
h = 1f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(AN0, θ+h*dθ)
    f = loss(AN0, X, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# Gradient Test reversed layer

AN = reverse(ActNorm(nc; logdet=true))
X = randn(Float32, nx, ny, nc, batchsize)
X0 = randn(Float32, nx, ny, nc, batchsize)
dX = X - X0

# Forward pass
Y = AN.forward(X)[1]

# Gradient test for X
maxiter = 6
print("\nGradient test actnorm reverse\n")
f0, ΔX = loss(AN, X0, Y)[1:2]
h = .1f0
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(AN, X0 + h*dX, Y)[1]
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for parameters
AN0 = reverse(ActNorm(nc; logdet=true)); AN0.forward(randn(Float32, nx, ny, nc, batchsize))
AN_ini = deepcopy(AN0)
θ = get_params(AN_ini)
dθ = get_params(AN)-get_params(AN0)
maxiter = 6
print("\nGradient test actnorm reverse\n")
f0, ΔX, Δθ = loss(AN0, X, Y)
h = 1f0
err7 = zeros(Float32, maxiter)
err8 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(AN0, θ + h*dθ)
    f = loss(AN0, X, Y)[1]
    err7[j] = abs(f - f0)
    err8[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err7[j], "; ", err8[j], "\n")
    global h = h/2f0
end

@test isapprox(err7[end] / (err7[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err8[end] / (err8[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# Jacobian adjoint test

AN = ActNorm(nc; logdet=false)
AN0 = ActNorm(nc; logdet=false)
X = randn(Float32, nx, ny, nc, batchsize); Y = AN.forward(X)
X0 = randn(Float32, nx, ny, nc, batchsize); AN0.forward(X0)
Δθ = get_params(AN)-get_params(AN0)
ΔX = randn(Float32, nx, ny, nc, batchsize)

# Jacobian forward pass J(X, θ)*(ΔX, Δθ)
J = Jacobian(AN, X)
ΔY = J*(ΔX, Δθ)

# Jacobian adjoint pass adjoint(J(X, θ)))*ΔY_
ΔY_ = randn(Float32, size(ΔY)); ΔY_ = norm(ΔY)/norm(ΔY_)*ΔY_
ΔX_, Δθ_ = adjoint(J)*ΔY_

# Adjoint test
a = dot(ΔY, ΔY_)
b = dot(ΔX, ΔX_)+dot(Δθ, Δθ_)
@test isapprox(a, b; rtol=1f-4)


###############################################################################
# Hessian test

# Loss function
function loss(X::Array{Float32, 4}, Y::Array{Float32, 4}, θ::Array{Parameter, 1}, Δθ::Array{Parameter, 1})

    # Jacobian evaluation
    AN = ActNorm(nc; logdet=true); set_params!(AN, θ)
    J = Jacobian(AN, X, false)
    JΔθ, FX, logdet, HlgdetΔθ = J*Δθ

    # Gradient of f
    ΔY = FX-Y
    # f = 0.5f0*norm(ΔY)^2f0/batchsize-logdet
    JT = adjoint(J; Y=FX)
    g = (JT*ΔY)[2]/batchsize # need logdet grads

    # Loss function
    f = dot(g, Δθ)

    # Hessians
    JtJΔθ = (JT*JΔθ)[2]/batchsize
    g = JtJΔθ#-HlgdetΔθ

    return f, g

end

# Gradient/Hessian test for parameters
AN = ActNorm(nc; logdet=true)
X = randn(Float32, nx, ny, nc, batchsize); Y, _ = AN.forward(X)
θ = deepcopy(get_params(AN))

AN_ = ActNorm(nc; logdet=true)
X_ = randn(Float32, nx, ny, nc, batchsize); Y, _ = AN_.forward(X_)
AN0_ = ActNorm(nc; logdet=true);
X0_ = randn(Float32, nx, ny, nc, batchsize); Y0_, _ = AN0_.forward(X0_)
Δθ = deepcopy(get_params(AN_)-get_params(AN0_))

AN_ = ActNorm(nc; logdet=true)
X_ = randn(Float32, nx, ny, nc, batchsize); Y, _ = AN_.forward(X_)
AN0_ = ActNorm(nc; logdet=true)
X0_ = randn(Float32, nx, ny, nc, batchsize); Y0_, _ = AN0_.forward(X0_)
Δθ_ = deepcopy(get_params(AN_)-get_params(AN0_))

maxiter = 6
print("\nGradient/Hessian test actnorm\n")
f0, g0 = loss(X, Y, θ, Δθ)
h = 1f0
err9 = zeros(Float32, maxiter)
err10 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(X, Y, θ+h*Δθ_, Δθ)[1]
    err9[j] = abs(f - f0)
    err10[j] = abs(f - f0 - h*dot(g0, Δθ_))
    print(err9[j], "; ", err10[j], "\n")
    global h = h/2f0
end

@test isapprox(err9[end] / (err9[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err10[end] / (err10[1]/4^(maxiter-1)), 1f0; atol=1f1)
